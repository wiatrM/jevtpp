#pragma once

#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace jevt::skills {

// Unknown is intentionally not a false observation, and never grants permission.
enum class truth { unknown, false_value, true_value };
struct condition {
    std::string id;
    truth value = truth::unknown;
    std::string detail;
};

struct snapshot {
    std::string run_id;
    std::uint64_t frame = 0;
    std::uint64_t context_version = 0;
    std::uint64_t candidates_version = 0;
};

// The domain adapter increments context_version when relevant facts change.
// Candidate versions additionally invalidate asynchronous rankings after replanning.
template <class Action> struct proposal {
    std::string id;
    std::string skill_id;
    std::string source;
    Action action;
    snapshot provenance;
    std::uint64_t max_age_frames = 0;
    int priority = 0;
    double utility = 0; // preference, NOT a calibrated probability
    bool emergency = false;
    std::vector<condition> conditions;
};

enum class proposal_status { selected, admissible, prohibited, unknown, stale, invalid };
[[nodiscard]] constexpr std::string_view status_name(proposal_status value) noexcept {
    switch (value) {
    case proposal_status::selected: return "selected";
    case proposal_status::admissible: return "admissible";
    case proposal_status::prohibited: return "prohibited";
    case proposal_status::unknown: return "unknown";
    case proposal_status::stale: return "stale";
    case proposal_status::invalid: return "invalid";
    }
    return "invalid";
}
struct proposal_trace {
    std::string id;
    std::string skill_id;
    std::string source;
    proposal_status status = proposal_status::invalid;
    std::string reason;
    std::vector<condition> conditions;
};
template <class Action> struct decision {
    snapshot provenance;
    std::optional<Action> action;
    std::string selected_id;
    bool emergency = false;
    // No action means no admissible proposal, not a guarantee that noop is safe.
    std::vector<proposal_trace> trace;
};

// Pure, deterministic final arbitration. No candidate callback can mutate a later
// candidate or the emitted action. The adapter must send this action unchanged.
template <class Action>
[[nodiscard]] decision<Action> arbitrate(const snapshot& current,
                                       std::span<const proposal<Action>> proposals) {
    if (current.run_id.empty()) throw std::invalid_argument("skill snapshot requires run_id");
    std::unordered_set<std::string> ids;
    for (const auto& p : proposals) {
        if (p.id.empty() || !ids.insert(p.id).second)
            throw std::invalid_argument("skill proposal IDs must be nonempty and unique");
    }
    decision<Action> result{current, {}, {}, false, {}};
    result.trace.reserve(proposals.size());
    std::optional<std::size_t> selected;
    for (std::size_t i = 0; i < proposals.size(); ++i) {
        const auto& p = proposals[i];
        proposal_trace entry{p.id, p.skill_id, p.source, proposal_status::admissible,
                             "admissible", p.conditions};
        if (p.skill_id.empty() || p.source.empty() || !std::isfinite(p.utility)) {
            entry.status = proposal_status::invalid;
            entry.reason = "missing skill/source or nonfinite utility";
        } else if (p.provenance.run_id != current.run_id ||
                   p.provenance.context_version != current.context_version ||
                   p.provenance.candidates_version != current.candidates_version ||
                   p.provenance.frame > current.frame ||
                   current.frame - p.provenance.frame > p.max_age_frames) {
            entry.status = proposal_status::stale;
            entry.reason = "run/context/candidates changed or frame age invalid";
        } else {
            for (const auto& c : p.conditions) {
                // A known prohibition is more informative than an unknown condition.
                if (c.value == truth::false_value) {
                    entry.status = proposal_status::prohibited;
                    entry.reason = c.id + ": " + c.detail;
                    break;
                }
                if (c.value == truth::unknown && entry.status == proposal_status::admissible) {
                    entry.status = proposal_status::unknown;
                    entry.reason = c.id + ": " + c.detail;
                }
            }
        }
        result.trace.push_back(std::move(entry));
        if (result.trace.back().status != proposal_status::admissible) continue;
        if (!selected) { selected = i; continue; }
        const auto& best = proposals[*selected];
        if ((best.emergency && !p.emergency) ||
            (p.emergency == best.emergency &&
             (p.priority > best.priority || (p.priority == best.priority && p.utility > best.utility))))
            selected = i;
    }
    if (selected) {
        const auto& winner = proposals[*selected];
        result.action = winner.action;
        result.selected_id = winner.id;
        result.emergency = winner.emergency;
        result.trace[*selected].status = proposal_status::selected;
        result.trace[*selected].reason = winner.emergency
            ? "emergency only: no regular admissible proposal; safety not guaranteed"
            : "highest priority then utility; stable input-order ties";
    }
    return result;
}

enum class execution_status { ready, running, verifying, succeeded, failed, blocked, interrupted };
[[nodiscard]] constexpr std::string_view status_name(execution_status value) noexcept {
    switch (value) {
    case execution_status::ready: return "ready";
    case execution_status::running: return "running";
    case execution_status::verifying: return "verifying";
    case execution_status::succeeded: return "succeeded";
    case execution_status::failed: return "failed";
    case execution_status::blocked: return "blocked";
    case execution_status::interrupted: return "interrupted";
    }
    return "interrupted";
}
struct outcome_evidence {
    std::string attempt_id;
    snapshot observed_at;
    truth effect_observed = truth::unknown;
    std::string evidence_id;
    std::string reason;
};

// One attempt of one temporally extended skill. The domain verifier, not a model
// score or disappearance event, supplies effect_observed. A rejected proposal
// must not be started or recorded as a failed executed attempt.
class skill_execution {
public:
    [[nodiscard]] execution_status status() const noexcept { return status_; }
    [[nodiscard]] const std::string& attempt_id() const noexcept { return attempt_id_; }
    [[nodiscard]] const std::string& reason() const noexcept { return reason_; }
    [[nodiscard]] const std::optional<outcome_evidence>& evidence() const noexcept { return evidence_; }
    bool start(std::string attempt_id, snapshot at, std::span<const condition> initiation) {
        if (status_ == execution_status::running || status_ == execution_status::verifying)
            throw std::logic_error("cannot overwrite an active skill attempt");
        if (attempt_id.empty() || at.run_id.empty()) throw std::invalid_argument("attempt and run IDs required");
        attempt_id_ = std::move(attempt_id);
        started_ = std::move(at);
        evidence_.reset();
        reason_.clear();
        for (const auto& c : initiation) if (c.value != truth::true_value) {
            status_ = execution_status::blocked;
            reason_ = c.id + (c.value == truth::unknown ? ": unknown initiation" : ": prohibited initiation");
            return false;
        }
        status_ = execution_status::running;
        return true;
    }
    void begin_verification() {
        if (status_ != execution_status::running) throw std::logic_error("only a running attempt can finish execution");
        status_ = execution_status::verifying;
    }
    // Returns whether evidence was accepted, not whether the skill succeeded.
    bool verify(outcome_evidence observed) {
        if (status_ != execution_status::verifying || observed.attempt_id != attempt_id_ ||
            observed.observed_at.run_id != started_.run_id || observed.observed_at.frame <= started_.frame ||
            observed.evidence_id.empty() ||
            (evidence_ && observed.observed_at.frame < evidence_->observed_at.frame)) return false;
        reason_ = observed.reason;
        evidence_ = std::move(observed);
        if (evidence_->effect_observed == truth::true_value) status_ = execution_status::succeeded;
        if (evidence_->effect_observed == truth::false_value) status_ = execution_status::failed;
        return true; // Unknown remains verifying and is not a failure label.
    }
    void interrupt(std::string reason) {
        if (status_ != execution_status::running && status_ != execution_status::verifying)
            throw std::logic_error("only an active attempt can be interrupted");
        status_ = execution_status::interrupted;
        reason_ = std::move(reason);
    }
private:
    execution_status status_ = execution_status::ready;
    std::string attempt_id_;
    snapshot started_;
    std::string reason_;
    std::optional<outcome_evidence> evidence_;
};

} // namespace jevt::skills
