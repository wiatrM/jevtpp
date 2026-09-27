#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace jevt {

enum class knowledge_scope { observation, run, session };
enum class knowledge_evidence { ram_observation, verified_outcome, model_hypothesis };
enum class knowledge_status { observed, confirmed, hypothesis, superseded };

[[nodiscard]] constexpr std::string_view name(knowledge_scope value) noexcept {
    switch (value) {
    case knowledge_scope::observation: return "observation";
    case knowledge_scope::run: return "run";
    case knowledge_scope::session: return "session";
    }
    return "observation";
}
[[nodiscard]] constexpr std::string_view name(knowledge_evidence value) noexcept {
    switch (value) {
    case knowledge_evidence::ram_observation: return "ram_observation";
    case knowledge_evidence::verified_outcome: return "verified_outcome";
    case knowledge_evidence::model_hypothesis: return "model_hypothesis";
    }
    return "model_hypothesis";
}

struct knowledge_fact {
    std::string id;
    std::string subject;
    std::string relation;
    std::string object;
    std::string source;
    // Optional opaque JSON object payload. The generic graph preserves it as
    // text so this header remains independent of a JSON library; adapters must
    // validate/canonicalize it before ingestion.
    std::string attributes_json;
    std::string event_id;
    std::string run_id;
    knowledge_scope scope = knowledge_scope::run;
    knowledge_evidence evidence = knowledge_evidence::model_hypothesis;
    std::uint64_t tick = 0;
    std::optional<std::uint64_t> expires_at;
    std::uint16_t priority = 0;
};

struct knowledge_limits {
    std::size_t facts = 4096;
    std::size_t deduplication_ids = 8192;
    std::size_t field_bytes = 512;
};

struct knowledge_ingest_result {
    bool accepted = false;
    std::string reason;
    std::uint64_t generation = 0;
};

// Bounded evidence graph for live observations and cross-run learning.
// Inputs are provenance-tagged records: model hypotheses remain hypotheses.
// The observation/outcome entry points are attestations by the caller after
// validating the corresponding RAM event or completed action feedback; this
// graph stores provenance but does not independently verify the game state.
// Call begin_run at GAME OVER/explicit reset, not on stage changes.
class knowledge_graph {
public:
    explicit knowledge_graph(knowledge_limits limits = {}) : limits_(limits) {
        if (!limits_.facts || !limits_.deduplication_ids || !limits_.field_bytes)
            throw std::invalid_argument("knowledge graph limits must be positive");
    }

    void begin_run(std::string run_id, bool retain_session = true) {
        if (run_id.empty() || run_id.size() > limits_.field_bytes ||
            run_id.find('\x1f') != std::string::npos)
            throw std::invalid_argument("run ID is empty or too long");
        current_run_ = std::move(run_id);
        auto out = facts_.begin();
        bool changed = false;
        for (auto it = facts_.begin(); it != facts_.end(); ++it) {
            if (it->scope != knowledge_scope::session || !retain_session) {
                superseded_.erase(key(*it));
                changed = true;
                continue;
            }
            if (out != it) *out = std::move(*it);
            ++out;
        }
        facts_.erase(out, facts_.end());
        seen_events_.clear();
        event_order_.clear();
        if (changed) ++generation_;
    }

    // Backward-compatible general entry point is intentionally hypothesis-only.
    // A caller cannot promote a model-generated label by setting the enum field.
    [[nodiscard]] knowledge_ingest_result ingest(knowledge_fact fact) {
        if (fact.evidence != knowledge_evidence::model_hypothesis)
            return {false, "evidence_requires_attested_ingest_path", generation_};
        return ingest_record(std::move(fact));
    }

    [[nodiscard]] knowledge_ingest_result ingest_hypothesis(knowledge_fact fact) {
        fact.evidence = knowledge_evidence::model_hypothesis;
        return ingest_record(std::move(fact));
    }

    // Call only for a directly observed RAM event. The caller owns the
    // validation boundary; source and event_id identify the evidence supplied.
    [[nodiscard]] knowledge_ingest_result ingest_ram_observation(knowledge_fact fact) {
        fact.evidence = knowledge_evidence::ram_observation;
        return ingest_record(std::move(fact));
    }

    // Call only after completed action feedback has verified the outcome.
    // A prediction or disappearance without the required event is not enough.
    [[nodiscard]] knowledge_ingest_result ingest_verified_outcome(knowledge_fact fact) {
        fact.evidence = knowledge_evidence::verified_outcome;
        return ingest_record(std::move(fact));
    }

private:
    [[nodiscard]] knowledge_ingest_result ingest_record(knowledge_fact fact) {
        if (auto invalid = validate(fact); !invalid.empty())
            return {false, std::move(invalid), generation_};
        if (fact.scope != knowledge_scope::session && fact.run_id != current_run_)
            return {false, "wrong_run_scope", generation_};
        if (fact.scope == knowledge_scope::observation &&
            (!fact.expires_at || *fact.expires_at < fact.tick))
            return {false, "observation_needs_valid_expiry", generation_};
        // Async completions may arrive out of order. Never let an older tick
        // supersede a newer value for the same logical fact.
        for (const auto& previous : facts_) {
            const bool same_identity = identity_key(previous) == identity_key(fact);
            if (same_identity && fact.tick < previous.tick)
                return {false, "stale_evidence_tick", generation_};
        }
        const auto event_key = fact.run_id + ":" + fact.event_id;
        if (!seen_events_.insert(event_key).second)
            return {false, "duplicate_event", generation_};
        event_order_.push_back(event_key);
        while (event_order_.size() > limits_.deduplication_ids) {
            seen_events_.erase(event_order_.front());
            event_order_.erase(event_order_.begin());
        }

        // A logical fact changing its object creates a new active version.
        // Re-observing an older object makes it active again and supersedes
        // every conflicting version (A -> B -> A must not leave B current).
        auto matching = facts_.end();
        for (auto it = facts_.begin(); it != facts_.end(); ++it) {
            const bool same_identity = identity_key(*it) == identity_key(fact);
            if (!same_identity) continue;
            if (key(*it) == key(fact)) matching = it;
            else superseded_.insert(key(*it));
        }
        if (matching != facts_.end()) {
            auto& previous = *matching;
            superseded_.erase(key(previous));
            previous.tick = fact.tick;
            previous.expires_at = fact.expires_at;
            previous.priority = std::max(previous.priority, fact.priority);
            // Keep provenance from the strongest evidence. Equal-strength
            // evidence may refresh provenance; weaker hypotheses cannot make
            // a verified result appear model-generated (or vice versa).
            if (evidence_rank(fact.evidence) >= evidence_rank(previous.evidence)) {
                previous.evidence = fact.evidence;
                previous.source = std::move(fact.source);
                previous.event_id = std::move(fact.event_id);
            }
            ++generation_;
            return {true, "evidence_appended", generation_};
        }
        if (facts_.size() == limits_.facts) {
            auto removable = std::find_if(facts_.begin(), facts_.end(), [](const auto& value) {
                return value.scope != knowledge_scope::session;
            });
            if (removable == facts_.end()) removable = facts_.begin();
            superseded_.erase(key(*removable));
            facts_.erase(removable);
        }
        facts_.push_back(std::move(fact));
        ++generation_;
        return {true, "fact_ingested", generation_};
    }

public:

    [[nodiscard]] std::vector<knowledge_fact> query(
        std::uint64_t tick, std::size_t limit = 128,
        std::string_view subject = {}, std::string_view relation = {}) const {
        if (limit == 0) return {};
        std::vector<const knowledge_fact*> candidates;
        candidates.reserve(facts_.size());
        for (const auto& fact : facts_) {
            // Historical versions remain inspectable through `status`, but
            // active model context must not present superseded values as live.
            if (superseded_.contains(key(fact))) continue;
            if (fact.scope == knowledge_scope::observation &&
                (!fact.expires_at || tick > *fact.expires_at)) continue;
            if (!subject.empty() && fact.subject != subject) continue;
            if (!relation.empty() && fact.relation != relation) continue;
            candidates.push_back(&fact);
        }
        std::stable_sort(candidates.begin(), candidates.end(), [](const auto* left, const auto* right) {
            if (left->priority != right->priority) return left->priority > right->priority;
            return left->tick > right->tick;
        });
        std::vector<knowledge_fact> result;
        result.reserve(std::min(limit, candidates.size()));
        for (std::size_t i = 0; i < std::min(limit, candidates.size()); ++i)
            result.push_back(*candidates[i]);
        return result;
    }

    [[nodiscard]] std::string status(const knowledge_fact& fact) const {
        const auto fact_key = key(fact);
        if (superseded_.contains(fact_key)) return "superseded";
        const auto stored = std::find_if(facts_.begin(), facts_.end(), [&](const auto& value) {
            return key(value) == fact_key;
        });
        const auto evidence = stored == facts_.end() ? fact.evidence : stored->evidence;
        if (evidence == knowledge_evidence::model_hypothesis) return "hypothesis";
        if (evidence == knowledge_evidence::verified_outcome) return "confirmed";
        return "observed";
    }
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
    [[nodiscard]] std::size_t size() const noexcept { return facts_.size(); }
    [[nodiscard]] std::string_view run_id() const noexcept { return current_run_; }
    // The application may serialize this bounded session snapshot and restore
    // it in a fresh process. Run/observation facts are deliberately excluded.
    [[nodiscard]] std::vector<knowledge_fact> session_snapshot() const {
        std::vector<knowledge_fact> result;
        for (const auto& fact : facts_)
            if (fact.scope == knowledge_scope::session && !superseded_.contains(key(fact)))
                result.push_back(fact);
        return result;
    }

    void restore_session_snapshot(std::span<const knowledge_fact> snapshot) {
        if (current_run_.empty())
            throw std::logic_error("begin_run must precede session snapshot restore");
        const auto retained_run_facts = static_cast<std::size_t>(std::count_if(
            facts_.begin(), facts_.end(), [](const auto& fact) {
                return fact.scope != knowledge_scope::session;
            }));
        if (snapshot.size() + retained_run_facts > limits_.facts)
            throw std::length_error("session snapshot exceeds knowledge graph limit");
        std::unordered_set<std::string> identities;
        identities.reserve(snapshot.size());
        for (const auto& fact : snapshot) {
            if (fact.scope != knowledge_scope::session)
                throw std::invalid_argument("session snapshot contains non-session fact");
            if (auto invalid = validate(fact); !invalid.empty())
                throw std::invalid_argument("invalid session snapshot fact: " + invalid);
            if (!identities.insert(identity_key(fact)).second)
                throw std::invalid_argument("session snapshot contains duplicate fact identity");
        }
        for (const auto& fact : facts_)
            if (fact.scope == knowledge_scope::session) superseded_.erase(key(fact));
        facts_.erase(std::remove_if(facts_.begin(), facts_.end(), [](const auto& fact) {
            return fact.scope == knowledge_scope::session;
        }), facts_.end());
        for (const auto& fact : snapshot) facts_.push_back(fact);
        ++generation_;
    }

    void clear() {
        facts_.clear(); seen_events_.clear(); event_order_.clear(); superseded_.clear();
        current_run_.clear(); ++generation_;
    }

private:
    static unsigned evidence_rank(knowledge_evidence evidence) noexcept {
        switch (evidence) {
        case knowledge_evidence::ram_observation: return 1;
        case knowledge_evidence::verified_outcome: return 2;
        case knowledge_evidence::model_hypothesis: return 0;
        }
        return 0;
    }

    static std::string key(const knowledge_fact& fact) {
        return fact.run_id + '\x1f' + std::string(name(fact.scope)) + '\x1f' + fact.id +
               '\x1f' + fact.subject + '\x1f' + fact.relation + '\x1f' + fact.object;
    }
    // A stable ID names one proposition; changes to its labels/value are
    // versions, not independent facts. Keep key() version-specific so old
    // versions remain inspectable as superseded.
    static std::string identity_key(const knowledge_fact& fact) {
        return fact.run_id + '\x1f' + std::string(name(fact.scope)) + '\x1f' + fact.id;
    }
    std::string validate(const knowledge_fact& fact) const {
        if (fact.scope != knowledge_scope::observation && fact.scope != knowledge_scope::run &&
            fact.scope != knowledge_scope::session) return "invalid_fact_scope";
        if (fact.evidence != knowledge_evidence::ram_observation &&
            fact.evidence != knowledge_evidence::verified_outcome &&
            fact.evidence != knowledge_evidence::model_hypothesis) return "invalid_fact_evidence";
        for (const auto* value : {&fact.id, &fact.subject, &fact.relation, &fact.object,
                                  &fact.source, &fact.event_id, &fact.run_id}) {
            if (value->empty()) return "required_fact_field_missing";
            if (value->size() > limits_.field_bytes) return "fact_field_too_long";
            if (value->find('\x1f') != std::string::npos) return "reserved_fact_separator";
        }
        if (fact.attributes_json.size() > limits_.field_bytes) return "fact_field_too_long";
        if (fact.priority > 1000) return "priority_out_of_range";
        if (fact.scope != knowledge_scope::session && current_run_.empty()) return "run_not_started";
        return {};
    }

    knowledge_limits limits_;
    std::string current_run_;
    std::vector<knowledge_fact> facts_;
    std::unordered_set<std::string> seen_events_;
    std::vector<std::string> event_order_;
    std::unordered_set<std::string> superseded_;
    std::uint64_t generation_ = 0;
};

struct knowledge_context_request {
    std::string_view question;
    std::string_view schema_id;
    std::string_view schema_revision;
    std::string_view prompt_revision;
    std::string_view observation;
    std::span<const knowledge_fact> facts;
    std::size_t token_budget = 0;
    std::function<std::size_t(std::string_view)> count_tokens;
    // The default retains the historical JSON-string encoding. Typed JSON is
    // opt-in and requires a formatter supplied by a JSON-aware adapter. That
    // formatter must validate one complete JSON value and return its compact
    // serialization; core JevT++ does not require a JSON parser.
    bool typed_json_observation = false;
    std::function<std::string(std::string_view)> format_typed_json_observation;
};

struct compiled_knowledge_context {
    std::string question;
    std::string input;
    std::string schema_revision;
    std::string prompt_revision;
    std::vector<std::string> admitted_fact_ids;
    std::vector<std::string> omitted_fact_ids;
    std::size_t token_count = 0;
    std::string input_digest;
    std::string tokenized_input_digest;
    std::string digest_algorithm = "fnv1a64-noncryptographic";
};

namespace knowledge_detail {
[[nodiscard]] inline std::string digest(std::string_view text) {
    std::uint64_t value = 14695981039346656037ULL;
    for (const unsigned char byte : text) {
        value ^= byte;
        value *= 1099511628211ULL;
    }
    constexpr char digits[] = "0123456789abcdef";
    std::array<char, 16> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        const auto shift = static_cast<unsigned>((result.size() - index - 1) * 4);
        result[index] = digits[(value >> shift) & 0x0f];
    }
    return {result.data(), result.size()};
}
} // namespace knowledge_detail

// Compile the full observation plus whole evidence records against the actual
// model tokenizer. The prompt is returned byte-for-byte unchanged. Mandatory
// fields never get truncated; optional knowledge is ranked and admitted whole.
[[nodiscard]] inline compiled_knowledge_context compile_knowledge_context(
    const knowledge_context_request& request) {
    if (request.question.empty() || request.schema_id.empty() ||
        request.schema_revision.empty() || request.prompt_revision.empty() ||
        request.observation.empty() || !request.count_tokens || request.token_budget == 0)
        throw std::invalid_argument("incomplete knowledge context contract");
    if (request.observation.size() > 1048576)
        throw std::invalid_argument("knowledge observation exceeds 1 MiB");
    std::string observation(request.observation);
    if (request.typed_json_observation) {
        if (!request.format_typed_json_observation)
            throw std::invalid_argument("typed JSON observation requires a validating formatter");
        observation = request.format_typed_json_observation(request.observation);
        if (observation.empty() || observation.size() > 1048576)
            throw std::invalid_argument("typed JSON formatter returned empty or oversized JSON");
    }
    const auto escape = [](std::string_view text) {
        std::string out; out.reserve(text.size() + 2); out.push_back('"');
        for (const unsigned char c : text) {
            switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    constexpr char hex[] = "0123456789abcdef";
                    out += "\\u00"; out.push_back(hex[c >> 4]); out.push_back(hex[c & 15]);
                } else out.push_back(static_cast<char>(c));
            }
        }
        out.push_back('"'); return out;
    };
    const auto make_input = [&](const std::vector<const knowledge_fact*>& admitted) {
        std::string out = "{\"schema\":" + escape(request.schema_id) +
            ",\"observation\":" +
            (request.typed_json_observation ? observation : escape(observation)) + ",\"knowledge\":[";
        for (std::size_t i = 0; i < admitted.size(); ++i) {
            if (i) out.push_back(',');
            const auto& fact = *admitted[i];
            // Only send semantic content to the model. Stable IDs, run/event
            // provenance, lifetime and source remain in the native graph and
            // response audit; repeating them in every head wastes scarce
            // sequence tokens without helping LAYA reason about the fact.
            const auto evidence = fact.evidence == knowledge_evidence::model_hypothesis ?
                "hypothesis" : fact.evidence == knowledge_evidence::verified_outcome ? "confirmed" : "observed";
            out += "{\"subject\":" + escape(fact.subject) +
                ",\"relation\":" + escape(fact.relation) + ",\"object\":" + escape(fact.object) +
                ",\"evidence\":" + escape(evidence) +
                (fact.attributes_json.empty() ? std::string{} : ",\"attributes\":" + escape(fact.attributes_json)) + "}";
        }
        out += "]}"; return out;
    };
    std::vector<const knowledge_fact*> candidates;
    candidates.reserve(request.facts.size());
    for (const auto& fact : request.facts) candidates.push_back(&fact);
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto* left, const auto* right) {
        if (left->priority != right->priority) return left->priority > right->priority;
        return left->tick > right->tick;
    });
    compiled_knowledge_context result;
    result.question = std::string(request.question);
    result.schema_revision = std::string(request.schema_revision);
    result.prompt_revision = std::string(request.prompt_revision);
    auto measure = [&](const std::string& input) {
        return request.count_tokens(result.question + "\n" + input);
    };
    result.input = make_input({});
    result.token_count = measure(result.input);
    if (result.token_count > request.token_budget)
        throw std::length_error("mandatory prompt, schema or observation exceeds model token budget");
    std::vector<const knowledge_fact*> admitted;
    for (const auto* candidate : candidates) {
        admitted.push_back(candidate);
        auto proposed = make_input(admitted);
        const auto tokens = measure(proposed);
        if (tokens > request.token_budget) {
            admitted.pop_back();
            result.omitted_fact_ids.push_back(candidate->id);
            continue;
        }
        result.input = std::move(proposed);
        result.token_count = tokens;
        result.admitted_fact_ids.push_back(candidate->id);
    }
    result.input_digest = knowledge_detail::digest(result.input);
    result.tokenized_input_digest = knowledge_detail::digest(result.question + "\n" + result.input);
    return result;
}

} // namespace jevt

