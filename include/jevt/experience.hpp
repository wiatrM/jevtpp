#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <map>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace jevt::experience {

// The adapter owns the semantics and units of each relative feature. Encoding
// is collision-free even when feature values contain separators. No implicit
// similarity or transfer across contexts is performed by this memory.
inline std::string context_key(const std::map<std::string, std::string>& features) {
    std::string key;
    for (const auto& [name, value] : features)
        key += std::to_string(name.size()) + ":" + name +
               std::to_string(value.size()) + ":" + value;
    return key;
}

struct moments {
    std::size_t samples = 0;
    double mean = 0, m2 = 0;
    bool observe(double value) {
        if (!std::isfinite(value) || samples==std::numeric_limits<std::size_t>::max()) return false;
        const auto next_samples=samples+1;
        const double delta = value - mean;
        const double next_mean=mean+delta/static_cast<double>(next_samples);
        const double next_m2=m2+delta*(value-next_mean);
        if(!std::isfinite(next_mean) || !std::isfinite(next_m2)) return false;
        samples=next_samples;mean=next_mean;m2=next_m2;
        return true;
    }
    double variance() const { return samples > 1 ? m2 / (samples - 1) : 0; }
};

// Scores each candidate forecast BEFORE learning from that observation.
// Promotion is reversible and based on a rolling paired error window, not
// merely the number of training samples. This is a prediction gate, NOT a
// policy improvement or safety guarantee; temporally adjacent samples remain
// correlated and the counts must not be advertised as independent trials.
class shadow_mean {
    struct error_pair { double prior, candidate; };
    moments observed_;
    std::deque<error_pair> checks_;
    double lower_,upper_;
    static constexpr std::size_t window = 32;
public:
    explicit shadow_mean(double lower=-std::numeric_limits<double>::infinity(),
                         double upper=std::numeric_limits<double>::infinity()) : lower_(lower),upper_(upper) {
        if(std::isnan(lower) || std::isnan(upper) || lower>upper)
            throw std::invalid_argument("invalid shadow estimator bounds");
    }
    bool observe(double value, double prior, bool admissible = true) {
        if (!admissible || !std::isfinite(value) || !std::isfinite(prior)) return false;
        auto next=observed_;
        if(!next.observe(value)) return false;
        if (observed_.samples >= 4) {
            const error_pair check{std::abs(value-prior),std::abs(value-candidate())};
            if(!std::isfinite(check.prior) || !std::isfinite(check.candidate)) return false;
            checks_.push_back(check);
            if (checks_.size() > window) checks_.pop_front();
        }
        observed_=next;return true;
    }
    std::size_t samples() const { return observed_.samples; }
    std::size_t checks() const { return checks_.size(); }
    double candidate() const { return std::clamp(observed_.mean,lower_,upper_); }
    double prior_error() const {
        double sum = 0; for (auto p : checks_) sum += p.prior;
        return checks_.empty() ? 0 : sum / checks_.size();
    }
    double candidate_error() const {
        double sum = 0; for (auto p : checks_) sum += p.candidate;
        return checks_.empty() ? 0 : sum / checks_.size();
    }
    bool promoted() const {
        std::size_t wins = 0;
        for (auto p : checks_) wins += p.candidate < p.prior;
        return checks_.size() == window && wins >= 24 &&
               candidate_error() + 1e-6 < prior_error() * .9;
    }
    double value_or(double prior) const { return promoted() ? candidate() : prior; }
};

enum class outcome { succeeded, failed, unknown };
struct evidence {
    std::size_t successes = 0, failures = 0, unknown = 0;
    std::map<std::string, std::size_t> failure_classes;
};

// Bounded, episode-owned evidence. One completed attempt contributes at most
// one label within the bounded deduplication window; missing observations are
// unknown, never negative evidence. The owner explicitly controls reset scope.
class outcome_memory {
    std::size_t capacity_;
    std::map<std::string, evidence> records_;
    std::deque<std::string> insertion_order_, seen_;
public:
    explicit outcome_memory(std::size_t capacity = 256) : capacity_(std::max<std::size_t>(1, capacity)) {}
    bool record(const std::string& context, const std::string& attempt,
                outcome result, const std::string& failure_class = {}) {
        // Attempt IDs are owner-assigned and unique within the episode. A
        // changed context label must not turn one execution into two outcomes.
        const auto& id = attempt;
        if (std::find(seen_.begin(), seen_.end(), id) != seen_.end()) return false;
        seen_.push_back(id);
        if (seen_.size() > capacity_ * 4) seen_.pop_front();
        if (!records_.contains(context)) {
            if (records_.size() >= capacity_) {
                records_.erase(insertion_order_.front()); insertion_order_.pop_front();
            }
            insertion_order_.push_back(context);
        }
        auto& value = records_[context];
        if (result == outcome::succeeded) ++value.successes;
        else if (result == outcome::unknown) ++value.unknown;
        else { ++value.failures; ++value.failure_classes[failure_class.empty() ? "unspecified" : failure_class]; }
        return true;
    }
    const evidence* find(const std::string& context) const {
        const auto it = records_.find(context);
        return it == records_.end() ? nullptr : &it->second;
    }
    const auto& records() const { return records_; }
    void clear() { records_.clear(); insertion_order_.clear(); seen_.clear(); }
};
} // namespace jevt::experience
