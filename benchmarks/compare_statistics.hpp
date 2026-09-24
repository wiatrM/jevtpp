#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace jevt::benchmark {

inline std::optional<double> percentile(std::vector<double> samples, double quantile) {
    if (!std::isfinite(quantile) || quantile <= 0 || quantile > 1)
        throw std::invalid_argument("quantile must be in (0, 1]");
    for (double value : samples) if (!std::isfinite(value) || value < 0)
        throw std::invalid_argument("latencies must be finite and nonnegative");
    if (samples.empty()) return std::nullopt;
    std::sort(samples.begin(), samples.end());
    return samples[static_cast<std::size_t>(std::ceil(quantile * static_cast<double>(samples.size()))) - 1];
}

struct parity_result {
    bool passed = false;
    double max_probability_delta = 0;
    std::size_t compared_probabilities = 0;
    std::size_t argmax_mismatches = 0;
    std::string reason;
};

// Ties choose the first maximum in both engines. A single NaN, infinity,
// missing row/score, or out-of-range probability makes a comparison invalid.
inline parity_result compare_probabilities(const std::vector<std::vector<float>>& reference,
                                          const std::vector<std::vector<float>>& candidate,
                                          double tolerance = 1e-4) {
    if (!std::isfinite(tolerance) || tolerance < 0) throw std::invalid_argument("invalid parity tolerance");
    parity_result result;
    if (reference.empty() || reference.size() != candidate.size()) {
        result.reason = "response row count mismatch or empty output";
        return result;
    }
    for (std::size_t row = 0; row < reference.size(); ++row) {
        const auto& a = reference[row]; const auto& b = candidate[row];
        if (a.empty() || a.size() != b.size()) {
            result.reason = "score count mismatch or empty row";
            return result;
        }
        for (std::size_t col = 0; col < a.size(); ++col) {
            if (!std::isfinite(a[col]) || !std::isfinite(b[col]) || a[col] < 0 || a[col] > 1 || b[col] < 0 || b[col] > 1) {
                result.reason = "nonfinite or out-of-range probability";
                return result;
            }
            result.max_probability_delta = std::max(result.max_probability_delta,
                std::abs(static_cast<double>(a[col]) - static_cast<double>(b[col])));
            ++result.compared_probabilities;
        }
        if (std::max_element(a.begin(), a.end()) - a.begin() != std::max_element(b.begin(), b.end()) - b.begin())
            ++result.argmax_mismatches;
    }
    result.passed = result.argmax_mismatches == 0 && result.max_probability_delta <= tolerance;
    if (!result.passed) result.reason = result.argmax_mismatches ? "argmax mismatch" : "probability tolerance exceeded";
    return result;
}

inline std::optional<double> throughput(std::size_t completed, double elapsed_seconds) {
    if (!std::isfinite(elapsed_seconds) || elapsed_seconds <= 0) return std::nullopt;
    return static_cast<double>(completed) / elapsed_seconds;
}

inline std::optional<double> accepted_speedup(bool parity_passed, std::size_t failures,
                                             double reference_seconds, double candidate_seconds) {
    if (!parity_passed || failures || !std::isfinite(reference_seconds) || !std::isfinite(candidate_seconds) ||
        reference_seconds <= 0 || candidate_seconds <= 0) return std::nullopt;
    return reference_seconds / candidate_seconds;
}

} // namespace jevt::benchmark
