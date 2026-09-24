#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace jevt::benchmark {

struct summary {
    std::uint64_t operations{};
    double elapsed_seconds{};
    double throughput_per_second{};
    double mean_us{};
    double p50_us{};
    double p95_us{};
    double p99_us{};
    double max_us{};
};

inline double percentile(const std::vector<double>& sorted, double quantile) {
    if (sorted.empty()) {
        return 0.0;
    }
    if (quantile < 0.0 || quantile > 1.0) {
        throw std::invalid_argument("quantile must be in [0, 1]");
    }
    const auto position = quantile * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const auto fraction = position - static_cast<double>(lower);
    return sorted[lower] + (sorted[upper] - sorted[lower]) * fraction;
}

inline summary summarize(std::vector<double> latencies_us, std::uint64_t operations,
                         double elapsed_seconds) {
    std::sort(latencies_us.begin(), latencies_us.end());
    double total = 0.0;
    for (const double latency : latencies_us) {
        total += latency;
    }
    summary result;
    result.operations = operations;
    result.elapsed_seconds = elapsed_seconds;
    result.throughput_per_second = elapsed_seconds > 0.0 ? operations / elapsed_seconds : 0.0;
    result.mean_us = latencies_us.empty() ? 0.0 : total / latencies_us.size();
    result.p50_us = percentile(latencies_us, 0.50);
    result.p95_us = percentile(latencies_us, 0.95);
    result.p99_us = percentile(latencies_us, 0.99);
    result.max_us = latencies_us.empty() ? 0.0 : latencies_us.back();
    return result;
}

} // namespace jevt::benchmark

