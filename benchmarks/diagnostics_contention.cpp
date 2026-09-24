#include <jevt/diagnostics.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

// Concurrent recording with one continuously sampling reader. All per-call
// timings include clock overhead; compare builds on the same idle machine.
int main(int argc, char** argv) {
    const unsigned threads = argc > 1 ? std::stoul(argv[1]) : 4;
    const unsigned calls = argc > 2 ? std::stoul(argv[2]) : 100000;
    const unsigned capacity = argc > 3 ? std::stoul(argv[3]) : 256;
    if (!threads || !calls) return 2;
    jevt::DiagnosticsOptions options;
    options.recent_capacity = capacity;
    options.recent_sample_every = argc > 4 ? std::stoul(argv[4]) : 1;
    jevt::Diagnostics diagnostics(options);
    std::atomic<bool> start{false}, done{false};
    std::vector<std::vector<std::uint64_t>> timings(threads);
    std::vector<std::uint64_t> snapshot_timings;
    using clock = std::chrono::steady_clock;
    auto ns_since = [](auto before) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - before).count();
    };
    std::thread reader([&] {
        while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
        while (!done.load(std::memory_order_acquire)) {
            const auto before = clock::now();
            auto snapshot = diagnostics.snapshot();
            snapshot_timings.push_back(ns_since(before));
        }
    });
    std::vector<std::thread> workers;
    for (unsigned t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            auto& timing = timings[t];
            timing.reserve(calls);
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            for (unsigned i = 0; i < calls; ++i) {
                const auto before = clock::now();
                diagnostics.record_call("contention.operation", jevt::CallOutcome::success,
                                        std::chrono::microseconds{123}, "opaque-benchmark-tag");
                timing.push_back(ns_since(before));
            }
        });
    }
    const auto before = clock::now();
    start.store(true, std::memory_order_release);
    for (auto& worker : workers) worker.join();
    const auto elapsed = std::chrono::duration<double>(clock::now() - before).count();
    done.store(true, std::memory_order_release);
    reader.join();
    const auto final = diagnostics.snapshot();
    const auto expected = std::uint64_t{threads} * calls;
    if (final.total.calls != expected ||
        std::accumulate(final.total.latency_histogram.counts.begin(),
                        final.total.latency_histogram.counts.end(), std::uint64_t{}) != expected) return 1;
    std::vector<std::uint64_t> records;
    records.reserve(expected);
    for (auto& timing : timings) records.insert(records.end(), timing.begin(), timing.end());
    auto print = [](const char* name, auto& samples) {
        std::sort(samples.begin(), samples.end());
        const auto percentile = [&](std::size_t percent) {
            return samples.empty() ? 0 : samples[(samples.size() - 1) * percent / 100];
        };
        std::cout << ' ' << name << "_p50_ns=" << percentile(50)
                  << ' ' << name << "_p95_ns=" << percentile(95);
    };
    std::cout << "threads=" << threads << " capacity=" << capacity
              << " sample_every=" << options.recent_sample_every
              << " calls_per_second=" << expected / elapsed;
    print("record", records);
    print("snapshot", snapshot_timings);
    std::cout << " snapshots=" << snapshot_timings.size() << '\n';
}
