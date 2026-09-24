#include <jevt/diagnostics.hpp>

#include "test_harness.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace {

const jevt::DecisionSnapshot& decision_named(const jevt::DiagnosticsSnapshot& snapshot,
                                             std::string_view name) {
    for (const auto& decision : snapshot.decisions) {
        if (decision.decision == name) {
            return decision;
        }
    }
    throw jevt::test::failure("decision exists", __FILE__, __LINE__, std::string{name});
}

} // namespace

JEVT_TEST("diagnostics report business outcomes and useful latency percentiles") {
    jevt::Diagnostics diagnostics;
    diagnostics.record_call("support.routing", jevt::CallOutcome::success, 1ms, "billing");
    diagnostics.record_call("support.routing", jevt::CallOutcome::success, 2ms, "technical");
    diagnostics.record_call("support.routing", jevt::CallOutcome::abstain, 40ms);
    diagnostics.record_call("support.needs_human", jevt::CallOutcome::error, 4ms);

    const auto snapshot = diagnostics.snapshot();
    JEVT_REQUIRE_EQ(snapshot.total.calls, 4u);
    JEVT_REQUIRE_EQ(snapshot.total.successes, 2u);
    JEVT_REQUIRE_EQ(snapshot.total.abstains, 1u);
    JEVT_REQUIRE_EQ(snapshot.total.errors, 1u);
    JEVT_REQUIRE(snapshot.total.latency_p50_ms >= 1.0);
    JEVT_REQUIRE(snapshot.total.latency_p95_ms >= snapshot.total.latency_p50_ms);
    JEVT_REQUIRE(snapshot.total.latency_p99_ms >= snapshot.total.latency_p95_ms);

    const auto& routing = decision_named(snapshot, "support.routing");
    JEVT_REQUIRE_EQ(routing.stats.calls, 3u);
    JEVT_REQUIRE_EQ(routing.stats.abstains, 1u);
    JEVT_REQUIRE_EQ(snapshot.recent_calls.front().decision, "support.needs_human");
}

JEVT_TEST("diagnostics JSON and Prometheus exports contain stable public metrics") {
    jevt::Diagnostics diagnostics;
    diagnostics.record_call("support.routing", jevt::CallOutcome::success, 250us, "billing");

    const std::string json = diagnostics.to_json();
    JEVT_REQUIRE(json.find("\"calls\":1") != std::string::npos);
    JEVT_REQUIRE(json.find("support.routing") != std::string::npos);
    JEVT_REQUIRE(json.find("billing") != std::string::npos);

    const std::string prometheus = diagnostics.to_prometheus();
    JEVT_REQUIRE(prometheus.find("support.routing") != std::string::npos);
    JEVT_REQUIRE(prometheus.find("jevt_") != std::string::npos);
}

JEVT_TEST("diagnostics recording is lossless under concurrent callers") {
    constexpr std::size_t thread_count = 8;
    constexpr std::size_t calls_per_thread = 5'000;
    jevt::Diagnostics diagnostics({.recent_capacity = 64, .max_decisions = 32});

    std::atomic<bool> start{false};
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (std::size_t thread = 0; thread < thread_count; ++thread) {
        workers.emplace_back([&, thread] {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            const auto outcome = thread % 4 == 0 ? jevt::CallOutcome::abstain
                                                  : jevt::CallOutcome::success;
            for (std::size_t call = 0; call < calls_per_thread; ++call) {
                diagnostics.record_call("load.concurrent", outcome,
                                        std::chrono::microseconds{10 + thread});
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto& worker : workers) {
        worker.join();
    }

    const auto snapshot = diagnostics.snapshot();
    JEVT_REQUIRE_EQ(snapshot.total.calls, thread_count * calls_per_thread);
    JEVT_REQUIRE_EQ(snapshot.total.abstains, 2u * calls_per_thread);
    JEVT_REQUIRE_EQ(snapshot.total.successes, 6u * calls_per_thread);
    JEVT_REQUIRE_EQ(snapshot.recent_calls.size(), 64u);
}

JEVT_TEST("diagnostics reset creates a clean measurement window") {
    jevt::Diagnostics diagnostics;
    diagnostics.record_call("support.routing", jevt::CallOutcome::success, 1ms);
    diagnostics.reset();

    const auto snapshot = diagnostics.snapshot();
    JEVT_REQUIRE_EQ(snapshot.total.calls, 0u);
    JEVT_REQUIRE(snapshot.decisions.empty());
    JEVT_REQUIRE(snapshot.recent_calls.empty());
}

JEVT_TEST("recent trace sampling preserves exact metrics and restarts on reset") {
    jevt::DiagnosticsOptions options;
    options.recent_capacity = 2;
    options.recent_sample_every = 3;
    options.max_decisions = 1;
    options.max_tag_length = 4;
    jevt::Diagnostics diagnostics(options);
    for (int i = 0; i < 8; ++i) {
        diagnostics.record_call(i % 2 ? "dropped" : "retained",
                                i % 2 ? jevt::CallOutcome::error : jevt::CallOutcome::success,
                                1ms, "long-opaque-tag");
    }
    auto snapshot = diagnostics.snapshot();
    JEVT_REQUIRE_EQ(snapshot.total.calls, 8u);
    JEVT_REQUIRE_EQ(snapshot.total.successes, 4u);
    JEVT_REQUIRE_EQ(snapshot.total.errors, 4u);
    JEVT_REQUIRE_EQ(snapshot.total.latency_mean_ms, 1.0);
    JEVT_REQUIRE_EQ(std::accumulate(snapshot.total.latency_histogram.counts.begin(),
                                   snapshot.total.latency_histogram.counts.end(),
                                   std::uint64_t{}), 8u);
    JEVT_REQUIRE_EQ(snapshot.decisions.front().stats.calls, 4u);
    JEVT_REQUIRE_EQ(snapshot.dropped_decisions, 4u);
    JEVT_REQUIRE_EQ(snapshot.recent_calls.size(), 2u);
    JEVT_REQUIRE_EQ(snapshot.recent_calls[0].sequence, 7u);
    JEVT_REQUIRE_EQ(snapshot.recent_calls[1].sequence, 4u);
    JEVT_REQUIRE_EQ(*snapshot.recent_calls[0].tag, "long");
    diagnostics.reset();
    diagnostics.record_call("new", jevt::CallOutcome::abstain, 2ms);
    snapshot = diagnostics.snapshot();
    JEVT_REQUIRE_EQ(snapshot.total.calls, 1u);
    JEVT_REQUIRE_EQ(snapshot.total.abstains, 1u);
    JEVT_REQUIRE_EQ(snapshot.total.latency_mean_ms, 2.0);
    JEVT_REQUIRE_EQ(snapshot.dropped_decisions, 0u);
    JEVT_REQUIRE_EQ(snapshot.recent_calls.size(), 1u);
    JEVT_REQUIRE_EQ(snapshot.recent_calls.front().sequence, 1u);
}

JEVT_TEST("either retention knob can disable traces without disabling metrics") {
    for (bool disable_capacity : {false, true}) {
        jevt::DiagnosticsOptions options;
        options.recent_capacity = disable_capacity ? 0 : 256;
        options.recent_sample_every = disable_capacity ? 1 : 0;
        jevt::Diagnostics diagnostics(options);
        diagnostics.record_call("operation", jevt::CallOutcome::success, 1ms, "opaque");
        const auto snapshot = diagnostics.snapshot();
        JEVT_REQUIRE(snapshot.recent_calls.empty());
        JEVT_REQUIRE_EQ(snapshot.total.calls, 1u);
        JEVT_REQUIRE_EQ(snapshot.decisions.front().stats.calls, 1u);
        JEVT_REQUIRE_EQ(std::accumulate(snapshot.total.latency_histogram.counts.begin(),
                                       snapshot.total.latency_histogram.counts.end(),
                                       std::uint64_t{}), 1u);
    }
}

JEVT_TEST("sampled concurrent snapshots keep counters histograms and traces consistent") {
    jevt::DiagnosticsOptions options;
    options.recent_capacity = 32;
    options.recent_sample_every = 7;
    jevt::Diagnostics diagnostics(options);
    std::atomic<bool> done{false};
    std::atomic<bool> valid{true};
    std::thread reader([&] {
        while (!done.load()) {
            const auto snapshot = diagnostics.snapshot();
            const auto histogram_calls = std::accumulate(
                snapshot.total.latency_histogram.counts.begin(),
                snapshot.total.latency_histogram.counts.end(), std::uint64_t{});
            std::uint64_t decision_calls = 0;
            for (const auto& decision : snapshot.decisions) decision_calls += decision.stats.calls;
            if (histogram_calls != snapshot.total.calls || decision_calls != snapshot.total.calls ||
                snapshot.total.successes != snapshot.total.calls) valid.store(false);
            auto previous = snapshot.total.calls + 1;
            for (const auto& call : snapshot.recent_calls) {
                if (call.sequence >= previous || (call.sequence - 1) % 7 != 0) valid.store(false);
                previous = call.sequence;
            }
        }
    });
    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) {
        workers.emplace_back([&] {
            for (int i = 0; i < 3000; ++i)
                diagnostics.record_call("concurrent", jevt::CallOutcome::success, 1ms);
        });
    }
    for (auto& worker : workers) worker.join();
    done.store(true);
    reader.join();
    JEVT_REQUIRE(valid.load());
    const auto snapshot = diagnostics.snapshot();
    JEVT_REQUIRE_EQ(snapshot.total.calls, 12000u);
    JEVT_REQUIRE_EQ(snapshot.recent_calls.size(), 32u);
    JEVT_REQUIRE_EQ(snapshot.recent_calls.front().sequence, 11999u);
}

JEVT_TEST("overflow percentiles use the observed maximum and reset with the window") {
    jevt::Diagnostics diagnostics;
    const auto record_window = [&](auto ordinary, auto tail) {
        for (int i = 0; i < 90; ++i)
            diagnostics.record_call("tail", jevt::CallOutcome::success, ordinary);
        for (int i = 0; i < 10; ++i)
            diagnostics.record_call("tail", jevt::CallOutcome::success, tail);
    };
    const auto check = [](const jevt::CounterSnapshot& stats, double tail) {
        JEVT_REQUIRE_EQ(stats.calls, 100u);
        JEVT_REQUIRE_EQ(stats.latency_p50_ms, 1000.0);
        JEVT_REQUIRE_EQ(stats.latency_p95_ms, tail);
        JEVT_REQUIRE_EQ(stats.latency_p99_ms, tail);
        JEVT_REQUIRE(stats.latency_p50_ms <= stats.latency_p95_ms);
        JEVT_REQUIRE(stats.latency_p95_ms <= stats.latency_p99_ms);
    };
    record_window(900ms, 2000ms);
    auto snapshot = diagnostics.snapshot();
    check(snapshot.total, 2000.0);
    check(decision_named(snapshot, "tail").stats, 2000.0);

    diagnostics.reset();
    snapshot = diagnostics.snapshot();
    JEVT_REQUIRE_EQ(snapshot.total.latency_p50_ms, 0.0);
    JEVT_REQUIRE_EQ(snapshot.total.latency_p95_ms, 0.0);
    JEVT_REQUIRE_EQ(snapshot.total.latency_p99_ms, 0.0);

    // A mean fallback would report 569 ms for p95, below the p50 bucket's
    // 1000 ms bound. The previous window's maximum must also be forgotten.
    record_window(510ms, 1100ms);
    snapshot = diagnostics.snapshot();
    check(snapshot.total, 1100.0);
    check(decision_named(snapshot, "tail").stats, 1100.0);
}

int main() {
    return jevt::test::run_all("diagnostics");
}
