#include <jevt/diagnostics.hpp>

#include "test_harness.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
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

int main() {
    return jevt::test::run_all("diagnostics");
}
