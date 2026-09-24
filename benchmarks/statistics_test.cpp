#include "statistics.hpp"
#include "../tests/test_harness.hpp"

JEVT_TEST("percentiles use linear interpolation") {
    const std::vector<double> samples{1.0, 2.0, 3.0, 4.0, 100.0};
    const auto result = jevt::benchmark::summarize(samples, 5, 0.5);
    JEVT_REQUIRE_EQ(result.p50_us, 3.0);
    JEVT_REQUIRE_NEAR(result.p95_us, 80.8, 1e-9);
    JEVT_REQUIRE_NEAR(result.p99_us, 96.16, 1e-9);
    JEVT_REQUIRE_EQ(result.throughput_per_second, 10.0);
}

int main() {
    return jevt::test::run_all("benchmark statistics");
}
