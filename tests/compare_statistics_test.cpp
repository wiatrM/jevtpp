#include "../benchmarks/compare_statistics.hpp"
#include "test_harness.hpp"
#include <limits>

JEVT_TEST("comparison percentiles use nearest rank and preserve empty samples") {
    using jevt::benchmark::percentile;
    JEVT_REQUIRE(!percentile({}, .5));
    JEVT_REQUIRE_EQ(*percentile({4, 1, 3, 2}, .5), 2.0);
    JEVT_REQUIRE_EQ(*percentile({4, 1, 3, 2}, .95), 4.0);
    JEVT_REQUIRE_EQ(*percentile({4, 1, 3, 2}, .99), 4.0);
    JEVT_REQUIRE_EQ(*percentile({0}, 1), 0.0);
    for (double q : {0., -1., 1.1, std::numeric_limits<double>::quiet_NaN()}) {
        bool threw = false;
        try { (void)percentile({1}, q); } catch (const std::invalid_argument&) { threw = true; }
        JEVT_REQUIRE(threw);
    }
    bool threw = false;
    try { (void)percentile({std::numeric_limits<double>::infinity()}, .5); }
    catch (const std::invalid_argument&) { threw = true; }
    JEVT_REQUIRE(threw);
}

JEVT_TEST("parity requires both probability tolerance and exact argmax") {
    using jevt::benchmark::compare_probabilities;
    auto exact = compare_probabilities({{.2f, .8f}, {.1f}}, {{.2f, .8f}, {.1f}});
    JEVT_REQUIRE(exact.passed);
    JEVT_REQUIRE_EQ(exact.compared_probabilities, 3u);
    JEVT_REQUIRE(compare_probabilities({{.2f, .8f}}, {{.20005f, .79995f}}).passed);
    auto delta = compare_probabilities({{.2f, .8f}}, {{.201f, .799f}});
    JEVT_REQUIRE(!delta.passed);
    JEVT_REQUIRE_EQ(delta.argmax_mismatches, 0u);
    auto choice = compare_probabilities({{.5f, .50001f}}, {{.50001f, .5f}});
    JEVT_REQUIRE(!choice.passed);
    JEVT_REQUIRE_EQ(choice.argmax_mismatches, 1u);
    JEVT_REQUIRE(choice.max_probability_delta < 1e-4);
    JEVT_REQUIRE(compare_probabilities({{.5f, .5f}}, {{.5f, .5f}}).passed);
}

JEVT_TEST("invalid probability output never passes parity") {
    using jevt::benchmark::compare_probabilities;
    JEVT_REQUIRE(!compare_probabilities({}, {}).passed);
    JEVT_REQUIRE(!compare_probabilities({{1}}, {}).passed);
    JEVT_REQUIRE(!compare_probabilities({{1}}, {{}}).passed);
    JEVT_REQUIRE(!compare_probabilities({{1}}, {{1, 0}}).passed);
    JEVT_REQUIRE(!compare_probabilities({{1}}, {{std::numeric_limits<float>::quiet_NaN()}}).passed);
    JEVT_REQUIRE(!compare_probabilities({{1}}, {{std::numeric_limits<float>::infinity()}}).passed);
    JEVT_REQUIRE(!compare_probabilities({{1}}, {{1.01f}}).passed);
    JEVT_REQUIRE(!compare_probabilities({{1}}, {{-.01f}}).passed);
}

JEVT_TEST("throughput uses actual elapsed time and speedup is gated") {
    using jevt::benchmark::throughput;
    using jevt::benchmark::accepted_speedup;
    JEVT_REQUIRE_EQ(*throughput(10, 2), 5.0);
    JEVT_REQUIRE_EQ(*throughput(0, 2), 0.0);
    JEVT_REQUIRE(!throughput(10, 0));
    JEVT_REQUIRE_EQ(*accepted_speedup(true, 0, 4, 2), 2.0);
    JEVT_REQUIRE(!accepted_speedup(false, 0, 4, 2));
    JEVT_REQUIRE(!accepted_speedup(true, 1, 4, 2));
    JEVT_REQUIRE(!accepted_speedup(true, 0, 4, 0));
    JEVT_REQUIRE(!accepted_speedup(true, 0, 4, std::numeric_limits<double>::infinity()));
}

int main() { return jevt::test::run_all("compare_statistics"); }
