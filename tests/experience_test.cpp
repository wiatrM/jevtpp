#include <jevt/experience.hpp>
#include "test_harness.hpp"
#include <limits>

using namespace jevt::experience;

JEVT_TEST("context encoding preserves feature boundaries and order") {
    JEVT_REQUIRE_EQ(context_key({{"a", "b:c"}, {"d", "e"}}), context_key({{"d", "e"}, {"a", "b:c"}}));
    JEVT_REQUIRE(context_key({{"a", "b:c"}}) != context_key({{"a:b", "c"}}));
}
JEVT_TEST("shadow estimator does not replace prior after twelve samples") {
    shadow_mean model;
    for(int i=0;i<12;++i) model.observe(.4,.8);
    JEVT_REQUIRE(!model.promoted());
    JEVT_REQUIRE_EQ(model.value_or(.8), .8);
    for(int i=0;i<24;++i) model.observe(.4,.8);
    JEVT_REQUIRE(model.promoted());
    JEVT_REQUIRE_NEAR(model.value_or(.8), .4, 1e-10);
}
JEVT_TEST("prequential error is measured before learning and promotion can revoke") {
    shadow_mean model;
    for(int i=0;i<40;++i) model.observe(.4,.8);
    JEVT_REQUIRE(model.promoted());
    model.observe(.8,.8);
    JEVT_REQUIRE(model.candidate_error()>0);
    for(int i=0;i<32;++i) model.observe(.8,.8);
    JEVT_REQUIRE(!model.promoted());
    JEVT_REQUIRE_EQ(model.value_or(.8), .8);
}
JEVT_TEST("invalid and inadmissible transitions never train") {
    shadow_mean model;
    JEVT_REQUIRE(!model.observe(std::numeric_limits<double>::quiet_NaN(), .8));
    JEVT_REQUIRE(!model.observe(.4, .8, false));
    JEVT_REQUIRE_EQ(model.samples(), 0u);
}
JEVT_TEST("shadow evaluation uses exactly the bounded value that can be deployed") {
    shadow_mean model(.2, 1.);
    for(int i=0;i<40;++i) model.observe(0., .5);
    JEVT_REQUIRE(model.promoted());
    JEVT_REQUIRE_NEAR(model.candidate_error(),.2,1e-10);
    JEVT_REQUIRE_EQ(model.value_or(.5),.2);
}
JEVT_TEST("finite arithmetic overflow rejects the observation transactionally") {
    moments stats;
    JEVT_REQUIRE(stats.observe(1e308));
    JEVT_REQUIRE(!stats.observe(-1e308));
    JEVT_REQUIRE_EQ(stats.samples,1u);
    JEVT_REQUIRE_EQ(stats.mean,1e308);
    JEVT_REQUIRE_EQ(stats.m2,0.);
    stats.samples=std::numeric_limits<std::size_t>::max();
    JEVT_REQUIRE(!stats.observe(0.));
    shadow_mean shadow;
    for(int i=0;i<4;++i) JEVT_REQUIRE(shadow.observe(0.,.8));
    JEVT_REQUIRE(!shadow.observe(1e308,.8));
    JEVT_REQUIRE_EQ(shadow.samples(),4u);
    JEVT_REQUIRE_EQ(shadow.checks(),0u);
}
JEVT_TEST("outcomes are context local deduplicated and unknown stays unknown") {
    outcome_memory memory;
    JEVT_REQUIRE(memory.record("narrow", "attempt-1", outcome::failed, "no_takeoff"));
    JEVT_REQUIRE(!memory.record("narrow", "attempt-1", outcome::failed, "no_takeoff"));
    JEVT_REQUIRE(!memory.record("wide", "attempt-1", outcome::succeeded));
    JEVT_REQUIRE(memory.find("wide")==nullptr);
    memory.record("wide", "attempt-2", outcome::unknown);
    JEVT_REQUIRE_EQ(memory.find("wide")->failures,0u);
    JEVT_REQUIRE_EQ(memory.find("wide")->unknown,1u);
    JEVT_REQUIRE_EQ(memory.find("narrow")->failures,1u);
    memory.clear();JEVT_REQUIRE(memory.find("narrow")==nullptr);
}
JEVT_TEST("outcome evidence storage is bounded") {
    outcome_memory memory(2);
    memory.record("a","1",outcome::failed);
    memory.record("b","2",outcome::succeeded);
    memory.record("c","3",outcome::unknown);
    JEVT_REQUIRE_EQ(memory.records().size(),2u);
    JEVT_REQUIRE(memory.find("a")==nullptr);
}
int main() { return jevt::test::run_all("experience"); }
