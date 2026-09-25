#include <jevt/skill.hpp>
#include <jevt/operator_registry.hpp>
#include "test_harness.hpp"
#include <limits>

namespace {
using namespace jevt::skills;
using P = proposal<std::string>;
const snapshot now{"run-1", 100, 2, 3};
P candidate(std::string id, int priority = 0) {
    return {id, "reach", "planner", id, now, 0, priority, 0, false,
            {{"path", truth::true_value, "observed clear"}}};
}
template <class F> bool throws(F f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}
}

JEVT_TEST("arbiter chooses a single admissible action and records all vetoes") {
    auto blocked = candidate("blocked", 100);
    blocked.conditions[0].value = truth::false_value;
    auto unknown = candidate("unobserved", 200);
    unknown.conditions[0].value = truth::unknown;
    const std::vector<P> candidates{candidate("baseline"), blocked, unknown, candidate("planner", 5)};
    const auto result = arbitrate<std::string>(now, candidates);
    JEVT_REQUIRE_EQ(result.selected_id, "planner");
    JEVT_REQUIRE_EQ(*result.action, "planner");
    JEVT_REQUIRE_EQ(result.trace[1].status, proposal_status::prohibited);
    JEVT_REQUIRE_EQ(result.trace[2].status, proposal_status::unknown);
    JEVT_REQUIRE_EQ(result.trace[0].status, proposal_status::admissible);
}

JEVT_TEST("arbiter rejects stale run context candidate future and old frame") {
    for (int change = 0; change < 5; ++change) {
        auto p = candidate("late");
        if (change == 0) p.provenance.run_id = "old";
        if (change == 1) p.provenance.context_version = 1;
        if (change == 2) p.provenance.candidates_version = 2;
        if (change == 3) p.provenance.frame = 101;
        if (change == 4) p.provenance.frame = 99;
        const auto result = arbitrate<std::string>(now, std::vector<P>{p});
        JEVT_REQUIRE(!result.action);
        JEVT_REQUIRE_EQ(result.trace.front().status, proposal_status::stale);
    }
    auto p = candidate("cached"); p.provenance.frame = 98; p.max_age_frames = 2;
    JEVT_REQUIRE(arbitrate<std::string>(now, std::vector<P>{p}).action.has_value());
}

JEVT_TEST("emergency cannot bypass constraints or displace a regular candidate") {
    auto emergency = candidate("brake", 1000); emergency.emergency = true;
    auto regular = candidate("normal");
    JEVT_REQUIRE_EQ(arbitrate<std::string>(now, std::vector<P>{emergency, regular}).selected_id, "normal");
    regular.conditions[0].value = truth::false_value;
    const auto selected = arbitrate<std::string>(now, std::vector<P>{emergency, regular});
    JEVT_REQUIRE(selected.emergency);
    JEVT_REQUIRE_EQ(selected.selected_id, "brake");
    emergency.conditions[0].value = truth::unknown;
    JEVT_REQUIRE(!arbitrate<std::string>(now, std::vector<P>{emergency, regular}).action);
}

JEVT_TEST("ties are stable invalid utility is rejected and IDs are unique") {
    auto bad = candidate("bad"); bad.utility = std::numeric_limits<double>::quiet_NaN();
    const auto result = arbitrate<std::string>(now, std::vector<P>{candidate("a"), candidate("b"), bad});
    JEVT_REQUIRE_EQ(result.selected_id, "a");
    JEVT_REQUIRE_EQ(result.trace.back().status, proposal_status::invalid);
    JEVT_REQUIRE(throws([&] { (void)arbitrate<std::string>(now, std::vector<P>{candidate("a"), candidate("a")}); }));
}

JEVT_TEST("unknown execution effect is never success disappearance or failed experience") {
    skill_execution attempt;
    const std::vector<condition> initiation{{"reachable", truth::true_value, "observed"}};
    JEVT_REQUIRE(attempt.start("attempt-1", now, initiation));
    JEVT_REQUIRE(throws([&] { (void)attempt.start("attempt-2", now, initiation); }));
    JEVT_REQUIRE(!attempt.verify({"attempt-1", {"run-1", 101}, truth::true_value, "ram-101", "premature"}));
    attempt.begin_verification();
    JEVT_REQUIRE(attempt.verify({"attempt-1", {"run-1", 101}, truth::unknown, "ram-101", "target unseen"}));
    JEVT_REQUIRE_EQ(attempt.status(), execution_status::verifying);
    JEVT_REQUIRE(!attempt.verify({"other", {"run-1", 102}, truth::true_value, "ram-102", "wrong attempt"}));
    JEVT_REQUIRE(!attempt.verify({"attempt-1", {"other-run", 102}, truth::true_value, "ram-102", "wrong run"}));
    JEVT_REQUIRE(!attempt.verify({"attempt-1", now, truth::true_value, "ram-100", "before action"}));
    JEVT_REQUIRE(attempt.verify({"attempt-1", {"run-1", 102}, truth::true_value, "ram-102", "effect confirmed"}));
    JEVT_REQUIRE_EQ(attempt.status(), execution_status::succeeded);
    JEVT_REQUIRE(!attempt.verify({"attempt-1", {"run-1", 103}, truth::false_value, "ram-103", "cannot rewrite terminal"}));
}

JEVT_TEST("blocked initiation and interruption do not fabricate executed failures") {
    skill_execution attempt;
    const std::vector<condition> unknown{{"support", truth::unknown, "unobserved"}};
    JEVT_REQUIRE(!attempt.start("blocked", now, unknown));
    JEVT_REQUIRE_EQ(attempt.status(), execution_status::blocked);
    JEVT_REQUIRE(!attempt.evidence());
    JEVT_REQUIRE(attempt.start("started", now, {}));
    attempt.interrupt("new run");
    JEVT_REQUIRE_EQ(attempt.status(), execution_status::interrupted);
    JEVT_REQUIRE(!attempt.evidence());
}

JEVT_TEST("same arbiter supports warehouse commands without game concepts") {
    using command = std::pair<std::string, int>;
    const std::vector<proposal<command>> candidates{
        {"dispatch", "move_pallet", "warehouse_planner", {"bay-A", 2}, now, 0, 50, 0, false,
            {{"aisle_clear", truth::unknown, "camera occluded"}}},
        {"charge", "dock", "battery_planner", {"charger-B", 0}, now, 0, 10, 0, false,
            {{"dock_clear", truth::true_value, "beacon observed"}}}};
    const auto result = arbitrate<command>(now, candidates);
    JEVT_REQUIRE_EQ(result.action->first, "charger-B");
    JEVT_REQUIRE_EQ(result.trace.front().status, proposal_status::unknown);
}

JEVT_TEST("registry reconfigures graph topology and parameters without new operators") {
    jevt::operator_registry registry;
    registry.add("echo.v1", [](const jevt::state_value& suffix) {
        return [suffix](const jevt::graph_context& ctx) {
            auto text = ctx.dependencies().empty() ? ctx.snapshot().input.content : ctx.dependencies().front()->output.content;
            return jevt::graph_node_result::success(jevt::text_state(text + suffix.content));
        };
    });
    const jevt::graph_spec spec{"jevt.graph.v1", {
        {"second", "echo.v1", {"first"}, jevt::text_state("B")},
        {"first", "echo.v1", {}, jevt::text_state("A")}}};
    const auto graph = registry.compile(spec);
    JEVT_REQUIRE_EQ(graph.evaluate({"s1", jevt::text_state("start-")}).at("second").output.content, "start-AB");
    auto changed = spec; changed.nodes[0].parameters = jevt::text_state("C");
    JEVT_REQUIRE_EQ(registry.compile(changed).evaluate({"s2", {}}).at("second").output.content, "AC");
    changed.nodes[0].operator_id = "not.registered";
    JEVT_REQUIRE(throws([&] { (void)registry.compile(changed); }));
    changed = spec; changed.nodes[1].dependencies = {"second"};
    JEVT_REQUIRE(throws([&] { (void)registry.compile(changed); }));
    changed = spec; changed.schema_version = "future.v99";
    JEVT_REQUIRE(throws([&] { (void)registry.compile(changed); }));
    changed = spec; changed.nodes[0].parameters = jevt::text_state(std::string(65537, 'x'));
    JEVT_REQUIRE(throws([&] { (void)registry.compile(changed); }));
    changed = spec; changed.nodes[0].id = std::string(129, 'x');
    JEVT_REQUIRE(throws([&] { (void)registry.compile(changed); }));
    JEVT_REQUIRE(throws([&] { registry.add("echo.v1", [](const auto&) { return jevt::operator_registry::evaluator{}; }); }));
}

int main() { return jevt::test::run_all("generic skill runtime"); }
