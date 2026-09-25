#include <jevt/graph_json.hpp>
#include "test_harness.hpp"

namespace {
using json = nlohmann::json;
template<class F> bool rejected(F f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}
json declaration() {
    return {{"schema_version", "jevt.graph.v1"}, {"nodes", json::array({
        {{"id", "one"}, {"operator", "echo.v1"}, {"params", {{"suffix", "A"}}}}
    })}};
}
}
JEVT_TEST("JSON adapter rejects unknown fields schema versions and mistyped topology") {
    auto spec = declaration(); spec["readz"] = json::array();
    JEVT_REQUIRE(rejected([&] { (void)jevt::parse_graph_spec(spec); }));
    spec = declaration(); spec["schema_version"] = "future";
    JEVT_REQUIRE(rejected([&] { (void)jevt::parse_graph_spec(spec); }));
    spec = declaration(); spec["nodes"][0]["depend_on"] = json::array();
    JEVT_REQUIRE(rejected([&] { (void)jevt::parse_graph_spec(spec); }));
    spec = declaration(); spec["nodes"][0]["depends_on"] = "one";
    JEVT_REQUIRE(rejected([&] { (void)jevt::parse_graph_spec(spec); }));
    spec = declaration(); spec["nodes"] = json::array();
    JEVT_REQUIRE(rejected([&] { (void)jevt::parse_graph_spec(spec); }));
}
JEVT_TEST("loaded JSON parameters configure captured operator behavior") {
    jevt::operator_registry registry;
    registry.add("echo.v1", [](const jevt::state_value& params) {
        const auto suffix = json::parse(params.content).at("suffix").get<std::string>();
        return [suffix](const jevt::graph_context& ctx) {
            return jevt::graph_node_result::success(jevt::text_state(ctx.snapshot().input.content + suffix));
        };
    });
    auto config = declaration();
    const auto first = registry.compile(jevt::parse_graph_spec(config));
    config["nodes"][0]["params"]["suffix"] = "B";
    const auto second = registry.compile(jevt::parse_graph_spec(config));
    JEVT_REQUIRE_EQ(first.evaluate({"s1", jevt::text_state("input-")}).at("one").output.content, "input-A");
    JEVT_REQUIRE_EQ(second.evaluate({"s2", jevt::text_state("input-")}).at("one").output.content, "input-B");
    config["nodes"][0]["operator"] = "shell.exec";
    JEVT_REQUIRE(rejected([&] { (void)registry.compile(jevt::parse_graph_spec(config)); }));
}
int main() { return jevt::test::run_all("optional graph JSON"); }
