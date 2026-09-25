#include <jevt/graph.hpp>
#include "test_harness.hpp"

namespace {
using R = jevt::graph_node_result;
R value(const jevt::graph_context&) { return R::success(jevt::text_state("ok")); }
bool rejected(std::vector<jevt::graph_node> nodes) {
    try { const jevt::decision_graph graph(std::move(nodes)); }
    catch (const std::invalid_argument&) { return true; }
    return false;
}
}

JEVT_TEST("DAG propagates owned named outputs in topological order") {
    std::vector<std::string> calls;
    const jevt::decision_graph graph({
        {"controller", {"ranking"}, [&](const auto& ctx) {
            calls.push_back("controller");
            JEVT_REQUIRE_EQ(ctx.dependencies().size(), 1U);
            JEVT_REQUIRE(ctx.find("goal") == nullptr);
            return R::success(jevt::text_state(ctx.at("ranking").output.content + " -> applied"));
        }},
        {"ranking", {"compact"}, [&](const auto& ctx) {
            calls.push_back("ranking");
            return R::success(jevt::text_state(ctx.at("compact").output.content + " -> jump"));
        }},
        {"goal", {}, [&](const auto& ctx) {
            calls.push_back("goal");
            JEVT_REQUIRE_EQ(ctx.snapshot().id, "frame:42");
            JEVT_REQUIRE_EQ(ctx.snapshot().input.content, "danger ahead");
            return R::success(jevt::text_state("survive"));
        }},
        {"compact", {"goal"}, [&](const auto& ctx) {
            calls.push_back("compact");
            return R::success(jevt::text_state(ctx.at("goal").output.content + ": " + ctx.snapshot().input.content));
        }},
    });
    auto trace = graph.evaluate({"frame:42", jevt::text_state("danger ahead")});
    JEVT_REQUIRE_EQ(calls, (std::vector<std::string>{"goal", "compact", "ranking", "controller"}));
    JEVT_REQUIRE_EQ(trace.at("controller").output.content, "survive: danger ahead -> jump -> applied");
    JEVT_REQUIRE_EQ(trace.edges.size(), 3U);
    JEVT_REQUIRE_EQ(trace.edges[0].from, "goal");
    JEVT_REQUIRE_EQ(trace.edges[0].to, "compact");
    for (const auto& node : trace.nodes) {
        JEVT_REQUIRE_EQ(node.snapshot_id, trace.snapshot_id);
        JEVT_REQUIRE_EQ(node.status, jevt::graph_status::succeeded);
        JEVT_REQUIRE(node.duration.count() >= 0);
        JEVT_REQUIRE(node.duration <= trace.duration);
    }
    const auto copied = trace;
    trace.nodes.clear();
    JEVT_REQUIRE_EQ(copied.at("goal").output.content, "survive");
}

JEVT_TEST("definition rejects missing dependencies duplicate names and cycles") {
    JEVT_REQUIRE(rejected({{"a", {"missing"}, value}}));
    JEVT_REQUIRE(rejected({{"a", {}, value}, {"a", {}, value}}));
    JEVT_REQUIRE(rejected({{"a", {"b"}, value}, {"b", {"a"}, value}}));
    JEVT_REQUIRE(rejected({{"a", {"a"}, value}}));
    JEVT_REQUIRE(rejected({{"a", {}, value}, {"b", {"a", "a"}, value}}));
    JEVT_REQUIRE(rejected({{"", {}, value}}));
    JEVT_REQUIRE(rejected({{"a", {}, {}}}));
}

JEVT_TEST("undeclared dependency is unavailable even if already computed") {
    const jevt::decision_graph graph({
        {"hidden", {}, value},
        {"consumer", {}, [](const auto& ctx) {
            return R::success(ctx.at("hidden").output);
        }},
    });
    const auto trace = graph.evaluate({"frame:1", jevt::text_state("input")});
    JEVT_REQUIRE_EQ(trace.at("consumer").status, jevt::graph_status::error);
    JEVT_REQUIRE(trace.at("consumer").message.find("undeclared") != std::string::npos);
}

JEVT_TEST("condition skips dependent branch but not independent branch") {
    int evaluated = 0;
    const jevt::decision_graph graph({
        {"optional", {}, [&](const auto&) { ++evaluated; return R::success(jevt::text_state("unused")); },
            [](const auto&) { return false; }},
        {"child", {"optional"}, [&](const auto&) { ++evaluated; return R::success(jevt::text_state("unused")); }},
        {"independent", {}, value},
    });
    const auto trace = graph.evaluate({"frame:1", {}});
    JEVT_REQUIRE_EQ(evaluated, 0);
    JEVT_REQUIRE_EQ(trace.at("optional").status, jevt::graph_status::skipped);
    JEVT_REQUIRE_EQ(trace.at("child").status, jevt::graph_status::skipped);
    JEVT_REQUIRE_EQ(trace.at("independent").status, jevt::graph_status::succeeded);
}

JEVT_TEST("errors block consumers and explicit fallback may recover") {
    const jevt::decision_graph graph({
        {"model", {}, [](const auto&) -> R { throw std::runtime_error("model unavailable"); }},
        {"controller", {"model"}, value},
        {"fallback", {"model"}, [](const auto& ctx) {
            JEVT_REQUIRE_EQ(ctx.at("model").status, jevt::graph_status::error);
            return R::success(jevt::text_state("safe action"));
        }, {}, false},
        {"condition_error", {}, value, [](const auto&) -> bool { throw std::runtime_error("invalid condition"); }},
    });
    const auto trace = graph.evaluate({"frame:1", {}});
    JEVT_REQUIRE_EQ(trace.at("model").message, "model unavailable");
    JEVT_REQUIRE_EQ(trace.at("controller").status, jevt::graph_status::skipped);
    JEVT_REQUIRE_EQ(trace.at("fallback").output.content, "safe action");
    JEVT_REQUIRE_EQ(trace.at("condition_error").status, jevt::graph_status::error);
}

JEVT_TEST("cached result from another snapshot is stale and never controls current frame") {
    int calls = 0;
    const jevt::decision_graph graph({
        {"cache", {}, [](const auto&) { return R::success(jevt::text_state("old action"), "frame:10"); }},
        {"controller", {"cache"}, [&](const auto&) { ++calls; return R::success(jevt::text_state("applied")); }},
    });
    const auto old = graph.evaluate({"frame:10", {}});
    JEVT_REQUIRE_EQ(old.at("controller").status, jevt::graph_status::succeeded);
    const auto current = graph.evaluate({"frame:11", {}});
    JEVT_REQUIRE_EQ(calls, 1);
    JEVT_REQUIRE_EQ(current.at("cache").status, jevt::graph_status::stale);
    JEVT_REQUIRE_EQ(current.at("cache").snapshot_id, "frame:10");
    JEVT_REQUIRE_EQ(current.at("controller").status, jevt::graph_status::stale);
    JEVT_REQUIRE_EQ(current.snapshot_id, "frame:11");
}

JEVT_TEST("expired evaluation records stale nodes without invoking callbacks") {
    int calls = 0;
    const jevt::decision_graph graph({{"a", {}, [&](const auto&) { ++calls; return R::success(jevt::text_state("value")); }}});
    const auto trace = graph.evaluate({"frame:1", {}}, {std::chrono::steady_clock::now() - std::chrono::seconds(1)});
    JEVT_REQUIRE_EQ(calls, 0);
    JEVT_REQUIRE_EQ(trace.at("a").status, jevt::graph_status::stale);
    JEVT_REQUIRE_EQ(jevt::graph_status_name(trace.at("a").status), "stale");
}

JEVT_TEST("evaluations do not leak outputs across snapshot identities") {
    const jevt::decision_graph graph({{"echo", {}, [](const auto& ctx) { return R::success(ctx.snapshot().input); }}});
    const auto first = graph.evaluate({"frame:1", jevt::json_state(R"({"x":1})")});
    const auto second = graph.evaluate({"frame:2", jevt::json_state(R"({"x":2})")});
    JEVT_REQUIRE_EQ(first.at("echo").output.content, R"({"x":1})");
    JEVT_REQUIRE_EQ(second.at("echo").output.content, R"({"x":2})");
    JEVT_REQUIRE_EQ(second.at("echo").output.kind, jevt::content_kind::json);
    bool threw = false;
    try { (void)graph.evaluate({"", {}}); } catch (const std::invalid_argument&) { threw = true; }
    JEVT_REQUIRE(threw);
}

int main() { return jevt::test::run_all("decision graph"); }
