#include <jevt/knowledge.hpp>
#include "test_harness.hpp"

namespace {
jevt::knowledge_fact fact(std::string object, std::string event, std::uint64_t tick) {
    jevt::knowledge_fact value;
    value.id = "enemy-1-motion";
    value.subject = "goomba-1";
    value.relation = "moves_left";
    value.object = std::move(object);
    value.source = "ram-tracker";
    value.event_id = std::move(event);
    value.run_id = "run-1";
    value.scope = jevt::knowledge_scope::session;
    value.tick = tick;
    return value;
}
}

JEVT_TEST("duplicate snapshot identities fail atomically") {
    jevt::knowledge_graph graph;
    graph.begin_run("run-1");
    auto original = fact("original", "original-event", 1);
    JEVT_REQUIRE(graph.ingest_ram_observation(original).accepted);
    auto duplicate = original;
    duplicate.object = "conflicting";
    duplicate.event_id = "conflicting-event";
    const std::vector<jevt::knowledge_fact> snapshot{original, duplicate};
    const auto generation = graph.generation();
    bool rejected = false;
    try { graph.restore_session_snapshot(snapshot); }
    catch (const std::invalid_argument&) { rejected = true; }
    JEVT_REQUIRE(rejected);
    JEVT_REQUIRE_EQ(graph.generation(), generation);
    JEVT_REQUIRE_EQ(graph.query(1).size(), 1u);
    JEVT_REQUIRE_EQ(graph.query(1)[0].object, "original");
}

JEVT_TEST("bounded knowledge query preserves insertion ties and returns owned copies") {
    jevt::knowledge_graph graph;
    graph.begin_run("run-1");
    for (int i = 0; i < 64; ++i) {
        auto value = fact("observed", "query-event-" + std::to_string(i), 5);
        value.id = "query-fact-" + std::to_string(i);
        value.priority = 10;
        JEVT_REQUIRE(graph.ingest_ram_observation(value).accepted);
    }
    const auto top = graph.query(5, 2);
    JEVT_REQUIRE_EQ(top.size(), 2u);
    JEVT_REQUIRE_EQ(top[0].id, "query-fact-0");
    JEVT_REQUIRE_EQ(top[1].id, "query-fact-1");
    JEVT_REQUIRE(graph.query(5, 0).empty());
    graph.begin_run("run-2");
    JEVT_REQUIRE_EQ(top[0].object, "observed");
}

JEVT_TEST("knowledge graph reactivates A after A to B to A and excludes superseded values from query") {
    jevt::knowledge_graph graph;
    graph.begin_run("run-1");
    auto a = fact("0.8 px/frame", "event-a1", 1);
    auto b = fact("0.2 px/frame", "event-b", 2);
    JEVT_REQUIRE(graph.ingest_ram_observation(a).accepted);
    JEVT_REQUIRE(graph.ingest_ram_observation(b).accepted);
    JEVT_REQUIRE_EQ(graph.status(a), "superseded");
    auto a_again = fact("0.8 px/frame", "event-a2", 3);
    JEVT_REQUIRE(graph.ingest_ram_observation(a_again).accepted);
    JEVT_REQUIRE_EQ(graph.status(a), "observed");
    JEVT_REQUIRE_EQ(graph.status(b), "superseded");
    const auto active = graph.query(3, 10);
    JEVT_REQUIRE_EQ(active.size(), 1u);
    JEVT_REQUIRE_EQ(active.front().object, "0.8 px/frame");
}

JEVT_TEST("stable proposition identity supersedes renamed or relabeled versions") {
    jevt::knowledge_graph graph;
    graph.begin_run("run-1");
    auto original = fact("moves left", "event-a", 1);
    JEVT_REQUIRE(graph.ingest_ram_observation(original).accepted);
    auto relabeled = fact("moves right", "event-b", 2);
    relabeled.subject = "Goomba type A";
    relabeled.relation = "observed_motion";
    JEVT_REQUIRE(graph.ingest_ram_observation(relabeled).accepted);
    JEVT_REQUIRE_EQ(graph.status(original), "superseded");
    const auto active = graph.query(2, 10);
    JEVT_REQUIRE_EQ(active.size(), 1u);
    JEVT_REQUIRE_EQ(active.front().subject, "Goomba type A");
    JEVT_REQUIRE_EQ(active.front().relation, "observed_motion");
    JEVT_REQUIRE_EQ(active.front().object, "moves right");
}

JEVT_TEST("knowledge graph preserves stronger evidence provenance and rejects older conflicts") {
    jevt::knowledge_graph graph;
    graph.begin_run("run-1");
    auto verified = fact("star contact defeats goomba", "verified-1", 10);
    verified.source = "confirmed-post-step-outcome";
    JEVT_REQUIRE(graph.ingest_verified_outcome(verified).accepted);

    auto weaker = fact("star contact defeats goomba", "hypothesis-1", 11);
    weaker.source = "model-hypothesis";
    JEVT_REQUIRE(graph.ingest_hypothesis(weaker).accepted);
    auto current = graph.query(11, 10);
    JEVT_REQUIRE_EQ(current.size(), 1u);
    JEVT_REQUIRE_EQ(current.front().evidence, jevt::knowledge_evidence::verified_outcome);
    JEVT_REQUIRE_EQ(current.front().source, "confirmed-post-step-outcome");

    const auto stale = graph.ingest_ram_observation(fact("old conflicting value", "late-old-event", 9));
    JEVT_REQUIRE(!stale.accepted);
    JEVT_REQUIRE_EQ(stale.reason, "stale_evidence_tick");
    current = graph.query(11, 10);
    JEVT_REQUIRE_EQ(current.size(), 1u);
    JEVT_REQUIRE_EQ(current.front().object, "star contact defeats goomba");
}

JEVT_TEST("typed observation is opt-in, keeps exact JSON bytes, budgets exactly and reports digests") {
    constexpr std::string_view question = "Choose a safe action.";
    constexpr std::string_view observation = R"({"line":"quote: \" slash: \\\\ newline: \\n"})";
    std::vector<std::string> measured;
    auto counter = [&](std::string_view text) {
        measured.emplace_back(text);
        return text.size();
    };
    jevt::knowledge_context_request legacy;
    legacy.question = question;
    legacy.schema_id = "mario.state";
    legacy.schema_revision = "state.v1";
    legacy.prompt_revision = "prompt.v1";
    legacy.observation = observation;
    legacy.count_tokens = counter;
    legacy.token_budget = 4096;
    const auto legacy_compiled = jevt::compile_knowledge_context(legacy);
    JEVT_REQUIRE(legacy_compiled.input.find("\\\"line\\\"") != std::string::npos);

    auto typed = legacy;
    typed.typed_json_observation = true;
    bool missing_formatter_rejected = false;
    try { (void)jevt::compile_knowledge_context(typed); }
    catch (const std::invalid_argument&) { missing_formatter_rejected = true; }
    JEVT_REQUIRE(missing_formatter_rejected);
    typed.format_typed_json_observation = [](std::string_view raw) { return std::string(raw); };
    const auto typed_compiled = jevt::compile_knowledge_context(typed);
    JEVT_REQUIRE(typed_compiled.input.find("\"observation\":" + std::string(observation)) != std::string::npos);
    JEVT_REQUIRE(typed_compiled.token_count < legacy_compiled.token_count);
    JEVT_REQUIRE_EQ(typed_compiled.input_digest, jevt::knowledge_detail::digest(typed_compiled.input));
    JEVT_REQUIRE_EQ(typed_compiled.tokenized_input_digest,
                    jevt::knowledge_detail::digest(std::string(question) + "\n" + typed_compiled.input));
    JEVT_REQUIRE_EQ(measured.back(), std::string(question) + "\n" + typed_compiled.input);
    JEVT_REQUIRE_EQ(typed_compiled.token_count, measured.back().size());

    typed.token_budget = typed_compiled.token_count;
    JEVT_REQUIRE(jevt::compile_knowledge_context(typed).token_count <= typed.token_budget);
    legacy.token_budget = typed.token_budget;
    bool legacy_budget_rejected = false;
    try { (void)jevt::compile_knowledge_context(legacy); }
    catch (const std::length_error&) { legacy_budget_rejected = true; }
    JEVT_REQUIRE(legacy_budget_rejected);
}

JEVT_TEST("knowledge rejects reserved identity separators without mutating state") {
    jevt::knowledge_graph graph;
    graph.begin_run("run-1");
    auto invalid = fact("value", "event", 1);
    invalid.id += '\x1f';
    const auto generation = graph.generation();
    const auto result = graph.ingest_ram_observation(invalid);
    JEVT_REQUIRE(!result.accepted);
    JEVT_REQUIRE_EQ(result.reason, "reserved_fact_separator");
    JEVT_REQUIRE_EQ(graph.generation(), generation);
    JEVT_REQUIRE_EQ(graph.size(), 0u);
}

JEVT_TEST("knowledge snapshot rejects invalid enums atomically") {
    jevt::knowledge_graph graph;
    graph.begin_run("run-1");
    auto invalid = fact("value", "event", 1);
    invalid.scope = static_cast<jevt::knowledge_scope>(99);
    JEVT_REQUIRE_EQ(graph.ingest_ram_observation(invalid).reason, "invalid_fact_scope");
    invalid.scope = jevt::knowledge_scope::session;
    invalid.evidence = static_cast<jevt::knowledge_evidence>(99);
    const auto generation = graph.generation();
    bool rejected = false;
    try { graph.restore_session_snapshot(std::span<const jevt::knowledge_fact>(&invalid, 1)); }
    catch (const std::invalid_argument&) { rejected = true; }
    JEVT_REQUIRE(rejected);
    JEVT_REQUIRE_EQ(graph.generation(), generation);
    JEVT_REQUIRE_EQ(graph.size(), 0u);
}

int main() { return jevt::test::run_all("knowledge graph"); }

