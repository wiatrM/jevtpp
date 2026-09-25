#include "../examples/mario_dashboard/plan_ranking.hpp"
#include "test_harness.hpp"
#include <limits>

namespace {
using J = nlohmann::json;
J plan(std::string id, std::string action, double utility = 100) {
    return {{"signature", id}, {"action", action}, {"objective_utility", utility},
        {"predicted_collision", false}, {"unknown_space", false}, {"unsafe_jump_cut", false},
        {"risk_cost", .1}, {"uncertainty_cost", .1}, {"supported_end", true},
        {"continuation_failed", false}, {"continuation", {{"feasible", true}}}};
}
J state() { return {{"player", {{"grounded", true}}}, {"level", {{"world", 1}, {"stage", 3}}}}; }
J plans() { return {{"candidates", J::array({plan("good", "right"), plan("other", "left", 90)})}}; }
J offered() { return mario::ranking::offer(state(), {{"mode", "score_attack"}}, plans(), J::object()); }
J request(const J& offer) {
    return {{"frame", 100}, {"epoch", 7}, {"strategy", {
        {"source_frame", 90}, {"epoch", 7}, {"mode", "score_attack"},
        {"judgment_kind", "plan"}, {"ok", true}, {"model_plan", "good"},
        {"plan_context", offer.at("context_id")}, {"plan_scores", {{"good", .6}, {"other", .4}}}}}};
}
J risk(const J& all) {
    return {{"apply", true}, {"emergency", false}, {"budget", .5}, {"selected", all.at("candidates").at(0)}};
}
}

JEVT_TEST("plan ranking validates provenance age objective and candidate availability") {
    const auto offer = offered();
    JEVT_REQUIRE(mario::ranking::validate(request(offer), offer).at("accepted").get<bool>());
    for (int variation = 0; variation < 8; ++variation) {
        auto input = request(offer);
        auto& strategy = input["strategy"];
        if (variation == 0) strategy["epoch"] = 6;
        if (variation == 1) strategy["source_frame"] = 54;
        if (variation == 2) strategy["source_frame"] = 101;
        if (variation == 3) strategy["mode"] = "hunter";
        if (variation == 4) strategy["plan_context"] = "old-geometry";
        if (variation == 5) strategy["model_plan"] = "absent-plan";
        if (variation == 6) strategy["model_plan"] = "";
        if (variation == 7) strategy["ok"] = false;
        JEVT_REQUIRE(!mario::ranking::validate(input, offer).at("accepted").get<bool>());
    }
}

JEVT_TEST("plan ranking refuses malformed and unoffered model scores") {
    const auto offer = offered();
    for (const auto& invalid : J::array({true, "high", -.1, 1.1, nullptr})) {
        auto input = request(offer); input["strategy"]["plan_scores"]["good"] = invalid;
        JEVT_REQUIRE(!mario::ranking::validate(input, offer).at("accepted").get<bool>());
    }
    auto input = request(offer);
    input["strategy"]["plan_scores"]["unoffered"] = .9;
    JEVT_REQUIRE(!mario::ranking::validate(input, offer).at("accepted").get<bool>());
    input = request(offer);
    input["strategy"]["plan_scores"]["good"] = std::numeric_limits<double>::quiet_NaN();
    JEVT_REQUIRE(!mario::ranking::validate(input, offer).at("accepted").get<bool>());
}

JEVT_TEST("offer excludes collision unknown unsafe cut and failed continuation") {
    for (const auto* prohibited : {"predicted_collision", "unknown_space", "unsafe_jump_cut", "continuation_failed"}) {
        auto all = plans(); all["candidates"][1][prohibited] = true;
        const auto offer = mario::ranking::offer(state(), {{"mode", "score_attack"}}, all, J::object());
        JEVT_REQUIRE_EQ(offer.at("plan_candidates").size(), 1U);
        JEVT_REQUIRE_EQ(offer.at("plan_candidates")[0].at("id").get<std::string>(), "good");
    }
}

JEVT_TEST("hunter only offers a plan with predicted verified task progress") {
    auto all=plans();
    auto none=mario::ranking::offer(state(),{{"mode","hunter"}},all,
        {{"objective","eliminate_observed_hostiles"}});
    JEVT_REQUIRE(none.at("plan_candidates").empty());
    all["candidates"][1]["predicted_stomp"]=true;
    auto one=mario::ranking::offer(state(),{{"mode","hunter"}},all,
        {{"objective","eliminate_observed_hostiles"}});
    JEVT_REQUIRE_EQ(one.at("plan_candidates").size(),1U);
    JEVT_REQUIRE_EQ(one.at("plan_candidates")[0].at("id").get<std::string>(),"other");
}

JEVT_TEST("model preference cannot bypass live safety gates or failed next transfer") {
    const J gate{{"accepted", true}, {"scores", {{"good", .1}, {"other", .9}}}};
    for (int variation = 0; variation < 6; ++variation) {
        auto all = plans(); auto& unsafe = all["candidates"][1];
        if (variation == 0) unsafe["predicted_collision"] = true;
        if (variation == 1) unsafe["unknown_space"] = true;
        if (variation == 2) unsafe["unsafe_jump_cut"] = true;
        if (variation == 3) unsafe["continuation_failed"] = true;
        if (variation == 4) unsafe["risk_cost"] = .8;
        if (variation == 5) unsafe["risk_cost"] = std::numeric_limits<double>::quiet_NaN();
        const auto result = mario::ranking::apply(risk(all), all, gate, J::object());
        JEVT_REQUIRE_EQ(result.at("selected").at("signature").get<std::string>(), "good");
    }
}

JEVT_TEST("model cannot override emergency or inactive execution and hunter requires stomp") {
    const auto all = plans();
    const J gate{{"accepted", true}, {"scores", {{"good", .1}, {"other", .9}}}};
    auto emergency = risk(all); emergency["emergency"] = true;
    JEVT_REQUIRE(!mario::ranking::apply(emergency, all, gate, J::object()).at("model_rank_applied").get<bool>());
    auto inactive = risk(all); inactive["apply"] = false;
    JEVT_REQUIRE(!mario::ranking::apply(inactive, all, gate, J::object()).at("model_rank_applied").get<bool>());
    JEVT_REQUIRE(!mario::ranking::apply(risk(all), all, gate,
        {{"objective", "eliminate_observed_hostiles"}}).at("model_rank_applied").get<bool>());
    const auto applied = mario::ranking::apply(risk(all), all, gate, J::object());
    JEVT_REQUIRE_EQ(applied.at("selected").at("signature").get<std::string>(), "other");
    JEVT_REQUIRE(applied.at("model_changed_selected_action").get<bool>());
}

JEVT_TEST("candidate reorder preserves named ranking context but changed geometry invalidates it") {
    auto all = plans();
    all["candidates"][0]["landing_target"] = {{"left_x", 128}, {"y", 64}};
    all["candidates"][1]["landing_target"] = {{"left_x", 256}, {"y", 112}};
    const auto first = mario::ranking::offer(state(), {{"mode", "score_attack"}}, all, J::object());
    // New local utility changes slot order; asynchronous scores remain bound to IDs.
    all["candidates"][0]["objective_utility"] = 80;
    const auto reordered = mario::ranking::offer(state(), {{"mode", "score_attack"}}, all, J::object());
    JEVT_REQUIRE_EQ(first.at("plan_candidates")[0].at("id").get<std::string>(), "good");
    JEVT_REQUIRE_EQ(reordered.at("plan_candidates")[0].at("id").get<std::string>(), "other");
    JEVT_REQUIRE_EQ(first.at("context_id"), reordered.at("context_id"));
    const auto gate = mario::ranking::validate(request(first), reordered);
    JEVT_REQUIRE(gate.at("accepted").get<bool>());
    const auto applied = mario::ranking::apply(risk(all), all, gate, J::object());
    JEVT_REQUIRE_EQ(applied.at("selected").at("signature").get<std::string>(), "good");
    all["candidates"][0]["landing_target"]["y"] = 80;
    const auto moved = mario::ranking::offer(state(), {{"mode", "score_attack"}}, all, J::object());
    JEVT_REQUIRE(first.at("context_id") != moved.at("context_id"));
    JEVT_REQUIRE(!mario::ranking::validate(request(first), moved).at("accepted").get<bool>());
}

JEVT_TEST("offer excludes malformed nonfinite utilities before sorting but allows finite negative utility") {
    for (const auto& invalid : J::array({true, "high", nullptr,
            std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity()})) {
        auto all = plans(); all["candidates"][1]["objective_utility"] = invalid;
        const auto offer = mario::ranking::offer(state(), {{"mode", "score_attack"}}, all, J::object());
        JEVT_REQUIRE_EQ(offer.at("plan_candidates").size(), 1U);
        JEVT_REQUIRE_EQ(offer.at("plan_candidates")[0].at("id").get<std::string>(), "good");
    }
    auto all = plans(); all["candidates"][1].erase("objective_utility");
    JEVT_REQUIRE_EQ(mario::ranking::offer(state(), {{"mode", "score_attack"}}, all,
        J::object()).at("plan_candidates").size(), 1U);
    all["candidates"][1]["objective_utility"] = -100;
    JEVT_REQUIRE_EQ(mario::ranking::offer(state(), {{"mode", "score_attack"}}, all,
        J::object()).at("plan_candidates").size(), 2U);
}

JEVT_TEST("context preserves candidate target association and support width") {
    auto all = plans();
    all["candidates"][0]["landing_target"] = {{"left_x", 128}, {"right_x", 160}, {"y", 64}};
    all["candidates"][1]["landing_target"] = {{"left_x", 256}, {"right_x", 288}, {"y", 112}};
    const auto first = mario::ranking::offer(state(), {{"mode", "score_attack"}}, all, J::object());
    auto reassigned = all;
    std::swap(reassigned["candidates"][0]["landing_target"], reassigned["candidates"][1]["landing_target"]);
    const auto swapped = mario::ranking::offer(state(), {{"mode", "score_attack"}}, reassigned, J::object());
    JEVT_REQUIRE(first.at("context_id") != swapped.at("context_id"));
    JEVT_REQUIRE(!mario::ranking::validate(request(first), swapped).at("accepted").get<bool>());
    all["candidates"][0]["landing_target"]["right_x"] = 144;
    const auto narrowed = mario::ranking::offer(state(), {{"mode", "score_attack"}}, all, J::object());
    JEVT_REQUIRE(first.at("context_id") != narrowed.at("context_id"));
    JEVT_REQUIRE(!mario::ranking::validate(request(first), narrowed).at("accepted").get<bool>());
}

JEVT_TEST("selected plan survives competing rollout churn but not changed support or task") {
    auto all=plans();
    all["candidates"][0]["landing_target"]={{"left_x",128},{"right_x",160},{"y",64}};
    const auto first=mario::ranking::offer(state(),{{"mode","score_attack"}},all,J::object());
    auto input=request(first);
    input["strategy"]["plan_facts"]=first.at("context_facts");
    all["candidates"].erase(all["candidates"].begin()+1);
    all["candidates"].push_back(plan("new","jump",95));
    const auto changed=mario::ranking::offer(state(),{{"mode","score_attack"}},all,J::object());
    JEVT_REQUIRE(first.at("context_id")!=changed.at("context_id"));
    const auto gate=mario::ranking::validate(input,changed);
    JEVT_REQUIRE(gate.at("accepted").get<bool>());
    JEVT_REQUIRE_EQ(gate.at("reason").get<std::string>(),"selected_plan_and_relevant_context_verified");
    all["candidates"][0]["landing_target"]["right_x"]=144;
    const auto narrower=mario::ranking::offer(state(),{{"mode","score_attack"}},all,J::object());
    JEVT_REQUIRE(!mario::ranking::validate(input,narrower).at("accepted").get<bool>());
    auto different=state();different["level"]["stage"]=4;
    const auto next_stage=mario::ranking::offer(different,{{"mode","score_attack"}},all,J::object());
    JEVT_REQUIRE(!mario::ranking::validate(input,next_stage).at("accepted").get<bool>());
}

int main() { return jevt::test::run_all("Mario plan ranking contracts"); }
