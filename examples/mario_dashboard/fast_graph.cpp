// Persistent frame-rate C++ graph. Open-JEV is an asynchronous intent producer.
#include <jevt/graph.hpp>
#include <nlohmann/json.hpp>
#include "fast_controller.hpp"
#include "experience_memory.hpp"
#include "adaptive_agent.hpp"
#include "plan_ranking.hpp"
#include <chrono>
#include <cmath>
#include <iostream>
#include <string>

namespace {
using Json = nlohmann::json;
using Context = jevt::graph_context;
using Result = jevt::graph_node_result;

Json output(const Context& context, std::string_view name) {
    return Json::parse(context.at(name).output.content);
}
Result success(const Json& value) { return Result::success(jevt::json_state(value.dump())); }

Json digital_twin(const Json& physics, const Json& geometry, const Json& threats, const Json& experience) {
    constexpr int horizon = 8;
    const auto player = physics.at("player"), collision = geometry.at("collision");
    const auto trajectory = physics.at("trajectory");
    const auto observed_number = [](const Json& value, const char* key) -> std::optional<double> {
        const auto it = value.find(key);
        if (it == value.end() || !it->is_number()) return std::nullopt;
        const auto number = it->get<double>();
        return std::isfinite(number) ? std::optional<double>(number) : std::nullopt;
    };
    const auto player_vx = observed_number(player, "vx"), player_x = observed_number(player, "x");
    const bool grounded = player.value("grounded", false), ram = collision.value("available", false);
    bool limited = !grounded || !player_vx;
    Json enemy{{"present", false}, {"kind_id", nullptr}, {"distance_px", nullptr},
               {"velocity_px_per_frame", nullptr}, {"velocity_source", "unknown"}, {"velocity_samples", 0},
               {"closing_speed_px_per_frame", nullptr}, {"contact_eta_frames", nullptr}, {"projected_distance_px", nullptr}};
    const auto enemies = threats.at("hazard").value("upcoming_enemies", Json::array());
    if (!enemies.empty()) {
        const auto& current = enemies.front();
        const auto distance = observed_number(current, "relative_x_pixels");
        const int kind = current.value("kind_id", -1);
        enemy["present"] = true;
        enemy["kind_id"] = kind;
        if (distance) enemy["distance_px"] = *distance;
        std::optional<double> relative_vx;
        if (ram && current.value("velocity_observed", false)) {
            relative_vx = observed_number(current, "relative_velocity_x");
            if (relative_vx && std::abs(*relative_vx) > 32) relative_vx.reset();
            if (relative_vx) {
                enemy["velocity_source"] = "current_ram";
                enemy["velocity_samples"] = 1;
                if (player_vx) enemy["velocity_px_per_frame"] = *relative_vx + *player_vx;
            }
        }
        if (!relative_vx && ram && player_vx) {
            const auto speeds = experience.value("enemy_speeds", Json::object());
            const auto key = std::to_string(kind);
            if (speeds.contains(key)) {
                const auto& statistic = speeds.at(key);
                const auto mean = observed_number(statistic, "mean");
                const auto count = statistic.value("samples", 0);
                if (count >= 5 && mean && std::abs(*mean) <= 16) {
                    relative_vx = *mean - *player_vx;
                    enemy["velocity_px_per_frame"] = *mean;
                    enemy["velocity_source"] = "empirical_same_type";
                    enemy["velocity_samples"] = count;
                    limited = true;
                }
            }
        }
        if (relative_vx && distance) {
            const double closing = std::max(0., -*relative_vx);
            enemy["closing_speed_px_per_frame"] = closing;
            enemy["projected_distance_px"] = std::max(0., *distance + *relative_vx * horizon);
            if (closing > 0 && *distance >= 0) enemy["contact_eta_frames"] = *distance / closing;
        } else limited = true;
    }
    const int jump_samples = experience.value("jump_samples", 0);
    Json landing{{"status", "unknown"}, {"eta_frames", nullptr}, {"projected_x", nullptr},
                 {"source", "insufficient_samples"}, {"samples", jump_samples}};
    if (ram && grounded) {
        landing["status"] = "grounded"; landing["eta_frames"] = 0; landing["source"] = "observed_ground";
        if (player_x) landing["projected_x"] = *player_x;
    } else if (ram && jump_samples >= 5 && player_vx && player_x &&
               !trajectory.value("engaging_enemy", false)) {
        const auto mean = observed_number(experience, "jump_duration_mean_frames");
        const auto elapsed = observed_number(trajectory, "airborne_frames");
        if (mean && elapsed && *mean >= 8 && *mean <= 120 && *elapsed > 0 && *elapsed < *mean) {
            const double remaining = *mean - *elapsed;
            landing["status"] = "estimated"; landing["eta_frames"] = remaining;
            landing["projected_x"] = *player_x + *player_vx * remaining;
            landing["source"] = "empirical_jump_duration";
        }
    }
    return {{"source", "current_ram_projection"}, {"horizon_frames", horizon},
            {"reliability", !ram ? "unavailable" : limited ? "limited" : "current_ram"},
            {"enemy", enemy}, {"landing", landing},
            {"limits", Json::array({"Constant-velocity projection; acceleration and future input may change ETA.",
                "Empirical landing time is not collision simulation or a safe-landing guarantee.",
                "Current RAM has priority; remembered geometry does not steer the controller."})}};
}

Json serialize(const jevt::graph_trace& trace) {
    Json nodes = Json::array(), edges = Json::array();
    for (const auto& node : trace.nodes) {
        Json value = node.output.content.empty() ? Json(nullptr) : Json::parse(node.output.content);
        nodes.push_back({{"id", node.id}, {"dependencies", node.dependencies},
                         {"status", jevt::graph_status_name(node.status)}, {"output", value},
                         {"message", node.message}, {"snapshot_id", node.snapshot_id},
                         {"duration_ms", std::chrono::duration<double, std::milli>(node.duration).count()}});
    }
    for (const auto& edge : trace.edges) {
        const auto& source = trace.at(edge.from);
        const auto& target = trace.at(edge.to);
        edges.push_back({{"from", edge.from}, {"to", edge.to}, {"payload_ref", edge.from},
                         {"bytes", source.output.content.size()},
                         {"active", source.status == jevt::graph_status::succeeded &&
                                    target.status == jevt::graph_status::succeeded}});
    }
    return {{"snapshot_id", trace.snapshot_id}, {"nodes", nodes}, {"edges", edges},
            {"duration_ms", std::chrono::duration<double, std::milli>(trace.duration).count()}};
}
}

int main(int argc, char** argv) {
    std::string environment = "SuperMarioBros-1-1-v0";
    if (argc == 3 && std::string_view(argv[1]) == "--env") environment = argv[2];
    else if (argc != 1) { std::cerr << "usage: jevt_mario_control [--env ENV]\n"; return 1; }
    mario::FastController controller;
    mario::KnowledgeMemory memory(environment);
    mario::AdaptiveAgent learner;
    jevt::decision_graph graph({
        {"telemetry", {}, [](const Context& ctx) {
            const auto input = Json::parse(ctx.snapshot().input.content);
            if (!input.at("state").is_object()) throw std::invalid_argument("state must be an object");
            return success(input.at("state"));
        }},
        {"knowledge_affordances", {"telemetry"}, [](const Context& ctx) {
            return success(output(ctx,"telemetry").value("knowledge_affordances",Json::object()));
        }},
        {"physics", {"telemetry"}, [](const Context& ctx) {
            const auto state = output(ctx, "telemetry");
            return success(Json{{"player", state.at("player")}, {"trajectory", state.value("trajectory", Json::object())},
                                {"recent_control", state.value("recent_control", Json::object())},
                                {"session", state.value("session", Json::object())},
                                {"episode", state.value("episode", Json::object())}});
        }},
        {"game_phase", {"telemetry"}, [](const Context& ctx) {
            const auto state = output(ctx, "telemetry");
            const auto session = state.value("session", Json::object());
            const auto episode = state.value("episode", Json::object());
            const auto level = state.value("level", Json::object());
            const bool complete = session.value("game_complete", Json(nullptr)) == true || episode.value("full_game_completed", Json(nullptr)) == true;
            const bool over = session.value("game_over", Json(nullptr)) == true || episode.value("game_over", Json(nullptr)) == true;
            const bool dead = episode.value("dead", Json(nullptr)) == true || session.value("life_lost", Json(nullptr)) == true;
            const bool clear = episode.value("stage_clear", Json(nullptr)) == true || session.value("stage_transition", Json(nullptr)) == true;
            const bool known = !session.empty() || episode.value("dead", Json(nullptr)).is_boolean() || episode.value("stage_clear", Json(nullptr)).is_boolean();
            const auto coordinate = [&](const char* key) {
                auto value = level.value(key, Json(nullptr));
                if (!value.is_number_integer() || value.get<int>() < 1) value = session.value(key, Json(nullptr));
                return value.is_number_integer() && value.get<int>() >= 1 ? value : Json(nullptr);
            };
            const auto lives = session.value("lives_remaining", episode.value("lives_remaining", Json(nullptr)));
            const bool lives_known = lives.is_number_integer() && lives.get<int>() >= 0;
            return success(Json{{"phase", complete ? "game_complete" : over ? "game_over" : dead ? "life_lost" : clear ? "stage_clear" : known ? "playing" : "unknown"},
                {"run_id", session.value("run_id", 0)}, {"world", coordinate("world")}, {"stage", coordinate("stage")},
                {"lives_remaining", lives_known ? lives : Json(nullptr)}, {"life_lost", dead},
                {"stage_clear", clear}, {"game_over", over}, {"game_complete", complete},
                {"memory_generation", session.value("memory_generation", 0)}, {"source", "current_session_and_ram"}});
        }},
        {"geometry", {"telemetry"}, [](const Context& ctx) {
            const auto state = output(ctx, "telemetry");
            return success(Json{{"terrain", state.at("terrain")}, {"collision", state.value("collision", Json::object())}});
        }},
        {"threats", {"telemetry"}, [](const Context& ctx) {
            return success(Json{{"hazard", output(ctx, "telemetry").at("hazard")}});
        }},
        {"objectives", {"telemetry"}, [](const Context& ctx) {
            const auto state = output(ctx, "telemetry");
            return success(Json{{"strategy", state.at("strategy")}, {"detectors", state.value("detectors", Json::object())},
                                {"optimization", state.value("optimization", Json::object())}});
        }},
        {"experience_memory", {"telemetry"}, [&memory,&environment](const Context& ctx) {
            const auto input = Json::parse(ctx.snapshot().input.content);
            const auto state=output(ctx,"telemetry");
            const auto session=state.value("session",Json::object());
            if ((session.value("game_over",Json(nullptr))==true ||
                 state.value("episode",Json::object()).value("game_over",Json(nullptr))==true) &&
                !session.value("retain_knowledge",false))
                memory=mario::KnowledgeMemory(environment);
            else if(session.value("game_over",Json(nullptr))!=true)
                memory.observe(state, input.at("epoch").get<int>(), input.at("frame").get<int>());
            auto context = memory.context();
            context["delta"] = memory.learning_delta();
            return success(context);
        }},
        {"outcome_update", {"telemetry"}, [&learner](const Context& ctx) {
            const auto input=Json::parse(ctx.snapshot().input.content);
            return success(learner.observe(output(ctx,"telemetry"),input.at("frame").get<int>()));
        }},
        {"world_belief", {"outcome_update", "experience_memory", "telemetry"}, [&learner](const Context& ctx) {
            auto belief=learner.world();
            const auto observed=output(ctx,"telemetry").value("collision",Json::object());
            belief["current_platforms"]=observed.value("landing_surfaces",Json::array());
            belief["known_columns_ahead"]=output(ctx,"experience_memory").value("known_columns_ahead",0);
            belief["geometry_policy"]="current visible RAM for collision; empirical map is observational context, not a replay route";
            return success(belief);
        }},
        {"action_dynamics", {"outcome_update"}, [&learner](const Context&) { return success(learner.dynamics()); }},
        {"persistent_tasks", {"telemetry", "world_belief"}, [&learner](const Context& ctx) {
            return success(learner.tasks(output(ctx,"telemetry")));
        }},
        {"reachability", {"telemetry", "persistent_tasks", "action_dynamics"}, [&learner](const Context& ctx) {
            return success(learner.reachability(output(ctx,"telemetry"),output(ctx,"persistent_tasks")));
        }},
        {"learning_delta", {"experience_memory"}, [](const Context& ctx) {
            return success(output(ctx, "experience_memory").at("delta"));
        }},
        {"digital_twin", {"physics", "geometry", "threats", "experience_memory"}, [](const Context& ctx) {
            return success(digital_twin(output(ctx, "physics"), output(ctx, "geometry"),
                                        output(ctx, "threats"), output(ctx, "experience_memory")));
        }},
        {"risk_policy", {"game_phase", "digital_twin", "learning_delta"}, [](const Context& ctx) {
            const auto phase = output(ctx, "game_phase"), twin = output(ctx, "digital_twin");
            const auto delta = output(ctx, "learning_delta");
            Json reasons = Json::array();
            if (phase.at("phase") != "playing") reasons.push_back("lifecycle_transition_or_terminal");
            if (phase.at("lives_remaining").is_number_integer() && phase.at("lives_remaining").get<int>() <= 1)
                reasons.push_back("last_life");
            if (twin.at("reliability") == "unavailable") reasons.push_back("unavailable_ram_projection");
            if (twin.at("enemy").at("present") == true && twin.at("enemy").at("velocity_source") == "unknown")
                reasons.push_back("unobserved_enemy_motion");
            if (delta.at("new_enemy_types").get<int>() > 0) reasons.push_back("new_enemy_type");
            return success(Json{{"stance", reasons.empty() ? "normal" : "cautious"},
                {"allow_aggressive_exploration", reasons.empty()}, {"reasons", reasons},
                {"source", "current_lives_phase_forecast_and_learning_delta"}});
        }},
        {"compact_state", {"physics", "geometry", "threats", "objectives", "experience_memory", "digital_twin", "game_phase", "learning_delta", "risk_policy", "persistent_tasks", "action_dynamics"}, [](const Context& ctx) {
            const auto physics = output(ctx, "physics"), geometry = output(ctx, "geometry");
            const auto objectives = output(ctx, "objectives"), threats = output(ctx, "threats");
            const auto player = physics.at("player"), terrain = geometry.at("terrain");
            const auto detectors = objectives.at("detectors");
            const auto experience = output(ctx, "experience_memory");
            const auto twin = output(ctx, "digital_twin");
            const auto phase = output(ctx, "game_phase"), delta = output(ctx, "learning_delta"), policy = output(ctx, "risk_policy");
            // Keep the persistent map/graph local. Only a bounded statistical
            // summary enters each candidate-scoring request.
            const auto rounded = [](const Json& value) -> Json {
                return value.is_number() ? Json(std::round(value.get<double>() * 10.) / 10.) : Json(nullptr);
            };
            Json memory_context{
                {"jump_n", experience.value("jump_samples", 0)},
                {"air_frames", rounded(experience.value("jump_duration_mean_frames", Json(nullptr)))},
                {"range_px", rounded(experience.value("jump_range_mean_pixels", Json(nullptr)))},
                {"known_ahead", experience.value("known_columns_ahead", 0)}};
            const auto enemies = threats.at("hazard").value("upcoming_enemies", Json::array());
            if (!enemies.empty()) {
                const auto kind = std::to_string(enemies.front().value("kind_id", -1));
                const auto speeds = experience.value("enemy_speeds", Json::object());
                if (speeds.contains(kind)) {
                    memory_context["enemy_type"] = kind;
                    memory_context["enemy_vx"] = rounded(speeds.at(kind).at("mean"));
                    memory_context["enemy_n"] = speeds.at(kind).at("samples");
                }
            }
            // This is the actual next model request, composed from predecessor
            // outputs. Full RAM/grid stays local and is never repeated per head.
            return success(Json{
                {"mode", objectives.at("strategy").value("mode", "speedrun")},
                {"phase", phase.at("phase")}, {"lives", phase.at("lives_remaining")},
                {"risk", policy.at("stance")},
                {"learned", {{"columns", delta.at("new_columns")}, {"enemy_v", delta.at("enemy_velocity_samples")}, {"jumps", delta.at("jump_samples")}}},
                {"grounded", player.value("grounded", false)},
                {"velocity", Json::array({player.value("vx", 0.0), player.value("vy", 0.0)})},
                {"enemy_distance", threats.at("hazard").value("enemy_distance", 999.0)},
                {"enemy_stompable", threats.at("hazard").value("nearest_enemy_stompable", false)},
                {"attack_path_blocked",output(ctx,"persistent_tasks").value("status",std::string()).find("blocked_")==0},
                {"gap_distance", terrain.value("gap_distance", 999.0)},
                {"obstacle_distance", terrain.value("obstacle_distance", 999.0)},
                {"powerup_distance", detectors.value("powerup_distance", 999.0)},
                {"coins_collected", detectors.value("coins", 0)},
                {"stalled", physics.at("recent_control").value("stalled", 0)},
                {"trajectory", physics.at("trajectory").value("committed_geometry", "none")},
                {"support_reliability", terrain.value("observation_reliability", "unknown")},
                {"experience", memory_context},
                {"online", {{"task",output(ctx,"persistent_tasks").at("status")},
                            {"samples",output(ctx,"action_dynamics").at("samples")},
                            {"error_px",rounded(output(ctx,"action_dynamics").at("prediction_error_px").at("mean"))}}},
                {"forecast", {{"horizon_frames", twin.at("horizon_frames")},
                              {"enemy_eta_frames", rounded(twin.at("enemy").at("contact_eta_frames"))},
                              {"enemy_distance_px", rounded(twin.at("enemy").at("projected_distance_px"))},
                              {"landing_eta_frames", rounded(twin.at("landing").at("eta_frames"))},
                              {"reliability", twin.at("reliability")}}}});
        }},
        {"model_mailbox", {}, [](const Context& ctx) {
            const auto input = Json::parse(ctx.snapshot().input.content);
            // Snapshot identifies this receipt, NOT a new inference. Preserve
            // the old producer frame/epoch inside the explicit envelope.
            return success(Json{{"cached_response", input.value("strategy", Json::object())},
                                {"epoch", input.at("epoch")}, {"frame", input.at("frame")},
                                {"role", "asynchronous selected-model intent; no inference in this node"}});
        }},
        {"intent_gate", {"model_mailbox", "compact_state"}, [](const Context& ctx) {
            const auto mailbox = output(ctx, "model_mailbox"), compact = output(ctx, "compact_state");
            const auto cached = mailbox.at("cached_response");
            const int age = mailbox.at("frame").get<int>() - cached.value("source_frame", -10000);
            bool accepted = cached.value("judgment_kind", "goal") == "goal" && cached.value("ok", false) && cached.value("epoch", -1) == mailbox.at("epoch").get<int>() &&
                            cached.value("mode", "") == compact.at("mode").get<std::string>() && age >= 0 && age <= 90;
            std::string reason = accepted ? "current episode; intent within 90-frame TTL" : "missing, stale or wrong-episode intent";
            const std::string raw_goal = cached.value("active_goal", "");
            const std::string mode = compact.at("mode");
            const double enemy_distance = compact.at("enemy_distance"), powerup_distance = compact.at("powerup_distance");
            const bool attack_blocked=compact.value("attack_path_blocked",false);
            const bool enemy_present = enemy_distance >= 0 && enemy_distance < 200 && compact.value("enemy_stompable", false) && !attack_blocked;
            const bool powerup_present = powerup_distance >= 0 && powerup_distance < 200;
            const auto target_available = [&](std::string_view goal) {
                return goal == "finish_fast" || (goal == "stomp_enemy" && enemy_present) ||
                    (goal == "collect_powerup" && powerup_present) ||
                    (goal == "score_attack" && (enemy_present || powerup_present)) ||
                    (goal == "recover_momentum" && (compact.at("stalled").get<int>() >= 8 || attack_blocked));
            };
            const auto mode_allows = [&](std::string_view goal) {
                if (goal == "recover_momentum") return true;
                if (mode == "hunter") return enemy_present ? goal == "stomp_enemy" : goal == "finish_fast";
                if (mode == "collector") return powerup_present ? goal == "collect_powerup" : goal == "finish_fast";
                if (mode == "score_attack") return enemy_present || powerup_present
                    ? goal == "score_attack" || goal == "stomp_enemy" || goal == "collect_powerup"
                    : goal == "finish_fast";
                return mode == "speedrun" && goal == "finish_fast";
            };
            const auto feasible = [&](std::string_view goal) {
                return target_available(goal) && mode_allows(goal);
            };
            const auto probabilities = cached.value("goal_probabilities", Json::object());
            Json candidates = Json::array(), selected_probability = nullptr;
            std::string goal = raw_goal, selection = "raw_goal_without_distribution";
            bool valid_scores = probabilities.is_object();
            double best_score = -1;
            std::string best_goal;
            // The model supplies ordering within the user's objective and
            // current target capabilities. Preserve scores without renormalizing
            // a win probability or using remembered map positions as targets.
            for (const auto* candidate : {"finish_fast", "stomp_enemy", "collect_powerup", "collect_coins", "score_attack", "recover_momentum"}) {
                const bool available = feasible(candidate);
                Json probability = nullptr;
                if (probabilities.is_object() && probabilities.contains(candidate)) {
                    const auto& score = probabilities.at(candidate);
                    if (!score.is_number() || !std::isfinite(score.get<double>()) || score.get<double>() < 0 || score.get<double>() > 1)
                        valid_scores = false;
                    else {
                        const double p = score.get<double>();
                        probability = p;
                        if (available && (p > best_score || (p == best_score && candidate == raw_goal))) {
                            best_score = p;
                            best_goal = candidate;
                        }
                    }
                }
                candidates.push_back({{"goal", candidate}, {"feasible", available}, {"probability", probability},
                    {"target_available", target_available(candidate)}, {"mode_allowed", mode_allows(candidate)},
                    {"constraint", !target_available(candidate) ? (attack_blocked && (std::string_view(candidate)=="stomp_enemy" || std::string_view(candidate)=="score_attack") ? "observed attack route blocked or attempt budget exhausted" : "target absent in current RAM") :
                        !mode_allows(candidate) ? "excluded by selected objective mode" : "current target and objective mode permit goal"}});
            }
            if (accepted && !valid_scores) {
                accepted = false; reason = "invalid model goal probability distribution";
            } else if (accepted && !probabilities.empty()) {
                if (best_goal.empty() || best_score <= 0) {
                    accepted = false; reason = "no positively scored model goal has a current feasible target";
                } else {
                    goal = best_goal;
                    selected_probability = best_score;
                    selection = goal == raw_goal ? "raw_model_goal_feasible" : "best_feasible_model_score";
                    reason += goal == raw_goal ? "; raw model goal is feasible" :
                        "; selected " + goal + " from model scores within " + mode + " objective and current targets (raw " + raw_goal + ")";
                }
            } else if (accepted && !feasible(raw_goal)) {
                accepted = false; reason = !target_available(raw_goal) ? "model target not present in current RAM" : "model goal conflicts with selected objective mode";
            }
            if (!accepted) selection = "rejected";
            return success(Json{{"accepted", accepted}, {"active_goal", accepted ? goal : ""},
                                {"raw_model_goal", raw_goal}, {"intent_selection", selection},
                                {"selected_goal_probability", accepted ? selected_probability : Json(nullptr)},
                                {"goal_candidates", candidates},
                                {"mode", compact.at("mode")}, {"age_frames", cached.empty() ? Json(nullptr) : Json(age)},
                                {"source_snapshot", cached.value("source_snapshot", "")},
                                {"model_id", cached.value("model_id", "")}, {"reason", reason},
                                {"goal_probabilities", probabilities}});
        }},
        {"reactive_baseline", {"physics", "geometry", "threats", "objectives", "intent_gate", "game_phase", "learning_delta", "risk_policy", "persistent_tasks"}, [&controller](const Context& ctx) {
            Json state = Json::object();
            for (const auto name : {"physics", "geometry", "threats", "objectives"}) state.update(output(ctx, name));
            for (const auto name : {"game_phase", "learning_delta", "risk_policy"}) state[name] = output(ctx, name);
            const auto phase = state.at("game_phase");
            if (phase.at("game_over") == true || phase.at("life_lost") == true) state["episode"]["dead"] = true;
            if (phase.at("game_complete") == true || phase.at("stage_clear") == true) state["episode"]["stage_clear"] = true;
            const auto intent = output(ctx, "intent_gate");
            state["attack_task_blocked"]=output(ctx,"persistent_tasks").value("status",std::string()).find("blocked_")==0;
            auto action = controller.step(state, intent);
            action["active_goal"] = intent.at("active_goal");
            action["intent_accepted"] = intent.at("accepted");
            action["intent_reason"] = intent.at("reason");
            action["objective_mode"] = intent.at("mode");
            action["raw_model_goal"] = intent.at("raw_model_goal");
            action["intent_selection"] = intent.at("intent_selection");
            action["selected_goal_probability"] = intent.at("selected_goal_probability");
            return success(action);
        }},
        {"candidate_plans", {"telemetry", "persistent_tasks", "reachability", "action_dynamics", "reactive_baseline", "intent_gate"}, [&learner](const Context& ctx) {
            return success(learner.plans(output(ctx,"telemetry"),output(ctx,"persistent_tasks"),output(ctx,"reachability"),output(ctx,"reactive_baseline"),output(ctx,"intent_gate")));
        }},
        {"plan_candidates", {"telemetry", "compact_state", "candidate_plans", "persistent_tasks"}, [](const Context& ctx) {
            return success(mario::ranking::offer(output(ctx,"telemetry"),output(ctx,"compact_state"),output(ctx,"candidate_plans"),output(ctx,"persistent_tasks")));
        }},
        {"plan_intent_gate", {"plan_candidates", "model_mailbox"}, [](const Context& ctx) {
            return success(mario::ranking::validate(Json::parse(ctx.snapshot().input.content),output(ctx,"plan_candidates")));
        }},
        {"risk_budget", {"telemetry", "candidate_plans", "persistent_tasks", "reactive_baseline", "outcome_update", "plan_intent_gate"}, [&learner](const Context& ctx) {
            const auto task=output(ctx,"persistent_tasks"),plans=output(ctx,"candidate_plans");
            return success(mario::ranking::apply(learner.risk(output(ctx,"telemetry"),plans,output(ctx,"reactive_baseline"),task),plans,output(ctx,"plan_intent_gate"),task));
        }},
        {"hunter_engagement", {"telemetry", "persistent_tasks", "action_dynamics"}, [&learner](const Context& ctx) {
            return success(learner.engage(output(ctx,"telemetry"),output(ctx,"persistent_tasks")));
        }},
        {"controller", {"telemetry", "knowledge_affordances", "reactive_baseline", "risk_budget", "persistent_tasks", "game_phase", "learning_delta", "risk_policy", "hunter_engagement"}, [&learner,&controller](const Context& ctx) {
            auto action=learner.execute(output(ctx,"telemetry"),output(ctx,"reactive_baseline"),output(ctx,"risk_budget"),output(ctx,"persistent_tasks"),output(ctx,"hunter_engagement"));
            controller.feedback_action(action.at("action").get<std::string>());
            return success(action);
        }}
    });
    for (std::string line; std::getline(std::cin, line);) {
        try {
            const auto input = Json::parse(line);
            if (input.value("command", "") == "memory_export") {
                std::cout << Json{{"ok", true}, {"memory", memory.export_json()}}.dump() << std::endl;
                continue;
            }
            if (input.value("command", "") == "memory_reset") {
                // GAME OVER starts discovery from zero. Life/stage changes
                // use reset_episode via a new epoch and retain observations.
                memory = mario::KnowledgeMemory(environment);
                learner.reset();
                controller.reset();
                std::cout << Json{{"ok", true}, {"memory", memory.snapshot()}}.dump() << std::endl;
                continue;
            }
            if (input.value("command", "") == "memory_import") {
                std::string error;
                const bool ok = memory.import_json(input.at("memory"), &error);
                std::cout << Json{{"ok", ok}, {"error", error}}.dump() << std::endl;
                continue;
            }
            if (input.value("reset", false)) controller.reset();
            const auto snapshot = std::to_string(input.at("epoch").get<int>()) + ":" + std::to_string(input.at("frame").get<int>());
            const auto trace = graph.evaluate({snapshot, jevt::json_state(line)});
            const auto& result = trace.at("controller");
            if (result.status != jevt::graph_status::succeeded) {
                std::cout << Json{{"ok", false}, {"error", result.message}, {"graph", serialize(trace)}}.dump() << std::endl;
                continue;
            }
            auto response = Json::parse(result.output.content);
            response["ok"] = true;
            response["compact_state"] = Json::parse(trace.at("compact_state").output.content);
            response["model_request"] = Json::parse(trace.at("plan_candidates").output.content);
            response["plan_intent"] = Json::parse(trace.at("plan_intent_gate").output.content);
            response["digital_twin"] = Json::parse(trace.at("digital_twin").output.content);
            for (const auto name : {"game_phase", "learning_delta", "risk_policy"})
                response[name] = Json::parse(trace.at(name).output.content);
            for (const auto name : {"outcome_update", "world_belief", "action_dynamics", "persistent_tasks", "risk_budget"})
                response[name] = Json::parse(trace.at(name).output.content);
            response["graph"] = serialize(trace);
            if (input.value("include_memory", false)) response["knowledge"] = memory.snapshot();
            std::cout << response.dump() << std::endl;
        } catch (const std::exception& error) {
            std::cout << Json{{"ok", false}, {"error", error.what()}}.dump() << std::endl;
        }
    }
}
