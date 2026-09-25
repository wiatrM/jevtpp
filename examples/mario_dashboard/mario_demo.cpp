#include <jevt/core.hpp>
#include <jevt/system_one.hpp>

#if JEVT_MARIO_HAS_LAYA
#include <jevt/laya.hpp>
#endif
#if JEVT_MARIO_HAS_OPEN_JEV
#include <jevt/remote.hpp>
#include <nlohmann/json.hpp>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

enum class ControllerAction { noop, right, right_jump, right_run, right_run_jump, jump, left };
enum class DangerLevel { safe = 0, watch = 1, critical = 2 };
enum class ActiveGoal { finish_fast, stomp_enemy, collect_powerup, collect_coins, score_attack, recover_momentum };
enum class PlanSlot { slot0, slot1, slot2, slot3, slot4, slot5, defer };
constexpr auto plan_slots=jevt::schema<PlanSlot,"runner.plan_slots">(
    jevt::option<PlanSlot::slot0>("Execute the plan in plan_candidates with slot 0, only if present and suitable."),
    jevt::option<PlanSlot::slot1>("Execute the plan in plan_candidates with slot 1, only if present and suitable."),
    jevt::option<PlanSlot::slot2>("Execute the plan in plan_candidates with slot 2, only if present and suitable."),
    jevt::option<PlanSlot::slot3>("Execute the plan in plan_candidates with slot 3, only if present and suitable."),
    jevt::option<PlanSlot::slot4>("Execute the plan in plan_candidates with slot 4, only if present and suitable."),
    jevt::option<PlanSlot::slot5>("Execute the plan in plan_candidates with slot 5, only if present and suitable."),
    jevt::option<PlanSlot::defer>("Defer to the verified local controller when no listed plan is appropriate. Never invent an absent slot."));
constexpr auto plan_model=jevt::decision_model<"jevt.plan_rank.v1">(
    "Rank the concrete observed plans for observation.mode. Preserve life and prefer verifiable task progress. Risk costs are not probabilities.",
    jevt::choice<"selected_plan">("Read each candidate's action sequence, landing support, continuation and uncertainty. Choose one existing slot or defer.",plan_slots));

constexpr auto actions = jevt::schema<ControllerAction, "runner.actions">(
    jevt::option<ControllerAction::noop>("Release controls and preserve the current trajectory."),
    jevt::option<ControllerAction::right>("Walk right with controlled forward speed."),
    jevt::option<ControllerAction::right_jump>("Move right and begin or sustain a jump."),
    jevt::option<ControllerAction::right_run>("Run right quickly on safe open ground."),
    jevt::option<ControllerAction::right_run_jump>("Run right while beginning or sustaining a long jump."),
    jevt::option<ControllerAction::jump>("Jump vertically without adding forward acceleration."),
    jevt::option<ControllerAction::left>("Move left only to recover from a blocked or unsafe position."));

constexpr auto danger_levels = jevt::schema<DangerLevel, "runner.danger">(
    jevt::option<DangerLevel::safe>("Safe open movement with no immediate hazard."),
    jevt::option<DangerLevel::watch>("An obstacle, gap, or enemy will matter soon."),
    jevt::option<DangerLevel::critical>("Immediate collision, fall, or enemy threat."));

constexpr auto goals = jevt::schema<ActiveGoal, "runner.goals">(
    jevt::option<ActiveGoal::finish_fast>("Maximize safe rightward progress and reach the flag quickly."),
    jevt::option<ActiveGoal::stomp_enemy>("Eliminate the nearest hostile by landing on it from above."),
    jevt::option<ActiveGoal::collect_powerup>("Route toward a visible mushroom, flower, or star."),
    jevt::option<ActiveGoal::collect_coins>("Increase coins and score without sacrificing the current life."),
    jevt::option<ActiveGoal::score_attack>("Prefer enemies, blocks, coins, and powerups that increase score."),
    jevt::option<ActiveGoal::recover_momentum>("Escape a blocked position and restore rightward movement."));

constexpr auto runner_model = jevt::decision_model<"mario.runner">(
    "Control an original side-scrolling platform runner. Advance to the finish without falling or touching enemies.",
    jevt::choice<"active_goal">(
        "Choose the current tactical goal. Follow strategy.mode, detected targets, pace, score delta, and recent control outcome.",
        goals),
    jevt::choice<"next_action">(
        "Choose the next controller macro. Respect projected contact time, jump phase, trusted terrain, and recent control outcome. A gap or obstacle within the reaction horizon requires a forward jump.",
        actions),
    jevt::noul<"jump_needed">(
        "Should a forward jump begin or remain held now? A near gap, obstacle, projected enemy contact, or an active gap crossing means yes.",
        {0.25F, 0.75F}),
    jevt::score<"danger">("How dangerous is the immediate situation?", danger_levels));

// A single compact judgment chooses intent; frame-accurate reflexes belong to
// the fast C++ graph, not to a remote inference round trip.
constexpr auto compact_goals = jevt::schema<ActiveGoal, "runner.compact_goals">(
    jevt::option<ActiveGoal::finish_fast>("Advance safely to the flag."),
    jevt::option<ActiveGoal::stomp_enemy>("Defeat a nearby enemy from above."),
    jevt::option<ActiveGoal::collect_powerup>("Collect a nearby mushroom or flower."),
    jevt::option<ActiveGoal::collect_coins>("Collect nearby coins safely."),
    jevt::option<ActiveGoal::score_attack>("Gain score from available targets."),
    jevt::option<ActiveGoal::recover_momentum>("Escape a blocked position."));
constexpr auto strategy_model = jevt::decision_model<"mario.strategy.v2">(
    "Choose Mario's goal for the selected mode.",
    jevt::choice<"active_goal">("Preserve life. Distances are pixels; 999 means absent.", compact_goals));

constexpr std::array<std::string_view, 7> action_names{
    "noop", "right", "right_jump", "right_run", "right_run_jump", "jump", "left"};
constexpr std::array<std::string_view, 6> goal_names{
    "finish_fast", "stomp_enemy", "collect_powerup", "collect_coins", "score_attack", "recover_momentum"};

struct Observation {
  double x{}, y{}, vx{}, vy{};
  double enemy_distance{999.0}, gap_distance{999.0}, obstacle_distance{999.0};
  int stalled{}, tick{};
  bool grounded{}, crossing_gap{};
};

double number_after(std::string_view input, std::string_view key, double fallback) {
  const auto at = input.find(key);
  if (at == std::string_view::npos) return fallback;
  auto value = input.substr(at + key.size());
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.remove_prefix(1);
  const auto end = value.find_first_of(",}");
  value = value.substr(0, end);
  double parsed{};
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  return result.ec == std::errc{} ? parsed : fallback;
}

bool bool_after(std::string_view input, std::string_view key) {
  const auto at = input.find(key);
  if (at == std::string_view::npos) return false;
  auto value = input.substr(at + key.size());
  while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.remove_prefix(1);
  return value.substr(0, 4) == "true";
}

class DemoBackend final : public jevt::backend {
 public:
  [[nodiscard]] jevt::result<jevt::inference_response> predict(
      const jevt::inference_request& request) override {
    const double enemy = number_after(request.input, "\"enemy_distance\":", 999.0);
    const double gap = number_after(request.input, "\"gap_distance\":", 999.0);
    const double obstacle = number_after(request.input, "\"obstacle_distance\":", 999.0);
    const double vy = number_after(request.input, "\"vy\":", 0.0);
    const double stalled = number_after(request.input, "\"stalled\":", 0.0);
    const bool grounded = bool_after(request.input, "\"grounded\":");
    const bool crossing = bool_after(request.input, "\"crossing_gap\":");
    const double nearest = std::min({enemy, gap, obstacle});
    const double urgency = std::clamp((150.0 - nearest) / 150.0, 0.0, 1.0);
    const bool jump_now = crossing || gap < 145.0 || obstacle < 112.0 || enemy < 96.0;

    std::vector<float> scores;
    if (request.question_kind == jevt::inference_request::kind::choice && request.options.size() == 6) {
      scores = {0.7F, 0.05F, 0.05F, 0.05F, 0.05F, 0.1F};
    } else if (request.question_kind == jevt::inference_request::kind::choice) {
      scores = {0.025F, 0.08F, 0.09F, 0.18F, 0.10F, 0.025F, 0.02F};
      if (jump_now) {
        scores[4] = static_cast<float>(0.62 + urgency * 0.45);
        scores[2] = static_cast<float>(0.28 + urgency * 0.18);
        scores[3] = 0.06F;
      } else if (!grounded && vy < 0.0) {
        scores[4] = 0.52F;
        scores[2] = 0.23F;
        scores[3] = 0.16F;
      } else {
        scores[3] = 0.74F;
      }
      if (stalled > 24.0) {
        scores[4] += 0.38F;
        scores[3] *= 0.3F;
      }
    } else if (request.question_kind == jevt::inference_request::kind::noul) {
      const float yes = static_cast<float>(jump_now ? std::max(0.78, 0.62 + urgency * 0.34) : 0.12);
      scores = {1.0F - yes, yes};
    } else {
      const float critical = static_cast<float>(std::clamp((82.0 - nearest) / 82.0, 0.0, 0.92));
      const float watch = static_cast<float>(std::clamp(urgency * 0.85, 0.04, 0.82));
      const float safe = static_cast<float>(std::max(0.04, 1.0 - watch - critical * 0.55));
      scores = {safe, watch, critical};
    }
    return jevt::inference_response{std::move(scores), "jevt-demo-policy"};
  }
  [[nodiscard]] std::string_view name() const noexcept override { return "jevt-demo-policy"; }
};

std::string observation_json(const Observation& o) {
  std::ostringstream out;
  out << "{\"player\":{\"x\":" << o.x << ",\"y\":" << o.y
      << ",\"vx\":" << o.vx << ",\"vy\":" << o.vy
      << ",\"grounded\":" << std::boolalpha << o.grounded << "},"
      << "\"trajectory\":{\"crossing_gap\":" << o.crossing_gap
      << ",\"jump_phase\":\"" << (o.grounded ? "grounded" : o.vy < 0 ? "rising" : "falling") << "\"},"
      << "\"hazard\":{\"enemy_distance\":" << o.enemy_distance
      << ",\"contact_within_reaction_horizon\":" << (o.enemy_distance < 96.0) << "},"
      << "\"terrain\":{\"gap_distance\":" << o.gap_distance
      << ",\"obstacle_distance\":" << o.obstacle_distance << ",\"observation_reliability\":\"high\"},"
      << "\"recent_control\":{\"stalled\":" << o.stalled << "},"
      << "\"episode\":{\"tick\":" << o.tick << ",\"goal\":\"reach_finish_beacon\"}}";
  return out.str();
}

std::string json_escape(std::string_view input) {
  std::string out;
  out.reserve(input.size() + 16);
  for (const char c : input) {
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (c == '\n') out += "\\n";
    else out += c;
  }
  return out;
}

struct DecisionEngine {
  explicit DecisionEngine(std::shared_ptr<jevt::backend> backend, std::string backend_label)
      : brain(jevt::bind_system_one(runner_model, backend)),
        strategist(jevt::bind_system_one(strategy_model, backend)),
        plan_ranker(jevt::bind_system_one(plan_model, backend)), backend_(std::move(backend)), label(std::move(backend_label)) {}

  std::string evaluate_plan(const std::string& state) {
#if JEVT_MARIO_HAS_OPEN_JEV
    const auto started=std::chrono::steady_clock::now();
    using Json=nlohmann::json;
    try {
    const auto input=Json::parse(state);
    if(!input.is_object()||!input.contains("plan_candidates")||!input.at("plan_candidates").is_array())
      return "{\"ok\":false,\"error\":\"missing_plan_candidates\"}";
    const auto& candidates=input.at("plan_candidates");
    if(candidates.empty())return "{\"ok\":false,\"error\":\"no_plan_candidates\"}";
    if(candidates.size()>6)return "{\"ok\":false,\"error\":\"too_many_plan_candidates\"}";
    std::vector<std::string> descriptions;
    descriptions.reserve(candidates.size()+1);
    for(std::size_t i=0;i<candidates.size();++i) {
      const auto& candidate=candidates.at(i);
      if(!candidate.is_object()||candidate.value("slot",99U)!=i||
         !candidate.contains("id")||!candidate.at("id").is_string()||
         candidate.at("id").get<std::string>().empty()||
         !candidate.contains("action")||!candidate.at("action").is_string())
        return "{\"ok\":false,\"error\":\"invalid_plan_candidate\"}";
      const auto landing=candidate.value("landing_target",Json(nullptr));
      Json concise={{"slot",i},{"action",candidate.at("action")},
                    {"skill",candidate.value("skill",std::string())},
                    {"sequence",candidate.value("sequence",Json::array())},
                    {"landing",landing},
                    {"continuation",candidate.value("continuation",Json(nullptr))},
                    {"risk_cost",candidate.value("risk_cost",Json(nullptr))},
                    {"uncertainty_cost",candidate.value("uncertainty_cost",Json(nullptr))},
                    {"predicted_stomp",candidate.value("predicted_stomp",false)}};
      descriptions.push_back("Execute this observed candidate: "+concise.dump(-1, ' ', true));
      if(descriptions.back().size()>2048)return "{\"ok\":false,\"error\":\"plan_option_too_large\"}";
    }
    descriptions.push_back("Defer: let the local C++ controller select its next action from the current observation.");
    std::vector<std::string_view> options;
    for(const auto& description:descriptions)options.push_back(description);
    const std::string question="Which current candidate best advances the stated mode and task while preserving remaining lives? Read knowledge_graph statements as evidence with provenance, not guaranteed causal laws; use named entities, observed action effects, current RAM, movement sequences, landing support, continuation, and risks. Unknown or stale facts cannot override current RAM. Choose defer if none is suitable. Scores are preferences, not win probabilities.";
    jevt::inference_request request;
    request.decision_id="jevt.plan_rank.v2";
    request.question=question;
    request.input=state;
    request.input_kind=jevt::content_kind::json;
    request.options=options;
    request.question_kind=jevt::inference_request::kind::choice;
    const auto result=backend_->predict(request);
    if(!result)return "{\"ok\":false,\"error\":\""+json_escape(result.error_value().message)+"\"}";
    if(result->scores.size()!=options.size())return "{\"ok\":false,\"error\":\"invalid_plan_score_count\"}";
    double total=0;
    for(float score:result->scores){
      if(!std::isfinite(score)||score<0)return "{\"ok\":false,\"error\":\"invalid_plan_scores\"}";
      total+=score;
    }
    if(!std::isfinite(total)||total<=0)return "{\"ok\":false,\"error\":\"zero_plan_score_mass\"}";
    std::array<double,7> scores{};
    for(std::size_t i=0;i<candidates.size();++i)scores[i]=result->scores[i]/total;
    scores[6]=result->scores.back()/total;
    std::size_t selected=6;
    double highest=scores[6];
    for(std::size_t i=0;i<candidates.size();++i)if(scores[i]>highest){highest=scores[i];selected=i;}
    std::ostringstream out;
    out<<"{\"ok\":true,\"judgment_kind\":\"plan\",\"selected_slot\":"<<selected
       <<",\"backend\":\""<<json_escape(label)<<"\",\"model_id\":\""<<json_escape(result->model_id)
       <<"\",\"latency_ms\":"<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()
       <<",\"slot_scores\":[";
    for(std::size_t i=0;i<7;++i){if(i)out<<',';out<<scores[i];}
    out<<"]}";return out.str();
    } catch(const std::exception& error) {
      return "{\"ok\":false,\"error\":\"invalid_plan_request\"}";
    }
#else
    (void)state;
    return "{\"ok\":false,\"error\":\"plan_ranking_unavailable\"}";
#endif
  }

  std::string evaluate_strategy(const std::string& state) {
    const auto started = std::chrono::steady_clock::now();
    const auto result = strategist.evaluate(jevt::json_state(state));
    if (!result) return "{\"ok\":false,\"error\":\"" + json_escape(result.error_value().message) + "\"}";
    const auto& goal = result->get<"active_goal">();
    std::ostringstream out;
    out << "{\"ok\":true,\"active_goal\":\"" << goal_names[static_cast<std::size_t>(goal.value())]
        << "\",\"goal_confidence\":" << goal.confidence()
        << ",\"latency_ms\":" << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count()
        << ",\"backend\":\"" << json_escape(label) << "\",\"model_id\":\"" << json_escape(goal.model_id())
        << "\",\"goal_probabilities\":{";
    for (std::size_t i = 0; i < goal_names.size(); ++i) {
      if (i) out << ',';
      out << '\"' << goal_names[i] << "\":" << goal.probabilities()[i];
    }
    out << "}}";
    return out.str();
  }

  std::string evaluate(const Observation& observation) { return evaluate_json(observation_json(observation)); }

  std::string evaluate_json(std::string state) {
    const auto started = std::chrono::steady_clock::now();
    const auto result = brain.evaluate(jevt::json_state(state));
    const double latency = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    if (!result) {
      return "{\"ok\":false,\"error\":\"" + json_escape(result.error_value().message) + "\"}";
    }
    const auto& action = result->get<"next_action">();
    const auto& goal = result->get<"active_goal">();
    const auto& jump = result->get<"jump_needed">();
    const auto& danger = result->get<"danger">();
    const auto raw_selected = static_cast<std::size_t>(action.value());
    std::size_t selected = raw_selected;
    std::string_view gate_reason = "choice accepted";

    // System One deliberately exposes independent typed judgments. The controller
    // composes them here instead of pretending that the largest Choice logit is
    // always safe. The final action remains model-backed: when a jump gate opens,
    // it selects the highest-probability jump-capable macro from Choice.
    const bool projected_contact = bool_after(state, "\"contact_within_reaction_horizon\":");
    const bool takeoff_due = bool_after(state, "\"jump_must_start_this_decision\":");
    const bool gap_ahead = bool_after(state, "\"gap_ahead\":");
    const bool obstacle_ahead = bool_after(state, "\"obstacle_ahead\":");
    const bool crossing_gap = bool_after(state, "\"crossing_gap\":");
    const bool crossing_obstacle = bool_after(state, "\"crossing_obstacle\":");
    const bool engaging_enemy = bool_after(state, "\"engaging_enemy\":");
    const bool grounded = bool_after(state, "\"grounded\":");
    const double takeoff_deadline = number_after(state, "\"takeoff_deadline_frames\":", 999.0);
    const double enemy_distance = number_after(state, "\"enemy_distance\":", 999.0);
    const double stalled = number_after(state, "\"stalled\":", 0.0);
    const double powerup_distance = number_after(state, "\"powerup_distance\":", 999.0);
    const bool mode_speedrun = bool_after(state, "\"mode_speedrun\":");
    const bool mode_hunter = bool_after(state, "\"mode_hunter\":");
    const bool mode_collector = bool_after(state, "\"mode_collector\":");
    const bool mode_score_attack = bool_after(state, "\"mode_score_attack\":");
    const bool hard_jump_gate = crossing_gap || crossing_obstacle || engaging_enemy || takeoff_due || projected_contact ||
                                (gap_ahead && takeoff_deadline <= 10.0) ||
                                (obstacle_ahead && takeoff_deadline <= 10.0) ||
                                (grounded && enemy_distance <= 72.0) ||
                                (grounded && mode_hunter && enemy_distance <= 176.0) ||
                                (grounded && mode_score_attack && enemy_distance <= 128.0) ||
                                (grounded && mode_collector && powerup_distance <= 112.0);
    // The local checkpoint's Noul head can saturate on out-of-domain prose.
    // It may vote, but only inside a hazard context independently verified
    // from NES RAM. This prevents a high Noul score from causing endless jumps
    // on open ground while keeping the decision trace fully visible.
    const bool noul_gate_eligible = grounded &&
        (enemy_distance <= 160.0 || gap_ahead || obstacle_ahead || stalled >= 4.0);
    const bool semantic_jump_gate = noul_gate_eligible &&
        jump.probability_true() >= 0.72F && danger.score() >= 0.72F;
    if (!hard_jump_gate && !semantic_jump_gate && jump.probability_true() >= 0.72F)
      gate_reason = "jump signal ignored without verified hazard";

    if (hard_jump_gate || semantic_jump_gate) {
      constexpr std::array<std::size_t, 3> jump_actions{2, 4, 5};
      selected = jump_actions.front();
      for (const auto candidate : jump_actions) {
        if (action.probabilities()[candidate] > action.probabilities()[selected]) selected = candidate;
      }
      if (crossing_gap) gate_reason = "hold jump through active gap crossing";
      else if (crossing_obstacle) gate_reason = "hold jump until obstacle is cleared";
      else if (engaging_enemy) gate_reason = "hold jump through active enemy engagement";
      else if (takeoff_due || (gap_ahead && takeoff_deadline <= 10.0)) gate_reason = "terrain takeoff window reached";
      else if (projected_contact || (grounded && enemy_distance <= 72.0)) gate_reason = "enemy contact inside reaction horizon";
      else if (obstacle_ahead && takeoff_deadline <= 10.0) gate_reason = "obstacle takeoff window reached";
      else gate_reason = "jump judgment and danger agree";
    } else if (mode_speedrun && grounded && (selected == 0 || selected == 6) &&
               !gap_ahead && !obstacle_ahead && enemy_distance > 160.0) {
      selected = action.probabilities()[3] >= action.probabilities()[1] ? 3U : 1U;
      gate_reason = "speedrun guard restores safe forward progress";
    }
    const bool override_applied = selected != raw_selected;
    std::ostringstream out;
    out << "{\"ok\":true,\"action\":\"" << action_names[selected] << "\","
        << "\"active_goal\":\"" << goal_names[static_cast<std::size_t>(goal.value())] << "\","
        << "\"goal_confidence\":" << goal.confidence() << ','
        << "\"raw_action\":\"" << action_names[raw_selected] << "\","
        << "\"override_applied\":" << std::boolalpha << override_applied
        << ",\"noul_gate_eligible\":" << noul_gate_eligible
        << ",\"jump_signal_used\":" << (hard_jump_gate || semantic_jump_gate)
        << ",\"gate_reason\":\"" << gate_reason << "\","
        << "\"confidence\":" << action.probabilities()[selected]
        << ",\"raw_confidence\":" << action.confidence() << ",\"jump_probability\":"
        << jump.probability_true() << ",\"danger_score\":" << danger.score()
        << ",\"latency_ms\":" << latency << ",\"backend\":\"" << json_escape(label)
        << "\",\"model_id\":\"" << json_escape(action.model_id()) << "\",\"probabilities\":{ ";
    for (std::size_t i = 0; i < action_names.size(); ++i) {
      if (i) out << ',';
      out << '\"' << action_names[i] << "\":" << action.probabilities()[i];
    }
    out << "},\"goal_probabilities\":{";
    for (std::size_t i = 0; i < goal_names.size(); ++i) {
      if (i) out << ',';
      out << '\"' << goal_names[i] << "\":" << goal.probabilities()[i];
    }
    out << "},\"state\":" << state << '}';
    return out.str();
  }

  decltype(jevt::bind_system_one(runner_model, std::shared_ptr<jevt::backend>{})) brain;
  decltype(jevt::bind_system_one(strategy_model, std::shared_ptr<jevt::backend>{})) strategist;
  decltype(jevt::bind_system_one(plan_model, std::shared_ptr<jevt::backend>{})) plan_ranker;
  std::shared_ptr<jevt::backend> backend_;
  std::string label;
  std::string provider = "unverified";
};

std::unordered_map<std::string, std::string> parse_query(std::string_view query) {
  std::unordered_map<std::string, std::string> values;
  while (!query.empty()) {
    const auto amp = query.find('&');
    const auto pair = query.substr(0, amp);
    const auto eq = pair.find('=');
    if (eq != std::string_view::npos) values.emplace(std::string(pair.substr(0, eq)), std::string(pair.substr(eq + 1)));
    if (amp == std::string_view::npos) break;
    query.remove_prefix(amp + 1);
  }
  return values;
}

double query_number(const std::unordered_map<std::string, std::string>& values,
                    std::string_view key, double fallback) {
  const auto it = values.find(std::string(key));
  if (it == values.end()) return fallback;
  double value{};
  const auto result = std::from_chars(it->second.data(), it->second.data() + it->second.size(), value);
  return result.ec == std::errc{} ? value : fallback;
}

std::optional<std::string> read_file(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) return std::nullopt;
  return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

#ifdef _WIN32
using Socket = SOCKET;
constexpr Socket invalid_socket = INVALID_SOCKET;
void close_socket(Socket socket) { closesocket(socket); }
struct SocketRuntime { SocketRuntime() { WSADATA d{}; if (WSAStartup(MAKEWORD(2, 2), &d)) throw std::runtime_error("WSAStartup failed"); } ~SocketRuntime() { WSACleanup(); } };
#else
using Socket = int;
constexpr Socket invalid_socket = -1;
void close_socket(Socket socket) { ::close(socket); }
struct SocketRuntime {};
#endif

void send_all(Socket socket, std::string_view bytes) {
  while (!bytes.empty()) {
#ifdef _WIN32
    const int count = ::send(socket, bytes.data(), static_cast<int>(bytes.size()), 0);
#else
    const auto count = ::send(socket, bytes.data(), bytes.size(), MSG_NOSIGNAL);
#endif
    if (count <= 0) return;
    bytes.remove_prefix(static_cast<std::size_t>(count));
  }
}

#if JEVT_MARIO_HAS_OPEN_JEV
// Small loopback-only transport keeps Open-JEV an implementation detail of the
// JevT++ backend and avoids adding a second HTTP dependency to the NES demo.
class LoopbackTransport final : public jevt::http_transport {
 public:
  jevt::result<jevt::http_response> perform(const jevt::http_request& request) override {
    constexpr std::string_view prefix = "http://127.0.0.1:";
    if (!request.url.starts_with(prefix))
      return jevt::error{jevt::error_code::invalid_request, "Open-JEV transport accepts loopback only"};
    const auto path_at = request.url.find('/', prefix.size());
    const auto port_text = std::string_view(request.url).substr(prefix.size(), path_at - prefix.size());
    unsigned port{};
    if (const auto parsed = std::from_chars(port_text.data(), port_text.data() + port_text.size(), port);
        parsed.ec != std::errc{} || port > 65535 || path_at == std::string::npos)
      return jevt::error{jevt::error_code::invalid_request, "invalid Open-JEV loopback URL"};
    SocketRuntime runtime;
    const Socket socket = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (socket == invalid_socket)
      return jevt::error{jevt::error_code::connection_failure, "cannot create Open-JEV socket"};
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    ::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
    if (::connect(socket, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
      close_socket(socket);
      return jevt::error{jevt::error_code::connection_failure, "cannot connect to local Open-JEV"};
    }
    std::ostringstream head;
    head << request.method << ' ' << request.url.substr(path_at) << " HTTP/1.1\r\n"
         << "Host: 127.0.0.1:" << port << "\r\nContent-Type: application/json\r\n"
         << "Content-Length: " << request.body.size() << "\r\nConnection: close\r\n\r\n";
    send_all(socket, head.str());
    send_all(socket, request.body);
    std::string response;
    std::array<char, 8192> buffer{};
    while (response.size() <= request.max_response_bytes + 16384) {
      if (request.cancellation.stop_requested() || std::chrono::steady_clock::now() >= request.deadline) {
        close_socket(socket);
        return jevt::error{request.cancellation.stop_requested() ? jevt::error_code::cancelled : jevt::error_code::timeout,
                           "Open-JEV request cancelled or deadline exceeded"};
      }
      // Poll in bounded slices: a wedged local model must not block forever.
      fd_set readable;
      FD_ZERO(&readable);
      FD_SET(socket, &readable);
      timeval slice{0, 100000};
      const auto ready = ::select(static_cast<int>(socket) + 1, &readable, nullptr, nullptr, &slice);
      if (ready == 0) continue;
      if (ready < 0) {
        close_socket(socket);
        return jevt::error{jevt::error_code::connection_failure, "Open-JEV socket polling failed"};
      }
#ifdef _WIN32
      const int count = ::recv(socket, buffer.data(), static_cast<int>(buffer.size()), 0);
#else
      const auto count = ::recv(socket, buffer.data(), buffer.size(), 0);
#endif
      if (count <= 0) break;
      response.append(buffer.data(), static_cast<std::size_t>(count));
    }
    close_socket(socket);
    const auto first_space = response.find(' ');
    const auto body_at = response.find("\r\n\r\n");
    int status{};
    if (first_space == std::string::npos || first_space + 4 > response.size() || body_at == std::string::npos ||
        std::from_chars(response.data() + first_space + 1, response.data() + first_space + 4, status).ec != std::errc{})
      return jevt::error{jevt::error_code::invalid_backend_output, "invalid Open-JEV HTTP response"};
    if (response.size() - body_at - 4 > request.max_response_bytes)
      return jevt::error{jevt::error_code::invalid_backend_output, "Open-JEV response exceeds configured limit"};
    return jevt::http_response{status, response.substr(body_at + 4), {}};
  }
};
#endif

void respond(Socket socket, int status, std::string_view type, std::string_view body) {
  std::ostringstream head;
  head << "HTTP/1.1 " << status << (status == 200 ? " OK" : " Not Found") << "\r\n"
       << "Content-Type: " << type << "\r\nContent-Length: " << body.size()
       << "\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n"
       << "Content-Security-Policy: default-src 'self'; img-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'\r\n"
       << "Connection: close\r\n\r\n";
  const auto header = head.str();
  send_all(socket, header);
  send_all(socket, body);
}

class DashboardServer {
 public:
  DashboardServer(DecisionEngine& engine, std::filesystem::path root, std::uint16_t port)
      : engine_(engine), root_(std::move(root)), port_(port) {}

  void run() {
    SocketRuntime runtime;
    const Socket server = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server == invalid_socket) throw std::runtime_error("cannot create server socket");
    int yes = 1;
#ifdef _WIN32
    ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&yes), sizeof(yes));
#else
    ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
#endif
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port_);
    ::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
    if (::bind(server, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 || ::listen(server, 32) != 0) {
      close_socket(server);
      throw std::runtime_error("cannot bind dashboard port " + std::to_string(port_));
    }
    std::cout << "JevT++ Runner Lab: http://127.0.0.1:" << port_ << "/\n"
              << "backend: " << engine_.label << "\n" << std::flush;
    while (true) {
      const Socket client = ::accept(server, nullptr, nullptr);
      if (client == invalid_socket) continue;
      handle(client);
      close_socket(client);
    }
  }

 private:
  void handle(Socket client) {
    std::string storage;
    storage.reserve(16384);
    std::array<char, 4096> buffer{};
    std::size_t expected = 0;
    while (storage.size() < 65536) {
#ifdef _WIN32
      const int count = ::recv(client, buffer.data(), static_cast<int>(buffer.size()), 0);
#else
      const auto count = ::recv(client, buffer.data(), buffer.size(), 0);
#endif
      if (count <= 0) break;
      storage.append(buffer.data(), static_cast<std::size_t>(count));
      const auto split = storage.find("\r\n\r\n");
      if (split == std::string::npos) continue;
      if (expected == 0) {
        const auto length_at = storage.find("Content-Length:");
        if (length_at != std::string::npos && length_at < split) {
          auto value = std::string_view(storage).substr(length_at + 15);
          while (!value.empty() && value.front() == ' ') value.remove_prefix(1);
          std::from_chars(value.data(), value.data() + value.size(), expected);
        }
      }
      if (storage.size() >= split + 4 + expected) break;
    }
    if (storage.empty()) return;
    const std::string_view request(storage);
    const auto first = request.find(' ');
    const auto second = first == std::string_view::npos ? first : request.find(' ', first + 1);
    if (first == std::string_view::npos || second == std::string_view::npos) {
      respond(client, 404, "text/plain", "not found\n"); return;
    }
    const auto method = request.substr(0, first);
    auto target = request.substr(first + 1, second - first - 1);
    const auto query_at = target.find('?');
    const auto path = target.substr(0, query_at);
    const auto query = query_at == std::string_view::npos ? std::string_view{} : target.substr(query_at + 1);
    if (path == "/api/health" && method == "GET") {
      respond(client, 200, "application/json", "{\"protocol\":2,\"capabilities\":[\"goal\""
#if JEVT_MARIO_HAS_OPEN_JEV
          ",\"plan_rank_v1\""
#endif
          "],\"backend\":\"" + json_escape(engine_.label) + "\",\"provider\":\"" + engine_.provider + "\"}");
      return;
    }
    if ((path == "/api/decision" || path == "/api/strategy" || path == "/api/plan") && method == "POST") {
      const auto split = request.find("\r\n\r\n");
      if (split == std::string_view::npos) { respond(client, 404, "text/plain", "missing body\n"); return; }
      const auto state = std::string(request.substr(split + 4));
      const auto body = path == "/api/plan" ? engine_.evaluate_plan(state) : path == "/api/strategy" ? engine_.evaluate_strategy(state) : engine_.evaluate_json(state);
      respond(client, 200, "application/json; charset=utf-8", body); return;
    }
    if (path == "/api/decision" && method == "GET") {
      const auto values = parse_query(query);
      Observation o;
      o.x = query_number(values, "x", 0); o.y = query_number(values, "y", 0);
      o.vx = query_number(values, "vx", 0); o.vy = query_number(values, "vy", 0);
      o.enemy_distance = query_number(values, "enemy", 999); o.gap_distance = query_number(values, "gap", 999);
      o.obstacle_distance = query_number(values, "obstacle", 999); o.stalled = static_cast<int>(query_number(values, "stalled", 0));
      o.tick = static_cast<int>(query_number(values, "tick", 0));
      o.grounded = query_number(values, "grounded", 0) != 0; o.crossing_gap = query_number(values, "crossing", 0) != 0;
      const auto body = engine_.evaluate(o);
      respond(client, 200, "application/json; charset=utf-8", body); return;
    }
    if (method != "GET") { respond(client, 404, "text/plain", "not found\n"); return; }
    const auto file = path == "/" ? root_ / "index.html" : root_ / std::string(path.substr(1));
    const auto normalized = file.lexically_normal();
    if (normalized.string().find(root_.lexically_normal().string()) != 0) {
      respond(client, 404, "text/plain", "not found\n"); return;
    }
    const auto body = read_file(normalized);
    if (!body) { respond(client, 404, "text/plain", "not found\n"); return; }
    std::string_view type = "application/octet-stream";
    if (normalized.extension() == ".html") type = "text/html; charset=utf-8";
    else if (normalized.extension() == ".js") type = "text/javascript; charset=utf-8";
    else if (normalized.extension() == ".css") type = "text/css; charset=utf-8";
    else if (normalized.extension() == ".png") type = "image/png";
    respond(client, 200, type, *body);
  }
  DecisionEngine& engine_;
  std::filesystem::path root_;
  std::uint16_t port_;
};

}  // namespace

int main(int argc, char** argv) try {
  std::filesystem::path web_root = "examples/mario_dashboard";
  std::filesystem::path model_directory;
  std::string open_jev_url;
  std::uint16_t port = 4173;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--web-root" && i + 1 < argc) web_root = argv[++i];
    else if (arg == "--model" && i + 1 < argc) model_directory = argv[++i];
    else if (arg == "--open-jev-url" && i + 1 < argc) open_jev_url = argv[++i];
    else if (arg == "--port" && i + 1 < argc) port = static_cast<std::uint16_t>(std::stoi(argv[++i]));
    else if (arg == "--help") {
      std::cout << "usage: jevt_mario_demo [--web-root DIR] [--port PORT] "
                   "[--model LAYA_DIR | --open-jev-url http://127.0.0.1:8791]\n";
      return 0;
    }
  }
  std::shared_ptr<jevt::backend> backend;
  std::string label;
  if (!model_directory.empty() && !open_jev_url.empty())
    throw std::runtime_error("choose exactly one model backend: --model or --open-jev-url");
  if (!open_jev_url.empty()) {
#if JEVT_MARIO_HAS_OPEN_JEV
    jevt::remote_options options;
    options.api_key = "local-open-jev";  // Open-JEV ignores Authorization; JevT++ requires an explicit credential value.
    options.base_url = open_jev_url;
    options.model = "open-jev";
    options.allow_insecure_loopback = true;
    options.call_timeout = std::chrono::seconds{15};
    options.attempt_timeout = std::chrono::seconds{15};
    options.max_retries = 0;
    backend = std::make_shared<jevt::typesafe_backend>(std::move(options), std::make_shared<LoopbackTransport>());
    label = "Open-JEV model · JevT++ controller";
#else
    throw std::runtime_error("--open-jev-url requires JEVT_ENABLE_REMOTE=ON");
#endif
  } else if (!model_directory.empty()) {
#if JEVT_MARIO_HAS_LAYA
    jevt::laya_options options;
    options.model_directory = model_directory;
    options.provider = jevt::laya_provider::cuda;
    options.intra_op_threads = 2;
    options.inter_op_threads = 1;
    options.use_tf32 = false;
    backend = std::make_shared<jevt::laya_backend>(std::move(options));
    label = "Laya CUDA · device 0";
#else
    throw std::runtime_error("--model requires a build with JEVT_ENABLE_LAYA=ON and ONNX Runtime GPU");
#endif
  } else {
    backend = std::make_shared<DemoBackend>();
    label = "JevT++ local policy · preview";
  }
  if (!std::filesystem::exists(web_root / "index.html"))
    throw std::runtime_error("dashboard assets not found: " + web_root.string());
  DecisionEngine engine(std::move(backend), std::move(label));
  if (!model_directory.empty()) engine.provider = "CUDAExecutionProvider";
  DashboardServer(engine, std::filesystem::absolute(web_root), port).run();
  return 0;
} catch (const std::exception& exception) {
  std::cerr << "jevt_mario_demo: " << exception.what() << '\n';
  return 1;
}
