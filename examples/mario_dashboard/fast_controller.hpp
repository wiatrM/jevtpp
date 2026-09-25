#pragma once

#include <nlohmann/json.hpp>
#include <algorithm>
#include <set>
#include <cmath>
#include <string>

namespace mario {

// Physical reaction distances, not level coordinates. They can be overridden
// through the graph controller node's config without rebuilding the policy.
struct FastControllerConfig {
    int gap_takeoff_px = 28;
    int wall_takeoff_px = 40;
    int enemy_takeoff_px = 52;
    int jump_hold_frames = 32;
    int input_latch_frames = 4;
    int stalled_recovery_frames = 8;
    int runup_stall_frames = 80;
    int backoff_frames = 24;
    int runup_frames = 32;
};

class FastController {
public:
    explicit FastController(FastControllerConfig config = {}) : config_(config) {}
    // The planner can override a proposed action. Latch the action actually
    // sent to the emulator, not the discarded reactive proposal.
    void feedback_action(const std::string& action) {
        previous_jump_=action.find("jump")!=std::string::npos;
    }
    void reset() {
        jump_frames_ = 0;
        previous_jump_ = false;
        observed_airborne_ = false;
        reason_ = "cruise";
        backoff_remaining_ = runup_remaining_ = recovery_attempts_ = 0;
        attempted_blocks_.clear();
        hunt_target_slot_ = -1;
        resource_target_x_ = resource_target_y_ = -1;
        resource_frames_ = 0; resource_airborne_ = false;
        powerup_track_frames_ = 0;
        lower_route_frames_ = 0; lower_route_x_ = lower_route_floor_ = -1;
        pipe_frames_=0; pipe_floor_=pipe_mouth_=pipe_approach_=-1;
        pipe_aligned_=false;
        landing_target_=nlohmann::json::object();
        moving_takeoff_feet_=0;
    }

    nlohmann::json step(const nlohmann::json& state,
                        const nlohmann::json& strategy = nlohmann::json::object()) {
        using json = nlohmann::json;
        const auto player = state.value("player", json::object());
        const auto collision = state.value("collision", json::object());
        const auto hazard = state.value("hazard", json::object());
        const auto recent = state.value("recent_control", json::object());
        const auto episode = state.value("episode", json::object());
        const auto detectors = state.value("detectors", json::object());
        const std::string goal = strategy.value("active_goal", "finish_fast");
        const std::string mode = state.value("strategy",json::object()).value("mode","speedrun");
        const bool explore = state.value("risk_policy",json::object()).value("allow_aggressive_exploration",true);
        const bool intent_accepted = strategy.value("accepted", false);
        const bool grounded = player.value("grounded", false);
        const bool reliable = collision.value("available", false);
        const int gap = distance(collision, "gap_distance_pixels");
        const int wall = distance(collision, "obstacle_distance_pixels");
        bool low_ceiling = false;
        bool gap_column_seen = false, gap_headroom_clear = false;
        int drop_edge = 9999;
        for (const auto& column : collision.value("columns",json::array())) {
            const int dx = column.value("x",0) - player.value("x",0);
            if (!gap_column_seen && dx>=-8 && column.contains("floor_y") && column["floor_y"].is_null()) {
                gap_column_seen=true; gap_headroom_clear=true;
                for (const auto& tile_y : column.value("solid_y",json::array())) {
                    const int headroom=player.value("screen_y",0)-tile_y.get<int>();
                    if (headroom>=16 && headroom<=64) gap_headroom_clear=false;
                }
            }
            if (dx >= -8 && column.contains("floor_y") && column["floor_y"].is_number() &&
                column["floor_y"].get<int>() > player.value("feet_y",208) + 16)
                drop_edge = std::min(drop_edge,std::max(0,dx-12));
            if (dx < -8 || dx > 64) continue;
            for (const auto& tile_y : column.value("solid_y",json::array())) {
                const int headroom = player.value("screen_y",0) - tile_y.get<int>();
                if (headroom >= 16 && headroom <= 64) low_ceiling = true;
            }
        }
        int enemy = 9999;
        int lower_enemy = 9999;
        int nonstompable_ahead = 9999;
        int nonstompable_contact = 9999;
        int grounded_hostiles_ahead = 0;
        for (const auto& item : hazard.value("upcoming_enemies", json::array())) {
            const int dy = item.value("relative_y_pixels", 9999);
            const int dx = item.value("relative_x_pixels", 9999);
            if (item.value("stompable",false) && dy >= -32 && dy <= 40) {
                enemy = std::min(enemy, dx);
                if (dx >= 0 && dx <= 128) ++grounded_hostiles_ahead;
            }
            if (!item.value("stompable",false) && dx >= 0 && dx <= 200) {
                nonstompable_ahead = std::min(nonstompable_ahead,dx);
                if (dy >= -8 && dy <= 32)
                    nonstompable_contact = std::min(nonstompable_contact,dx);
            }
            if (dy > 40 && dy <= 128)
                lower_enemy = std::min(lower_enemy,dx);
        }
        if (episode.value("dead", json(false)) == true || episode.value("stage_clear", json(false)) == true) {
            reset();
            return {{"action", "noop"}, {"reason", "episode_terminal"},
                    {"phase", "terminal"}, {"source", "jevtpp_reactive_controller"},
                    {"strategy_applied", false}, {"active_goal", goal}, {"baseline_action","noop"},
                    {"model_changed_action",false}, {"model_effect","episode_terminal"}, {"mode_effect","none"}};
        }
        if (!reliable) {
            return {{"action", "noop"}, {"reason", "collision_evidence_unavailable"},
                    {"phase", "waiting"}, {"source", "jevtpp_reactive_controller"},
                    {"strategy_applied", false}, {"active_goal", goal}, {"baseline_action","noop"},
                    {"model_changed_action",false}, {"model_effect","missing_collision_evidence"}, {"mode_effect","none"}};
        }

        // A released A during a high-wall jump can leave Mario stuck without
        // running speed. Retry from a short, RAM-verified safe runway. This is
        // triggered by lack of progress, never a saved level coordinate.
        const bool safe_retreat = collision.value("behind_floor_safe",false) &&
                                  hazard.value("enemy_behind_distance",0) > 64;
        const auto side_pipe=collision.value("side_pipe",json(nullptr));
        if (pipe_floor_<0 && side_pipe.is_object() && safe_retreat && grounded &&
            player.value("feet_y",0)<=side_pipe.value("floor_y",0)+4 &&
            std::abs(side_pipe.value("mouth_x",0)-player.value("x",0))<=96) {
            pipe_floor_=side_pipe.value("floor_y",0);
            pipe_mouth_=side_pipe.value("mouth_x",0);
            pipe_approach_=side_pipe.value("approach_x",0);
            pipe_frames_=0; pipe_aligned_=false;
            previous_jump_=observed_airborne_=false; jump_frames_=0;
        }
        if (pipe_floor_>=0) {
            ++pipe_frames_;
            if (pipe_frames_>240 || hazard.value("enemy_behind_distance",999)<24 ||
                player.value("x",0)>pipe_mouth_+64 || player.value("x",0)<pipe_approach_-64) {
                pipe_floor_=-1;
            } else {
                if (grounded && std::abs(player.value("feet_y",0)-pipe_floor_)<=4)
                    pipe_aligned_=true;
                const auto action=pipe_aligned_ ? "right" :
                    player.value("x",0)>pipe_approach_ ? "left" : "noop";
                auto result=recovery_action(action,pipe_aligned_ ? "enter_observed_side_pipe" : "align_observed_side_pipe",goal);
                result["route_target"]={{"x",pipe_mouth_},{"floor_y",pipe_floor_},
                                        {"source","current_RAM_side_pipe"}};
                return result;
            }
        }
        const auto passage=collision.value("lower_passage",json(nullptr));
        if (lower_route_floor_<0 && grounded && safe_retreat && passage.is_object() && wall<=48) {
            lower_route_x_=passage.value("entrance_x",player.value("x",0));
            lower_route_floor_=passage.value("floor_y",208); lower_route_frames_=0;
            previous_jump_=observed_airborne_=false; jump_frames_=0;
        }
        if (lower_route_floor_>=0) {
            ++lower_route_frames_;
            if ((grounded && player.value("feet_y",0)>=lower_route_floor_-4) || lower_route_frames_>180 ||
                hazard.value("enemy_behind_distance",999)<24) {
                lower_route_floor_=-1;
            } else {
                const auto action=player.value("x",0)>lower_route_x_ ? "left" : "noop";
                auto result=recovery_action(action,"descend_to_observed_lower_passage",goal);
                result["route_target"]={{"x",lower_route_x_},{"floor_y",lower_route_floor_},
                                        {"source","current_RAM_corridor"}};
                return result;
            }
        }
        if (!backoff_remaining_ && !runup_remaining_ && grounded && safe_retreat &&
            wall <= 16 && recent.value("stalled",0) >= config_.runup_stall_frames) {
            backoff_remaining_ = config_.backoff_frames;
            runup_remaining_ = config_.runup_frames;
            ++recovery_attempts_;
            previous_jump_ = observed_airborne_ = false;
            jump_frames_ = 0;
        }
        if (backoff_remaining_) {
            if (grounded && safe_retreat) {
                --backoff_remaining_;
                return recovery_action("left","recovery_backoff",goal);
            }
            backoff_remaining_ = 0;
        }
        if (runup_remaining_) {
            --runup_remaining_;
            if (grounded && player.value("physics_vx",0.) < 1.5 && wall > 4)
                return recovery_action("right_run","recovery_runup",goal);
            runup_remaining_ = 0;
        }

        if (grounded && observed_airborne_) landing_target_=json::object();
        bool jump = false;
        std::string phase = grounded ? "grounded" : "airborne";
        if (grounded) {
            // SMB polls the previous frame's joypad value. A just-issued A can
            // still observe grounded for one frame. Do not misread it as a
            // landing and release A before the emulator accepts the jump.
            if (previous_jump_ && !observed_airborne_ && jump_frames_ < config_.input_latch_frames) {
                ++jump_frames_;
                jump = true;
                phase = "takeoff";
            } else {
                jump_frames_ = 0;
                observed_airborne_ = false;
                reason_ = "cruise";
                if (!previous_jump_) {
                    // When a low roof ends at the pit edge, an early jump hits
                    // its underside. Move to the edge before taking off into
                    // the RAM-verified open headroom above the gap.
                    int gap_takeoff = low_ceiling && gap_headroom_clear ? 8 : config_.gap_takeoff_px;
                    // A short gap between two observed supports permits a
                    // later takeoff. Jumping at the generic 28 px warning
                    // distance wastes horizontal range before the far edge
                    // and can miss a higher support beyond it.
                    if (gap > 0 && gap <= config_.gap_takeoff_px &&
                        player.value("physics_vx",0.) >= 1.5 && enemy > 64) {
                        const int px=player.value("x",0), feet=player.value("feet_y",208);
                        for (const auto& current : collision.value("landing_surfaces",json::array())) {
                            const int edge=current.value("right_x",0);
                            if (current.value("left_x",0)>px+8 || edge<px+24 ||
                                std::abs(current.value("y",0)-feet)>2) continue;
                            for (const auto& next : collision.value("landing_surfaces",json::array())) {
                                const int span=next.value("left_x",0)-edge;
                                if (span>=8 && span<=32 && next.value("right_x",0)-next.value("left_x",0)>=32 &&
                                    std::abs(next.value("y",0)-feet)<=16)
                                    gap_takeoff=std::min(gap_takeoff,12);
                            }
                        }
                    }
                    if (gap <= gap_takeoff) {
                        jump = true; reason_ = "gap";
                    } else if (wall <= config_.wall_takeoff_px) {
                        jump = true; reason_ = "obstacle";
                    } else if (nonstompable_contact <= 64 && wall > 64) {
                        // A moving non-stompable entity on Mario's current
                        // lane requires an avoidance jump, not a stomp plan.
                        // Waiting until contact distance is too late at run
                        // speed even if the entity is visible in RAM.
                        jump = true; reason_ = "avoid_nonstompable_contact";
                    } else if (drop_edge <= 28 && lower_enemy <= 128) {
                        jump = true; reason_ = "enemy_below_edge";
                    } else if (!low_ceiling && grounded_hostiles_ahead >= 2 &&
                               enemy <= 100 && nonstompable_ahead > enemy &&
                               nonstompable_ahead <= 192 && wall > 96 && wall <= 192 &&
                               player.value("physics_vx",0.) >= 2.) {
                        // A late stomp of the first enemy in a wave can land
                        // directly on a non-stompable foe atop the next wall.
                        // Use the observed wave and obstacle as one flight.
                        jump = true; reason_ = "enemy_wave_before_nonstompable";
                    } else if (enemy <= (low_ceiling ? 32 : config_.enemy_takeoff_px)) {
                        jump = true; reason_ = "enemy";
                    } else if (recent.value("stalled", 0) >= config_.stalled_recovery_frames) {
                        jump = true; reason_ = "unstick";
                    }
                } else reason_ = "release_landing";
            }
        } else {
            observed_airborne_ = true;
            ++jump_frames_;
            // Release A on descent, including a short ceiling-bumped jump.
            // This re-arms the NES edge-triggered jump before landing among
            // multiple enemies rather than wasting the next grounded frame.
            jump = previous_jump_ && jump_frames_ <= config_.jump_hold_frames &&
                   player.value("vy", 0.) >= 0.;
        }
        if (grounded && jump && reason_=="gap") {
            const int px=player.value("x",0), feet=player.value("feet_y",0);
            for (const auto& support : collision.value("moving_surfaces",json::array())) {
                const int left=support.value("left_x",0), right=support.value("right_x",0);
                if (left>px+8 || right<px+8 || std::abs(support.value("y",0)-feet)>3) continue;
                bool next_moving=false;
                for (const auto& next : collision.value("moving_surfaces",json::array()))
                    next_moving |= next.value("left_x",0)>right+64 &&
                        next.value("left_x",0)<px+220;
                if (next_moving && right-px>=18 && player.value("physics_vx",0.)<1.8) {
                    jump=false; reason_="runup_on_observed_moving_support";
                }
                break;
            }
        }
        const bool safe_corridor = grounded && !jump && gap > 96 && wall > 96 && enemy > 96;
        const bool collect = goal == "collect_powerup";
        const bool nearby_powerup = detectors.value("powerups_visible", 0) > 0 &&
            distance(detectors, "powerup_distance") <= 48;
        const std::string baseline_action = jump ? "right_run_jump" : "right_run";
        if (grounded && jump && reason_=="gap" && landing_target_.empty() &&
            distance(collision,"gap_width_pixels")>=96) {
            const int px=player.value("x",0), feet=player.value("feet_y",208);
            for (const auto& surface : collision.value("moving_surfaces",json::array())) {
                const int left=surface.value("left_x",0), right=surface.value("right_x",0);
                if (left>px+24 && left<px+160 && right>px+48 &&
                    surface.value("y",0)>=feet-64 && surface.value("y",0)<=feet+96) {
                    landing_target_=surface;
                    landing_target_["target_x"]=left+10;
                    moving_takeoff_feet_=feet;
                    break;
                }
            }
        }
        if (grounded && jump && reason_=="gap" && landing_target_.empty()) {
            const int edge=player.value("x",0)+12+gap;
            for (const auto& surface : collision.value("landing_surfaces",json::array())) {
                const int left=surface.value("left_x",0), right=surface.value("right_x",0);
                if (left>=edge && left-player.value("x",0)<200 && right-left>=32 &&
                    surface.value("y",0)>=player.value("feet_y",0)-64) {
                    landing_target_=surface;
                    landing_target_["target_x"]=(left+right)/2-8;
                    break;
                }
            }
        }
        if (!grounded && player.value("vy",0.)<0 && landing_target_.empty()) {
            // Walking down to a lower platform is also a landing problem;
            // it must not require that this flight began with an A press.
            for (const auto& surface : collision.value("landing_surfaces",json::array())) {
                const int left=surface.value("left_x",0), right=surface.value("right_x",0);
                if (right>player.value("x",0)+16 && left<=player.value("x",0)+80 &&
                    surface.value("y",0)>=player.value("feet_y",0)-4) {
                    landing_target_=surface;
                    landing_target_["target_x"]=(left+right)/2-8;
                    break;
                }
            }
        }
        if (!grounded && landing_target_.value("source",std::string())=="observed_ram_moving_support") {
            bool still_visible=false;
            for (const auto& surface : collision.value("moving_surfaces",json::array())) {
                if (surface.value("slot",-1)!=landing_target_.value("slot",-2)) continue;
                landing_target_=surface;
                still_visible=true;
                break;
            }
            if (!still_visible) landing_target_=json::object();
        }
        std::string action = baseline_action, mode_effect = "none", tactical_effect = "none";
        const bool attack_allowed=!state.value("attack_task_blocked",false);
        const bool hunt_mode = explore && attack_allowed && (mode == "hunter" || mode == "score_attack");
        const bool hunt_model = explore && attack_allowed && intent_accepted && goal == "stomp_enemy";
        const bool hunt = hunt_mode || hunt_model;
        const bool collect_mode = explore && (mode == "collector" || mode == "score_attack");
        const bool collect_model = explore && intent_accepted && (collect || goal == "score_attack");
        bool used_mode = false;
        // Targeted short hops replace a full-speed avoidance leap only on a
        // verified continuous floor. Geometry/recovery remain higher priority.
        bool model_modulation = false;
        std::string mode_action = baseline_action;
        std::string mode_tactic = "none";
        const int enemy_spacing = distance(hazard,"spacing_to_second_enemy_pixels");
        const bool room_to_target = hunt_model || !(low_ceiling && enemy_spacing <= 48);
        if (hunt && reason_ != "enemy_wave_before_nonstompable" &&
            reason_ != "avoid_nonstompable_contact" && room_to_target &&
            hazard.value("nearest_enemy_stompable",true) && gap > 96 && wall > 64 &&
            ((grounded && enemy <= 128) || (!grounded && reason_ == "enemy"))) {
            if (!grounded && jump_frames_ > 8) jump = false;
            action = jump ? "right_jump" : "right";
            tactical_effect = jump ? "targeted_stomp_hop" : "hostile_approach_walk";
            used_mode = hunt_mode;
            mode_action = action;
            mode_tactic = tactical_effect;
            if (grounded && !jump) hunt_target_slot_ = -1;
            if (grounded && jump && hunt_target_slot_ < 0 && !hazard.value("upcoming_enemies",json::array()).empty())
                hunt_target_slot_ = hazard.at("upcoming_enemies").front().value("slot",-1);
            if (hunt_model && !grounded && hunt_target_slot_ >= 0) {
                for (const auto& target : hazard.value("nearby_enemies",json::array())) {
                    if (target.value("slot",-1) != hunt_target_slot_) continue;
                    const double height = target.value("screen_y",0.) + 8 - player.value("feet_y",208.);
                    if (height < 0 || !target.value("velocity_observed",false)) break;
                    const double vy = player.value("vy",0.);
                    const double remaining = std::clamp((vy+std::sqrt(vy*vy+1.5*height))/.75,2.,24.);
                    const double target_vx = target.value("relative_velocity_x",0.) + player.value("vx",0.);
                    const double desired_vx = std::clamp(target.value("relative_x_pixels",0.)/remaining+target_vx,-1.5,1.75);
                    const double vx = player.value("physics_vx",0.);
                    if (vx > desired_vx + .25) { action = "left"; jump = false; }
                    else if (vx < desired_vx - .25) action = jump ? "right_jump" : "right";
                    else action = jump ? "jump" : "noop";
                    tactical_effect = "stomp_air_alignment";
                    model_modulation = action != mode_action;
                    break;
                }
            }
        } else if ((collect_mode || collect_model) && safe_corridor && nearby_powerup) {
            action = "right";
            tactical_effect = "slower_powerup_approach";
            used_mode = collect_mode;
            mode_action = action; mode_tactic = tactical_effect;
        } else if ((collect_mode || collect_model) && safe_corridor) {
            for (const auto& block : collision.value("overhead_blocks",json::array())) {
                const int dx = block.value("relative_x",9999);
                const auto key = std::to_string(block.value("x",0))+":"+std::to_string(block.value("y",0));
                if (dx >= 8 && dx <= 32 && !attempted_blocks_.count(key)) {
                    if (attempted_blocks_.size() < 128) attempted_blocks_.insert(key);
                    jump = true; reason_ = "observed_block_probe"; action = "right_jump";
                    tactical_effect = "probe_observed_overhead_block"; used_mode = collect_mode;
                    mode_action = action; mode_tactic = tactical_effect;
                    break;
                }
            }
        }
        // The RAM C1 metatile is the power-up question block. Align below a
        // currently observed block, hit its underside, then release it. The
        // finite attempt budget prevents farming an empty or unreachable box.
        const bool resources_allowed = (collect_mode || collect_model) && gap > 96 && enemy > 64 && lower_enemy > 64;
        if (resources_allowed && grounded && resource_target_x_ < 0 && !jump) {
            for (const auto& block : collision.value("overhead_blocks",json::array())) {
                const int dx = block.value("relative_x",9999);
                const auto key = std::to_string(block.value("x",0))+":"+std::to_string(block.value("y",0));
                if (block.value("tile_id",0) == 0xC1 && std::abs(dx) <= 64 && !attempted_blocks_.count(key) &&
                    (dx >= 0 || collision.value("behind_floor_safe",false))) {
                    // Player RAM x is the sprite's left edge; its head sample
                    // is x+8, so matching the tile's left edge hits its center.
                    resource_target_x_ = block.value("x",0); resource_target_y_ = block.value("y",0);
                    resource_frames_ = 0; resource_airborne_ = false;
                    break;
                }
            }
        }
        if (resource_target_x_ >= 0) {
            ++resource_frames_;
            if (!grounded) resource_airborne_ = true;
            if (!resources_allowed || resource_frames_ > 140 || (grounded && resource_airborne_)) {
                attempted_blocks_.insert(std::to_string(resource_target_x_)+":"+std::to_string(resource_target_y_));
                resource_target_x_ = -1;
            } else {
                const double dx = resource_target_x_ - player.value("x",0);
                const double vx = player.value("physics_vx",0.);
                const double desired = std::clamp(dx*.15,-1.5,1.5);
                jump = grounded ? (std::abs(dx)<=6 && std::abs(vx)<.75) : player.value("vy",0.) >= 0.;
                if (vx > desired+.2) { action = "left"; jump = false; }
                else if (vx < desired-.2) action = jump ? "right_jump" : "right";
                else action = jump ? "jump" : "noop";
                if (jump) reason_ = "powerup_block_hit";
                tactical_effect = jump ? "powerup_block_hit" : "powerup_block_alignment";
                used_mode = collect_mode; mode_action = action;
                mode_tactic = tactical_effect;
            }
        }
        if (detectors.value("powerups_visible",0)==0) powerup_track_frames_=0;
        if ((collect_mode || collect_model) && resource_target_x_ < 0 && grounded && powerup_track_frames_<300 &&
            detectors.value("powerups_visible",0)>0 && gap>96 && enemy>64 && lower_enemy>64) {
            const double pdx = distance(detectors,"powerup_distance");
            const double pdy = detectors.value("powerup_relative_y",0.);
            if (std::abs(pdx)<96) {
                ++powerup_track_frames_;
                // Stay within reach while the power-up emerges and walks off
                // its support. Do not let an unstick jump run away from it.
                const double desired = wall<=16 ? 0. : std::clamp((pdx+(pdy < -24 ? 12 : 0))*.15,-1.5,1.5);
                const double vx = player.value("physics_vx",0.);
                action = vx>desired+.2 ? "left" : vx<desired-.2 ? "right" : "noop";
                jump=false; used_mode=collect_mode; mode_action=action;
                tactical_effect="track_visible_powerup";
                mode_tactic=tactical_effect;
            }
        }
        // Maximum jump hold alone overshoots short treetop platforms. On
        // descent, steer towards a current-RAM landing interval, not a route
        // coordinate. This safety plan takes precedence over optional goals.
        if (!grounded && landing_target_.value("source",std::string())=="observed_ram_moving_support" &&
            state.value("trajectory",json::object()).value("airborne_frames",0)>=12) {
            const double duration=std::clamp(42.+(landing_target_.value("y",0)-moving_takeoff_feet_)/5.,32.,54.);
            const double remaining=std::max(5.,duration-
                state.value("trajectory",json::object()).value("airborne_frames",0));
            // Land near the inner half of the moving support. Touching only
            // its trailing edge leaves no runway for the following transfer.
            const double target=landing_target_.value("left_x",0.)+10.+
                landing_target_.value("vx",0.)*remaining;
            const double desired=std::clamp((target-player.value("x",0.))/remaining,-1.5,3.);
            const double vx=player.value("physics_vx",0.);
            action=vx>desired+.25 ? "left" : vx<desired-.25 ? "right_run" : "noop";
            jump=false; reason_="intercept_observed_moving_support";
            tactical_effect=reason_; used_mode=false; model_modulation=false;
            mode_action=action; mode_tactic="none";
            landing_target_["target_x"]=target;
            landing_target_["estimated_frames_to_contact"]=remaining;
        } else if (!grounded && !landing_target_.empty() && player.value("vy",0.)<0) {
            // NES screen_y wraps at 256 while y_pos remains continuous over
            // the top of the viewport. Comparing a wrapped feet_y=251 with a
            // platform at y=208 falsely declares that a rising jump already
            // passed the platform. The observed RAM y_pos maps to unwrapped
            // screen feet as (255 - y_pos) + sprite height 32.
            const double raw_unwrapped_feet=287.-player.value("y",0.);
            const double observed_feet=player.value("feet_y",0.);
            const bool y_consistent=std::abs(std::remainder(raw_unwrapped_feet-observed_feet,256.))<=2.;
            const double unwrapped_feet=y_consistent?raw_unwrapped_feet:observed_feet;
            const double height=landing_target_.value("y",0.)-unwrapped_feet;
            if (height>=-4 && player.value("x",0)<landing_target_.value("right_x",0)+16) {
                const double remaining=std::clamp(height/5.+2.,2.,32.);
                const double desired=std::clamp((landing_target_.value("target_x",0.)-player.value("x",0.))/remaining,-1.5,3.);
                const double vx=player.value("physics_vx",0.);
                action=vx>desired+.25 ? "left" : vx<desired-.25 ? "right_run" : "noop";
                jump=false; reason_="align_observed_platform_landing";
                tactical_effect=reason_; used_mode=false; model_modulation=false;
                mode_action=action; mode_tactic="none";
            } else {landing_target_=json::object();reason_="landing_target_missed";}
        }
        previous_jump_ = jump;
        const bool changed = action != baseline_action;
        const bool mode_changed = used_mode && mode_action != baseline_action;
        const bool landing_control=reason_=="align_observed_platform_landing" ||
            reason_=="intercept_observed_moving_support";
        const bool applied = !landing_control && (model_modulation || (!used_mode && intent_accepted && changed));
        if (mode_changed) mode_effect = mode_tactic;
        const std::string model_effect = applied ? "slower_powerup_approach" :
            !intent_accepted ? "no_current_accepted_intent" :
            !safe_corridor ? "reactive_safety_priority" :
            goal == "finish_fast" ? "agrees_with_baseline" :
            used_mode ? "mode_already_handles_goal" :
            collect ? "no_safe_nearby_powerup" : "goal_has_no_motor_handler";
        return {{"action", action}, {"reason", changed ? tactical_effect : reason_},
                {"phase", phase}, {"source", "jevtpp_reactive_controller"},
                {"active_goal", goal}, {"strategy_applied", applied},
                {"strategy_reason", model_effect}, {"baseline_action",landing_control ? action : baseline_action},
                {"model_changed_action",applied}, {"model_effect",applied ? tactical_effect : model_effect},
                {"mode_baseline_action",mode_action},
                {"risk_effect",explore ? "optional_tactics_allowed" : "optional_tactics_vetoed_keep_reactive_safety"},
                {"mode_effect",mode_effect}, {"mode_changed_action",mode_changed},
                {"jump_hold_frames", jump_frames_},
                {"recovery_attempts",recovery_attempts_},
                {"landing_plan",landing_target_.empty() ? json(nullptr) : landing_target_},
                {"evidence", {{"ground_source", player.value("ground_source", "unknown")},
                              {"landing_target",landing_target_.empty() ? json(nullptr) : landing_target_},
                              {"gap_distance_pixels", gap == 9999 ? json(nullptr) : json(gap)},
                              {"wall_distance_pixels", wall == 9999 ? json(nullptr) : json(wall)},
                              {"enemy_distance_pixels", enemy == 9999 ? json(nullptr) : json(enemy)}}}};
    }

private:
    nlohmann::json recovery_action(const std::string& action,const std::string& phase,const std::string& goal) const {
        return {{"action",action},{"baseline_action",action},{"phase",phase},{"reason",phase},
                {"source","jevtpp_reactive_controller"},{"active_goal",goal},{"strategy_applied",false},
                {"model_changed_action",false},{"model_effect","reactive_safety_priority"},
                {"strategy_reason","verified_safe_runup_recovery"},{"mode_effect","none"},
                {"recovery_attempts",recovery_attempts_},{"jump_hold_frames",0}};
    }
    static int distance(const nlohmann::json& value, const char* key) {
        const auto it = value.find(key);
        return it != value.end() && it->is_number() ? it->get<int>() : 9999;
    }
    FastControllerConfig config_;
    int jump_frames_ = 0;
    bool previous_jump_ = false;
    bool observed_airborne_ = false;
    std::string reason_ = "cruise";
    int backoff_remaining_ = 0, runup_remaining_ = 0, recovery_attempts_ = 0;
    std::set<std::string> attempted_blocks_;
    int hunt_target_slot_ = -1;
    int resource_target_x_ = -1, resource_target_y_ = -1, resource_frames_ = 0;
    bool resource_airborne_ = false;
    int powerup_track_frames_ = 0;
    int lower_route_x_ = -1, lower_route_floor_ = -1, lower_route_frames_ = 0;
    int pipe_floor_=-1, pipe_mouth_=-1, pipe_approach_=-1, pipe_frames_=0;
    bool pipe_aligned_=false;
    nlohmann::json landing_target_=nlohmann::json::object();
    int moving_takeoff_feet_=0;
};

} // namespace mario
