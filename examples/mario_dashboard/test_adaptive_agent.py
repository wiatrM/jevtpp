"""C++ online learner contracts; no emulator or model fabrication."""
import copy
import unittest
from test_fast_graph import BINARY, evaluate_many, node, observation
from judgment_contract import bind_plan_judgment


def state(frame=0, *, mode="hunter", lives=3):
    s=observation()
    s["player"].update(x=80, feet_y=208, screen_y=176, physics_vx=0, vx=0,
                       powerup_status="small")
    s["session"]={"run_id":1,"deaths":3-lives,"lives_remaining":lives,
                  "world":1,"stage":1,"session_frame":frame}
    s["collision"].update(behind_floor_safe=True, planning_columns=[
        {"x":x,"solid_y":[208,224]} for x in range(0,256,16)])
    s["strategy"]["mode"]=mode
    s["recent_control"]={"action":"right","stalled":0}
    return s


def enemy(x=120, *, slot=0, kind=6, stompable=True):
    return {"slot":slot,"kind_id":kind,"x":x,"screen_y":184,"stompable":stompable,
            "velocity_observed":True,"relative_velocity_x":-1,"relative_x_pixels":x-80,
            "relative_y_pixels":8}


def add(s,e):
    s["hazard"].update(nearby_enemies=[e],upcoming_enemies=[e],enemy_distance=e["x"]-80,
                       nearest_enemy_stompable=e["stompable"])
    return s


def run(*states):
    requests=[s if "command" in s else {"state":s,"frame":i,"epoch":1,"strategy":{}}
              for i,s in enumerate(states)]
    outputs=evaluate_many(requests)
    for out in outputs:
        if not out["ok"]: raise AssertionError(out)
    return outputs


@unittest.skipUnless(BINARY.is_file(), "Build controller first")
class AdaptiveAgentTests(unittest.TestCase):
    def test_tall_step_from_narrow_support_needs_early_vertical_clearance(self):
        s=state(mode="speedrun")
        s["player"].update(x=521,feet_y=192,screen_y=160,grounded=True,
                           physics_vx=1.4375)
        s["collision"].update(gap_distance_pixels=39,
            planning_columns=[{"x":x,"solid_y":[192] if x<560 else [128]}
                              for x in range(512,672,16)],
            landing_surfaces=[{"left_x":512,"right_x":560,"y":192},
                              {"left_x":560,"right_x":640,"y":128}])
        decision=run(s)[0]
        self.assertEqual(decision["arbitration"]["selected"],"platform_stage")
        self.assertEqual(decision["action"],"right_run_jump")

    def test_wide_support_with_32px_gap_stages_high_ledge_takeoff(self):
        s=state(mode="speedrun")
        s["player"].update(x=300,feet_y=192,screen_y=160,grounded=True,
                           physics_vx=3.0)
        s["collision"].update(gap_distance_pixels=40,
            planning_columns=[{"x":x,"solid_y":[192] if x<352 else
                               [144] if x>=384 else []}
                              for x in range(288,480,16)],
            landing_surfaces=[{"left_x":288,"right_x":352,"y":192},
                              {"left_x":384,"right_x":480,"y":144}])
        staging=run(s)[0]
        self.assertEqual(staging["arbitration"]["selected"],"platform_stage")
        self.assertEqual(staging["action"],"right_run")
        s["player"]["x"]=322
        ready=run(s)[0]
        self.assertEqual(ready["arbitration"]["selected"],"platform_stage")
        self.assertEqual(ready["action"],"right_run_jump")

    def test_large_two_slot_hazard_blocks_optimistic_crossing_plan(self):
        s=state(mode="speedrun")
        s["player"].update(x=2145,feet_y=160,screen_y=128,physics_vx=1.75,
                           vx=2,grounded=True)
        s["collision"].update(obstacle_distance_pixels=99,
            planning_columns=[{"x":x,"solid_y":[32,160]}
                              for x in range(2016,2256,16)],
            landing_surfaces=[{"left_x":2048,"right_x":2256,"y":160}])
        parts=[]
        for slot,x,y in ((2,2194,99),(3,2210,107)):
            parts.append({"slot":slot,"kind_id":45,"x":x,"screen_y":y,
                          "stompable":False,"velocity_observed":True,
                          "relative_velocity_x":-2,"vx":0,"vy":1,
                          "relative_x_pixels":x-2145,"relative_y_pixels":y-128})
        s["hazard"].update(upcoming_enemies=parts,nearby_enemies=parts,
                           enemy_distance=49,nearest_enemy_stompable=False)
        plans=node(run(s)[0],"candidate_plans")["output"]["candidates"]
        delayed=next(p for p in plans if p["signature"]=="traverse:right:28:6")
        self.assertTrue(delayed["predicted_collision"])

    def test_higher_support_across_gap_stages_takeoff(self):
        s=state(0,mode="speedrun")
        s["level"].update(world=1,stage=3)
        s["session"].update(world=1,stage=3)
        s["player"].update(x=1120,feet_y=144,screen_y=112,
                           grounded=True,physics_vx=1.5)
        s["collision"].update(gap_distance_pixels=48,
            planning_columns=[{"x":x,"solid_y":[144] if x<1168 else [96]}
                              for x in range(1120,1328,16)],
            landing_surfaces=[{"left_x":1120,"right_x":1168,"y":144},
                              {"left_x":1216,"right_x":1312,"y":96}])
        staging=run(s)[0]
        self.assertEqual(staging["arbitration"]["selected"],"platform_stage")
        self.assertEqual(staging["action"],"right_run")
        s["player"].update(x=1138,physics_vx=2.6)
        ready=run(s)[0]
        self.assertEqual(ready["arbitration"]["selected"],"platform_stage")
        self.assertEqual(ready["action"],"right_run_jump")

    def test_higher_support_requires_acceleration_before_takeoff(self):
        s=state(0,mode="speedrun")
        s["level"].update(world=1,stage=3)
        s["session"].update(world=1,stage=3)
        s["player"].update(x=551,feet_y=128,screen_y=96,
                           grounded=True,physics_vx=.8)
        s["collision"].update(gap_distance_pixels=89,
            planning_columns=[{"x":x,"solid_y":[128] if x<640 else [64]}
                              for x in range(560,752,16)],
            landing_surfaces=[{"left_x":560,"right_x":640,"y":128},
                              {"left_x":640,"right_x":752,"y":64}])
        staging=run(s)[0]
        self.assertEqual(staging["arbitration"]["selected"],"platform_stage")
        self.assertEqual(staging["action"],"right_run")
        s["player"].update(x=582,physics_vx=2.6)
        ready=run(s)[0]
        self.assertEqual(ready["arbitration"]["selected"],"platform_stage")
        self.assertEqual(ready["action"],"right_run_jump")

    def test_confirmed_ascent_keeps_jump_held_until_observed_platform_clearance(self):
        start=state(0,mode="speedrun")
        start["level"].update(world=1,stage=3)
        start["session"].update(world=1,stage=3)
        start["player"].update(x=580,feet_y=128,screen_y=96,grounded=True,
                               vy=0,physics_vx=2.6)
        start["collision"].update(gap_distance_pixels=43,
            planning_columns=[{"x":x,"solid_y":[128] if x<640 else [64]}
                              for x in range(560,752,16)],
            landing_surfaces=[{"left_x":560,"right_x":640,"y":128},
                              {"left_x":640,"right_x":752,"y":64}])
        start["recent_control"]={"action":"right_run"}
        states=[start]
        for frame,x,feet,vy in [(1,583,128,0),(2,586,122,6),
                                (3,589,116,5.8),(12,617,84,3)]:
            flight=copy.deepcopy(start)
            flight["session"]["session_frame"]=frame
            flight["player"].update(x=x,feet_y=feet,screen_y=feet-32,
                                    grounded=frame==1,vy=vy)
            flight["recent_control"]={"action":"right_jump"}
            states.append(flight)
        outputs=evaluate_many([{"state":s,"frame":s["session"]["session_frame"],
                               "epoch":1,"strategy":{}} for s in states])
        self.assertIn("jump",outputs[0]["action"])
        self.assertEqual(outputs[-1]["arbitration"]["selected"],"platform_hold")
        self.assertIn("jump",outputs[-1]["action"])
        self.assertEqual(outputs[-1]["phase"],"ascent_to_observed_support")

    def test_failed_high_target_wait_retreats_only_with_safe_floor_behind(self):
        s=add(state(),enemy(x=122))
        s["hazard"]["upcoming_enemies"][0]["screen_y"]=120
        s["collision"]["behind_floor_safe"]=True
        s["knowledge_affordances"]={"schema":"jevt.mario_affordance.v1",
            "star_contact_kinds":[],"death_zones":[{"stage":"1-1","x":112,
                "unsafe_waits":1,"target_kinds":[6],"warning_radius_px":112}]}
        result=run(s)[0]
        self.assertEqual(result["hunter_engagement"]["phase"],
                         "retreat_from_failed_wait")
        self.assertEqual(result["hunter_engagement"]["action"],"left")
        s["collision"]["behind_floor_safe"]=False
        unsafe=run(s)[0]
        self.assertNotEqual(unsafe["hunter_engagement"]["phase"],
                            "retreat_from_failed_wait")

    def test_fresh_model_rank_remains_advisory_to_hunter_target_skill(self):
        s=add(state(),enemy())
        offered=run(s)[0]["model_request"]
        selected=next(i for i,c in enumerate(offered["plan_candidates"])
                      if c["action"]=="jump")
        scores=[.05]*7
        scores[selected]=.7
        ranked=bind_plan_judgment({"ok":True,"judgment_kind":"plan",
            "selected_slot":selected,"slot_scores":scores},offered)
        ranked.update(epoch=1,source_frame=0,mode="hunter")
        result=evaluate_many([{"state":s,"frame":0,"epoch":1,
                               "reset":True,"strategy":ranked}])[0]
        self.assertTrue(result["plan_intent"]["accepted"])
        self.assertTrue(result["risk_budget"]["model_rank_applied"])
        self.assertEqual(result["arbitration"]["selected"],"target")
        self.assertNotEqual(result["action"],"jump")

    def test_cross_run_death_zone_vetoes_repeated_airborne_intercept(self):
        s=add(state(mode="hunter"),enemy(x=122))
        s["player"].update(x=80,grounded=False,feet_y=120,screen_y=88,
                           physics_vx=1.5,vy=-5)
        s["session"].update(world=1,stage=2,run_id=2)
        s["level"].update(world=1,stage=2)
        s["knowledge_affordances"]={"schema":"jevt.mario_affordance.v1",
            "star_contact_kinds":[],"death_zones":[{"stage":"1-2","x":112,
                "events":2,"runs":1,"warning_radius_px":112,
                "unsafe_intercepts":1,"target_kinds":[6]}]}
        result=run(s)[0]
        self.assertEqual(result["hunter_engagement"]["phase"],
                         "observed_death_zone_replan")
        self.assertFalse(result["hunter_engagement"]["apply"])
        self.assertNotEqual(result["source"],"jevtpp_target_skill")
        s["knowledge_affordances"]["death_zones"][0]["unsafe_intercepts"]=0
        unrelated=run(s)[0]
        self.assertNotEqual(unrelated["hunter_engagement"]["phase"],
                            "observed_death_zone_replan")

    def test_airborne_attack_cannot_override_observed_block_wall(self):
        for mode in ('hunter','score_attack'):
            s=add(state(mode=mode),enemy(x=148))
            s['player'].update(grounded=False,feet_y=185,screen_y=153,vy=-1)
            s['collision']['planning_columns'].append({'x':96,'solid_y':[144,160,176,192]})
            s['collision']['obstacle_distance_pixels']=4
            out=run(s)[0]
            self.assertEqual(out['persistent_tasks']['status'],'blocked_observed_blocks')
            self.assertTrue(out['persistent_tasks']['approach']['blocked'])
            self.assertFalse(out['hunter_engagement']['apply'])
            self.assertNotEqual(out['source'],'jevtpp_target_skill')
            self.assertEqual(out['persistent_tasks']['unresolved_targets'],1)

    def test_expired_task_does_not_continue_airborne_hunting(self):
        a=add(state(0),enemy());b=add(state(400),enemy())
        b['player'].update(grounded=False,feet_y=170,vy=-1)
        middle=[add(state(f),enemy()) for f in range(60,400,60)]
        out=run(a,*middle,b)[-1]
        self.assertEqual(out['persistent_tasks']['status'],'blocked_attempt_budget')
        self.assertFalse(out['hunter_engagement']['apply'])

    def test_observed_wall_removal_reenables_attack_without_fake_kill(self):
        a=add(state(0),enemy(x=148));b=add(state(1),enemy(x=148))
        a['collision']['planning_columns'].append({'x':112,'solid_y':[160,176,192]})
        first,second=run(a,b)
        self.assertEqual(first['persistent_tasks']['status'],'blocked_observed_blocks')
        self.assertEqual(second['persistent_tasks']['status'],'active')
        self.assertEqual(second['world_belief']['confirmed_eliminations'],0)

    def test_undershoot_feedback_survives_life_but_resets_at_game_over(self):
        a=state(0,mode="speedrun");a["collision"]["gap_distance_pixels"]=10
        a["collision"]["landing_surfaces"]=[{"left_x":144,"right_x":208,"y":192}]
        b=state(1,mode="speedrun");b["player"].update(x=90,grounded=False,feet_y=202,vy=6)
        b["recent_control"]["action"]="right_run_jump"
        c=state(2,mode="speedrun");c["player"].update(x=100,grounded=False,feet_y=217,vy=-5)
        c["recent_control"]["action"]="right_run"
        d=state(3,mode="speedrun",lives=2);d["session"]["life_lost"]=True
        e=state(4,mode="speedrun",lives=2)
        f=state(5,lives=0);f["session"]["game_over"]=True
        out=run(a,b,c,d,e,f)
        context=out[4]["action_dynamics"]["last_flight"]["context"]
        self.assertGreater(out[4]["action_dynamics"]["learned_minimum_hold_frames"][context],0)
        self.assertEqual(out[4]["action_dynamics"]["last_flight"]["failure_hypothesis"],"undershot_observed_support")
        self.assertEqual(out[5]["action_dynamics"]["learned_minimum_hold_frames"],{})

    def test_walk_off_cannot_teach_longer_jump_hold(self):
        a=state(0,mode="speedrun");a["collision"]["gap_distance_pixels"]=10
        a["collision"]["landing_surfaces"]=[{"left_x":144,"right_x":208,"y":192}]
        b=state(1,mode="speedrun");b["player"].update(x=90,grounded=False,feet_y=214,vy=-5)
        b["recent_control"]["action"]="right_run_jump"
        c=state(2,mode="speedrun");c["player"].update(x=100,grounded=False,feet_y=225,vy=-5)
        d=state(3,mode="speedrun",lives=2);d["session"]["life_lost"]=True
        out=run(a,b,c,d)[-1]["action_dynamics"]
        self.assertEqual(out["learned_minimum_hold_frames"],{})
        self.assertEqual(out["last_flight"]["failure_hypothesis"],"takeoff_not_observed")
        self.assertFalse(out["last_flight"].get("takeoff_confirmed",False))
        self.assertEqual(sum(e["failures"] for e in out["contextual_evidence"].values()),1)

    def test_learning_context_is_relative_but_distinguishes_target_geometry(self):
        a=state(0,mode="speedrun");a["collision"]["gap_distance_pixels"]=10
        a["collision"]["landing_surfaces"]=[{"left_x":144,"right_x":208,"y":192}]
        translated=copy.deepcopy(a);translated["player"]["x"]+=256
        for col in translated["collision"]["planning_columns"]: col["x"]+=256
        for surface in translated["collision"]["landing_surfaces"]:
            surface["left_x"]+=256;surface["right_x"]+=256
        different=copy.deepcopy(a);different["collision"]["landing_surfaces"][0]["right_x"]=160
        def key(s):
            following=copy.deepcopy(s);following["session"]["session_frame"]=1
            return run(s,following)[-1]["action_dynamics"]["active_flight"]["credit_key"]
        self.assertEqual(key(a),key(translated))
        self.assertNotEqual(key(a),key(different))

    def test_learning_physics_remains_shadow_until_predictive_evidence(self):
        samples=[state(i,mode="speedrun") for i in range(40)]
        for i,s in enumerate(samples):
            s["player"].update(x=80+3*i,grounded=False,physics_vx=3,vx=3,feet_y=160,vy=0)
            s["recent_control"]["action"]="left"
        early=run(*samples[:13])[-1]["action_dynamics"]["learning_supervisor"]["small:air:left:ax"]
        self.assertEqual(early["state"],"shadow")
        mature=run(*samples)[-1]["action_dynamics"]["learning_supervisor"]["small:air:left:ax"]
        self.assertEqual(mature["state"],"promoted")
        self.assertLess(mature["candidate_mae"],mature["prior_mae"])

    def test_ceiling_transition_does_not_train_gravity(self):
        a,b=state(0),state(1)
        for s in (a,b):
            s["player"].update(grounded=False,feet_y=160)
            s["recent_control"]["action"]="right_jump"
        a["player"]["vy"]=2;b["player"].update(vy=0,feet_y=160)
        out=run(a,b)[-1]["action_dynamics"]
        self.assertNotIn("small:gravity_hold",out["conditioned_statistics"])
        self.assertGreater(out["rejected_parameter_transitions"],0)

    def test_horizon_error_is_future_conditional_and_cancelled_on_action_change(self):
        samples=[state(i,mode="speedrun") for i in range(9)]
        for s in samples:
            s["recent_control"]["action"]="right"
        first=run(*samples)[-1]["action_dynamics"]
        self.assertGreater(first["horizon_prediction_error_px"]["8"]["samples"],0)
        samples[4]["recent_control"]["action"]="left"
        changed=run(*samples)[-1]["action_dynamics"]
        self.assertEqual(changed["horizon_prediction_error_px"]["8"]["samples"],0)
        self.assertGreater(changed["cancelled_forecasts"],0)

    def test_braking_is_learned_even_above_walking_speed(self):
        a,b=state(0),state(1)
        for s in (a,b):
            s["player"].update(grounded=False,physics_vx=3,vx=3,feet_y=160,vy=0)
            s["recent_control"]["action"]="left"
        b["player"]["x"]=83
        stats=run(a,b)[-1]["action_dynamics"]["conditioned_statistics"]
        self.assertEqual(stats["small:air:left:ax"]["samples"],1)
        self.assertEqual(stats["small:air:left:ax"]["mean"],0)

    def test_integer_two_pixel_acceleration_is_not_dropped(self):
        a,b=state(0),state(1)
        for s in (a,b):
            s["player"].update(grounded=False,feet_y=160)
            s["recent_control"]["action"]="right"
        a["player"]["vy"]=3;b["player"].update(vy=1,feet_y=159)
        stats=run(a,b)[-1]["action_dynamics"]["conditioned_statistics"]
        self.assertEqual(stats["small:gravity_release"]["samples"],1)
        self.assertEqual(stats["small:gravity_release"]["mean"],2)

    def test_repressing_jump_does_not_restore_hold_physics(self):
        a,b,c=state(0),state(1),state(2)
        for s in (a,b,c): s["player"].update(grounded=False,vy=3,feet_y=160)
        a["recent_control"]["action"]="right"
        b["recent_control"]["action"]="right_jump"
        c["recent_control"]["action"]="right_jump"
        out=run(a,b,c)[-1]
        self.assertTrue(out["action_dynamics"]["jump_cut"])

    def test_discontinuous_sensor_frame_does_not_train(self):
        a,b=state(0),state(1);b["player"]["velocity_valid"]=False
        self.assertEqual(run(a,b)[-1]["action_dynamics"]["samples"],0)

    def test_death_credit_uses_takeoff_not_last_falling_action(self):
        a=state(0,mode="speedrun");a["collision"]["gap_distance_pixels"]=10
        b=state(1,mode="speedrun");b["player"].update(grounded=False,feet_y=202,vy=6)
        b["recent_control"]["action"]="right_run_jump"
        c=state(2,mode="speedrun");c["player"].update(grounded=False,feet_y=210,vy=-5)
        c["recent_control"]["action"]="right_run"
        d=state(3,mode="speedrun",lives=2);d["session"]["life_lost"]=True
        out=run(a,b,c,d)[-1]
        self.assertEqual(out["risk_budget"]["failures_by_skill"].get("baseline:right_run_jump"),1)
        self.assertNotIn("baseline:right_run",out["risk_budget"]["failures_by_skill"])
        self.assertEqual(out["action_dynamics"]["last_flight"]["start_x"],80)

    def test_disappearance_not_elimination_and_target_persists(self):
        a,b=run(add(state(0),enemy()),state(1))
        self.assertEqual(a["persistent_tasks"]["target"]["id"],b["persistent_tasks"]["target"]["id"])
        self.assertEqual(b["world_belief"]["confirmed_eliminations"],0)
        self.assertEqual(b["persistent_tasks"]["status"],"unresolved_target_unobserved")

    def test_identity_specific_confirmation_counted_once(self):
        b=state(1);b["hazard"]["elimination_events"]=[{"slot":0,"kind_id":6,"x":119}]
        c=copy.deepcopy(b);c["session"]["session_frame"]=2
        outputs=run(add(state(),enemy()),b,c)
        self.assertEqual([o["world_belief"]["confirmed_eliminations"] for o in outputs],[0,1,1])
        self.assertIsNone(outputs[-1]["persistent_tasks"]["target"])

    def test_slot_reuse_does_not_resurrect_old_identity(self):
        a,b=run(add(state(),enemy()),add(state(1),enemy(x=250,kind=0)))
        self.assertEqual(len(b["world_belief"]["entities"]),2)
        self.assertEqual(b["world_belief"]["entities"][0]["status"],"unobserved")
        self.assertNotEqual(*[e["id"] for e in b["world_belief"]["entities"]])

    def test_unsupported_attack_is_explicitly_blocked(self):
        out=run(add(state(),enemy(kind=13,stompable=False)))[0]
        self.assertEqual(out["persistent_tasks"]["status"],"blocked_no_supported_attack")
        self.assertEqual(out["persistent_tasks"]["completion"],"not_proven")

    def test_hunter_execution_ownership_stays_with_active_target_skill(self):
        out=run(add(state(),enemy()))[0]
        combat=node(out,"hunter_engagement")["output"]
        self.assertTrue(combat["apply"])
        self.assertFalse(out["planner_applied"])
        self.assertEqual(out["action"],combat["action"])
        self.assertFalse(out["model_changed_action"])
        self.assertEqual(out["source"],"jevtpp_target_skill")

    def test_hunter_brakes_before_target_instead_of_running_past_it(self):
        s=add(state(),enemy(x=180));s["player"]["physics_vx"]=3
        out=run(s)[0]
        self.assertEqual(out["hunter_engagement"]["action"],"left")
        if out["arbitration"]["selected"]=="rollout":
            self.assertTrue(out["risk_budget"]["selected"]["predicted_stomp"])
        self.assertEqual(out["hunter_engagement"]["phase"],"approach_target")

    def test_hunter_waits_for_elevated_enemy_and_reports_incomplete(self):
        s=add(state(),enemy(x=140));s["hazard"]["nearby_enemies"][0]["screen_y"]=96
        out=run(s)[0]
        self.assertEqual(out["action"],"noop")
        self.assertEqual(out["hunter_engagement"]["phase"],"wait_for_reachable_target")
        self.assertEqual(out["persistent_tasks"]["clearance"],"incomplete")

    def test_hunter_release_and_verified_kill_use_ram_evidence(self):
        s=add(state(),enemy());s["recent_control"]["action"]="right_jump"
        a=run(s)[0]
        self.assertNotIn("jump",a["action"])
        killed=state(1);killed["hazard"]["elimination_events"]=[{"slot":0,"kind_id":6,"x":119}]
        a,b=run(add(state(),enemy()),killed)
        self.assertEqual(b["hunter_engagement"]["completion"],"RAM_confirmed")
        self.assertEqual(b["persistent_tasks"]["clearance"],"observed_targets_cleared")

    def test_unobserved_target_does_not_block_new_visible_hostile(self):
        a,b=run(add(state(),enemy()),add(state(1),enemy(x=160,slot=1)))
        self.assertNotEqual(a["persistent_tasks"]["target"]["id"],b["persistent_tasks"]["target"]["id"])
        self.assertEqual(b["world_belief"]["unresolved"],2)
        self.assertEqual(b["world_belief"]["confirmed_eliminations"],0)

    def test_only_executed_skill_is_credited_not_rejected_plan(self):
        a=add(state(0,mode="hunter"),enemy(x=168));b=state(1,lives=2,mode="hunter");b["session"]["life_lost"]=True
        c=add(state(2,lives=2,mode="hunter"),enemy(x=168))
        first,lost,retry=run(a,b,c)
        signature=first["skill_signature"]
        self.assertEqual(lost["risk_budget"]["failures_by_skill"][signature],1)
        self.assertEqual(sum(lost["risk_budget"]["failures_by_skill"].values()),1)
        if first["arbitration"]["selected"]=="rollout":
            self.assertEqual(signature,first["risk_budget"]["selected"]["signature"])
        else:
            self.assertNotIn(first["risk_budget"]["selected"]["signature"],lost["risk_budget"]["failures_by_skill"])

    def test_committed_gap_jump_cannot_be_cancelled_by_planner(self):
        a=add(state(),enemy())
        a["collision"]["gap_distance_pixels"]=10
        out=run(a)[0]
        self.assertFalse(out["planner_applied"])
        self.assertEqual(out["planner_reason"],"committed_geometry_guard")
        self.assertEqual(out["action"],"right_run_jump")

    def test_online_dynamics_uses_actual_action_and_measures_error(self):
        a=state(0,mode="speedrun");b=state(1,mode="speedrun")
        b["recent_control"]["action"]="right_run";b["player"].update(x=82,physics_vx=.2,vx=2)
        out=run(a,b)[-1]
        self.assertEqual(out["action_dynamics"]["samples"],1)
        self.assertIn("small:ground:right:dx",out["action_dynamics"]["conditioned_statistics"])
        self.assertEqual(out["action_dynamics"]["prediction_error_px"]["samples"],1)

    def test_duplicate_and_skipped_frames_do_not_train(self):
        a=state();b=state(4)
        out=run(a,a,b)[-1]
        self.assertEqual(out["action_dynamics"]["samples"],0)

    def test_memory_survives_life_loss_but_not_game_over_or_restart(self):
        a=state(0);b=state(1);c=state(2,lives=2);c["session"]["life_lost"]=True
        d=state(3,lives=2)
        e=state(4,lives=0);e["session"]["game_over"]=True
        outputs=run(a,b,c,d,e,{"command":"memory_reset"},state(5))
        self.assertEqual(outputs[1]["action_dynamics"]["samples"],1)
        self.assertEqual(outputs[2]["action_dynamics"]["samples"],1)
        self.assertEqual(outputs[2]["outcome_update"]["deaths_this_run"],1)
        self.assertEqual(outputs[4]["action_dynamics"]["samples"],0)
        self.assertEqual(outputs[6]["world_belief"]["unresolved"],0)

    def test_rollouts_have_sequences_and_risk_selects_feasible_candidate(self):
        out=run(add(state(mode="score_attack"),enemy()))[0]
        plans=node(out,"candidate_plans")["output"]["candidates"]
        self.assertGreater(len(plans),20)
        self.assertTrue(all(sum(x["frames"] for x in p["sequence"])<=48 for p in plans))
        selected=out["risk_budget"]["selected"]
        self.assertIsNotNone(selected)
        self.assertFalse(selected["predicted_collision"])
        self.assertLessEqual(selected["risk_cost"],out["risk_budget"]["budget"])
        self.assertEqual(out["arbitration"]["selected"],"rollout")
        self.assertTrue(out["planner_applied"])
        self.assertEqual(out["action"],selected["action"])

    def test_last_life_reduces_budget_not_blanket_disables_task(self):
        a=run(add(state(lives=3),enemy()))[0]
        b=run(add(state(lives=1),enemy()))[0]
        self.assertLess(b["risk_budget"]["budget"],a["risk_budget"]["budget"])
        self.assertEqual(b["persistent_tasks"]["status"],"active")

    def test_unknown_space_is_not_safe_empty_floor(self):
        a=add(state(),enemy());a["collision"]["planning_columns"]=[{"x":80,"solid_y":[208]}]
        out=run(a)[0];plans=node(out,"candidate_plans")["output"]["candidates"]
        self.assertTrue(any(p["unknown_space"] for p in plans))
        self.assertTrue(all(p["risk_cost"]>=.55 for p in plans if p["unknown_space"]))

    def test_terminal_guard_and_missing_geometry(self):
        a=add(state(),enemy());a["episode"]["dead"]=True
        out=run(a)[0]
        self.assertEqual(out["action"],"noop")
        self.assertFalse(out["planner_applied"])
        b=state();b["collision"]["available"]=False
        out=run(b)[0]
        self.assertEqual(out["action"],"noop")
        self.assertEqual(node(out,"candidate_plans")["output"]["candidates"],[])


if __name__=="__main__": unittest.main()
