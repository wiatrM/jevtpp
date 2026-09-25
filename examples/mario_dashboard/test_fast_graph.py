"""Black-box tests against the C++ graph executable; no emulator/model needed.

Set JEVT_CONTROL_BINARY to override the default CUDA build path.
"""
import json
import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ.get("JEVT_CONTROL_BINARY", ROOT / "build-mario-cuda/examples/jevt_mario_control"))


def observation():
    return {
        "level": {"world": 1, "stage": 1},
        "player": {"grounded": True, "vx": 3, "vy": 0},
        "terrain": {"gap_distance": 999, "obstacle_distance": 999},
        "collision": {"available": True, "gap_distance_pixels": None, "obstacle_distance_pixels": None},
        "hazard": {"enemy_distance": 999, "upcoming_enemies": []},
        "strategy": {"mode": "speedrun"},
        "episode": {"dead": False, "stage_clear": False},
    }


def evaluate(state=None, *, epoch=2, frame=100, strategy=None):
    request = {"state": observation() if state is None else state, "epoch": epoch,
               "frame": frame, "reset": True, "strategy": strategy or {}}
    return evaluate_many([request])[0]


def evaluate_many(requests):
    result = subprocess.run([str(BINARY)], input="".join(json.dumps(request) + "\n" for request in requests),
                            text=True, capture_output=True, check=True, timeout=5)
    responses = [json.loads(line) for line in result.stdout.splitlines()]
    if len(responses) != len(requests):
        raise AssertionError("controller did not return one response per request")
    return responses


def node(result, name):
    return next(item for item in result["graph"]["nodes"] if item["id"] == name)


@unittest.skipUnless(BINARY.is_file(), f"Build controller first: {BINARY}")
class FastGraphTests(unittest.TestCase):
    def test_verified_star_contact_skill_requires_live_timer_evidence_and_clear_terrain(self):
        state=observation()
        state["strategy"]={"mode":"hunter"}
        state["player"].update(x=100,screen_y=176,feet_y=208,
                               star_active=True,star_timer_ram_0x079f=100)
        state["collision"].update(gap_distance_pixels=200,obstacle_distance_pixels=200)
        state["hazard"].update(enemy_distance=32,upcoming_enemies=[{
            "kind_id":6,"category":"hostile","x":132,"slot":0,
            "relative_x_pixels":32,"relative_y_pixels":16,"stompable":True}])
        state["knowledge_affordances"]={"schema":"jevt.mario_affordance.v1",
                                        "star_contact_kinds":[6]}
        accepted=evaluate(state)
        self.assertTrue(accepted["ok"],accepted.get("error"))
        self.assertEqual(accepted["arbitration"]["selected"],"star_contact")
        self.assertEqual(accepted["action"],"right_run")
        short=json.loads(json.dumps(state))
        short["player"]["star_timer_ram_0x079f"]=3
        rejected=evaluate(short)
        self.assertTrue(rejected["ok"],rejected.get("error"))
        self.assertNotEqual(rejected["arbitration"]["selected"],"star_contact")
        threshold=json.loads(json.dumps(state))
        threshold["player"]["star_timer_ram_0x079f"]=8
        self.assertEqual(evaluate(threshold)["arbitration"]["selected"],"star_contact")
        gap=json.loads(json.dumps(state))
        gap["collision"]["gap_distance_pixels"]=16
        rejected=evaluate(gap)
        self.assertTrue(rejected["ok"],rejected.get("error"))
        self.assertNotEqual(rejected["arbitration"]["selected"],"star_contact")

    def test_opt_in_memory_survives_game_over_but_default_clears(self):
        state=observation()
        state["session"]={"run_id":1,"deaths":0,"world":1,"stage":1,
                          "session_frame":1,"game_over":False,"retain_knowledge":True}
        state["player"].update(x=64,screen_y=176,feet_y=208,movement_state=0)
        state["collision"].update(visible_left_world_x=32,visible_right_world_x=288,
                                   columns=[{"x":80,"solid_y":[208,224]}])
        over=json.loads(json.dumps(state))
        over["session"].update(session_frame=2,game_over=True)
        again=json.loads(json.dumps(state))
        again["session"].update(run_id=2,session_frame=0)
        retained=evaluate_many([
            {"state":state,"epoch":1,"frame":1,"strategy":{},"include_memory":True},
            {"state":over,"epoch":1,"frame":2,"strategy":{},"include_memory":True},
            {"state":again,"epoch":2,"frame":0,"strategy":{},"include_memory":True},
        ])
        self.assertTrue(all(row["ok"] for row in retained))
        self.assertGreater(retained[1]["knowledge"]["observed_frames"],0)
        self.assertGreater(retained[2]["knowledge"]["observed_frames"],1)
        over["session"]["retain_knowledge"]=False
        cleared=evaluate_many([
            {"state":state,"epoch":1,"frame":1,"strategy":{},"include_memory":True},
            {"state":over,"epoch":1,"frame":2,"strategy":{},"include_memory":True},
        ])
        self.assertEqual(cleared[1]["knowledge"]["observed_frames"],0)

    def intent(self, **overrides):
        return {"ok": True, "epoch": 2, "source_frame": 50, "mode": "speedrun",
                "active_goal": "finish_fast", "source_snapshot": "2:50", **overrides}

    def test_live_payload_and_trace_come_from_current_snapshot(self):
        result = evaluate(strategy=self.intent())
        self.assertTrue(result["ok"])
        self.assertEqual(result["action"], "right_run")
        self.assertTrue(result["intent_accepted"])
        self.assertEqual(result["graph"]["snapshot_id"], "2:100")
        self.assertEqual(result["compact_state"]["velocity"], [3, 0])
        self.assertEqual(node(result, "intent_gate")["output"]["source_snapshot"], "2:50")
        self.assertTrue(all(item["snapshot_id"] == "2:100" for item in result["graph"]["nodes"]))
        edges = {(edge["from"], edge["to"]) for edge in result["graph"]["edges"]}
        self.assertIn(("intent_gate", "reactive_baseline"), edges)
        self.assertIn(("risk_budget", "controller"), edges)
        self.assertIn(("physics", "compact_state"), edges)

    def test_expired_future_wrong_epoch_and_wrong_mode_intent_rejected(self):
        for changes in ({"source_frame": 9}, {"source_frame": 101}, {"epoch": 1}, {"mode": "hunter"}):
            with self.subTest(changes=changes):
                result = evaluate(strategy=self.intent(**changes))
                self.assertTrue(result["ok"])
                self.assertFalse(result["intent_accepted"])

    def test_absent_target_rejects_model_goal(self):
        result = evaluate(strategy=self.intent(active_goal="stomp_enemy"))
        self.assertFalse(result["intent_accepted"])
        self.assertIn("not present", result["intent_reason"])

    def test_plan_judgment_cannot_leak_into_legacy_goal_authority(self):
        result = evaluate(strategy=self.intent(judgment_kind="plan", selected_slot=6,
                          goal_probabilities={"finish_fast": 1.0}))
        self.assertFalse(result["intent_accepted"])

    def test_hazard_overrides_model_cruise_intent(self):
        state = observation()
        state["collision"]["gap_distance_pixels"] = 10
        result = evaluate(state, strategy=self.intent())
        self.assertTrue(result["ok"])
        self.assertEqual(result["action"], "right_run_jump")
        self.assertEqual(result["reason"], "gap")

    def test_terminal_observation_reaches_controller(self):
        for terminal_key in ("dead", "stage_clear"):
            with self.subTest(terminal_key=terminal_key):
                state = observation()
                state["episode"][terminal_key] = True
                result = evaluate(state, strategy=self.intent())
                self.assertTrue(result["ok"])
                self.assertEqual(result["action"], "noop")
                self.assertEqual(result["phase"], "terminal")

    def test_invalid_input_fails_without_controller_output(self):
        result = evaluate({"player": {}})
        self.assertFalse(result["ok"])
        self.assertNotIn("action", result)
        self.assertEqual(node(result, "controller")["status"], "skipped")

    def test_model_collect_goal_reaches_controller_only_while_fresh_and_safe(self):
        state = observation()
        state["strategy"]["mode"] = "collector"
        state["detectors"] = {"powerups_visible": 1, "powerup_distance": 32}
        baseline = evaluate(state)
        fresh = evaluate(state, strategy=self.intent(mode="collector", active_goal="collect_powerup"))
        stale = evaluate(state, strategy=self.intent(mode="collector", active_goal="collect_powerup", source_frame=9))
        self.assertFalse(baseline["intent_accepted"])
        self.assertTrue(fresh["intent_accepted"])
        self.assertEqual(fresh["active_goal"], "collect_powerup")
        self.assertEqual(fresh["strategy_applied"], fresh["action"] != fresh["mode_baseline_action"])
        self.assertEqual(fresh["action"], baseline["action"])
        self.assertEqual(fresh["model_changed_action"], fresh["strategy_applied"])
        self.assertFalse(stale["intent_accepted"])
        self.assertFalse(stale["strategy_applied"])
        state["collision"]["gap_distance_pixels"] = 10
        hazard = evaluate(state, strategy=self.intent(mode="collector", active_goal="collect_powerup"))
        self.assertEqual(hazard["action"], "right_run_jump")
        self.assertFalse(hazard["strategy_applied"])

    def test_model_agreement_is_distinct_from_model_changing_controller_action(self):
        agreed = evaluate(strategy=self.intent())
        self.assertTrue(agreed["intent_accepted"])
        self.assertFalse(agreed["strategy_applied"])
        self.assertFalse(agreed["model_changed_action"])
        self.assertEqual(agreed["action"], agreed["baseline_action"])
        self.assertEqual(agreed["model_effect"], "agrees_with_baseline")
        state = observation()
        state["hazard"]["enemy_distance"] = 160
        state["hazard"]["nearest_enemy_stompable"] = True
        state["hazard"]["upcoming_enemies"] = [{"relative_x_pixels": 160, "relative_y_pixels": 0}]
        unsupported = evaluate(state, strategy=self.intent(active_goal="stomp_enemy"))
        self.assertFalse(unsupported["intent_accepted"])
        self.assertFalse(unsupported["model_changed_action"])
        self.assertIn("objective mode", unsupported["intent_reason"])

    def test_collector_mode_does_not_fabricate_model_goal(self):
        state = observation()
        state["strategy"]["mode"] = "collector"
        state["detectors"] = {"powerups_visible": 1, "powerup_distance": 32}
        result = evaluate(state)
        self.assertFalse(result["intent_accepted"])
        self.assertFalse(result["strategy_applied"])
        self.assertEqual(result["active_goal"], "")

    def test_intent_ttl_boundary_is_inclusive(self):
        self.assertTrue(evaluate(strategy=self.intent(source_frame=10))["intent_accepted"])
        self.assertFalse(evaluate(strategy=self.intent(source_frame=9))["intent_accepted"])

    def test_best_feasible_model_score_replaces_missing_raw_target_without_changing_scores(self):
        state = observation()
        state["strategy"]["mode"] = "collector"
        state["detectors"] = {"powerups_visible": 1, "powerup_distance": 32}
        scores = {"stomp_enemy": .6, "collect_powerup": .3, "finish_fast": .1}
        result = evaluate(state, strategy=self.intent(mode="collector", active_goal="stomp_enemy", goal_probabilities=scores))
        self.assertTrue(result["intent_accepted"])
        self.assertEqual(result["raw_model_goal"], "stomp_enemy")
        self.assertEqual(result["active_goal"], "collect_powerup")
        self.assertEqual(result["intent_selection"], "best_feasible_model_score")
        self.assertAlmostEqual(result["selected_goal_probability"], .3)
        self.assertEqual(node(result, "intent_gate")["output"]["goal_probabilities"], scores)
        candidates = {item["goal"]: item for item in node(result, "intent_gate")["output"]["goal_candidates"]}
        self.assertFalse(candidates["stomp_enemy"]["feasible"])
        self.assertTrue(candidates["collect_powerup"]["feasible"])

    def test_objective_mode_changes_goal_for_identical_ram_and_model_distribution(self):
        state = observation()
        state["hazard"].update(enemy_distance=100, nearest_enemy_stompable=True, upcoming_enemies=[
            {"relative_x_pixels": 100, "relative_y_pixels": 0, "kind_id": 6, "kind": "goomba", "stompable": True,
             "velocity_observed": True, "relative_velocity_x": -4}])
        scores = {"finish_fast": .9, "stomp_enemy": .1}
        speedrun = evaluate(state, strategy=self.intent(goal_probabilities=scores))
        state["strategy"]["mode"] = "hunter"
        hunter = evaluate(state, strategy=self.intent(mode="hunter", goal_probabilities=scores))
        self.assertEqual(speedrun["active_goal"], "finish_fast")
        self.assertEqual(hunter["active_goal"], "stomp_enemy")
        self.assertEqual(hunter["raw_model_goal"], "finish_fast")
        self.assertEqual(hunter["objective_mode"], "hunter")
        self.assertEqual(speedrun["action"], "right_run")
        self.assertEqual(hunter["action"], "right")
        self.assertTrue(hunter["mode_changed_action"])
        self.assertEqual(hunter["mode_effect"], "hostile_approach_walk")
        self.assertFalse(hunter["model_changed_action"])
        self.assertEqual(node(hunter, "objectives")["output"]["strategy"]["mode"], "hunter")
        choices = {item["goal"]: item for item in node(hunter, "intent_gate")["output"]["goal_candidates"]}
        self.assertTrue(choices["finish_fast"]["target_available"])
        self.assertFalse(choices["finish_fast"]["mode_allowed"])
        self.assertEqual(choices["finish_fast"]["probability"], .9)
        self.assertTrue(choices["stomp_enemy"]["mode_allowed"])

    def test_feasible_ranking_never_bypasses_ttl_epoch_or_mode(self):
        scores = {"stomp_enemy": .8, "finish_fast": .2}
        for changed in ({"source_frame": 9}, {"source_frame": 101}, {"epoch": 1}, {"mode": "hunter"}):
            with self.subTest(changed=changed):
                result = evaluate(strategy=self.intent(active_goal="stomp_enemy", goal_probabilities=scores, **changed))
                self.assertFalse(result["intent_accepted"])
                self.assertEqual(result["active_goal"], "")
                self.assertEqual(result["raw_model_goal"], "stomp_enemy")
                self.assertEqual(result["intent_selection"], "rejected")

    def test_hunter_can_rank_current_hostile_and_rejects_invalid_model_scores(self):
        state = observation()
        state["strategy"]["mode"] = "hunter"
        state["hazard"].update(enemy_distance=160, nearest_enemy_stompable=True, upcoming_enemies=[
            {"relative_x_pixels": 160, "relative_y_pixels": 0}])
        scores = {"collect_coins": .5, "stomp_enemy": .4, "finish_fast": .1}
        result = evaluate(state, strategy=self.intent(mode="hunter", active_goal="collect_coins", goal_probabilities=scores))
        self.assertTrue(result["intent_accepted"])
        self.assertEqual(result["active_goal"], "stomp_enemy")
        self.assertEqual(result["raw_model_goal"], "collect_coins")
        for invalid in ({"finish_fast": -1}, {"finish_fast": "0.8"}, {"finish_fast": 0}, [1]):
            with self.subTest(invalid=invalid):
                rejected = evaluate(state, strategy=self.intent(mode="hunter", goal_probabilities=invalid))
                self.assertFalse(rejected["intent_accepted"])

    def test_hunter_does_not_treat_non_stompable_or_unknown_hostile_as_stomp_target(self):
        state = observation()
        state["strategy"]["mode"] = "hunter"
        state["hazard"].update(enemy_distance=80, nearest_enemy_stompable=False,
            upcoming_enemies=[{"relative_x_pixels": 80, "relative_y_pixels": 0, "kind_id": 13, "stompable": False}])
        intent = self.intent(mode="hunter", active_goal="stomp_enemy",
            goal_probabilities={"stomp_enemy": .9, "finish_fast": .1})
        for has_capability in (True, False):
            with self.subTest(explicit_capability=has_capability):
                if not has_capability:
                    del state["hazard"]["nearest_enemy_stompable"]
                result = evaluate(state, strategy=intent)
                self.assertTrue(result["intent_accepted"])
                self.assertEqual(result["raw_model_goal"], "stomp_enemy")
                self.assertEqual(result["active_goal"], "finish_fast")
                self.assertFalse(result["compact_state"]["enemy_stompable"])
                choices = {item["goal"]: item for item in node(result, "intent_gate")["output"]["goal_candidates"]}
                self.assertFalse(choices["stomp_enemy"]["target_available"])

    def test_digital_twin_projects_current_observed_ram_and_enters_next_model_state(self):
        state = observation()
        state["player"].update(x=64, screen_y=176)
        state["hazard"].update(enemy_distance=80, upcoming_enemies=[
            {"kind_id": 6, "kind": "goomba", "category": "hostile", "slot": 0, "x": 144,
             "relative_x_pixels": 80, "relative_y_pixels": 0, "relative_velocity_x": -4, "velocity_observed": True}])
        result = evaluate(state)
        twin = result["digital_twin"]
        self.assertEqual(twin["source"], "current_ram_projection")
        self.assertEqual(twin["enemy"]["velocity_source"], "current_ram")
        self.assertEqual(twin["enemy"]["velocity_px_per_frame"], -1)
        self.assertEqual(twin["enemy"]["contact_eta_frames"], 20)
        self.assertEqual(twin["enemy"]["projected_distance_px"], 48)
        self.assertEqual(twin["landing"]["status"], "grounded")
        self.assertEqual(twin["landing"]["eta_frames"], 0)
        self.assertEqual(result["compact_state"]["forecast"]["enemy_eta_frames"], 20)
        self.assertIn("digital_twin", node(result, "compact_state")["dependencies"])
        self.assertEqual(node(result, "digital_twin")["output"], twin)
        state["collision"]["available"] = False
        unavailable = evaluate(state)["digital_twin"]
        self.assertEqual(unavailable["reliability"], "unavailable")
        self.assertIsNone(unavailable["enemy"]["contact_eta_frames"])

    def test_digital_twin_uses_empirical_velocity_only_after_samples_and_current_ram_wins(self):
        requests = []
        for frame in range(7):
            state = observation()
            state["player"].update(x=64 + frame * 3, screen_y=176)
            distance = 160 - 4 * frame
            state["hazard"].update(enemy_distance=distance, upcoming_enemies=[
                {"kind_id": 6, "kind": "goomba", "category": "hostile", "slot": 0, "x": 224 - frame,
                 "relative_x_pixels": distance, "relative_y_pixels": 0,
                 "relative_velocity_x": -8, "velocity_observed": frame == 6}])
            requests.append({"state": state, "epoch": 1, "frame": frame, "strategy": {}})
        results = evaluate_many(requests)
        self.assertIsNone(results[0]["digital_twin"]["enemy"]["contact_eta_frames"])
        self.assertIsNone(results[4]["digital_twin"]["enemy"]["contact_eta_frames"])
        empirical = results[5]["digital_twin"]["enemy"]
        self.assertEqual(empirical["velocity_source"], "empirical_same_type")
        self.assertEqual(empirical["velocity_samples"], 5)
        self.assertEqual(empirical["velocity_px_per_frame"], -1)
        self.assertEqual(empirical["contact_eta_frames"], 35)
        measured = results[6]["digital_twin"]["enemy"]
        self.assertEqual(measured["velocity_source"], "current_ram")
        self.assertEqual(measured["contact_eta_frames"], 17)
        self.assertEqual(measured["velocity_px_per_frame"], -5)

    def test_jump_latch_survives_epoch_change_without_physical_reset(self):
        grounded = observation()
        grounded["collision"]["gap_distance_pixels"] = 10
        airborne = json.loads(json.dumps(grounded))
        airborne["player"]["grounded"] = False
        inputs = [(2, grounded), (2, grounded), (3, airborne), (3, grounded), (3, grounded)]
        requests = [{"state": state, "epoch": epoch, "frame": frame, "reset": frame == 0, "strategy": {}}
                    for frame, (epoch, state) in enumerate(inputs)]
        results = evaluate_many(requests)
        self.assertTrue(all(result["ok"] for result in results))
        self.assertEqual([result["action"] for result in results],
                         ["right_run_jump", "right_run_jump", "right_run_jump", "right_run", "right_run_jump"])
        self.assertEqual(results[2]["graph"]["snapshot_id"], "3:2")

    def test_memory_uses_observed_columns_and_projects_bounded_current_context(self):
        state = observation()
        state["level"] = {"world": 1, "stage": 1}
        state["player"].update(x=64, screen_y=176, feet_y=208, movement_state=0)
        state["collision"].update(visible_left_world_x=32, visible_right_world_x=288,
            columns=[{"x": 80, "solid_y": [208, 224]}, {"x": 400, "solid_y": []}])
        results = evaluate_many([
            {"state": state, "epoch": 2, "frame": 1, "reset": True, "strategy": {}, "include_memory": True},
            {"state": state, "epoch": 2, "frame": 2, "strategy": {}, "include_memory": False},
        ])
        self.assertTrue(all(result["ok"] for result in results))
        memory = results[0]["knowledge"]
        self.assertEqual([column["x"] for column in memory["columns"]], [80])
        self.assertEqual(memory["context"]["known_columns_ahead"], 1)
        self.assertNotIn("knowledge", results[1])
        for frame, result in enumerate(results, 1):
            memory_node = node(result, "experience_memory")
            compact_node = node(result, "compact_state")
            self.assertEqual(memory_node["snapshot_id"], f"2:{frame}")
            self.assertIn("experience_memory", compact_node["dependencies"])
            self.assertEqual(result["compact_state"]["experience"]["known_ahead"], 1)
            self.assertNotIn("columns", result["compact_state"]["experience"])
            self.assertNotIn("graph", result["compact_state"]["experience"])
            self.assertLess(len(json.dumps(result["compact_state"])), 1100)

    def test_memory_retains_life_and_stage_observations_but_explicit_game_over_reset_clears_everything(self):
        requests = [{"command": "memory_export"}]
        for frame in range(4):
            state = observation()
            state["level"] = {"world": 1, "stage": 1}
            state["player"].update(x=64 + frame * 3, feet_y=176 if frame == 1 else 208,
                                   grounded=frame != 1, movement_state=1 if frame == 1 else 0)
            state["collision"].update(visible_left_world_x=32, visible_right_world_x=288,
                                      columns=[{"x": 80, "solid_y": [208, 224]}])
            state["hazard"]["upcoming_enemies"] = [{"kind_id": 6, "kind": "goomba", "category": "hostile",
                "slot": 0, "x": 160 - frame, "relative_x_pixels": 96 - 4 * frame}]
            state["episode"]["dead"] = frame == 3
            requests.append({"state": state, "epoch": 1, "frame": frame, "strategy": {}})
        stage_two = observation()
        stage_two["level"] = {"world": 1, "stage": 2}
        stage_two["player"].update(x=64, feet_y=208)
        stage_two["collision"].update(visible_left_world_x=32, visible_right_world_x=288,
                                      columns=[{"x": 80, "solid_y": []}])
        stage_two["episode"]["stage_clear"] = True
        requests.extend([
            {"state": stage_two, "epoch": 2, "frame": 0, "reset": True, "strategy": {}, "include_memory": True},
            {"command": "memory_export"},
        ])
        results = evaluate_many(requests)
        self.assertTrue(all(result["ok"] for result in results))
        empty, populated = results[0]["memory"], results[-1]["memory"]
        self.assertEqual(results[1]["learning_delta"]["new_columns"], 1)
        self.assertEqual(results[1]["learning_delta"]["new_enemy_types"], 1)
        self.assertEqual(results[1]["learning_delta"]["enemy_velocity_samples"], 0)
        self.assertIn("new_enemy_type", results[1]["risk_policy"]["reasons"])
        self.assertEqual(results[2]["learning_delta"]["enemy_velocity_samples"], 1)
        self.assertEqual(results[3]["learning_delta"]["jump_samples"], 1)
        self.assertEqual(results[3]["compact_state"]["learned"]["jumps"], 1)
        self.assertEqual(results[-2]["learning_delta"]["new_columns"], 1)
        self.assertEqual(results[-2]["learning_delta"]["enemy_velocity_samples"], 0)
        self.assertEqual({(col["world"], col["stage"], col["x"]) for col in populated["columns"]},
                         {(1, 1, 80), (1, 2, 80)})
        self.assertEqual(populated["jumps"]["duration_frames"]["count"], 1)
        self.assertGreater(populated["enemy_types"][0]["velocity_x_px_per_frame"]["count"], 0)
        self.assertEqual(populated["outcomes"], {"successes": 1, "failures": 1})
        self.assertEqual(results[-2]["knowledge"]["context"]["known_columns_ahead"], 1)
        self.assertEqual(results[-2]["knowledge"]["context"]["remembered_empty_columns_ahead"], 1)

        invalid = {**populated, "environment": "wrong-environment"}
        restored = evaluate_many([
            {"command": "memory_import", "memory": populated},
            {"command": "memory_export"},
            {"command": "memory_import", "memory": invalid},
            {"command": "memory_export"},
            {"command": "memory_reset"},
            {"command": "memory_export"},
            {"state": stage_two, "epoch": 1, "frame": 0, "strategy": {}, "include_memory": True},
        ])
        self.assertTrue(restored[0]["ok"])
        self.assertTrue(all(col["imported"] for col in restored[1]["memory"]["columns"]))
        self.assertFalse(restored[2]["ok"])
        self.assertEqual(restored[1]["memory"], restored[3]["memory"])
        self.assertTrue(restored[4]["ok"])
        self.assertEqual(restored[4]["memory"]["epoch"], -1)
        self.assertEqual(restored[4]["memory"]["context"]["jump_samples"], 0)
        self.assertEqual(restored[5]["memory"], empty)
        fresh = restored[6]["knowledge"]
        self.assertEqual(fresh["observed_frames"], 1)
        self.assertEqual(len(fresh["columns"]), 1)
        self.assertEqual(fresh["enemy_types"], [])
        self.assertEqual(fresh["jumps"]["duration_frames"]["count"], 0)

    def test_duplicate_observation_does_not_fabricate_learning_delta(self):
        state = observation()
        state["player"]["x"] = 64
        state["collision"]["columns"] = [{"x": 80, "solid_y": [208]}]
        request = {"state": state, "epoch": 1, "frame": 0, "strategy": {}}
        fresh, duplicate = evaluate_many([request, request])
        self.assertTrue(fresh["learning_delta"]["observation_accepted"])
        self.assertEqual(fresh["learning_delta"]["new_columns"], 1)
        self.assertFalse(duplicate["learning_delta"]["observation_accepted"])
        for key in ("new_columns", "columns_observed", "changed_columns", "new_enemy_types", "enemy_velocity_samples", "jump_samples"):
            self.assertEqual(duplicate["learning_delta"][key], 0)

    def test_lives_phase_and_uncertainty_policy_reach_model_and_motor(self):
        state = observation()
        state["strategy"]["mode"] = "hunter"
        state["session"] = {"run_id": 4, "world": 1, "stage": 2, "lives_remaining": 2, "memory_generation": 3}
        state["level"]["stage"] = 2
        state["hazard"].update(enemy_distance=100, nearest_enemy_stompable=True,
            upcoming_enemies=[{"relative_x_pixels": 100, "relative_y_pixels": 0, "stompable": True,
                              "velocity_observed": True, "relative_velocity_x": -4}])
        normal = evaluate(state)
        self.assertEqual(normal["game_phase"]["phase"], "playing")
        self.assertEqual(normal["game_phase"]["stage"], 2)
        self.assertEqual(normal["compact_state"]["lives"], 2)
        self.assertEqual(normal["risk_policy"]["stance"], "normal")
        self.assertEqual(normal["action"], "right")
        state["session"]["lives_remaining"] = 1
        cautious = evaluate(state)
        self.assertEqual(cautious["compact_state"]["risk"], "cautious")
        self.assertFalse(cautious["risk_policy"]["allow_aggressive_exploration"])
        self.assertIn("last_life", cautious["risk_policy"]["reasons"])
        self.assertNotEqual(cautious["action"], normal["action"])
        self.assertEqual(cautious["risk_effect"], "optional_tactics_vetoed_keep_reactive_safety")
        self.assertTrue({"game_phase", "learning_delta", "risk_policy"}.issubset(node(cautious, "controller")["dependencies"]))
        state["collision"]["gap_distance_pixels"] = 10
        self.assertEqual(evaluate(state)["action"], "right_run_jump")
        state["session"]["game_over"] = True
        terminal = evaluate(state)
        self.assertEqual(terminal["game_phase"]["phase"], "game_over")
        self.assertEqual(terminal["action"], "noop")
        state["session"]["game_over"] = False
        state["session"]["game_complete"] = True
        complete = evaluate(state)
        self.assertEqual(complete["compact_state"]["phase"], "game_complete")
        self.assertEqual(complete["action"], "noop")

    def test_unknown_ram_lifecycle_does_not_invent_world_or_evidence(self):
        state = observation()
        state["level"] = {"world": None, "stage": None}
        state["episode"] = {"dead": None, "stage_clear": None, "lives_remaining": None, "game_over": None, "full_game_completed": None}
        state["collision"]["available"] = False
        result = evaluate_many([{"state": state, "epoch": 1, "frame": 0, "strategy": {}, "include_memory": True}])[0]
        self.assertTrue(result["ok"])
        self.assertEqual(result["action"], "noop")
        self.assertEqual(result["game_phase"]["phase"], "unknown")
        self.assertIsNone(result["game_phase"]["world"])
        self.assertIsNone(result["game_phase"]["lives_remaining"])
        self.assertFalse(result["learning_delta"]["observation_accepted"])
        self.assertEqual(result["knowledge"]["observed_frames"], 0)
        self.assertEqual(result["knowledge"]["columns"], [])

    def test_post_skip_session_outcomes_are_counted_once_even_after_mode_epoch_change(self):
        state = observation()
        state["session"] = {"run_id": 1, "session_frame": 10, "stage_transition": True, "life_lost": False}
        requests = [{"state": state, "epoch": epoch, "frame": 10, "strategy": {}, "include_memory": True}
                    for epoch in (2, 3)]
        after_death = json.loads(json.dumps(state))
        after_death["session"].update(session_frame=20, stage_transition=False, life_lost=True)
        requests.extend({"state": after_death, "epoch": epoch, "frame": 20, "strategy": {}, "include_memory": True}
                        for epoch in (4, 5))
        results = evaluate_many(requests)
        self.assertTrue(all(result["ok"] for result in results))
        self.assertEqual([result["knowledge"]["outcomes"]["successes"] for result in results], [1, 1, 1, 1])
        self.assertEqual([result["knowledge"]["outcomes"]["failures"] for result in results], [0, 0, 1, 1])

    def test_mode_epoch_does_not_break_physical_samples_or_reobserve_same_frame(self):
        requests = []
        for epoch, frame in ((1, 0), (1, 1), (2, 1), (2, 2)):
            state = observation()
            state["player"].update(x=64 + frame * 3, feet_y=176 if frame == 1 else 208,
                                   grounded=frame != 1, movement_state=1 if frame == 1 else 0)
            state["session"] = {"run_id": 1, "deaths": 0, "world": 1, "stage": 1, "session_frame": frame}
            state["collision"]["columns"] = [{"x": 80, "solid_y": [208]}]
            state["hazard"]["upcoming_enemies"] = [{"kind_id": 6, "kind": "goomba", "category": "hostile",
                "slot": 0, "x": 160 - frame, "relative_x_pixels": 96 - 4 * frame}]
            requests.append({"state": state, "epoch": epoch, "frame": frame, "strategy": {}, "include_memory": True})
        results = evaluate_many(requests)
        self.assertTrue(all(result["ok"] for result in results))
        self.assertFalse(results[2]["learning_delta"]["observation_accepted"])
        self.assertEqual(results[2]["knowledge"]["epoch"], 2)
        final = results[-1]["knowledge"]
        self.assertEqual(final["observed_frames"], 3)
        self.assertEqual(final["columns"][0]["observations"], 3)
        self.assertEqual(final["jumps"]["duration_frames"]["count"], 1)
        self.assertEqual(final["enemy_types"][0]["velocity_x_px_per_frame"]["count"], 2)

    def test_full_game_flag_then_transition_is_one_clear_and_final_stage_clear_still_counts(self):
        before = observation()
        before["level"] = {"world": 8, "stage": 3}
        before["episode"]["stage_clear"] = True
        before["session"] = {"run_id": 1, "deaths": 0, "world": 8, "stage": 3, "session_frame": 10,
                             "stage_transition": False, "life_lost": False, "game_complete": False}
        after = json.loads(json.dumps(before))
        after["level"]["stage"] = 4
        after["episode"]["stage_clear"] = False
        after["session"].update(stage=4, session_frame=11, stage_transition=True)
        complete = json.loads(json.dumps(after))
        complete["session"].update(session_frame=12, stage_transition=False, game_complete=True)
        requests = [{"state": state, "epoch": epoch, "frame": frame, "strategy": {}, "include_memory": True}
                    for state, epoch, frame in ((before, 1, 10), (after, 2, 11), (after, 3, 11), (complete, 3, 12))]
        results = evaluate_many(requests)
        self.assertTrue(all(result["ok"] for result in results))
        self.assertEqual([result["knowledge"]["outcomes"]["successes"] for result in results], [0, 1, 1, 2])


if __name__ == "__main__":
    unittest.main()
