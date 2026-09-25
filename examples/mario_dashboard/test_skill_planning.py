"""Contracts for skill arbitration, active budgets and support continuation."""
import copy
import unittest
from test_adaptive_agent import state, enemy, add, run
from test_fast_graph import BINARY, node


@unittest.skipUnless(BINARY.is_file(), "Build controller first")
class SkillPlanningTests(unittest.TestCase):
    def test_final_action_has_one_named_writer(self):
        for mode in ("hunter", "score_attack", "speedrun"):
            out = run(add(state(mode=mode), enemy()))[0]
            arbitration = out["arbitration"]
            selected = [p for p in arbitration["proposals"] if p["status"] == "selected"]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["id"], arbitration["selected"])
            self.assertTrue(arbitration["single_writer"])
            self.assertFalse(arbitration["post_selection_mutation"])

    def test_score_attack_uses_outcome_predicted_plan_before_heuristic_attack(self):
        out = run(add(state(mode="score_attack"), enemy()))[0]
        self.assertFalse(out["hunter_engagement"]["apply"])
        self.assertTrue(out["risk_budget"]["selected"]["predicted_stomp"])
        self.assertEqual(out["arbitration"]["selected"], "rollout")
        self.assertFalse(out["model_changed_action"])

    def test_terminal_wins_over_every_target_proposal(self):
        s = add(state(), enemy())
        s["session"]["life_lost"] = True
        out = run(s)[0]
        self.assertEqual(out["action"], "noop")
        self.assertEqual(out["arbitration"]["selected"], "reactive")

    def test_verified_takeoff_has_bounded_clearance_phase(self):
        a = add(state(0), enemy(x=110))
        b = add(state(1), enemy(x=110))
        b["player"].update(x=82,feet_y=203,vy=5,grounded=False)
        b["recent_control"]["action"]="right_jump"
        out=run(a,b)[-1]
        self.assertTrue(out["action_dynamics"]["active_flight"]["takeoff_confirmed"])
        self.assertEqual(out["arbitration"]["selected"],"takeoff_commitment")
        self.assertEqual(out["action"],"right_jump")
        self.assertFalse(out["model_changed_action"])

    def test_takeoff_clearance_cannot_hold_into_observed_ceiling(self):
        a = add(state(0), enemy(x=110))
        b = add(state(1), enemy(x=110))
        b["player"].update(x=82,feet_y=203,vy=5,grounded=False)
        b["recent_control"]["action"]="right_jump"
        b["collision"]["planning_columns"].append({"x":80,"solid_y":[160]})
        out=run(a,b)[-1]
        self.assertNotEqual(out["arbitration"]["selected"],"takeoff_commitment")

    def test_blocked_navigation_is_explicit_and_does_not_consume_attack_budget(self):
        snapshots = []
        for frame in range(0, 501, 50):
            s = add(state(frame), enemy(x=148))
            s["collision"]["planning_columns"].append({"x":112,"solid_y":[160,176,192]})
            snapshots.append(s)
        out = run(*snapshots)[-1]
        task = out["persistent_tasks"]
        self.assertEqual(task["status"], "blocked_observed_blocks")
        self.assertEqual(task["active_attempt_frames"], 0)
        self.assertEqual(task["subtasks"][0]["skill"], "navigate_around_observed_block")
        self.assertEqual(task["subtasks"][-1]["status"], "requires_RAM_evidence")
        self.assertEqual(task["confirmed_eliminations"], 0)

    def test_real_approach_progress_renews_active_budget(self):
        snapshots = []
        for frame in range(0, 401, 50):
            s = add(state(frame), enemy(x=180))
            s["player"]["x"] = 80 + frame/5
            snapshots.append(s)
        out = run(*snapshots)[-1]
        self.assertEqual(out["persistent_tasks"]["status"], "active")
        self.assertLess(out["persistent_tasks"]["active_attempt_frames"], 100)

    def test_platform_plans_report_second_support_and_contact_reserve(self):
        s = state(mode="speedrun")
        s["player"].update(x=285, feet_y=160, grounded=False, physics_vx=3, vx=3, vy=-1)
        s["recent_control"]["action"]="right_run_jump"
        s["collision"].update(gap_distance_pixels=999, obstacle_distance_pixels=999,
            planning_columns=[{"x":x,"solid_y":([192] if 288<=x<352 else [144] if 384<=x<512 else [])}
                              for x in range(192,576,16)],
            landing_surfaces=[{"left_x":288,"right_x":352,"y":192},
                              {"left_x":384,"right_x":512,"y":144}])
        plans = node(run(s)[0], "candidate_plans")["output"]["candidates"]
        self.assertTrue(any("brake:" in p["signature"] for p in plans))
        evaluated = [p["continuation"] for p in plans if p["continuation"]["status"] != "not_applicable"]
        self.assertTrue(evaluated)
        for c in evaluated:
            self.assertIn("required_input_contact_reserve_px", c)
            if c["status"] == "insufficient_takeoff_reserve":
                self.assertFalse(c["feasible"])
                self.assertEqual(c["evaluated_sequences"], 0)


if __name__ == "__main__":
    unittest.main()
