"""Integration regressions for the persistent C++ controller and strategy input."""
from __future__ import annotations
import argparse
import unittest
from pathlib import Path
from async_runner import FastControllerProcess
from nes_state import StateTracker
from test_nes_state import ram_fixture

EXECUTABLE = Path(__file__).resolve().parents[2] / "build-mario-cuda/examples/jevt_mario_control"


class ControllerRegressionTest(unittest.TestCase):
    def test_enemy_wave_before_nonstompable_wall_takes_off_early(self):
        self.state["player"].update(x=1479,physics_vx=3.0,grounded=True)
        self.state["collision"].update(obstacle_distance_pixels=157)
        self.state["hazard"].update(upcoming_enemies=[
            {"relative_x_pixels":99,"relative_y_pixels":8,"stompable":True},
            {"relative_x_pixels":123,"relative_y_pixels":8,"stompable":True},
            {"relative_x_pixels":177,"relative_y_pixels":-27,"stompable":False},
        ])
        result=self.step()
        self.assertEqual(result["action"],"right_run_jump")
        self.assertEqual(result["reason"],"enemy_wave_before_nonstompable")

    def test_nonstompable_moving_contact_triggers_avoidance_not_stomp(self):
        self.state["player"].update(x=1549,screen_y=128,feet_y=160,
                                    physics_vx=1.3125,grounded=True)
        self.state["collision"].update(obstacle_distance_pixels=135)
        self.state["hazard"].update(upcoming_enemies=[
            {"relative_x_pixels":47,"relative_y_pixels":16,"stompable":False,
             "kind_id":21,"kind":"enemy_0x15"}])
        result=self.step()
        self.assertEqual(result["action"],"right_run_jump")
        self.assertEqual(result["reason"],"avoid_nonstompable_contact")

    def test_short_observed_gap_uses_late_takeoff_window(self):
        self.state["player"].update(x=985,feet_y=208,screen_y=176,
                                    grounded=True,physics_vx=2.0)
        self.state["collision"].update(gap_distance_pixels=27,
            landing_surfaces=[{"left_x":944,"right_x":1024,"y":208},
                              {"left_x":1040,"right_x":1120,"y":208}])
        self.assertEqual(self.step()["action"],"right_run")
        self.state["player"].update(x=1000,physics_vx=2.6)
        self.state["collision"]["gap_distance_pixels"]=12
        self.assertEqual(self.step()["action"],"right_run_jump")

    def setUp(self):
        if not Path(EXECUTABLE).is_file():
            self.skipTest("Build jevt_mario_control before integration tests")
        self.control = FastControllerProcess(EXECUTABLE)
        self.state, _ = StateTracker(1).parse({"x_pos":160,"y_pos":79},ram_fixture(),
                                             previous_action="right_run",previous_reward=0,
                                             response_delay_frames=0)
        self.frame = 0
        # These tests pin the existing reactive fallback. The online planner is
        # covered separately with full trajectory/identity/outcome fixtures.
        self.state["planner_enabled"] = False
        self.state["episode"]["lives_remaining"] = 3

    def tearDown(self):
        self.control.close()

    def step(self, goal="finish_fast"):
        strategy = {"ok":True,"epoch":1,"source_frame":self.frame,
                    "mode":self.state["strategy"]["mode"],"active_goal":goal}
        if goal is None:
            strategy = {}
        result = self.control.step(self.state,strategy,1,self.frame,self.frame == 0)
        self.frame += 1
        return result

    def test_takeoff_input_latches_until_air_and_releases_after_landing(self):
        self.state["collision"]["gap_distance_pixels"] = 12
        self.assertEqual(self.step()["action"],"right_run_jump")
        self.assertEqual(self.step()["action"],"right_run_jump")
        self.state["player"]["grounded"] = False
        self.assertEqual(self.step()["action"],"right_run_jump")
        self.state["player"]["grounded"] = True
        self.assertEqual(self.step()["action"],"right_run")
        self.assertEqual(self.step()["action"],"right_run_jump")

    def test_collector_mode_changes_safe_collectible_approach(self):
        self.state["detectors"]["powerups_visible"] = 1
        self.state["detectors"]["powerup_distance"] = 32
        self.state["detectors"]["powerup_relative_y"] = 0
        baseline = self.step("finish_fast")
        self.assertEqual(baseline["action"],"right_run")
        self.assertFalse(baseline["strategy_applied"])
        self.assertFalse(baseline["model_changed_action"])
        self.assertEqual(baseline["model_effect"],"agrees_with_baseline")
        self.state["strategy"]["mode"] = "collector"
        result = self.step("collect_powerup")
        self.assertTrue(result["intent_accepted"])
        self.assertFalse(result["strategy_applied"])
        self.assertFalse(result["model_changed_action"])
        self.assertTrue(result["mode_changed_action"])
        self.assertEqual(result["baseline_action"],"right_run")
        self.assertEqual(result["action"],"right")

    def test_accepted_stomp_adds_alignment_beyond_mode_baseline(self):
        self.state["strategy"]["mode"] = "hunter"
        target = {"slot":0,"kind_id":6,"stompable":True,"relative_x_pixels":32,
                  "relative_y_pixels":8,"screen_y":184,"relative_velocity_x":-2,"velocity_observed":True}
        self.state["hazard"].update(enemy_distance=32,nearest_enemy_stompable=True,
                                   upcoming_enemies=[target],nearby_enemies=[target])
        self.assertEqual(self.step(None)["action"],"right_jump")
        self.state["player"].update(grounded=False,feet_y=174,screen_y=142,vy=4,vx=2,physics_vx=1.75)
        target.update(relative_x_pixels=10,relative_y_pixels=42)
        absent=self.step(None)
        self.assertFalse(absent["model_changed_action"])
        result=self.step("stomp_enemy")
        self.assertTrue(result["intent_accepted"])
        self.assertEqual(result["action"],"left")
        self.assertEqual(result["mode_baseline_action"],"right_jump")
        self.assertTrue(result["model_changed_action"])
        self.assertEqual(result["model_effect"],"stomp_air_alignment")

    def test_last_life_vetoes_optional_tactics_not_safety_jump(self):
        self.state["episode"]["lives_remaining"] = 1
        self.state["strategy"]["mode"] = "hunter"
        self.state["hazard"].update(enemy_distance=100,nearest_enemy_stompable=True,
                                   upcoming_enemies=[{"relative_x_pixels":100,"relative_y_pixels":8}])
        result=self.step("stomp_enemy")
        self.assertEqual(result["action"],"right_run")
        self.assertFalse(result["mode_changed_action"])
        self.state["collision"]["gap_distance_pixels"] = 12
        self.assertEqual(self.step()["action"],"right_run_jump")

    def test_hazard_overrides_optional_collectible_approach(self):
        self.state["detectors"]["powerups_visible"] = 1
        self.state["detectors"]["powerup_distance"] = 32
        self.state["collision"]["gap_distance_pixels"] = 12
        result = self.step("collect_powerup")
        self.assertEqual(result["action"],"right_run_jump")
        self.assertFalse(result["strategy_applied"])

    def test_missing_collision_evidence_fails_closed(self):
        self.state["collision"]["available"] = False
        self.assertEqual(self.step()["action"],"noop")

    def test_low_roof_ending_at_pit_delays_takeoff_until_clear_headroom(self):
        self.state["collision"].update(gap_distance_pixels=20,columns=[
            {"x":160,"floor_y":208,"solid_y":[144,208,224]},
            {"x":192,"floor_y":None,"solid_y":[32]}])
        self.assertEqual(self.step()["action"],"right_run")
        self.state["collision"]["gap_distance_pixels"]=4
        self.assertEqual(self.step()["action"],"right_run_jump")

    def test_wall_with_lower_passage_routes_down_instead_of_jumping(self):
        self.state["player"].update(feet_y=144,screen_y=112)
        self.state["collision"].update(obstacle_distance_pixels=4,behind_floor_safe=True,
            lower_passage={"entrance_x":144,"floor_y":208,"clearance_pixels":48})
        self.state["hazard"]["enemy_behind_distance"]=999
        result=self.step()
        self.assertEqual(result["action"],"left")
        self.assertEqual(result["reason"],"descend_to_observed_lower_passage")
        self.state["player"].update(x=144,feet_y=208,screen_y=176)
        self.state["collision"].update(obstacle_distance_pixels=None,lower_passage=None)
        self.assertEqual(self.step()["action"],"right_run")

    def test_observed_side_pipe_aligns_below_lip_then_enters_without_jump(self):
        self.state["player"].update(x=190,feet_y=128,screen_y=96)
        self.state["collision"].update(obstacle_distance_pixels=4,behind_floor_safe=True,
            side_pipe={"mouth_x":176,"approach_x":152,"floor_y":160})
        self.state["hazard"]["enemy_behind_distance"]=999
        self.assertEqual(self.step()["action"],"left")
        self.state["player"].update(x=152,feet_y=150,screen_y=118,grounded=False)
        self.assertEqual(self.step()["action"],"noop")
        self.state["player"].update(feet_y=160,screen_y=128,grounded=True)
        result=self.step()
        self.assertEqual(result["action"],"right")
        self.assertEqual(result["reason"],"enter_observed_side_pipe")

    def test_gap_landing_plan_brakes_for_observed_isolated_platform(self):
        self.state["player"].update(x=220,feet_y=208,physics_vx=3)
        self.state["collision"].update(gap_distance_pixels=24,
            landing_surfaces=[{"left_x":288,"right_x":352,"y":192,
                               "source":"observed_ram_isolated_support"}])
        takeoff=self.step()
        self.assertEqual(takeoff["action"],"right_run_jump")
        self.assertEqual(takeoff["landing_plan"]["target_x"],312)
        self.state["player"].update(x=310,grounded=False,feet_y=128,screen_y=96,vy=-2)
        result=self.step()
        self.assertEqual(result["action"],"left")
        self.assertEqual(result["reason"],"align_observed_platform_landing")
        self.assertFalse(result["model_changed_action"])
        self.assertEqual(result["baseline_action"],"left")

    def test_long_gap_intercepts_live_moving_support(self):
        self.state["player"].update(x=1280,feet_y=96,screen_y=64,
                                    grounded=True,physics_vx=3)
        self.state["collision"].update(gap_distance_pixels=20,gap_width_pixels=176,
            moving_surfaces=[{"left_x":1371,"right_x":1403,"y":128,
                              "vx":-1,"slot":2,"source":"observed_ram_moving_support"}])
        takeoff=self.step()
        self.assertEqual(takeoff["action"],"right_run_jump")
        self.assertEqual(takeoff["landing_plan"]["slot"],2)
        self.state["player"].update(x=1337,grounded=False,physics_vx=3,vy=2)
        self.state["trajectory"]={"airborne_frames":20}
        self.state["collision"]["moving_surfaces"][0].update(left_x=1361,right_x=1393)
        alignment=self.step()
        self.assertEqual(alignment["action"],"left")
        self.assertEqual(alignment["reason"],"intercept_observed_moving_support")
        self.assertEqual(alignment["landing_plan"]["right_x"],1393)
        self.assertFalse(alignment["model_changed_action"])

    def test_landing_plan_survives_nes_screen_y_wrap(self):
        self.state["player"].update(x=1001,y=207,feet_y=80,screen_y=48,
                                    grounded=True,physics_vx=2.1)
        self.state["collision"].update(gap_distance_pixels=11,
            landing_surfaces=[{"left_x":944,"right_x":1024,"y":208},
                              {"left_x":1040,"right_x":1120,"y":208}])
        self.assertEqual(self.step()["action"],"right_run_jump")
        self.state["player"].update(x=1091,y=292,feet_y=251,screen_y=219,
                                    grounded=False,vy=-1,physics_vx=3)
        result=self.step()
        self.assertEqual(result["action"],"left")
        self.assertEqual(result["reason"],"align_observed_platform_landing")

    def test_moving_support_builds_speed_before_far_transfer(self):
        self.state["player"].update(x=1344,y=159,feet_y=128,screen_y=96,
                                    grounded=True,physics_vx=0.9)
        self.state["collision"].update(gap_distance_pixels=0,gap_width_pixels=208,
            moving_surfaces=[{"left_x":1336,"right_x":1368,"y":128,"slot":2,
                              "source":"observed_ram_moving_support"},
                             {"left_x":1497,"right_x":1529,"y":144,"slot":3,
                              "source":"observed_ram_moving_support"}])
        runup=self.step()
        self.assertEqual(runup["action"],"right_run")
        self.assertEqual(runup["reason"],"runup_on_observed_moving_support")
        self.state["player"].update(x=1354,physics_vx=1.9)
        takeoff=self.step()
        self.assertEqual(takeoff["action"],"right_run_jump")

    def test_walkoff_to_lower_platform_also_acquires_landing_plan(self):
        self.state["player"].update(x=502,feet_y=148,screen_y=116,grounded=False,vy=-2,physics_vx=3)
        self.state["collision"].update(gap_distance_pixels=44,
            landing_surfaces=[{"left_x":512,"right_x":560,"y":192,
                               "source":"observed_ram_isolated_support"}])
        result=self.step()
        self.assertEqual(result["landing_plan"]["target_x"],528)
        self.assertEqual(result["reason"],"align_observed_platform_landing")
        self.assertFalse(result["model_changed_action"])

    def test_stalled_wall_recovery_requires_safe_rear_floor_and_no_enemy(self):
        self.state["recent_control"]["stalled"] = 90
        self.state["collision"]["obstacle_distance_pixels"] = 0
        self.state["collision"]["behind_floor_safe"] = True
        self.state["hazard"]["enemy_behind_distance"] = 999
        self.assertEqual(self.step()["action"],"left")
        self.state["hazard"]["enemy_behind_distance"] = 20
        self.assertNotEqual(self.step()["action"],"left")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-jsonl",type=Path,default=EXECUTABLE)
    args, remaining = parser.parse_known_args()
    EXECUTABLE = args.controller_jsonl
    unittest.main(argv=[__file__,*remaining])
