"""Deterministic worker tests; no CUDA, model service or emulator required."""
from concurrent.futures import Future
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest
from unittest.mock import MagicMock, patch

import async_runner
from run_nes_demo import native_info, current_environment_info

try:
    import numpy as np
except ImportError:
    np = None


class ManualExecutor:
    def __init__(self, **_):
        self.jobs = []

    def submit(self, function):
        future = Future()
        self.jobs.append((future, function))
        return future

    def complete(self):
        future, function = self.jobs.pop(0)
        try:
            future.set_result(function())
        except Exception as error:
            future.set_exception(error)

    def shutdown(self, **_):
        pass


class CurrentEnvironmentInfoTests(unittest.TestCase):
    def test_reads_native_state_past_env_compatibility_unwrapped(self):
        native=SimpleNamespace(_get_info=lambda: {"world":1,"stage":3,"x_pos":40})
        compatibility=SimpleNamespace(env=native)
        environment=SimpleNamespace(env=compatibility,unwrapped=compatibility)
        event={"world":1,"stage":2,"x_pos":3114}
        self.assertEqual(current_environment_info(environment,event),
                         {"world":1,"stage":3,"x_pos":40})


class StrategyWorkerTests(unittest.TestCase):
    def test_knowledge_context_is_front_loaded_without_mutating_canonical_graph(self):
        offered={"context_id":"ctx","plan_candidates":[{"id":"route","slot":0}]}
        graph={"schema":"jevt.knowledge_context.v1","statements":[
            {"subject":f"ENEMY {i}","relation":"observed_failure_context",
             "object":"WORLD 1-2","status":"empirical","samples":i,
             "source":"verified RAM"} for i in range(5)]}
        request=async_runner.attach_knowledge_context(offered,graph)
        self.assertEqual(next(iter(request)),"knowledge_head")
        self.assertEqual(len(request["knowledge_head"]),3)
        self.assertEqual(len(request["knowledge_graph"]["statements"]),5)
        self.assertEqual(request["knowledge_head"][0]["relation"],
                         "observed_failure_context")
        self.assertNotIn("knowledge_head",offered)

    @staticmethod
    def plan_request():
        return {"context_id":"support-A", "plan_candidates":[
            {"slot":0,"id":"jump:short","skill":"transfer","action":"right_jump"}]}

    @staticmethod
    def plan_response(**extra):
        return {"ok":True,"selected_slot":0,"slot_scores":[.6,.05,.05,.05,.05,.05,.15],**extra}

    def setUp(self):
        self.executor_patch = patch.object(async_runner, "ThreadPoolExecutor", ManualExecutor)
        self.executor_patch.start()
        self.call_patch = patch.object(async_runner, "call_model", return_value={
            "ok": True, "active_goal": "finish_fast", "model_id": "fixture"
        })
        self.call = self.call_patch.start()
        self.worker = async_runner.StrategyWorker("http://fixture/api/strategy")
        self.addCleanup(self.executor_patch.stop)
        self.addCleanup(self.call_patch.stop)
        self.addCleanup(self.worker.close)

    def test_single_inflight_request_does_not_queue_old_observations(self):
        self.assertTrue(self.worker.submit({"grounded": True}, 1, 10, "speedrun"))
        self.assertFalse(self.worker.submit({"grounded": False}, 1, 20, "speedrun"))
        self.assertEqual(len(self.worker.pool.jobs), 1)
        self.assertFalse(self.worker.poll(1, "speedrun"))
        self.worker.pool.complete()
        self.assertTrue(self.worker.poll(1, "speedrun"))
        self.assertEqual(self.worker.latest["source_frame"], 10)
        self.assertEqual(self.worker.latest["source_snapshot"], "1:10")
        self.assertEqual(self.worker.completed, 1)
        self.assertTrue(self.worker.submit({}, 1, 21, "speedrun"))

    def test_model_switch_captures_original_route_and_discards_old_result(self):
        self.worker.submit({},1,10,'speedrun')
        self.worker.switch_url('http://laya/api/strategy')
        self.worker.pool.complete()
        self.assertEqual(self.call.call_args.args[0],'http://fixture/api/strategy')
        self.worker.poll(1,'speedrun')
        self.assertEqual(self.worker.latest,{})
        self.assertEqual(self.worker.discarded,1)
        self.worker.submit({},1,11,'speedrun');self.worker.pool.complete();self.worker.poll(1,'speedrun')
        self.assertEqual(self.call.call_args.args[0],'http://laya/api/strategy')
        self.assertEqual(self.worker.latest['source_frame'],11)

    def test_reset_discards_old_episode_success(self):
        self.worker.submit({}, 1, 10, "speedrun")
        self.worker.reset()
        self.worker.pool.complete()
        self.worker.poll(2, "speedrun")
        self.assertEqual(self.worker.latest, {})
        self.assertEqual(self.worker.discarded, 1)
        self.assertEqual(self.worker.completed, 0)
        self.assertIsNone(self.worker.future)

    def test_mode_switch_discards_response_even_without_epoch_change(self):
        self.worker.submit({}, 1, 10, "speedrun")
        self.worker.pool.complete()
        self.worker.poll(1, "collector")
        self.assertEqual(self.worker.latest, {})
        self.assertEqual(self.worker.discarded, 1)

    def test_current_error_is_reported_and_next_success_clears_error(self):
        self.call.return_value = {"ok": False, "error": "GPU unavailable"}
        self.worker.submit({}, 1, 10, "speedrun")
        self.worker.pool.complete()
        self.worker.poll(1, "speedrun")
        self.assertEqual(self.worker.error, "GPU unavailable")
        self.assertIsNone(self.worker.future)
        self.call.return_value = {"ok": True, "active_goal": "finish_fast"}
        self.worker.submit({}, 1, 20, "speedrun")
        self.worker.pool.complete()
        self.worker.poll(1, "speedrun")
        self.assertIsNone(self.worker.error)
        self.assertEqual(self.worker.completed, 1)

    def test_old_episode_error_does_not_pollute_current_episode(self):
        self.call.side_effect = TimeoutError("old episode request timeout")
        self.worker.submit({}, 1, 10, "speedrun")
        self.worker.reset()
        self.worker.pool.complete()
        self.worker.poll(2, "speedrun")
        self.assertIsNone(self.worker.error)
        self.assertEqual(self.worker.discarded, 1)

    def test_response_cannot_override_request_provenance(self):
        self.call.return_value = {"ok": True, "epoch": 999, "mode": "hunter", "source_frame": 1000,
                                  "source_snapshot": "wrong", "active_goal": "finish_fast"}
        self.worker.submit({}, 2, 30, "collector")
        self.worker.pool.complete()
        self.worker.poll(2, "collector")
        self.assertEqual(self.worker.latest["epoch"], 2)
        self.assertEqual(self.worker.latest["mode"], "collector")
        self.assertEqual(self.worker.latest["source_frame"], 30)
        self.assertEqual(self.worker.latest["source_snapshot"], "2:30")

    def test_plan_request_routes_to_plan_endpoint_and_seals_slot_identity(self):
        self.call.return_value=self.plan_response(plan_context="forged",model_plan="unoffered",
            epoch=999,mode="hunter",source_frame=900,source_snapshot="forged")
        self.worker.submit(self.plan_request(),2,30,"speedrun")
        self.worker.pool.complete();self.worker.poll(2,"speedrun")
        self.assertEqual(self.call.call_args.args[0],"http://fixture/api/plan")
        result=self.worker.latest
        self.assertEqual(result["plan_context"],"support-A")
        self.assertEqual(result["model_plan"],"jump:short")
        self.assertEqual(result["source_snapshot"],"2:30")
        self.assertEqual(result["epoch"],2)
        self.assertEqual(result["mode"],"speedrun")

    def test_plan_snapshot_is_sealed_against_nested_caller_mutation(self):
        self.call.return_value=self.plan_response()
        request=self.plan_request()
        self.worker.submit(request,1,10,"speedrun")
        request["context_id"]="support-B"
        request["plan_candidates"][0]["id"]="different-plan"
        request["plan_candidates"][0]["action"]="left"
        self.worker.pool.complete();self.worker.poll(1,"speedrun")
        self.assertEqual(self.call.call_args.args[1]["context_id"],"support-A")
        self.assertEqual(self.worker.latest["model_plan"],"jump:short")
        self.assertEqual(self.worker.latest["plan_context"],"support-A")

    def test_plan_defer_cannot_smuggle_legacy_goal_authority(self):
        self.call.return_value=self.plan_response(selected_slot=6,active_goal="finish_fast",
            goal_probabilities={"finish_fast":1.},goal_scores={"finish_fast":1.})
        self.worker.submit(self.plan_request(),1,10,"speedrun")
        self.worker.pool.complete();self.worker.poll(1,"speedrun")
        result=self.worker.latest
        self.assertEqual(result["model_plan"],"")
        self.assertEqual(result["judgment_kind"],"plan")
        for legacy in ("active_goal","goal_probabilities","goal_scores"):
            self.assertNotIn(legacy,result)

    def test_empty_candidates_do_not_issue_model_call_or_manufacture_response(self):
        request=self.plan_request();request["plan_candidates"]=[]
        self.assertFalse(self.worker.submit(request,1,10,"speedrun"))
        self.call.assert_not_called()
        self.assertEqual(self.worker.pool.jobs,[])
        self.assertEqual(self.worker.latest,{})
        self.assertEqual(self.worker.last_submit_frame,10)

    def test_invalid_distribution_is_error_not_fresh_model_authority(self):
        self.call.return_value=self.plan_response(slot_scores=[float("nan")]*7)
        self.worker.submit(self.plan_request(),1,10,"speedrun")
        self.worker.pool.complete();self.worker.poll(1,"speedrun")
        self.assertEqual(self.worker.latest,{})
        self.assertEqual(self.worker.completed,0)
        self.assertIn("Invalid model plan distribution",self.worker.error)

    def test_model_switch_discards_pending_plan_and_keeps_original_route(self):
        self.call.return_value=self.plan_response()
        self.worker.submit(self.plan_request(),1,10,"speedrun")
        self.worker.switch_url("http://laya/api/strategy")
        self.worker.pool.complete();self.worker.poll(1,"speedrun")
        self.assertEqual(self.call.call_args.args[0],"http://fixture/api/plan")
        self.assertEqual(self.worker.latest,{})
        self.assertEqual(self.worker.discarded,1)


class AsyncLoopTests(unittest.TestCase):
    def run_scripted_episode(self, mode_switch=False, numpy_terminal_api=None, full_game_script=None, initial_info=None):
        """Execute the real loop with two observations and no model/emulator."""
        live = async_runner.LiveState()
        worker = SimpleNamespace(latest={}, completed=0, discarded=0, error=None,
                                 future=None, last_submit_frame=-10000,
                                 poll=MagicMock(), reset=MagicMock(), submit=MagicMock(), close=MagicMock())
        controller, environment, tracker = MagicMock(), MagicMock(), MagicMock()
        controller.exchange.return_value = {"ok": True, "memory": {"columns": [], "observed_frames": 0}}
        current_info = initial_info or {}
        actual_step_environment = async_runner.step_environment
        if numpy_terminal_api:
            info = {"flag_get": np.bool_(True), "x_pos": np.int64(3266),
                    "metadata": {"flags": [np.bool_(False), np.int16(2), np.float32(.5)]}}
            if numpy_terminal_api == 5:
                environment.step.return_value = ("frame", np.float32(2), np.bool_(True), np.bool_(False), info)
            else:
                environment.step.return_value = ("frame", np.float32(2), np.bool_(True), info)

        def parse(info, *_, **kwargs):
            terminal = bool(info.get("flag_get"))
            return {"episode": {"stage_clear": terminal, "dead": info.get("life") == 255},
                    "level": {"world": info.get("world", 1), "stage": info.get("stage", 1)},
                    "player": {"grounded": False},
                    "strategy": {"mode": kwargs["objective_mode"]}}, {}

        def control(state, strategy, epoch, frame, reset=False, include_memory=False):
            terminal = state["episode"]["stage_clear"] or state["session"]["game_over"] or state["session"]["game_complete"]
            action = "noop" if terminal else "right_run_jump"
            result = {"ok": True, "action": action, "reason": "episode_terminal" if terminal else "gap",
                      "phase": "terminal" if terminal else "airborne", "source": "jevtpp_reactive_controller",
                      "compact_state": {"mode": state["strategy"]["mode"]},
                      "digital_twin": {"source": "fixture_forecast"},
                      "game_phase": {"phase": "game_over" if state["session"]["game_over"] else "playing"},
                      "learning_delta": {"new_columns": 1}, "risk_policy": {"stance": "normal"},
                      "graph": {"snapshot_id": f"{epoch}:{frame}", "nodes": [], "edges": []}}
            if include_memory:
                result["knowledge"] = {"frame": frame, "columns": []}
            return result

        tracker.parse.side_effect = parse
        controller.step.side_effect = control
        steps = 0

        def step(*_):
            nonlocal steps, current_info
            steps += 1
            if full_game_script is not None:
                event, current_info, terminated = full_game_script[steps - 1]
                return "frame", 1., terminated, False, event
            if mode_switch and steps == 1:
                live.command = "mode:collector"
                return "frame", 1., False, False, {}
            return "frame", 2., True, False, {"flag_get": True}

        args = SimpleNamespace(controller=str(Path(__file__).resolve()), engine=str(Path(__file__).resolve()),
            open_jev_url=None, model_url="http://fixture/api/decision", env="SuperMarioBros-v0" if full_game_script is not None else "fixture-env", seed=1,
            dashboard_port=0, mode="speedrun", fps=100000, dashboard_fps=12,
            max_decisions=20, frames_per_decision=30, max_frames=20, exit_on_terminal=True)
        with TemporaryDirectory(prefix="jevt-async-test-") as temporary, ExitStack() as stack:
            replacements = {
                "__file__": str(Path(temporary) / "async_runner.py"),
                "start_engine": MagicMock(return_value=None),
                "FastControllerProcess": MagicMock(return_value=controller),
                "StrategyWorker": MagicMock(return_value=worker),
                "LiveState": MagicMock(return_value=live),
                "ThreadingHTTPServer": MagicMock(), "StateTracker": MagicMock(return_value=tracker),
                "create_environment": MagicMock(return_value=environment),
                "reset_environment": MagicMock(return_value=("frame", current_info)),
                "current_environment_info": lambda *_: current_info,
                "step_environment": actual_step_environment if numpy_terminal_api else step,
                "unwrap_ram": lambda _: None, "encode_frame": lambda _: b"frame",
            }
            for name, value in replacements.items():
                stack.enter_context(patch.object(async_runner, name, value))
            stack.enter_context(patch.object(async_runner.threading, "Thread", MagicMock()))
            stack.enter_context(patch.object(async_runner.time, "sleep", lambda _: None))
            stack.enter_context(patch("builtins.print"))
            async_runner.run(args)
            reset_count = replacements["reset_environment"].call_count
            run_log = next((Path(temporary) / "artifacts").glob("async-*.jsonl"))
            self.last_log_records = [json.loads(line) for line in run_log.read_text(encoding="utf-8").splitlines()]
            self.last_terminal_log = next(item for item in reversed(self.last_log_records) if item["type"] == "terminal")
            self.last_controller_factory = replacements["FastControllerProcess"]
        return live.payload, controller, worker, reset_count

    def test_terminal_decision_matches_terminal_graph_and_action(self):
        payload, controller, _, _ = self.run_scripted_episode()
        self.assertEqual(payload["status"], "clear")
        self.assertEqual(payload["action"], "noop")
        self.assertEqual(payload["decision"]["action"], "noop")
        self.assertEqual(payload["decision"]["reason"], "episode_terminal")
        self.assertEqual(payload["decision"]["gate_reason"], "episode_terminal")
        self.assertEqual(payload["decision"]["phase"], "terminal")
        self.assertEqual(payload["graph"]["snapshot_id"], "1:1")
        self.assertTrue(payload["graph_current"])
        self.assertEqual(controller.save_memory.call_count, 2)

    def test_active_mode_switch_invalidates_intent_without_resetting_jump_latch(self):
        payload, controller, worker, reset_count = self.run_scripted_episode(mode_switch=True)
        calls = controller.step.call_args_list
        self.assertEqual(calls[0].args[2:5], (1, 0, True))
        self.assertEqual(calls[1].args[2:5], (2, 1, False))
        self.assertEqual(reset_count, 1)
        worker.reset.assert_called_once()
        self.assertEqual(payload["mode"], "collector")
        self.assertEqual(payload["graph"]["snapshot_id"], "2:2")

    def test_decision_does_not_duplicate_trace_memory_or_model_input(self):
        payload, _, _, _ = self.run_scripted_episode()
        self.assertIn("graph", payload)
        self.assertIn("knowledge", payload)
        for duplicated in ("graph", "knowledge", "compact_state", "digital_twin", "game_phase", "learning_delta", "risk_policy"):
            self.assertNotIn(duplicated, payload["decision"])
        self.assertEqual(payload["digital_twin"]["source"], "fixture_forecast")
        self.assertEqual(payload["learning_delta"]["new_columns"], 1)

    def test_full_game_continues_stage_and_life_then_clears_memory_only_at_game_over(self):
        initial = {"world": 1, "stage": 1, "life": 2}
        script = [
            ({**initial, "flag_get": True}, {"world": 1, "stage": 2, "life": 2}, False),
            ({"world": 1, "stage": 2, "life": 2}, {"world": 1, "stage": 2, "life": 1}, False),
            ({"world": 1, "stage": 2, "life": 1}, {"world": 1, "stage": 2, "life": 0}, False),
            ({"world": 1, "stage": 2, "life": 255}, {"world": 1, "stage": 2, "life": 255}, True),
        ]
        payload, controller, worker, resets = self.run_scripted_episode(full_game_script=script, initial_info=initial)
        self.assertEqual(payload["status"], "game_over")
        self.assertEqual(payload["session"]["stages_cleared"], 1)
        self.assertEqual(payload["session"]["deaths"], 3)
        self.assertEqual(payload["session"]["lives_remaining"], 0)
        self.assertEqual(payload["session"]["memory_generation"], 2)
        self.assertEqual(resets, 1)
        self.assertIsNone(self.last_controller_factory.call_args.args[1])
        self.assertEqual(worker.reset.call_count, 3)
        controller.exchange.assert_called_once_with({"command": "memory_reset"})
        self.assertEqual([call.args[2:5] for call in controller.step.call_args_list[:4]],
                         [(1, 0, True), (2, 1, True), (3, 2, True), (4, 3, True)])
        self.assertEqual(controller.step.call_args_list[1].args[0]["level"]["stage"], 2)
        self.assertFalse(controller.step.call_args_list[1].args[0]["episode"]["stage_clear"])
        self.assertEqual(payload["knowledge"], {"columns": [], "observed_frames": 0})
        self.assertFalse(payload["graph_current"])
        self.assertEqual(sum(item["type"] == "knowledge_reset" for item in self.last_log_records), 1)

    def test_final_8_4_clear_finishes_full_game_without_erasing_successful_run_knowledge(self):
        initial = {"world": 8, "stage": 4, "life": 1}
        final = {**initial, "flag_get": True}
        payload, controller, _, resets = self.run_scripted_episode(
            full_game_script=[(final, final, False)], initial_info=initial)
        self.assertEqual(payload["status"], "game_complete")
        self.assertEqual(payload["session"]["stages_cleared"], 1)
        self.assertEqual(payload["session"]["memory_generation"], 1)
        self.assertEqual(payload["action"], "noop")
        self.assertEqual(resets, 1)
        self.assertTrue(payload["graph_current"])
        controller.exchange.assert_not_called()

    @unittest.skipIf(np is None, "NumPy is required; run with .venv-mario/bin/python")
    def test_numpy_terminal_scalars_pass_real_emulator_boundary_and_json_log(self):
        # Keep the real step_environment/native_info path in the real async loop.
        # Gym's old four-tuple and new five-tuple APIs both produce NumPy scalars.
        for api in (4, 5):
            with self.subTest(gym_step_tuple_size=api):
                payload, controller, _, _ = self.run_scripted_episode(numpy_terminal_api=api)
                self.assertEqual(payload["status"], "clear")
                self.assertIs(type(payload["terminated"]), bool)
                self.assertIs(type(payload["truncated"]), bool)
                self.assertIs(type(payload["info"]["flag_get"]), bool)
                self.assertIs(type(payload["info"]["x_pos"]), int)
                self.assertEqual(payload["info"]["metadata"]["flags"], [False, 2, .5])
                self.assertEqual(json.loads(json.dumps(payload))["action"], "noop")
                self.assertEqual(self.last_terminal_log["info"], payload["info"])
                self.assertEqual(self.last_terminal_log["episode_reward"], 2.)
                self.assertTrue(self.last_terminal_log["flag_get"])
                self.assertEqual(controller.save_memory.call_count, 2)


@unittest.skipIf(np is None, "NumPy is required; run with .venv-mario/bin/python")
class NativeInfoTests(unittest.TestCase):
    def test_nested_numpy_scalars_normalize_without_mutating_input(self):
        original = {"flag_get": np.bool_(True), "nested": (np.int32(7), {"fraction": np.float32(.5)})}
        normalized = native_info(original)
        self.assertEqual(json.loads(json.dumps(normalized)), {"flag_get": True, "nested": [7, {"fraction": .5}]})
        self.assertIs(type(original["flag_get"]), np.bool_)
        self.assertIsInstance(original["nested"], tuple)


if __name__ == "__main__":
    unittest.main()
