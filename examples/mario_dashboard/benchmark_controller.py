#!/usr/bin/env python3
"""Reproducible closed-loop emulator benchmark, including terminal next state."""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from nes_state import StateTracker, unwrap_ram
from run_nes_demo import create_environment, reset_environment, step_environment, ACTION_INDEX


class JsonlController:
    def __init__(self, executable, mode="speedrun", goal="finish_fast", memory_path=None):
        from async_runner import FastControllerProcess
        self.process = FastControllerProcess(executable,memory_path=memory_path)
        self.frame, self.mode, self.goal = 0, mode, goal
        self.latencies = []
        initial = self.process.exchange({"command":"memory_export"}).get("memory",{})
        self.memory_seed = {"observed_columns":len(initial.get("columns",[])),"outcomes":initial.get("outcomes"),
                            "jump_samples":initial.get("jumps",{}).get("duration_frames",{}).get("count",0)}

    def step(self, state):
        started = time.perf_counter()
        # Fixed synthetic intent validates the locomotion implementation only;
        # it is explicitly not scored as a successful Open-JEV inference.
        intent = {"ok":True,"epoch":1,"source_frame":self.frame,"mode":self.mode,
                  "active_goal":self.goal,"model_id":"benchmark_fixed_intent",
                  "source_snapshot":f"1:{self.frame}"}
        if self.goal == "none":
            intent = {}
        result = self.process.step(state,intent,1,self.frame,reset=self.frame == 0)
        self.latencies.append((time.perf_counter()-started)*1000)
        self.frame += 1
        return {k:v for k,v in result.items() if k not in ("graph","compact_state")}

    def close(self):
        self.process.close()

    def finalize(self, state):
        result = self.process.step(state,{},1,self.frame,include_memory=True)
        knowledge = result.get("knowledge",{})
        self.process.save_memory()
        return {"outcomes":knowledge.get("outcomes"), "jump_samples":knowledge.get("context",{}).get("jump_samples"),
                "observed_columns":len(knowledge.get("columns",[])), "enemy_types":len(knowledge.get("enemy_types",[]))}


class ProbeController:
    """RAM reactive baseline; no level coordinates, seed tables or action replay."""
    def __init__(self):
        self.jump_frames = 0
        self.previous_jump = False
        self.observed_airborne = False
        self.reason = "cruise"

    def step(self, state):
        player, collision = state["player"], state["collision"]
        jump = False
        if player["grounded"]:
            if self.previous_jump and not self.observed_airborne and self.jump_frames < 4:
                self.jump_frames += 1
                return {"action":"right_run_jump", "reason":"takeoff_input_latch", "phase":"takeoff"}
            self.jump_frames = 0
            self.observed_airborne = False
            gap = collision["gap_distance_pixels"]
            wall = collision["obstacle_distance_pixels"]
            enemy = next((e for e in state["hazard"]["upcoming_enemies"]
                          if -32 <= e["relative_y_pixels"] <= 40), None)
            self.reason = "cruise"
            if not self.previous_jump:
                if gap is not None and gap <= 28:
                    jump, self.reason = True, "gap"
                elif wall is not None and wall <= 40:
                    jump, self.reason = True, "obstacle"
                elif enemy and enemy["relative_x_pixels"] <= 52:
                    jump, self.reason = True, "enemy"
                elif state["recent_control"]["stalled"] >= 8:
                    jump, self.reason = True, "unstick"
            else:
                self.reason = "release_landing"
        else:
            self.observed_airborne = True
            self.jump_frames += 1
            jump = self.previous_jump and self.jump_frames <= 32
        self.previous_jump = jump
        return {"action": "right_run_jump" if jump else "right_run", "reason": self.reason,
                "phase": "grounded" if player["grounded"] else "airborne"}


def episode(seed, offset, max_frames, trace=False, controller=None, mode="speedrun", perturb_every=0,
            environment="SuperMarioBros-1-1-v0"):
    env = create_environment(environment)
    _, info = reset_environment(env, seed)
    for _ in range(offset):
        _, _, _, _, info = step_environment(env, 0)
    tracker = StateTracker(1)
    controller = controller or ProbeController()
    previous_action, reward = "noop", 0.
    max_x, outcome, rows = 0, "frame_limit", []
    action_counts, mode_effects, model_effects = Counter(), Counter(), Counter()
    action_hash = hashlib.sha256()
    started = time.perf_counter()
    try:
        for frame in range(max_frames):
            state, _ = tracker.parse(info, unwrap_ram(env), previous_action=previous_action,
                                     previous_reward=reward, response_delay_frames=0, objective_mode=mode)
            decision = controller.step(state)
            applied_action = "noop" if perturb_every and frame and frame % perturb_every == 0 else decision["action"]
            action_counts[applied_action] += 1
            action_hash.update((applied_action+"\n").encode())
            if decision.get("mode_changed_action"):
                mode_effects[decision.get("mode_effect","unknown")] += 1
            if decision.get("model_changed_action"):
                model_effects[decision.get("model_effect","unknown")] += 1
            _, reward, terminated, truncated, next_info = step_environment(env, ACTION_INDEX[applied_action])
            max_x = max(max_x, int(next_info.get("x_pos", 0)))
            ram = unwrap_ram(env)
            dead = int(ram[0x0e]) in (6, 11) or int(ram[0xb5]) > 1
            perturbed = applied_action != decision["action"]
            if trace and (decision["action"] != previous_action or frame % 30 == 0 or dead or perturbed):
                rows.append({"frame":frame,"decision":decision,"player":state["player"],
                             "applied_action":applied_action,"input_perturbed":perturbed,
                             "collision":{k:v for k,v in state["collision"].items() if k != "columns"},
                             "enemy":state["hazard"]["upcoming_enemies"],"detectors":state["detectors"],"next_info":dict(next_info),
                             "terminated":terminated,"truncated":truncated,"dead":dead})
            info, previous_action = next_info, applied_action
            if next_info.get("flag_get"):
                outcome = "flag_get"; break
            if terminated or truncated or dead:
                outcome = "death" if dead else "truncated" if truncated else "terminated"; break
        latencies = getattr(controller,"latencies",[])
        timing = {"p50_ms":statistics.median(latencies), "p95_ms":sorted(latencies)[int(.95*(len(latencies)-1))]} if latencies else None
        final_state, _ = tracker.parse(info,unwrap_ram(env),previous_action=previous_action,
                                       previous_reward=reward,response_delay_frames=0,objective_mode=mode)
        memory = controller.finalize(final_state) if hasattr(controller,"finalize") else None
        if outcome == "flag_get":
            failure_cause = None
        elif int(info.get("time",1)) <= 0:
            failure_cause = "game_timer_expired"
        elif int(ram[0xb5]) > 1:
            failure_cause = "fell_below_viewport"
        elif dead:
            failure_cause = "lethal_contact_inferred" if state["hazard"]["enemy_distance"] <= 40 else "death_cause_unresolved"
        elif outcome == "frame_limit":
            failure_cause = "stalled" if state["recent_control"]["stalled"] >= 30 else "benchmark_frame_budget"
        else:
            failure_cause = outcome
        return {"seed":seed,"start_delay_frames":offset,"mode":mode,"perturb_every":perturb_every,
                "outcome":outcome,"flag_get":bool(info.get("flag_get")),"control_roundtrip":timing,
                "failure_cause":failure_cause,"memory":memory,"memory_seed":getattr(controller,"memory_seed",None),
                "controller_recovery_attempts":decision.get("recovery_attempts",0),
                "objectives":{"score":int(info.get("score",0)),"coins":int(info.get("coins",0)),
                              "confirmed_eliminations":final_state["detectors"].get("eliminations_confirmed",0),
                              "confirmed_stomps":final_state["detectors"].get("stomps_confirmed",0),
                              "hostiles_seen":final_state["detectors"].get("hostiles_seen",0),
                              "hostiles_passed_alive":final_state["detectors"].get("hostiles_passed_alive",0),
                              "hostiles_passed_not_eliminated":final_state["detectors"].get("hostiles_passed_not_eliminated",0),
                              "powerups_collected":final_state["detectors"].get("powerups_collected",0)},
                "behavior":{"actions":dict(action_counts),"mode_effects":dict(mode_effects),
                            "model_effects":dict(model_effects),"action_sha256":action_hash.hexdigest()},
                "terminal_evidence":{"player_before":state["player"],"player_after":final_state["player"],
                                     "decision":decision,"applied_action":applied_action,
                                     "enemy_before":state["hazard"]["upcoming_enemies"],
                                     "ram_player_state":int(ram[0x0e]),"ram_y_viewport":int(ram[0xb5])},
                "max_x":max_x,"frames":frame+1,"wall_seconds":round(time.perf_counter()-started,3),
                "terminal_next_info":dict(info),"trace":rows}
    finally:
        env.close()
        if hasattr(controller,"close"):
            controller.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--environment", default="SuperMarioBros-1-1-v0",
                        help="Explicit stage fixture; use smoke_full_game.py for campaign evidence.")
    parser.add_argument("--offsets", default="0")
    parser.add_argument("--max-frames", type=int, default=6000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--controller-jsonl", type=Path)
    parser.add_argument("--mode", choices=("speedrun","hunter","collector","score_attack"), default="speedrun")
    parser.add_argument("--goal", default="finish_fast")
    parser.add_argument("--memory-path",type=Path,
                        help="Optional scoped memory JSON reused across episode processes; omit for independent episodes.")
    parser.add_argument("--perturb-every", type=int, default=0,
                        help="Stress test: inject one neutral input every N frames (0 disables).")
    args = parser.parse_args()
    results = []
    for seed in map(int, args.seeds.split(",")):
        for offset in map(int, args.offsets.split(",")):
            controller = JsonlController(args.controller_jsonl,args.mode,args.goal,args.memory_path) if args.controller_jsonl else None
            result = episode(seed,offset,args.max_frames,args.trace,controller,args.mode,args.perturb_every,args.environment)
            results.append(result)
            print(json.dumps({k:v for k,v in result.items() if k not in {"trace","terminal_evidence","terminal_next_info"}},
                             default=lambda v: v.item()), flush=True)
    if args.output:
        args.output.write_text(json.dumps({"environment":args.environment,
            "controller":"jevtpp-cpp-jsonl" if args.controller_jsonl else "reactive-ram-probe",
            "provenance":"Closed-loop emulator observations; no savestates, map coordinates or replay. JSONL test uses labeled fixed strategy intent, not a live model.",
            "limitation":"Seeds do not randomize the fixed level. Start delays and optional input perturbations test timing, not other levels.",
            "wins":sum(r["flag_get"] for r in results),"total":len(results),
            "source_sha256":{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ("nes_state.py","fast_controller.hpp","fast_graph.cpp","adaptive_agent.hpp","run_nes_demo.py","benchmark_controller.py")},
            "controller_binary_sha256":hashlib.sha256(args.controller_jsonl.read_bytes()).hexdigest() if args.controller_jsonl else None,
            "episodes":results}, indent=2, default=lambda v: v.item()))


if __name__ == "__main__":
    main()
