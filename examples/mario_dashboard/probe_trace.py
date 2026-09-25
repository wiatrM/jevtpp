#!/usr/bin/env python3
"""Replay a recorded NES action trace in a disposable emulator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_nes_demo import ACTION_INDEX, create_environment, native_info, reset_environment, step_environment
from nes_state import StateTracker, unwrap_ram


def samples(path: Path):
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("type") == "control_frame" and "action" in row:
            yield row


def run_action(environment, name):
    _, _, terminated, truncated, info = step_environment(environment, ACTION_INDEX[name])
    ram = unwrap_ram(environment)
    dead = int(ram[0x0E]) in (6, 11) or int(ram[0xB5]) > 1
    return info, bool(dead or terminated or truncated)


def native_environment(environment):
    wrappers = []
    native = environment
    while hasattr(native, "env"):
        wrappers.append(native)
        native = native.env
    return native, wrappers


def trial_actions(specification):
    for macro in specification.split(","):
        name, count_text = macro.split(":", 1)
        count = int(count_text)
        if name not in ACTION_INDEX or count < 0:
            raise ValueError(f"invalid trial macro: {macro}")
        for _ in range(count):
            yield name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--env", default="SuperMarioBros-1-2-v0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--macro", action="append", default=[], metavar="ACTION:FRAMES")
    parser.add_argument("--trial", action="append", default=[],
                        metavar="ACTION:FRAMES,...", help="Compare branches from one disposable NES backup")
    parser.add_argument("--sample-every", type=int, default=10)
    args = parser.parse_args()
    environment = create_environment(args.env)
    try:
        reset_environment(environment, args.seed)
        last_info = {}
        count = 0
        for row in samples(args.trace):
            if count >= args.frame:
                break
            expected = row.get("x")
            if count and expected is not None and last_info.get("x_pos") != expected:
                raise RuntimeError(f"trace diverged at {count}: emulator={last_info.get('x_pos')} trace={expected}")
            last_info, dead = run_action(environment, row["action"])
            count += 1
            if dead:
                raise RuntimeError(f"trace terminated before requested frame: {count}")
        if count != args.frame:
            raise RuntimeError(f"trace has only {count} action samples")
        print(json.dumps({"checkpoint_frame": count, "info": last_info}))
        tracker = StateTracker(1)
        checkpoint, _ = tracker.parse(last_info, unwrap_ram(environment), previous_action="noop",
                                      previous_reward=0., response_delay_frames=0,
                                      objective_mode="speedrun")
        print(json.dumps({"checkpoint_player": checkpoint["player"],
                          "collision": checkpoint["collision"],
                          "hazard": checkpoint["hazard"]["upcoming_enemies"]}))
        if args.trial:
            native, wrappers = native_environment(environment)
            native._backup()
            metadata = (native.done, native._time_last, native._x_position_last,
                        [getattr(wrapper,"_elapsed_steps",None) for wrapper in wrappers])
            for specification in args.trial:
                native._restore()
                native.done, native._time_last, native._x_position_last = metadata[:3]
                for wrapper, elapsed in zip(wrappers, metadata[3]):
                    if elapsed is not None:
                        wrapper._elapsed_steps = elapsed
                info = native_info(native._get_info())
                steps, peak, dead = 0, int(info.get("x_pos",0)), False
                for name in trial_actions(specification):
                    info, dead = run_action(environment,name)
                    steps += 1
                    peak = max(peak,int(info.get("x_pos",0)))
                    if dead or info.get("flag_get"):
                        break
                print(json.dumps({"trial":specification,"steps":steps,"max_x":peak,
                                  "final_x":info.get("x_pos"),"final_y":info.get("y_pos"),
                                  "dead":dead,"flag_get":bool(info.get("flag_get"))}))
            return
        elapsed = 0
        for macro in args.macro:
            name, count_text = macro.split(":", 1)
            repetitions = int(count_text)
            if name not in ACTION_INDEX or repetitions < 0:
                raise ValueError(f"invalid macro: {macro}")
            for _ in range(repetitions):
                state, _ = tracker.parse(last_info, unwrap_ram(environment), previous_action=name,
                                         previous_reward=0., response_delay_frames=0,
                                         objective_mode="speedrun")
                last_info, dead = run_action(environment, name)
                elapsed += 1
                if elapsed % args.sample_every == 0 or dead:
                    print(json.dumps({"step": elapsed, "action": name, "info": last_info,
                                      "player": state["player"],
                                      "hazard": state["hazard"]["upcoming_enemies"],
                                      "collision": {key: value for key, value in state["collision"].items()
                                                    if key != "columns"}, "dead": dead}))
                if dead:
                    return
        print(json.dumps({"final_step": elapsed, "info": last_info}))
    finally:
        environment.close()


if __name__ == "__main__":
    main()
