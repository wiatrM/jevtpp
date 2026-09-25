#!/usr/bin/env python3
"""Score Mario decision traces without pretending replay labels are an RL oracle.

The labels below come only from deterministic RAM/grid evidence.  They measure
whether a typed model agrees with obvious controller obligations (jump at a
verified hazard, run on verified safe ground), plus closed-loop progress and
terminal causes.  This is deliberately separate from generic JevBench scores.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


JUMP_ACTIONS = {"right_jump", "right_run_jump", "jump"}
FORWARD_ACTIONS = {"right", "right_run", "right_jump", "right_run_jump"}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return round(ordered[lo], 3)
    return round(ordered[lo] * (hi - position) + ordered[hi] * (position - lo), 3)


def evidence_label(state: dict[str, Any]) -> tuple[bool, bool, list[str]]:
    player, terrain = state.get("player", {}), state.get("terrain", {})
    hazard, trajectory = state.get("hazard", {}), state.get("trajectory", {})
    recent = state.get("recent_control", {})
    reasons = []
    if trajectory.get("crossing_gap"): reasons.append("crossing_gap")
    if trajectory.get("crossing_obstacle"): reasons.append("crossing_obstacle")
    if trajectory.get("engaging_enemy"): reasons.append("engaging_enemy")
    if hazard.get("jump_must_start_this_decision"): reasons.append("takeoff_window")
    if hazard.get("contact_within_reaction_horizon"): reasons.append("contact_horizon")
    if terrain.get("gap_ahead"): reasons.append("gap_ahead")
    if terrain.get("obstacle_ahead"): reasons.append("obstacle_ahead")
    if player.get("grounded") and hazard.get("enemy_distance", 999) <= 72: reasons.append("enemy_close")
    need_jump = bool(reasons)
    safe_ground = bool(
        player.get("grounded")
        and terrain.get("observation_reliability") == "high"
        and not terrain.get("gap_ahead")
        and not terrain.get("obstacle_ahead")
        and hazard.get("enemy_distance", 999) > 160
        and recent.get("stalled", 0) < 4
    )
    return need_jump, safe_ground, reasons


def death_cause(history: list[dict[str, Any]]) -> str:
    for item in reversed(history[-8:]):
        state = item.get("state", {})
        hazard, terrain = state.get("hazard", {}), state.get("terrain", {})
        trajectory, recent = state.get("trajectory", {}), state.get("recent_control", {})
        if hazard.get("enemy_distance", 999) <= 96 or hazard.get("contact_within_reaction_horizon"):
            return "enemy_contact"
        if terrain.get("gap_ahead") or trajectory.get("crossing_gap"):
            return "pit_or_gap_timing"
        if terrain.get("obstacle_ahead") or trajectory.get("crossing_obstacle") or recent.get("stalled", 0) >= 4:
            return "obstacle_or_stall"
    return "unclassified_terminal"


def analyze(paths: list[Path], model_name: str) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    deaths: Counter[str] = Counter()
    latencies: list[float] = []
    jump_errors: list[float] = []
    episodes = []

    for path in paths:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            continue
        max_progress = 0
        terminal = False
        for row_index, row in enumerate(rows):
            if not row.get("ok", True):
                counts["errors"] += 1
                continue
            state = row.get("state", {})
            need_jump, safe_ground, why = evidence_label(state)
            raw, final = row.get("raw_action", row.get("action")), row.get("action")
            raw_jump, final_jump = raw in JUMP_ACTIONS, final in JUMP_ACTIONS
            counts["decisions"] += 1
            counts["hazard_labels"] += int(need_jump)
            counts["safe_ground_labels"] += int(safe_ground)
            counts["raw_hazard_hits"] += int(need_jump and raw_jump)
            counts["final_hazard_hits"] += int(need_jump and final_jump)
            counts["raw_safe_forward_hits"] += int(safe_ground and raw in FORWARD_ACTIONS)
            counts["final_safe_forward_hits"] += int(safe_ground and final in FORWARD_ACTIONS)
            counts["raw_safe_false_jumps"] += int(safe_ground and raw_jump)
            counts["final_safe_false_jumps"] += int(safe_ground and final_jump)
            counts["overrides"] += int(row.get("override_applied", False))
            counts["noul_choice_contradictions"] += int((row.get("jump_probability", 0) >= .72) != raw_jump)
            for reason in why: reasons[reason] += 1
            if "latency_ms" in row: latencies.append(float(row["latency_ms"]))
            if "jump_probability" in row:
                jump_errors.append((float(row["jump_probability"]) - float(need_jump)) ** 2)
            episode = state.get("episode", {})
            max_progress = max(max_progress, int(episode.get("best_progress", episode.get("progress", 0)) or 0))
            if episode.get("dead"):
                terminal = True
                deaths[death_cause(rows[: row_index + 1])] += 1
                break
        episodes.append({"trace": path.name, "decisions": len(rows), "max_progress": max_progress,
                         "terminal_death_observed": terminal})

    def rate(hit: str, total: str) -> float | None:
        return round(counts[hit] / counts[total], 4) if counts[total] else None

    return {
        "model": model_name,
        "scope": "deterministic replay obligations plus observed closed-loop traces; not a learned-policy win-rate benchmark",
        "traces": len(episodes),
        "decisions": counts["decisions"],
        "typed_model": {
            "raw_jump_recall_on_verified_hazards": rate("raw_hazard_hits", "hazard_labels"),
            "raw_forward_rate_on_verified_safe_ground": rate("raw_safe_forward_hits", "safe_ground_labels"),
            "raw_false_jump_rate_on_verified_safe_ground": rate("raw_safe_false_jumps", "safe_ground_labels"),
            "jump_noul_brier": round(statistics.fmean(jump_errors), 4) if jump_errors else None,
            "noul_vs_choice_contradiction_rate": round(counts["noul_choice_contradictions"] / counts["decisions"], 4) if counts["decisions"] else None,
        },
        "jevtpp_composed_controller": {
            "final_jump_recall_on_verified_hazards": rate("final_hazard_hits", "hazard_labels"),
            "final_forward_rate_on_verified_safe_ground": rate("final_safe_forward_hits", "safe_ground_labels"),
            "final_false_jump_rate_on_verified_safe_ground": rate("final_safe_false_jumps", "safe_ground_labels"),
            "override_rate": round(counts["overrides"] / counts["decisions"], 4) if counts["decisions"] else None,
        },
        "latency_ms": {"p50": percentile(latencies, .5), "p95": percentile(latencies, .95),
                       "mean": round(statistics.fmean(latencies), 3) if latencies else None},
        "evidence_labels": {"hazard": counts["hazard_labels"], "safe_ground": counts["safe_ground_labels"],
                            "hazard_reasons": dict(reasons)},
        "closed_loop": {"best_progress": max((ep["max_progress"] for ep in episodes), default=0),
                        "death_causes": dict(deaths), "episodes": episodes},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--glob", default="examples/mario_dashboard/artifacts/run-*.jsonl")
    parser.add_argument("--model", default="laya-multilingual")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = args.paths or [Path(path) for path in sorted(glob.glob(args.glob))]
    report = analyze(paths, args.model)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
