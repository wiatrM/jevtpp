# v13: 1-3 trace, blocked attacks and model selection

## Findings

**1-3 is not solved.** In a fresh, real Open-JEV stage fixture, the raw model
selected `finish_fast` during the failure, but the C++ planner landed too close
to a platform edge and the next jump command produced no ascent. Changing the
six-way goal classifier alone cannot repair this motor-planning failure.

Evidence: `artifacts/async-20260925T064507Z.jsonl`, terminal frame 145, X=418.
The last 120 per-frame records all report `model_changed_action: false`.
This is one cold-start stage fixture, not a three-life campaign or benchmark
comparing model quality. No imported experience or saved-state replay was used.

| Frame | X | Feet Y (downward positive) | Observation / applied input |
| --- | ---: | ---: | --- |
| 120 | 343 | 189 | Descending; `left` |
| 121 | 346 | 192 | Grounded; `right_jump`; predicted next landing X=403.75, Y=144 |
| 122 | 349 | 192 | Airborne flag; `right_run_jump`; no upward velocity |
| 125 | 358 | 194 | Falling; `landing_target_missed` |
| 144 | 415 | 284 | Below the platforms; still no successful takeoff |

The current platform was [288,352] at Y=192, leaving only 6 pixels to its edge
at takeoff scheduling; the next observed platform began at X=384, Y=144.
The flight record's apex remained Y=192: the jump did not rise. The trace
supports a late edge takeoff / optimistic contact forecast, not an A-button
rearm failure (A was released before frame 121). A stronger controller needs
landing positions that reserve time and support for the *next* takeoff, plus
explicit verification of takeoff impulse rather than treating any airborne
state as a successful launch. This diagnosis is not yet an implemented fix.

The compact model request contains distances, velocity and aggregate jump
statistics, but not platform intervals, landing margin or the selected flight
trajectory. Model scores therefore cannot be read as evaluations of that route.
Current empirical adaptation does not establish reliable learning from this
failure; a low one-step prediction error is not evidence of safe long jumps.

The older v12 trace (`artifacts/async-20260925T062616Z.jsonl`, summarized in
`artifacts/trace-1-3-v12-analysis.json`) contains 13 approximately 1 Hz samples
per run: ten raw `collect_coins` responses and three pending responses. None of
those samples reports model-changed motor output; intent gating selected within
the speedrun objective. It cannot establish exact behavior between samples.
New lifecycle and terminal records retain a bounded 120-frame evidence window.

## Blocked attacks

A live hunter snapshot showed X=386, enemy X=452, solids at X=400,
`blocked_attempt_budget`, yet `hunter_engagement.apply=true` with more than
6,500 attack frames. The executor ignored task vetoes and checked walls only
while grounded. This was a controller integration bug, not proof of a bad
Open-JEV score.

The executor now checks observed body-height approach geometry and requires an
active task even in the air. Blocked or expired tasks cannot trigger optional
attack reflexes in hunter or score-attack; unresolved enemies stay unresolved.
The model context and intent gate receive `attack_path_blocked`. This is a
conservative direct-approach check, not a complete obstacle-navigation planner.

The real-emulator score-attack regression (`artifacts/score-obstruction-v13.json`)
passed the reported X=386 region and reached X=898, with two confirmed kills,
then died. It uses fixed labeled intent, **not live model inference**. A separate
controller-only hunter fixture (`artifacts/hunter-obstruction-v13.json`) died at
X=149 before reaching that region; it does not validate obstacle traversal.
Neither test proves kill-all or completion. Unit regressions cover airborne
blocking, exhausted attempt budgets and reactivation after a wall disappears.

## Actual model selection

`artifacts/model-switch-v13.json` records a browser test of Open-JEV -> LAYA ->
Open-JEV. Selecting while paused preserved frame, run, world, stage, lives,
knowledge generation and session frame. Resuming produced fresh responses from
each selected model; the previous model's response was cleared.

Both backends use the RTX 4090 at `cuda:0`. Open-JEV retains its actual
`Qwen/Qwen3.5-2B` base plus released adapter/head. LAYA uses the local
`laya-multilingual` ONNX bundle through an initialized CUDAExecutionProvider;
this is not an assertion that every ONNX operator resides on GPU.
Observed individual response times were 25.3 ms for LAYA and 173.8 ms for
Open-JEV. These are smoke-test samples, not latency percentiles or gameplay
quality comparisons. The same C++ controller and accumulated run knowledge
remain in charge across the switch.
