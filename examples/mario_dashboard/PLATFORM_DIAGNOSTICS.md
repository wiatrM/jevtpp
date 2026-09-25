# Platform-control investigation (v12)

This is a partial correction, **not a solved 1-3 or full-game controller**.

## Reproduced faults

- Isolated surfaces disappeared from telemetry when the player moved too far
  above/below them. Visible geometry now remains independent of player height.
- The rolling RAM nametable was treated as visible geometry beyond the camera.
  Landing supports now use the currently observed planning window only.
- A viewport/respawn Y wrap was reported as a +251 pixel jump. Such samples now
  carry `velocity_valid: false`, cannot imply ground contact, and do not train
  the action dynamics model.
- The planner could release A during the one-frame input latch, or assume that
  re-pressing A restored a shortened jump. Latching and jump-cut state are now
  explicit. Candidate platform landings include an interior margin and stop
  their rollout at first contact rather than walking off that platform.
- Braking samples above walking speed and two-pixel integer accelerations were
  excluded by learning filters. Both are now covered by regression tests.
- Failure credit used the last falling command. It now retains the takeoff
  signature, observed apex, target, and landing/life-loss result across the
  flight. Relative-height undershoots can adjust a bounded hold constraint in
  the same run. GAME OVER clears it. No route is imported from another run.

## Evidence and remaining failure

`artifacts/platform-feedback-campaign.json` is an actual full-ROM run without
model inference, replay, save states, or imported experience. It cleared 1-1
and 1-2, then exhausted three lives in 1-3. The maximum X in the three recorded
1-3 death windows was 694, 590 and 591. This is **not evidence of monotonic
learning or improved campaign success**. Subsequent changes only tightened
lifecycle/ablation guards and added tests.

The planner remains an approximate 48-frame, single-action-prefix rollout.
Moving supports, hostile vertical motion, multi-platform route selection, and
counterfactual failure credit are incomplete. A single-step error average does
not calibrate long-horizon landing uncertainty. Unit tests prove contracts,
not gameplay competence.

The dashboard reports flight feedback and actual controller authority. The
Open-JEV CUDA backend remains separate from these deterministic motor skills;
controller-only benchmark results are not model benchmark results.

Validation: 133 Python tests and 14 dashboard contract tests passed. The
`build-mario-cuda` tree has no registered CTest tests; do not count its empty
CTest invocation as a passing core suite.
The separate `build-graph-check` tree passed all 10 registered CTest tests.
Live Chromium QA verified v12, 23 executed nodes, CUDA on RTX 4090, and no
JavaScript errors at desktop, tablet, and mobile viewport sizes.

The final hunter controller-only fixture (`artifacts/hunter-v12-final.json`)
confirmed 5 eliminations out of 6 observed hostiles and ended in death with one
passed hostile unresolved. It does not satisfy kill-all either.

For a focused cold-start fixture, use `benchmark_controller.py --environment
SuperMarioBros-1-3-v0 --goal none`. For lives, stage continuation, and within-run
experience, use `smoke_full_game.py --stop-stage 9` instead.
