# Online discovery controller

The v12 demo executes an episode-local learning loop in C++. Open-JEV remains
the asynchronous CUDA judgment backend; no additional neural model is trained.

## Executed graph

`telemetry -> outcome_update -> world_belief / action_dynamics -> persistent_tasks
-> reachability -> candidate_plans -> risk_budget -> controller`

The next observation checks the previous prediction and updates the loop.
The existing `experience_memory`, compact Open-JEV request, intent gate and
`reactive_baseline` are connected to this graph, not separate visual decorations.
The runtime trace contains the actual inputs, outputs and durations.

- **World belief:** bounded entity tracks use RAM slot, type, spatial continuity
  and life/stage boundaries. Slot reuse starts a new identity. Losing visibility
  never confirms a kill. Only an observed alive-to-defeated RAM transition can
  close an elimination. Geometry outside the visible RAM window is unknown.
- **Action-conditioned dynamics:** online motion statistics distinguish action,
  power state, ground/air and jump hold/release. The model includes a one-frame
  input latch and reports one-step position error. Explicit physical priors
  bootstrap exploration; this is not learning physics without any prior.
  Releasing A cuts the current jump permanently: pressing it again in midair
  cannot restore hold gravity. Air speed is bounded by takeoff momentum.
  Duplicate, skipped, respawn and cross-stage transitions do not become samples.
- **Reachability:** observed tile surfaces and swept body collisions constrain
  action sequences. Ceiling contact, predicted hostile contact, unobserved
  space and unsupported endpoints are exposed separately.
- **Planning:** up to 29 candidate sequences are rolled forward for 48 frames;
  only their first requested input is executed before replanning. The current
  action already latched by the NES is simulated before that requested input.
  Platform forecasts stop at the first landing rather than running off the
  support with the same command. All visible isolated surfaces remain in
  telemetry regardless of Mario's height; reachable geometry gates planning.
  Clear corridors without a nearby task or platform skip expensive rollouts.
- **Persistent tasks:** hunter targets survive inference updates. Unobserved or
  unsupported targets stay unresolved; another visible target may be selected
  without counting the old target as a success. Attack capability currently
  covers stomping, not a complete shell/fireball/Bowser combat repertoire.
- **Outcome/risk adaptation:** a flight keeps its takeoff plan until a verified
  landing or life loss, so falling deaths do not blame the final rightward input.
  Confirmed life loss penalizes the associated plan
  signature within the same run. This is conservative outcome association,
  not proof of the cause of death. Risk budgets tighten as lives decrease.
  A model bonus may rank feasible plans but cannot bypass risk/lifecycle gates.
  A flight that falls below an observed support without reaching its near edge
  records an undershoot hypothesis. A bounded minimum hold is increased for the
  same power-state/relative-height context in the current run. This is a coarse
  motor adaptation, not a learned level route or proof of causal diagnosis.

## Safety and authority

The predictive model is approximate. Planner authority is limited to nearby
combat in a clear observed arena and observed isolated platforms. It cannot
cancel a committed gap/wall maneuver unless a collision-checked forward platform
landing is predicted, and cannot cancel a pipe/recovery macro. A combat override
requires a predicted stomp; unsupported approaches retain the reactive fallback.
If no feasible plan exists at immediate enemy contact, an explicitly labeled
emergency proposal maximizes predicted time to collision. It is not called safe.

Hunter additionally uses an explicitly labeled closed-loop target skill:
approach/brake, launch, airborne intercept, and RAM outcome verification.
This user-objective motor skill does not claim to be a model-selected action.
Elevated targets may be awaited instead of silently skipped. Unsafe geometry
or unsupported attacks remain unresolved; this does not guarantee kill-all.

`planner_enabled: false` in telemetry provides a baseline ablation. Tests of the
legacy motor latches use this setting; dedicated learner tests exercise default
enabled planning, actual action selection, memory boundaries and outcome credit.

The `risk_cost` and `utility` fields are heuristic costs, not calibrated death
or success probabilities. A predicted stomp is not a confirmed elimination.
The demo does not claim full-game completion, all enemies killed, or improvement
over the baseline without a measured campaign comparison.

## Lifetime and inspection

Learning statistics and failure costs survive life loss and stage transitions.
Temporal samples and entity tracking restart at those physical boundaries.
GAME OVER, a new run, and `memory_reset` clear adaptive learning. Changing the
objective or asynchronous model epoch does not erase the physical experience.
Adaptive state is not imported from prior runs or persisted as a learned route.

The online-learning panel shows unresolved enemies, transition sample count,
forecast error, feasible plans and the current task. The map overlays the
selected forecast in blue. Select a graph node to inspect its predecessor
payloads; the situation report includes all six layers. Open-JEV's score and
CUDA provenance remain separate from the controller's forecasts.

## Validation

```sh
cmake --build build-mario-cuda --target jevt_mario_control
.venv-mario/bin/python -m unittest discover -s examples/mario_dashboard -p 'test_*.py'
node examples/mario_dashboard/test_dashboard.cjs
.venv-mario/bin/python examples/mario_dashboard/benchmark_controller.py \
  --controller-jsonl build-mario-cuda/examples/jevt_mario_control \
  --mode hunter --goal stomp_enemy --offsets 0,17 --output /tmp/hunter.json
```

The last command uses an explicitly labeled fixed intent, not live Open-JEV.
Use the actual dashboard for CUDA/model integration validation. Frame limits,
unit fixtures and stage-1 benchmarks do not prove campaign completion.
