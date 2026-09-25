# JevT++ NES Runner Lab · v14

Real NES emulator with a C++ decision graph and asynchronous local Open-JEV
plan ranking. Open-JEV is only the model backend; typed contracts, freshness
checks, physical safety and controller output stay in JevT++.

Model identity is explicit: the released Open-JEV-2B adapter and scalar decision
head run on a pinned `Qwen/Qwen3.5-2B` base. JevT++ is the C++ orchestration
and control library, not a separately trained neural checkpoint. The dashboard
shows this stack, loaded checkpoint identity and actual RTX 4090 CUDA device.

See [OPTIMIZATION.md](OPTIMIZATION.md) for the diagnosis, measured latency
reduction, completion evidence and limitations. The legacy synchronous runner
is retained behind `--control legacy`; it is not the default.

## Build

The fast graph uses the nlohmann JSON dependency already provided by the remote
module. No paid Jev API or API key is used for local Open-JEV.

```sh
cmake -S . -B build-mario-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DJEVT_BUILD_EXAMPLES=ON \
  -DJEVT_ENABLE_REMOTE=ON -DJEVT_REMOTE_CURL=OFF
cmake --build build-mario-cuda --target jevt_mario_demo jevt_mario_control -j 2
```

For the existing optional Laya comparison build, retain `JEVT_ENABLE_LAYA=ON`
and the installed ONNX Runtime GPU path. Open-JEV itself does not need ONNX.

## Run one local CUDA model, then the demo

Use the existing pinned checkpoint and installed environments. Do not load a
second copy of the model for the runner: it connects to port 8791.

```sh
HF_HOME="$PWD/models/huggingface-cache" \
.venv-open-jev/bin/python examples/mario_dashboard/open_jev_runtime.py serve \
  --source .deps/open-jev-src \
  --checkpoint models/open-jev-2b/package/checkpoint \
  --port 8791

.venv-mario/bin/python examples/mario_dashboard/run_nes_demo.py \
  --engine build-mario-cuda/examples/jevt_mario_demo \
  --controller build-mario-cuda/examples/jevt_mario_control \
  --open-jev-url http://127.0.0.1:8791 \
  --laya-model models/laya-multilingual \
  --frames-per-decision 30 --fps 60 --dashboard-fps 12
```

Open http://127.0.0.1:4173/?dashboard=v16. The header identifies the model,
latency, C++ control latency and emulator FPS separately. The first model
judgment may take longer while CUDA warms up; current-RAM C++ control continues
without waiting for it. Model errors are visible instead of hidden.

The MODEL selector switches actual local judgment backends, not just labels.
The optional `--laya-model` bundle starts a second C++ engine on port 4175;
without it LAYA is disabled with an explanation. Switching preserves the NES,
lives, stage, C++ controller and same-run knowledge. In-flight responses from
the previous model are discarded. A paused game stays paused after switching.
LAYA requires an initialized ONNX Runtime CUDA session on RTX 4090 device 0;
some shape operators may run on CPU, so this is not an all-operators GPU claim.
See [TRACE_1_3.md](TRACE_1_3.md) for v13 failure evidence and selector validation.

The default `--model-policy plan` sends bounded, currently feasible candidate
plans to the C++ engine's `/api/plan` endpoint. The model ranks concrete action
sequences and their observed targets, not just six broad goal names. The local
plan gate checks candidate membership, context, source epoch, mode and age;
the physical risk gate still has authority to reject a preference. A model
score is not a probability of successful landing or survival. For an explicit
comparison with the older goal-only path, run with `--model-policy goal`;
that uses `/api/strategy`. Do not mix the two protocols in a model benchmark.

The v12 [online discovery controller](ONLINE_LEARNING.md) adds actual entity
tracking, action-conditioned motion learning, persistent tasks, collision-based
rollouts and outcome-dependent risk selection. The online-learning panel and
executed graph expose its decisions and its limitations.

The Open-JEV wrapper refuses startup unless the loaded parameters and buffers
are on CUDA and the actual GPU matches RTX 4090. Its /api/runtime endpoint
rechecks the same model, and the runner validates it before starting.
This is an enforced device contract, not a cosmetic CUDA label.

The Situation Report exposes actual model scores, source frame, age,
physical evidence, accepted/ignored intent, game phase, learning delta and
current risk stance. The observed-world map is built from visible RAM geometry.
Bounded empirical memory retains map columns, enemy-speed statistics, completed
jumps and outcomes across lives and stages of one full-game run by default. The
header's KNOWLEDGE selector enables an in-process **RETAIN ACROSS RUNS**
experiment: the observed map, C++ aggregate memory and named ontology survive
GAME OVER and manual restart. Switching back to RESET clears them. The action
learner still resets per-run state. The model request gets bounded,
source-bearing ontology relations, not a fabricated map or unverified claims.
See [KNOWLEDGE_ONTOLOGY.md](../../docs/KNOWLEDGE_ONTOLOGY.md).
The repeated x469–472 failure and cross-run feedback are reviewed in
[TRACE_X472_FEEDBACK.md](TRACE_X472_FEEDBACK.md).
This is online telemetry
adaptation, not neural weight training. See EXPERIENCE_MEMORY.md for provenance.

A different engine already on port 4174 is rejected if its protocol/backend
does not match. Restart that demo process explicitly; never silently reuse an
old preview or Laya backend for an Open-JEV run.

The repository does not distribute a ROM. Use only an installed game you are
legally entitled to run.

## What the graph means

The generic `jevt::decision_graph` validates dependencies and evaluates real
named outputs. The Mario specialization projects RAM into physics, geometry,
threats, objectives and game phase. `experience_memory` reports incremental
discoveries; `digital_twin` forecasts a short horizon; `learning_delta` and
`risk_policy` feed the compact model request and frame-rate controller. The
graph does not replay a prerecorded route or pretend unknown tiles are known.
The controller additionally derives a temporary landing target from isolated
RAM-observed platforms and reports that applied target in the controller node
and Situation Report. It is not a fixed stage-coordinate script and does not
imply reliable traversal of every platform level.

The model mailbox contains an earlier asynchronous response, preserving its
source epoch/frame. The intent gate validates episode, mode, age and current
target availability. The fast controller consumes these results each frame.

### v14 skill contracts and learning supervision

The domain-independent [skill runtime](../../docs/SKILLS.md) supplies three-state
conditions (`true`, `false`, `unknown`), snapshot-bound proposals and a pure
final arbiter. Mario supplies domain-specific observations, motor proposals and
verification rules. Only the arbiter emits the final action; its trace records
the selected proposal and the reason each alternative was excluded or lost.
Lifecycle and input-release constraints outrank speculative movement. No module
silently changes the actuator command after arbitration.

The planner now evaluates observed second-support continuation as well as the
first landing. Candidate summaries expose support margin, input/contact reserve,
braking sequences and an approximate next transfer. An unseen next platform is
unknown, not an empty floor or a guaranteed safe continuation. These forecasts
remain approximate; the demo is not a solved Mario agent.

Attack tasks retain identity-specific verification and separate active attempt
time from time blocked by geometry. Observed approach progress renews the active
budget. Blocked attacks expose navigation, attack-pose and elimination-verification
subtasks. An exposed subtask is not proof that a detour was synthesized or an
enemy was eliminated: without a feasible route it stays explicitly blocked.

Experience is keyed by relative context rather than applying a global penalty
to every similarly named jump. Only the executed skill receives outcome credit;
a rejected proposal is not recorded as an observed failure. Takeoff must be
observed before a failed flight can teach a longer jump hold. Walking off an
edge is a different failure class. Neural checkpoint weights are unchanged.

Action-conditioned physics updates are collected in shadow first. Frozen
predictions are scored on subsequent observations before those observations
train the estimator. Promotion depends on predictive evidence and guards, not
the sample count alone. Reported multi-horizon errors are conditional on the
assumed input continuing; action changes invalidate such forecasts. Neither
shadow promotion nor a low one-step error proves full-game competence.

Click a node to inspect its exact dependency inputs and output. Drag to pan,
use +/−/FIT to zoom, or use left/right arrow keys while the canvas is focused.
Green edges mean successful dependency delivery, not a neural attention weight.
Packets indicate newly received execution snapshots, not measured network bytes.
The terminal screen explicitly marks the last pre-terminal controller trace.

## Modes and limits

Speedrun is the best tested stage objective. Hunter, collector and score-attack
alter real motor choices but remain limited objectives, not guarantees of
exhaustive kills/collection or full-game completion.
The model can select safe intent (for example a slower powerup approach); it
does not own every frame-level action. Safety can decline model intent.

The current reflex controller has physical reaction parameters, not scripted
level coordinates or recorded actions. It is not a trained universal Mario
policy. The default `SuperMarioBros-v0` plays through stages, respecting the
ROM's three starting playable lives; a flag on an intermediate stage does not
terminate the session. Only 8-4 clear is full completion and raw life `255`
is GAME OVER. Both frame and decision limits default to 0 (unlimited).
The older `--control legacy` loop is explicitly single-stage only. SMB 1-1
success must not be advertised as completion of every world.

## Tests and reproducible gameplay benchmark

```sh
cmake -S . -B build-graph-check -DJEVT_BUILD_EXAMPLES=OFF \
  -DJEVT_BUILD_BENCHMARKS=OFF -DJEVT_ENABLE_HTTP=OFF
cmake --build build-graph-check --target jevt_graph_tests jevt_system_one_tests
ctest --test-dir build-graph-check -R 'graph|system_one' --output-on-failure

(cd examples/mario_dashboard && ../../.venv-mario/bin/python -m unittest -v \
  test_async_runner test_game_session test_nes_state test_adaptive_agent \
  test_skill_planning)
.venv-mario/bin/python examples/mario_dashboard/test_fast_graph.py
.venv-mario/bin/python examples/mario_dashboard/test_fast_controller.py \
  --controller-jsonl build-mario-cuda/examples/jevt_mario_control
.venv-mario/bin/python examples/mario_dashboard/benchmark_controller.py \
  --controller-jsonl build-mario-cuda/examples/jevt_mario_control \
  --seeds 0,1,42 --offsets 0,1,7,17,31,60 \
  --output examples/mario_dashboard/artifacts/controller-cpp-matrix.json
```

Use `--perturb-every 73` for injected neutral-input perturbations; do not mix
that protocol into the baseline completion rate. NES seeds do not randomize
the fixed level layout.

For a bounded diagnostic full-game run use `--max-frames N --exit-on-terminal`;
the default continues until real GAME OVER or 8-4 clear. Logs
`artifacts/async-*.jsonl` include sampled provenance, lifecycle transitions
and a separate post-action terminal record. On GAME OVER a `knowledge_reset`
event records the memory wipe. Each run gets a new log; there is no automatic
log deletion.

Use `--trace-every-frame` for diagnostic full-action logging. It increases disk
usage; the default retains sampled logs plus bounded failure windows. Preserve
the source/model identities, executed input and lifecycle boundary when comparing
traces. A sampled log cannot establish what happened on every intermediate frame.

### Paired architecture regression screen

Freeze each built controller under a distinct filename before running this
comparison. Do not rebuild or overwrite those binaries during evaluation.
For example, with preserved v13 and v14 executables:

```sh
.venv-mario/bin/python examples/mario_dashboard/evaluate_architecture.py \
  --baseline examples/mario_dashboard/artifacts/jevt_mario_control-v13-baseline \
  --candidate examples/mario_dashboard/artifacts/jevt_mario_control-v14-candidate \
  --cases SuperMarioBros-1-1-v0:speedrun,SuperMarioBros-1-2-v0:score_attack,SuperMarioBros-1-2-v0:hunter,SuperMarioBros-1-3-v0:speedrun \
  --offsets 0,17 --max-frames 3000 \
  --output examples/mario_dashboard/artifacts/architecture-paired.json
```

The evaluator records binary SHA-256 hashes and checks for changes during the
run. Each revision starts a fresh independent stage episode without imported
experience, model inference, save states or action replay. It reports progress,
confirmed eliminations, score and regression signals separately. This isolates
controller revisions, not LAYA versus Open-JEV quality. Timing offsets are not
independent random maps, and this small screen does not establish statistical
significance or campaign completion.

In focused no-model 1-3 fixtures, the v14 motor changes reached X=702 and X=692
for offsets 0 and 17; the frozen v13 comparison reached X=418 in both cases.
**All four runs died.** This is partial progress past the earlier failure, not
a solved level. Hunter and score-attack also remain incomplete; the public
contract is inspectable decisions and measured limitations, not guaranteed
"kill all" or completion with three lives.

## Model latency ablation

```sh
.venv-open-jev/bin/python examples/mario_dashboard/open_jev_runtime.py benchmark \
  --trace examples/mario_dashboard/artifacts/run-20260924T172428Z.jsonl \
  --samples 16 --warmups 3 \
  --output examples/mario_dashboard/artifacts/latency-local.json
```

Pause the live demo first. Warm serial measurements compare the original
four-head workload against compact goal/tactical variants, not model accuracy.
Model service configuration, sampled requests and responses are preserved in
the JSON artifact. Do not compare a CPU/memory-contended run to an idle run
without reporting that limitation.

## Live evidence ontology

The dashboard includes a clickable, run scoped knowledge ontology beneath the
discovered map. It connects verified jump/landing records, enemy motion samples,
pickup and size transitions, enemy identities and outcomes. Current accepted
model judgments can highlight existing facts without changing their evidence
status. It resets on GAME OVER. See
[the view contract](../../docs/KNOWLEDGE_ONTOLOGY.md).
