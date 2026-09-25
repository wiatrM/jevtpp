# Mario v10: diagnosis and optimization (2026-09-24)

## Scope and current verification

This is a real NES SMB full-game emulator demo, not a recreation. Open-JEV remains a CUDA model
under JevT++; it does not replace the C++ orchestrator. No ROM was downloaded
or modified during this optimization.

Implemented architecture:
NES RAM → C++ DAG projections → compact strategic request → asynchronous
Open-JEV → epoch/age/target validation → frame-rate C++ reflex controller.

The generic runtime is in include/jevt/graph.hpp. It supports named dependency
outputs, topological validation, conditions, failures, stale results, snapshot
provenance and execution traces. It is a first reusable library primitive,
not a finished visual graph-builder product, scheduler or arbitrary JSON DSL.

Later additions: strict RTX 4090/CUDA loaded-tensor validation, a Situation
Report, and bounded empirical experience scoped to one game run. The final
graph has real `game_phase`, `learning_delta`, `digital_twin` and `risk_policy`
outputs in the causal path to model context and motor control. The latency
ablation below predates this expanded v10 input. A separate measured final-v10
run appears below.

## Confirmed causes, not a claim that model weights are broken

- Enemy type 0 was incorrectly treated as an empty slot. It is a green Koopa.
  The reactive prototype died at x≈1675 until this sensor bug was fixed.
- Grounded status was inferred from nearly constant Y. This can mistake a jump
  apex for contact. Ground contact now comes from SMB RAM movement state 0x1D.
- Short A-button releases could cancel takeoff before NES input polling accepted
  it. The controller now handles input latching, airborne commitment and landing
  release explicitly.
- The old loop blocked emulation on every inference. A 900 ms request slowed the
  game; it did NOT advance 54 simulated frames. The previous conversion of wall
  latency into state age was incorrect and contaminated collision predictions.
- Four SystemOne fields were independent judgments over the same verbose JSON.
  They were not successive layers consuming predecessor decisions. Adding more
  fields multiplied model work without creating a causal decision hierarchy.
- Old benchmark traces mixed different sensor revisions, horizons and episode
  counts. They cannot establish a Laya-versus-Open-JEV gameplay winner.
- Full-game testing exposed an overhead-wall / lower-passage ambiguity in 1-2.
  A repeated jump at world x≈978 spent lives despite successful 1-1 tests.
  A RAM-observed lower-passage recovery now passes this geometry; the next
  failure is a pit under a low ceiling (death routine later reports x≈1204).
  These are perception/control
  generalization failures, not evidence that CUDA model weights are wrong.

No evidence here demonstrates an error in Open-JEV's neural weights or JevT++
tensor inference. The confirmed bugs are observation/control/orchestration bugs.

## What changed

### Fast/slow separation

async_runner.py runs a persistent C++ JSONL graph every emulated frame. One
background worker requests strategic intent at most once per configured interval
(default 30 frames). It allows only one in-flight request, with no stale queue.
Results retain source epoch/frame/mode; old episode responses AND errors are
discarded. The C++ gate checks a 90-frame intent TTL and current target feasibility.

The Open-JEV goal can influence safe controller choices, e.g. aligning for a
stomp or slowing for a nearby powerup. Immediate jumps and safe locomotion are
deterministic C++ rules.
This is deliberately a hybrid controller, not evidence that the model alone
learned to beat Mario. Hunter/collector/score modes are limited objective hints;
they do not guarantee every enemy, coin or mushroom is collected.

Wall-clock model latency, model age in simulated frames, C++ graph duration and
emulated FPS are separate metrics. Image encoding/publishing is capped at 12Hz;
physics pacing subtracts actual processing time instead of sleeping an additional
full frame after each operation. Terminal logs contain the observation AFTER
the terminal action, not the preceding live observation.

### Actual DAG trace on the dashboard

The UI lays out the nodes and edges returned by graph.evaluate(), not an invented
fixed graph. Click/keyboard inspection exposes exact dependency JSON, output,
status, snapshot, edge payload size and node duration. Pan/zoom and fit controls
are available. Packet animation stops without fresh running snapshots. The
asynchronous model mailbox is explicitly labelled as cached intent, not a new
model inference every frame. Terminal screens label the last pre-terminal trace.

### Latency measurement

Pinned Open-JEV-2B checkpoint, same already-loaded CUDA service, 16 sampled
observations per variant, three warmups, serial round-robin HTTP requests:

| Run | Legacy 4-head P50/P95 ms | Compact goal6 P50/P95 ms |
| --- | ---: | ---: |
| First completed 16-state run | 758.5 / 931.4 | 196.3 / 236.9 |
| Later corrected coin-field run, host under pressure | 1081.5 / 1292.4 | 274.3 / 299.6 |

About 3.9× faster in both runs. This is a workload ablation, NOT equivalent
gameplay quality or model-vs-model comparison. The first compact request used a
constant zero coins_visible field; the corrected request uses actual coins_collected.
The latest C++ prompt splits the same instruction with a double newline to match
SystemOne serialization; a final exact-format rerun was interrupted by WSL failure.

The original batch scored 17 sequences, with four prefix forwards and four suffix
forwards; compact goal6 uses six sequences, one prefix and one suffix forward.
Logical input tokens fell from roughly 22,178–23,878 to ~950; compact state was
87–93 tokens. Upstream prefix caching is request-local, not persistent across
game frames. FLA kernels are present; causal_conv1d is absent, so the warning
about fallback convolution does not mean all FLA acceleration is unavailable.

Raw last completed samples: artifacts/latency-open-jev-original-service.json.
Older measurements are retained as observations, not cherry-picked best latency.
A second simultaneously loaded service failed with OOM and was not used. Do not
load duplicate checkpoints on this host while compiling/benchmarking.

Final v10 C++ `compact_state` → local HTTP → Open-JEV on RTX 4090 CUDA,
16 serial trace-sampled requests, three warmups: **P50 152.559 ms, P95
176.485 ms**, mean 155.892 ms, input range 1795–1879 tokens. This is model
request latency, not emulator frame latency or a gameplay win rate. Artifact:
`artifacts/latency-jevtpp-v10-rtx4090.json` (records runtime tensor device,
prompt hashes and each inference). The C++ reflex continues at 60 emulated FPS
while requests are in flight.

## Completion evidence and limits

The final C++ baseline completed 18/18 SMB 1-1 episodes: seeds 0,1,42 ×
initial neutral delays 0,1,7,17,31,60. It used current observations, no
savestates, recorded actions or level coordinates. Artifacts:
`artifacts/controller-cpp-final-matrix.json` and the per-mode traces. Accepted
model stomp intent in hunter completed 3/3 tested offsets with 6/3/4 observed
kills; collector and score-attack each completed 3/3, collected a real
mushroom and showed different scores/actions. Artifacts:
`controller-hunter-final-stomp.json`, `controller-hunter-safe-fallback.json`,
`controller-collector-final.json`, and `controller-score-final.json`. These are fixed-level tests,
not a guarantee of exhaustive kills or generalization.

The full-game `SuperMarioBros-v0` live CUDA run cleared 1-1 and entered 1-2
with three lives, carrying knowledge across the transition. Before the lower
passage fix, it got stuck at x≈978, lost all three lives, reached actual
GAME OVER at frame 19051, and reset memory (`memory_generation` 1→2,
observed_frames=0, columns=0). This verifies the lifecycle, not full-game
success. A subsequent RAM-derived low-ceiling takeoff and side-pipe entry
advanced through 1-2: `controller-full-game-sidepipe.json` records
1-1→1-2→1-3 in 2862 frames, 0 deaths, three lives, and memory spanning all
three stages. The actual CUDA/Open-JEV live run also reached 1-3, then lost
its three lives at the first elevated-platform crossing; its GAME OVER again
cleared memory. An observed-platform landing plan subsequently passed that
first crossing in the no-model emulator trace but still failed farther on
1-3 (vertical approach to a higher platform); the last trace is
`controller-full-game-landings-next.json`. No claim of full-game completion is made. Earlier failed
traces remain in `controller-full-game-tunnel.json` and
`controller-stage2-contact.json`. A
controlled dropped-input stress test passed 3/3 with a neutral
input every 31 frames but failed 3/3 at intervals 17 and 73; the policy is
not fault tolerant.

These seeds do not randomize the fixed NES level. This is NOT evidence of 100%
completion on all worlds, arbitrary start states or perturbations. No guarantee
of “always completes the game” is justified by this matrix.

Latest Python regressions: 106/106 passed; native
memory tests and graph/SystemOne CTest passed. Chromium QA covered desktop,
tablet and mobile with zero JavaScript exceptions and the original 256×240
NES frame visible. The live `/api/runtime` confirmed loaded Open-JEV base and
head on `cuda:0` NVIDIA GeForce RTX 4090.

## Research relevance

Open-JEV's Doom example uses a narrow observation and frozen small contract
(movement, attack, alignment), not an unconstrained long-horizon gameplay policy.
The useful transfer is bounded evidence and measurable control, not more
independent questions. A richer generic graph should compose outputs, validate
contracts and deadlines, and expose provenance; model calls are one node kind.

Primary sources:
- https://github.com/Zefan-Cai/Open-Jev/blob/main/docs/inference-latency.md
- https://github.com/Zefan-Cai/Open-Jev/blob/main/docs/doom-case.md
- https://github.com/Zefan-Cai/Open-Jev/blob/main/docs/games.md
- https://zefan-cai.github.io/open-jev/
