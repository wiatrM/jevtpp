# NES Mario 1-3: controller verification (2026-09-25)

Goal: complete the full original SMB episode with three lives and no scripted trajectory. Status: **not achieved**.

## Reproducible evidence

- Isolated full-game run: `artifacts/async-20260925T102224Z.jsonl`, seed 123, speedrun, Open-JEV judgment on RTX 4090, C++ JevT++ controller, 24-frame asynchronous model interval, 240 emulator frames/s.
- Stages 1-1 and 1-2 cleared; GAME OVER at session frame 3590 in 1-3 with 0 lives and 2/32 stages cleared.
- Three confirmed 1-3 life-loss knowledge outcomes were at x=634, 715, and 758. The knowledge graph retained these observations across lives, but they did not produce a successful corrective action.
- Isolated 1-3 A/B traces: `async-20260925T100950Z.jsonl` reached x=1543 with jump staging/commitment; `async-20260925T101652Z.jsonl` reached x=1231 after correcting NES ID 0x28 to moving platform; `async-20260925T101906Z.jsonl` reached only x=960 after a conservative landing-contact rule, so that rule was reverted.

## Causal findings and limits

1. Earlier rollouts replanned every frame and repeatedly selected the first action of a multistep jump. Bounded continuation of a verified support-to-support plan reached farther in the isolated stage.
2. NES enemy-slot ID 0x28 is a moving platform, not a hostile. It is now exposed separately as `moving_surfaces`. Its RAM motion is projected in C++, but it is not promoted to a static `landing_surfaces` target until a reliable reachability contract exists.
3. A plan landing on *some* support previously bypassed the baseline gap guard even if its chosen target was another, unreachable platform. The guard now requires `target_landed`.
4. At x≈1228 on one trace, Mario touched the upper platform beside a Goomba and died. A naive three-frame enemy-contact penalty worsened earlier traversals and was reverted. The planner still needs validated swept collision/landing safety and enough control authority to brake before contact.
5. Death-zone facts in the retained knowledge graph are evidence of loss location, not proof of cause. The current speedrun controller does not use them to change platform traversal; merely injecting them into model context is insufficient to demonstrate learning.

No full-game completion claim should be made from unit-test success or a local stage-position gain. Next acceptance gate is repeated seeded full-game runs with per-life death causes and explicit action-difference evidence before/after each learned fact.

## Follow-up, same day

- LAYA CUDA with observed short-gap late takeoff and narrow-support run-up reached x=1378 in isolated 1-3 (`async-20260925T103516Z.jsonl`) but still fell at the next unmapped gap.
- A complete 60 FPS LAYA run (`async-20260925T104832Z.jsonl`) still ended at 2/32 stages. Three 1-3 losses were observed near x=695, 694, 694. This is direct evidence that retaining death-zone facts did not alter the outcome.
- Frame trace `async-20260925T104345Z.jsonl` showed the airborne planner carrying a takeoff target x=512–560 while Mario was already seeing supports x=560–640 and x=640–752. A naive nearest-support retarget passed a unit contract but regressed both isolated and full-game runs (`async-20260925T105151Z.jsonl`, `async-20260925T105550Z.jsonl`), so it was reverted. Correct retargeting requires a verified landing/second-transfer cost, not merely a nearer tile.
- A blanket high-ledge waiting rule also regressed the real run because it activated on an earlier, traversable platform. It was reverted. Unknown geometry and moving-platform timing still need a risk policy grounded in observed outcomes.
- The asynchronous model makes accelerated 240 FPS A/B runs sensitive to wall-clock response timing. For acceptance, evaluate the 60 FPS product rate and repeat seeds/runs; isolated stage starts are also shifted by an initialization frame relative to full-game stage transitions.
