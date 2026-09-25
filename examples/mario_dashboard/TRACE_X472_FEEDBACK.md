# Mario 1-2: repeated loss at x469–472

This is a local trace review, not a claim that Mario can now finish the game.
The runs use the NES emulator with the C++ JevT++ controller and RTX 4090
model backends. The ontology is empirical memory; neither model is trained
online.

## Failure in the original trace

In `artifacts/async-20260925T082159Z.jsonl`, two consecutive GAME OVER runs
lose their final life in world 1-2 at x472. Immediately beforehand, the
controller's `hunter_track_intercept` skill brakes left and then issues
`noop` through the long descent toward an enemy around x479. In one recorded
attempt, Mario is at x455 with feet y168 while the target is at screen y184;
the controller continues `noop` until x467, feet y203. A protective jump
starts only after landing at x468, roughly 12 pixels from the target. The
enemy is not verified as killed. The model's candidate scores are preferences,
not proof that a stomp is feasible.

The feedback path was incomplete: the ontology was updated before
`environment.step()`, but a life-loss event arrived after the step. The
post-step event was not joined to the last observed state. On the collision
frame the enemy tracker may also change from visible to unobserved; using
only that final frame discards the entire preceding intercept sequence.
`unobserved` must never be interpreted as `eliminated`.

## Implemented feedback

The runner now records `knowledge_outcome` after a confirmed life-counter
transition. The graph attaches a stage-scoped x32 danger zone to
`outcome:life_lost`, with source and repeated-event count. A narrow
`observed_failure_context` edge is added only when at least eight
consecutive executed `noop` intercept frames occur within the last 48
frames, and a recently visible target remains near the loss location.
This is a skill/sequence association, not a causal claim that the enemy
caused the loss.

The typed `death_zones` affordance is sent to the C++ decision graph on
every frame. When the same named enemy kind is encountered near a zone with
that failure context, the target skill yields its airborne intercept and
forces replanning. Unrelated deaths do not suppress target pursuit.
The later 1-2 loss at x1104 had a different precursor: Mario repeatedly
waited below an elevated Goomba. A separate `unsafe_waits` relation now
requires at least 12 preceding `hunter_wait_for_reachable_target` frames,
and authorizes a short retreat only when the floor behind is observed safe.
It does not relabel the enemy as killed or claim a collision cause that RAM
does not prove.
The full relation is sent to Open-JEV; a short `knowledge_head` is placed
at the start of model JSON so the 1024-token LAYA sequence can see it.

The model itself has a separate actuator bottleneck. In sampled live frames,
LAYA's plan was fresh and risk-vetted, yet the target skill won the
single-writer arbiter by priority. We tested promoting such plans above the
target skill, including an end-to-end regression that verified a changed
final action. That change regressed full-game outcomes with both backends,
so the experiment was rolled back. Model ranking remains advisory to hunter
target execution while the verified C++ feedback skill can change actions.

## Observed validation and limits

- Before the correction, paired runs died again at x472. One retained run
  showed the same sequence for more than 25 `noop` frames.
- An overly broad first danger-zone guard caused an additional death in 1-1
  and was removed. This is why the shipped guard requires the specific
  intercept-coasting precursor.
- In `artifacts/async-20260925T090430Z.jsonl`, a same-seed Open-JEV run
  lost a life at x469, recorded one such failure context, executed
  `observed_death_zone_replan` on the next approach, and reached x615 in
  1-2 with one life remaining when the 3100-frame diagnostic limit stopped
  it. That is local improvement, not a completed game.
- The visible mixed-backend run in
  `artifacts/async-20260925T091451Z.jsonl` reached x1117 in 1-2 and then
  GAME OVER. Its graph preserved the failure relation after GAME OVER and
  showed `runs_observed=2` after RESTART. The progression cannot be
  attributed solely to LAYA: the C++ controller and ontology changed too.
- An isolated, same-seed Open-JEV session recorded the x1104 wait failure in
  run 1. After GAME OVER and RESTART, run 2 retained the zone and emitted
  `retreat_from_failed_wait` at frame 3245 near x1094. It cleared 1-2
  without another recorded death there, but repeated a 1-1 death near x1488
  and ended GAME OVER in 1-3 at x682. This demonstrates a specific
  cross-run feedback effect, not reliable completion or general learning.
- In the direct-model-actuation experiment, a same-seed LAYA run had two
  deaths and remained in 1-1 at x1799 at the 3100-frame limit. The Open-JEV
  version reached 1-2 but lost its last life at x149 at frame 2602. Both were
  worse than the advisory-controller run above. The priority promotion was
  therefore reverted, not shipped as a gameplay improvement.
- On three frozen, distinct old requests, Open-JEV's warm median was about
  202 ms and LAYA's about 28 ms. Removing the graph changed Open-JEV's
  ranking in one frozen test. LAYA's old tail-positioned graph had identical
  scores with/without it; a short front-positioned head changed LAYA scores
  but not its selected slot in that test. This proves input sensitivity,
  not gameplay superiority. The accepted model plan changes no sampled final
  hunter actions in the retained advisory design. This limit is intentional
  after the failed actuation experiment, not hidden as an autonomy claim.

Remaining work: calibrate model-ranked plans against real outcomes before
giving them hunter actuator priority; then measure full-game outcomes over
multiple seeds, track final model-controlled actions and actual life/clear
rates, diagnose the repeated 1-1 x1488 death and 1-3 falls, and validate
whether these narrow failure relations transfer to genuinely different
enemy/geometry contexts without overgeneralizing.
