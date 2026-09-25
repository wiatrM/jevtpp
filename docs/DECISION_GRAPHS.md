# Explicit decision graphs

`<jevt/graph.hpp>` adds an application-level DAG above the existing typed
decision API. Use it when a later evaluation must consume an earlier answer.
Fields within one `decision_model` still receive the same input independently;
declaring several fields does not make a reasoning chain.

The graph runtime handles scheduling, declared dependencies, snapshot identity,
failure propagation and traces. Each node owns its output `state_value`; JSON
remains a string serialized and validated by the application. No JSON library,
game rules or model provider is built into the graph runtime.

## Small example

```cpp
#include <jevt/graph.hpp>

using Result = jevt::graph_node_result;
jevt::decision_graph graph({
    {"goal", {}, [](const jevt::graph_context& ctx) {
        // May instead invoke a typed System One model on ctx.snapshot().input.
        return Result::success(jevt::json_state(R"({"goal":"survive"})"));
    }},
    {"focused_state", {"goal"}, [](const jevt::graph_context& ctx) {
        // Here JSON concatenation is safe only because these are complete JSON
        // values. Real projections should use the application's JSON serializer.
        return Result::success(jevt::json_state(
            "{\"goal_answer\":" + ctx.at("goal").output.content +
            ",\"observation\":" + ctx.snapshot().input.content + "}"));
    }},
    {"ranking", {"focused_state"}, [](const jevt::graph_context& ctx) {
        const auto& state_for_model = ctx.at("focused_state").output;
        // Replace this example answer with model.evaluate(state_for_model),
        // check its result, and serialize the probabilities and selected value.
        (void)state_for_model;
        return Result::success(jevt::json_state(R"({"action":"jump"})"));
    }},
    {"controller", {"ranking"}, [](const jevt::graph_context& ctx) {
        // An application would validate allowed actions and current constraints.
        return Result::success(ctx.at("ranking").output);
    }},
});

const auto trace = graph.evaluate({
    "episode-3:frame-42", jevt::json_state(R"({"gap_ahead":true})")
});
if (trace.at("controller").status == jevt::graph_status::succeeded) {
    const auto& command = trace.at("controller").output.content;
    // Apply command through the application's controller.
}
```

Node registration order need not be topological. Construction rejects missing
predecessors, cycles (including self-dependencies), repeated node IDs, duplicate
dependencies, empty IDs and missing callbacks. An empty graph is valid.

## Data and execution contract

- Every evaluation takes a nonempty snapshot ID and one immutable input. Include
  the episode/session identity as well as the frame/revision to avoid collisions
  after reset. The runtime does not assign IDs or inspect their meaning.
- A callback can read that snapshot and only its declared direct predecessors.
  `context.at("name")` throws for undeclared predecessors, even when they have
  already executed. References returned by the context last only for the callback.
- The application projects and merges selected predecessor outputs explicitly.
  This avoids accidental key collisions and sending the whole growing trace to
  the model at every layer. A declared edge establishes ordering, not an automatic
  recursive JSON merge or proof that a callback used the input.
- Callbacks run sequentially in a stable topological order. One slow callback
  delays its consumers. Keep high-frequency control outside slow strategic model
  evaluations when running a real-time system.
- Outputs computed for the current snapshot can omit `snapshot_id`. Cached or
  asynchronously produced outputs must include their actual source snapshot ID:
  `Result::success(payload, source_snapshot_id)`. A mismatch becomes `stale` and
  blocks normal consumers. The runtime cannot detect falsely labelled provenance.
- No result cache is maintained across evaluations. Graph evaluation is locally
  isolated; thread safety of captured mutable state or a model backend remains
  the callback author's responsibility.

## Conditions, failures and freshness

The optional fourth node field is `condition(context)`. A false condition records
`skipped` without invoking the evaluator. Exceptions from conditions or evaluators
are recorded as `error`; unrelated branches continue.

By default, any failed/skipped predecessor skips its consumers; a stale predecessor
makes its consumers stale. A fallback or merge node can set the fifth node field,
`require_successful_dependencies`, to `false`. It then receives all declared
predecessor records and must check each status before reading its output.

`graph_run_options::deadline` is an optional `steady_clock` deadline. Nodes that
have not started at expiry are stale. A successful callback returning after the
deadline is also stale, with its output preserved for diagnostics. The deadline
does **not** cancel an in-flight model call or roll back side effects. Pass backend
deadlines separately and keep external actions outside evaluators until checking
the final node's status and snapshot identity.

The four trace states are `succeeded`, `skipped`, `error`, and `stale`. Only
`succeeded` is actionable. `graph_status_name()` gives their stable text labels.

## Trace and dashboard integration

`graph_trace` owns the evaluation snapshot ID, topologically ordered node records,
declared `from`/`to` edges and total duration. Each node records its ID,
dependencies, status, owned output, diagnostic message, output snapshot ID and
duration in nanoseconds. Serialize this actual trace to the dashboard so node
labels and edge inspectors show the data that was evaluated. A stale output may
be useful for inspection but must not be presented as the current controller action.

This initial runtime returns a completed trace; it does not stream a `running`
event while a callback is executing. It also does not provide parallel scheduling,
incremental recomputation, durable replay, arbitrary workflow loading, a visual
editor or schema migration. Those require separate contracts and validation before
this becomes a general workflow product.

## From the Mario demo toward reusable workflows

Keep three concerns separate: reusable graph execution, typed model adapters, and
domain nodes (RAM interpretation, candidate generation, collision constraints,
controller commands). Making every domain predicate a model call increases latency
without improving physical accuracy. Deterministic measured facts belong in code;
model nodes can rank feasible alternatives from a compact shared snapshot.

A reusable product can build on a versioned registry of node types and input/output
schemas, explicit JSON projections on edges, replay fixtures, and a visual editor
that consumes the same graph definition and runtime traces. Shipping the runtime
alone is not evidence that arbitrary games or workflows will succeed. End-to-end
completion, survival and latency must be tested per domain and execution mode.

Run the focused tests with `ctest --test-dir <build> -R jevt_graph_tests`.

## Mario forecast and feasible model goals

The Mario graph has an explicit `digital_twin` node consuming current physics,
collision evidence, threats and the statistical experience summary. It projects
enemy distance eight frames ahead at constant relative velocity and reports a
contact ETA only when a velocity observation exists. Current measured RAM wins;
otherwise a same-type velocity estimate requires at least five empirical samples.
Landing time is only a rough estimate from at least five completed clean jumps,
with no estimate while engaging an enemy or outside the observed mean airtime.
Missing evidence produces `null` forecasts, not invented confidence. This is a
short-horizon state projection, not a full collision simulator or a promise of
safe landing. Remembered level geometry never drives the controller.

The full forecast is visible in the execution trace. Only a bounded summary is
passed through `compact_state.forecast` to the next model request. The intent gate
filters unavailable targets using current RAM and selects the highest original
model score among feasible goals. It retains `raw_model_goal`,
`selected_goal_probability`, `intent_selection` and per-goal eligibility in the
trace. Scores are never replaced with scripted outcomes or renormalized into win
probabilities. Epoch, mode and frame-age checks apply before any goal can be used;
an expired model result cannot become valid by choosing another goal.

The selected objective is a higher-level constraint: `hunter` prioritizes a
currently observed nearby stompable hostile, `collector` a visible power-up, and `score_attack`
allows scoring, stomping or collecting available targets. These modes permit
`finish_fast` when their target is absent; `speedrun` selects progress. Recovery
remains eligible when the current RAM watchdog reports a stall. Model scores rank
only the goals permitted by that objective and current observations. The trace
separates `target_available` from `mode_allowed`, preserving the raw model answer
even when a higher-scoring goal conflicts with the user's selected mode. Motor
handlers and physical safety remain separate from this strategic filtering.
The enemy gate requires the current telemetry's positive
`nearest_enemy_stompable` capability: missing/false capability does not authorize
a stomp. This prevents treating a Piranha Plant as a Goomba. The distance bound
is a candidate filter, not proof of a physically reachable or safe intercept;
the motor handler still checks geometry and the current jump phase.

## Discovery lifetime

Mario's JSONL process accepts `{"command":"memory_reset"}`. It creates a fresh
`KnowledgeMemory` for the same environment and returns `{"ok":true,"memory":...}`
with its empty diagnostic snapshot. This clears columns from every world/stage,
enemy statistics, jump samples, outcomes and sample history. It does not reset
the physical controller; the runner separately resets that when restarting play.

The runner should issue this command at **GAME OVER**, before starting a new
discovery run. A lost life or a cleared stage only changes the observation epoch:
`reset_episode` clears consecutive-sample history but preserves accumulated
knowledge. Column keys include world and stage; compact geometry summaries only
refer to the currently observed stage. Model weights are not changed by either
operation. Import is explicit, environment-validated and atomic; a rejected
import leaves existing memory untouched. Exported files are not deleted by the
in-process reset, so the runner must not silently import an old export after a
GAME OVER reset.

Full-game memory uses `(run_id, deaths, world, stage)` as its physical episode
identity. Changing objective mode invalidates model results without resetting
velocity/jump samples, and a repeated same-frame observation stays deduplicated.
Inputs without this session identity retain the legacy graph-epoch boundary.
Session outcomes are deduplicated by `run_id/session_frame/event`; when session
lifecycle is present, raw pre-skip RAM flags are not counted as another clear or
death. This avoids counting one flag and its ensuing stage transition twice.

## Incremental state and risk policy

`game_phase` resolves current session/RAM lifecycle data (world/stage, run and
memory generation, lives, life lost, stage clear, GAME OVER, game complete).
It feeds the compact model state and the controller's terminal flags. Unknown
lives remain `null`; they are not reported as zero lives.

`learning_delta` records actual evidence added by this observation: newly seen
or changed columns, new enemy types, accepted velocity samples and clean jumps.
Duplicate or out-of-order snapshots report `observation_accepted=false` and zero
increments. These are empirical memory updates, **not neural weight training**.
The small delta enters the next model request, while full observations stay local.

`risk_policy` consumes lifecycle, short-horizon forecast and learning delta.
The last life, a terminal/transition state, unavailable RAM projection, an enemy
with unknown motion or the first observation of a type make it cautious. This
policy is passed to the motor controller, which vetoes optional pursuit/alignment
and collection while keeping required reactive collision jumps. It does not
turn uncertain forecasts into safe trajectories or replace the model's scores.
