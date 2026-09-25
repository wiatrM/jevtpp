# Generic skill arbitration and configured graphs

`jevt/skill.hpp` supplies a domain-independent final-action arbiter and an
execution-result contract. `jevt/operator_registry.hpp` compiles a runtime graph
description into the existing `decision_graph`. Neither header requires a model,
CUDA, Mario, a JSON library, or a network service.

These are execution and evidence primitives, not a trained policy or a guarantee
of safety. An adapter still supplies observations, available actions, validity
conditions, effect predictions, and a verifier that understands its domain.

## One final action

```cpp
#include <jevt/skill.hpp>
using namespace jevt::skills;
const snapshot current{"run-7", 42, 3, 9};
std::vector<proposal<std::string>> proposals{
    {"route", "reach_target", "planner", "move_right", current, 0, 10, 0.4,
     false, {{"path_clear", truth::true_value, "sensor observation"}}},
    {"attack", "attack_target", "model", "jump", current, 0, 20, 0.9,
     false, {{"path_clear", truth::false_value, "observed wall"}}}
};
const auto result = arbitrate<std::string>(current, proposals);
// result.action == "move_right"; the higher-ranked attack is prohibited.
// Send result.action unchanged. Record any actuator mismatch separately.
```

The action type may be a string, an application struct, or an optional adapter's
JSON value. The arbiter copies the winning value and does not call actuators.
Only one returned action may be emitted by the integration. The C++ API cannot
stop unrelated application code from bypassing that boundary; test that wiring.

Admission precedes ranking:

1. Proposal IDs must be unique and nonempty; skill/source IDs and utility must be
   valid. Non-finite utility is rejected, not sorted unpredictably.
2. Run, relevant-context version, and candidate-set version must match. Future
   frames and frames older than `max_age_frames` are stale. Default age is zero.
3. Every supplied condition must be `true_value`. `unknown` is not permission;
   `false_value` is an observed prohibition. Both remain distinct in the trace.
4. Among admitted regular proposals, highest priority wins, followed by utility.
   Equal priority and utility preserve input order. Utility is not a probability.
5. Emergency proposals are considered only when no regular proposal is admitted.
   They do not bypass conditions or freshness. An emergency label is not a safety
   guarantee. No admitted proposal returns an empty optional, not a "safe noop".

`context_version` is adapter-owned: increment it when a fact relevant to the
proposal changes (support, obstruction, target phase), not arbitrarily on every
unrelated sensor update. Increment `candidates_version` when replacing the set
that an asynchronous model ranked. Capture provenance when inference starts;
never stamp an old response with the current frame on arrival. The arbiter cannot
detect forged provenance or undeclared domain conditions.

## Verified outcomes, separate from selection

`skill_execution` tracks one attempt:

`ready -> running -> verifying -> succeeded | failed`

Initiation with false or unknown conditions yields `blocked`. An active attempt
may instead become `interrupted`. It cannot be overwritten by another attempt.
After actual execution, call `begin_verification()`, then `verify()` with:

- the unique attempt ID and run ID;
- an observation frame strictly after the start;
- a nonempty evidence ID;
- a separately observed effect: true, false, or unknown.

Unknown evidence keeps the attempt in `verifying`; it is neither success nor a
failed training example. Stale/wrong-attempt evidence is rejected. Terminal
outcomes cannot be overwritten. `verify()` returns *evidence accepted*, not
*skill succeeded*. Read `status()` to determine the outcome.

Example: a target disappearing is unknown unless the domain independently
confirms its elimination. A rejected proposal was not executed and must not be
recorded as a failed attempt. A button request is not proof that a jump began.
Use unique attempt IDs across the run; ID generation, deadlines, active-time
budgets, progress estimation, and independent result verification belong to the
adapter. This small lifecycle deliberately does not claim to implement them.

## Runtime configuration without recompilation

Register compiled factories under versioned operator IDs. A factory validates
parameters and returns a callback; parameters must be captured by value. Then
call `operator_registry::compile(graph_spec)`. The existing graph evaluator
validates duplicate IDs, missing dependencies, cycles, undeclared dependency
reads, and predecessor status. The registry rejects unknown operators and schema
versions, and bounds the graph to 1–1024 nodes.

IDs are bounded to 128 bytes; parameters to 64 KiB per node and 1 MiB per graph.
These checks happen before operator factories run. Applications accepting external
configuration must additionally bound input bytes and parser nesting *before*
constructing the IR. Registered native callbacks are trusted code, not sandboxed
scripts; the graph's cooperative deadlines cannot interrupt a blocked callback.

`compile` here means assembling a validated in-memory DAG, **not invoking a C++
compiler**. Load a new specification, validate it completely, then replace the
old graph at an application-defined snapshot boundary. In-flight snapshots must
finish on their original graph or be invalidated; the registry performs no live
hot-swap coordination itself.

For JSON, explicitly include `jevt/graph_json.hpp` and provide nlohmann_json:

```cpp
#include <jevt/graph_json.hpp>
auto spec = jevt::parse_graph_spec(nlohmann::json::parse(config_text));
auto graph = operators.compile(spec);
auto trace = graph.evaluate({"run:frame:version", jevt::json_state(input_text)});
```

Accepted root fields are `schema_version` and `nodes`. Each node declares `id`,
`operator`, optional `depends_on`, `params`, and `require_successful_dependencies`.
Unknown fields are rejected, including misspelled dependency names. Operator
factories own parameter schemas; the IR does not pretend to type-check arbitrary
opaque payloads. Explicit fallback nodes may opt out of predecessor-success
requirements, but their callbacks must check status before reading output.

Changing topology or registered-operator parameters does not require rebuilding
the binary. Adding a new native operator does. JSON is configuration, not an
unrestricted code evaluator, a neural network, or meta-learning by itself.

## Executable second-domain examples

`examples/skill_runtime_demo.cpp` registers only `observe.v1`, `propose.v1`, and
`arbitrate.v1`. The *same binary* reads either:

- `examples/skills/warehouse.json`: an occluded aisle is unknown, so select a
  confirmed available dock instead of moving a pallet through it;
- `examples/skills/routing.json`: an unauthorized refund is prohibited, so select
  human review instead.

Both are deterministic synthetic fixtures. They emit commands and rejection
traces as JSON with `actuated:false`; they do not drive machinery, route tickets,
issue refunds, use learned physics, or demonstrate product effectiveness.

Build the opt-in JSON examples/tests using the existing remote dependency option
(no remote calls occur):

```sh
cmake -S . -B build-skills -DJEVT_BUILD_TESTS=ON -DJEVT_ENABLE_REMOTE=ON \
  -DJEVT_REMOTE_CURL=OFF -DJEVT_BUILD_EXAMPLES=OFF -DJEVT_BUILD_BENCHMARKS=OFF
cmake --build build-skills --target jevt_skill_tests jevt_graph_json_tests jevt_skill_runtime_demo
ctest --test-dir build-skills -R 'skill|graph_json' --output-on-failure
build-skills/tests/jevt_skill_runtime_demo examples/skills/warehouse.json \
  examples/skills/warehouse_state.json
build-skills/tests/jevt_skill_runtime_demo examples/skills/routing.json \
  examples/skills/routing_state.json
```

Core-only `jevt_skill_tests` requires no remote/JSON dependency. It covers vetoes,
unknown state, freshness, version invalidation, emergency behavior, stable ties,
verified outcomes, second-domain reuse, and reconfigured DAGs. JSON tests cover
schema validation, registry allowlisting, and changing parameters without changing
the binary. These tests establish contract behavior, not Mario completion or
learning effectiveness.
