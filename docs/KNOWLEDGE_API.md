# Evidence-backed knowledge API

Include `<jevt/knowledge.hpp>` for a dependency-free C++20 evidence store and
model-context compiler. This API stores experience; it does not train neural weights.

## Lifecycle and provenance

Facts have a stable proposition ID, subject, relation, object, source, event ID,
run ID, scope, tick, priority and evidence class. Ingest model suggestions as
hypotheses, telemetry as observations and environment-confirmed results as
verified outcomes. Disappearance alone is not evidence of elimination.

Call `begin_run` explicitly. Observation facts require expiry; run facts expire
on a new run, while session facts can be retained. Queries return owned copies
of current, unexpired facts ordered by priority and recency. Equal-ranked facts
retain insertion order. Storage is bounded by the configured capacity.

Session snapshots reject duplicate proposition identities and invalid entries
before replacing existing data. Failed validation leaves the graph unchanged.
Snapshot serialization is an application responsibility.

## Exact context budgets

`compile_knowledge_context` preserves the question and uses a caller-supplied
token counter on the complete question plus input. It admits whole facts within
the budget and reports included and omitted IDs, token count and input digests.
An oversized mandatory observation raises an error instead of silently truncating.

Typed JSON observation is opt-in and requires a caller-supplied validating
formatter. The library does not parse JSON or validate the formatter's output.
Digests are diagnostic identifiers, not collision-proof cache keys; retain exact
input bytes when caching.

The store is not internally synchronized. Serialize mutations and queries, or
protect it with application-level synchronization. Returned query copies may be
passed safely to another thread after the query finishes.

See `tests/knowledge_test.cpp` for executable lifecycle, snapshot, provenance and
token-budget examples. The API is independent of CUDA and inference backends.
