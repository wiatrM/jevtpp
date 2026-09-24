# Observability and HTTP dashboard

## Diagnostics are an API

Monitoring must remain available to embedded and network-restricted programs.
The core therefore exposes immutable snapshots; an HTTP server is only one
possible consumer.

A `jevt::Diagnostics` registry is configured with
`jevt::DiagnosticsOptions` (recent-call capacity, maximum decisions, maximum
tag length, histogram bounds and recent-trace sampling). `record_call()` accepts the decision ID,
`CallOutcome::success`, `error` or `abstain`, a nanosecond duration, and an
optional bounded tag. `snapshot()` returns:

- total, successful, abstained and failed call counts;
- the same counters grouped by decision ID;
- p50, p95 and p99 end-to-end latency;
- bounded newest-first recent calls;
- histogram buckets and the number of dropped decision IDs.

Percentiles are estimates from bounded aggregation and should report their
unit and sample count. The nearest-rank bucket (`ceil(q * calls)`) supplies its
upper bound; overflow uses the maximum observed latency for that aggregate
since construction or `reset()`, not its mean. These conservative estimates
are not exact sample percentiles. Empty windows use zero-valued latency statistics. Call
counts and histogram bucket counts include every recorded call, even when
recent traces are sampled or disabled. Snapshots taken while
calls are active are internally consistent but naturally become historical
immediately after capture.

`recent_sample_every` defaults to `1`, retaining every call up to
`recent_capacity` (default `256`). Set it to `N` to retain calls `1`, `1 + N`,
`1 + 2N`, and so on within each measurement window. Set either option to zero
to disable recent traces while keeping aggregate metrics. Recent-call
`sequence` values are call ordinals and can therefore have gaps. `reset()`
clears both metrics and traces and restarts the sampling window.

Do not use raw user text, customer IDs, arbitrary labels or exception messages
as metric dimensions. This prevents cardinality explosions and data leakage.

## HTTP surface

When `JEVT_ENABLE_HTTP=ON`, the optional service presents the same snapshot:

| Endpoint | Media type | Purpose |
|---|---|---|
| `GET /healthz` | `text/plain` | Dashboard process health only |
| `GET /api/stats` | `application/json` | Machine-readable snapshot |
| `GET /metrics` | Prometheus text | Monitoring-system scrape |
| `GET /` | `text/html` | Read-only operational dashboard |

The HTML view renders totals, outcome/error mix and latency charts
from the JSON endpoint. Assets are embedded so the dashboard has no CDN
or internet dependency. Unknown methods return 405; unknown paths return 404.
Incomplete or malformed headers return 400; headers that exhaust the configured
byte limit without a terminator return 431.
Responses should set `Content-Type`, `X-Content-Type-Options: nosniff`, a
restrictive content security policy, and `Cache-Control: no-store`.

The same representations are available without HTTP through `to_json()` and
`to_prometheus()`. The JSON uses `generated_at_unix_ms`, a `total` counter object,
`decisions` entries shaped as `{decision, stats}`, `recent_calls` and
`dropped_decisions`. Each counter object has the following shape (illustrated
with custom histogram bounds of 1 and 10 ms):

```json
{
  "calls": 1,
  "successes": 1,
  "errors": 0,
  "abstains": 0,
  "latency_ms": {
    "mean": 0.25,
    "p50": 1,
    "p95": 1,
    "p99": 1,
    "histogram": {"bounds": [1, 10], "counts": [1, 0, 0]}
  }
}
```

`reset()` clears the registry explicitly.

## Safe operation

`jevt::HttpServerOptions` defaults to `127.0.0.1` and port `0`, allowing the OS
to select a free port. It also configures the listen backlog and maximum
request size. Starting is explicit; `running()`, `port()` and `url()` expose
the current state, and `stop()` shuts the accept thread down.

The built-in service is not an authentication or TLS boundary. For remote
access, place it behind the host application's authenticated reverse proxy or
export snapshots into the application's existing telemetry stack. Apply
network policy and request limits there.

Dashboard failure must not fail inference. Server shutdown is bounded and
joins its worker thread after interrupting the active client socket. Reads and
writes have a two-second idle timeout. This is a single-client-at-a-time utility,
not a production HTTP server or total-request-deadline implementation.
Snapshot generation is bounded by the number of
registered schemas and fixed outcome/error dimensions.

## Advanced analysis without sensitive payloads

Call inspection means outcome counts, opaque bounded tags and timings. Storing
raw prompts, input text, token streams or
full model output is off by default. If future tracing is added, it should use
explicit sampling, caller-provided redaction, bounded retention and separate
access control.

The optional `batching_backend` also exposes `stats()` with queue depth,
in-flight requests, accepted/rejected/completed requests and batch count.
These scheduling counters are separate from decision outcome diagnostics.

## Benchmark expectations

`jevt_diagnostics_contention` runs concurrent writers and a continuously
snapshotting reader. Its arguments are writer count, calls per writer, recent
capacity and sampling interval; defaults are `4 100000 256 1`. It reports
calls per second and p50/p95 recording and snapshot latency. Compare full,
sampled and disabled recent traces on the same hardware; disabling traces
does not disable counters. No default-mode speedup is assumed.

Broader benchmark scenarios should include one hot schema, many schemas, mixed
outcomes and a reader taking snapshots under write load. A CI smoke test checks
correct execution; stable performance thresholds belong on controlled
hardware rather than shared hosted runners.

