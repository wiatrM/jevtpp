# Observability and HTTP dashboard

## Diagnostics are an API

Monitoring must remain available to embedded and network-restricted programs.
The core therefore exposes immutable snapshots; an HTTP server is only one
possible consumer.

A `jevt::Diagnostics` registry is configured with
`jevt::DiagnosticsOptions` (recent-call capacity, maximum decisions, maximum
tag length and histogram bounds). `record_call()` accepts the decision ID,
`CallOutcome::success`, `error` or `abstain`, a nanosecond duration, and an
optional bounded tag. `snapshot()` returns:

- total, successful, abstained and failed call counts;
- the same counters grouped by decision ID;
- p50, p95 and p99 end-to-end latency;
- bounded newest-first recent calls;
- histogram buckets and the number of dropped decision IDs.

Percentiles are estimates from bounded aggregation and should report their
unit and sample count. Empty windows use zero-valued latency statistics. Call
counts are exact where practical; snapshots taken while
calls are active are internally consistent but naturally become historical
immediately after capture.

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

The HTML view should render totals, outcome/error mix and p50/p95/p99 charts
from the JSON endpoint. Assets should be embedded so the dashboard has no CDN
or internet dependency. Unknown methods return 405; unknown paths return 404.
Responses should set `Content-Type`, `X-Content-Type-Options: nosniff`, a
restrictive content security policy, and `Cache-Control: no-store`.

The same representations are available without HTTP through `to_json()` and
`to_prometheus()`. A representative JSON payload is:

```json
{
  "schema_version": 1,
  "generated_at": "2026-09-24T12:00:00Z",
  "decisions": [
    {
      "id": "support.routing",
      "calls": 1200,
      "successes": 1140,
      "abstains": 45,
      "errors": 15,
      "latency_mean_ms": 0.96,
      "latency_p50_ms": 0.82,
      "latency_p95_ms": 2.4,
      "latency_p99_ms": 5.1
    }
  ]
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
joins its worker thread. Snapshot generation is bounded by the number of
registered schemas and fixed outcome/error dimensions.

## Advanced analysis without sensitive payloads

Call inspection means outcome counts, opaque bounded tags and timings. Storing
raw prompts, input text, token streams or
full model output is off by default. If future tracing is added, it should use
explicit sampling, caller-provided redaction, bounded retention and separate
access control.

## Benchmark expectations

Diagnostics benchmarks compare disabled recording, enabled recording and
concurrent snapshotting. Report operations per second and p50/p95/p99, not
only a mean. Scenarios should include one hot schema, many schemas, mixed
outcomes and a reader taking snapshots under write load. A CI smoke test checks
correct execution; stable performance thresholds belong on controlled
hardware rather than shared hosted runners.

