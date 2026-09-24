# Bounded execution and cross-request batching

`jevt::batching_backend` is an optional decorator for any `jevt::backend`. It
combines queued requests into backend batches and limits the number of concurrent
backend calls. Existing synchronous engine/backend APIs continue to work.

```cpp
#include <jevt/batching.hpp>
#include <jevt/laya.hpp>

auto native = std::make_shared<jevt::laya_backend>(
    jevt::laya_options{.model_directory = "models/laya"});
auto pooled = std::make_shared<jevt::batching_backend>(native,
    jevt::batching_options{
        .worker_count = 1,
        .queue_capacity = 256,
        .max_batch_size = 16,
        .max_delay = std::chrono::microseconds{1000},
        .length_bucket_width = 256,
    });

// Request strings and option strings are copied before submit returns.
auto pending = pooled->submit(request);
auto response = pending.get();
// Or: pooled->predict(request), pooled->predict_batch(requests).
pooled->shutdown();
```

`queue_capacity` counts waiting **rows**, not API calls or bytes. A four-field
`predict_batch` consumes four entries. In addition to the queue, at most
`worker_count * max_batch_size` rows execute. Very large individual inputs still
need an application-level size limit. Batch admission is atomic: if all rows do
not fit, the call returns `error_code::overloaded` without admitting any rows.
`submit` reports rejection through an already-ready future. Applications should
apply an explicit retry/backoff, shedding, or upstream admission policy.

One worker is the default. Configuring multiple workers requires the wrapped
backend to support concurrent `predict_batch` calls. All decorator entry points
are thread safe while the object remains alive. Synchronous recursive calls and
`shutdown` from its own worker are rejected to avoid self-waits. Do not destroy
the decorator from a backend callback or while other threads still access it.

The batching delay is measured from enqueue time. A full compatible batch runs
immediately; otherwise workers wait until the oldest queued row's deadline.
Backend contention can cause queue residence to exceed `max_delay`: this setting
limits the intentional accumulation delay, not total latency or a service-level
deadline. Zero disables intentional waiting.

Length bucketing is optional and disabled by default. A nonzero width groups
rows by total UTF-8 text bytes divided by that width, including decision ID,
question, input, and options. Byte length is a cheap token-length proxy, not an
exact token count. The oldest queued row selects the next bucket, preventing
continuous short requests from starving a longer one. Results are restored to
the original caller order. Bucketing can reduce padding but also produce smaller
batches; measure both effects on your workload.

`shutdown()` closes admission, wakes workers, drains all accepted work, and joins
them. Concurrent shutdown calls are supported. Destruction does the same.
Requests after admission closes return `error_code::shutting_down`. Shutdown
does not cancel a running backend call and requires that call to eventually
return. Backend errors are delivered to every row in the failed microbatch;
thrown exceptions become `backend_failure`, and a response-count mismatch becomes
`invalid_backend_output`.

`stats()` returns a synchronized snapshot of `queued`, `in_flight`, `accepted`,
`rejected`, `completed`, and `batches`. Counts are rows except `batches`, which
counts wrapped backend calls. `completed` includes unsuccessful execution.
These counters do not measure queue residence or distinguish backend execution
time from tokenization and batching overhead.

## Measure throughput and latency together

Build the Laya backend and benchmarks using the project's CMake options. The
`jevt_laya_load` executable accepts:

```text
jevt_laya_load MODEL cpu|cuda ORT_THREADS CLIENTS FIELDS ITERATIONS MAX_BATCH DELAY_US IO_BINDING
```

`FIELDS` is 1 or 4. `ITERATIONS` is the number of calls **per client**.
`MAX_BATCH=0` calls Laya directly; a positive value enables the decorator with
one worker and a 256-row queue. `IO_BINDING` is 0 or 1. For example:

```sh
jevt_laya_load models/laya cpu 2 4 4 100 16 1000 0
jevt_laya_load models/laya cuda 2 8 4 100 32 1000 1
```

The fixture matches `laya_latency.cpp` and `laya_latency.py`. The benchmark warms
the underlying backend with six calls, then starts all client threads together.
Each client sends another call immediately after the previous one finishes.
This is a closed-loop load test; it does not model independent arrivals or expose
all overload behavior of an open-loop production workload.

JSON output reports measured wall time, attempted/successful calls, errors,
overload rejections, exceptions, successful calls/second, fields/second, and
successful-call end-to-end p50/p95/p99. End-to-end latency starts before the API
call and includes request copying, queue residence, backend work, and waiting for
all fields. It excludes cold loading and warmup. Empty success samples produce
`null` percentiles. Error rate uses all attempted calls as the denominator;
percentiles alone exclude errors and must always be read with that rate.
Decorator counters additionally report accepted/rejected/completed rows, backend
batches, and mean rows per batch. Configured maximum batch size is not a measured
batch-size maximum.

Start with a direct-backend baseline, then sweep client concurrency, ORT thread
count, maximum batch size, and delay separately on the same model/provider and
input distribution. Compare p95/p99 and rejection rate as well as throughput.
Measure a realistic mixture of lengths before enabling buckets. Record hardware,
model hash, ONNX Runtime/provider versions, and all settings with results. CPU
thread counts and worker counts can oversubscribe cores; GPU batches can exhaust
device memory. Keep the smallest settings that meet the measured target.

Throughput is successful completed calls divided by measured wall time. It is
not `1 / p50`: concurrent calls overlap, while median latency describes an
individual call. Four-field calls are also distinct from four independent API
calls, so keep calls/second and fields/second labeled separately.
