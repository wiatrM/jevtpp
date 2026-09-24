# JevT++ architecture

## Goals

JevT++ lets an application express model-backed decisions as strongly typed
C++ schemas. Business code should know `init`, `bind`, and `choose/evaluate`;
advanced users can inject contexts, backends and a diagnostics registry. The
core stays usable without networking or a production ML
runtime.

The architecture separates five concerns:

```text
compile-time schema
        |
        v
binding/configuration -> prepared decision -> backend request
        |                     |                    |
        |                     v                    v
        +--------------> typed result       model/runtime
                              |
                              v
                    diagnostics recorder
                              |
                       snapshot API
                         /         \
                  application   HTTP dashboard
```

## 1. Compile-time schema

A choice schema contains:

- a stable identifier such as `support.routing`, encoded in its type;
- the application's enum type;
- a closed list of allowed enumerators and human-readable descriptions;
- optional prompt metadata that is configuration, not a new result type.

`option<EnumValue>` preserves sparse enum values. Concepts and `consteval`
validation reject duplicate values, invalid option types and empty schemas.
A predicate similarly owns a stable identifier but returns the explicit
states true, false or abstain.

The backend never defines business types. Its output is validated and mapped
through the prepared schema before reaching the caller. Consequently, a
result from one schema cannot be passed to a matcher for another schema.

## 2. Context, initialization and binding

`init()` constructs a shared application context. The context owns the backend,
default abstention threshold and diagnostics registry. A backend adapter can
in turn own expensive model resources. The returned RAII handle must remain
alive while creating bindings through the default context. Existing bindings
own shared backend/diagnostics references and can outlive that handle.

`bind(schema, options)` resolves the backend and abstention threshold and
returns a typed decision handle. It rejects a missing backend before the hot
path.

Configuration precedence is:

```text
binding override > explicit/default context
```

Changing context state does not mutate an existing binding. A binding holds a
shared backend reference and its resolved threshold.

Explicit contexts are preferred in libraries and tests:

```cpp
jevt::context context{{.inference_backend = backend}};
auto routing = context.bind(support);
```

The global `jevt::bind()` form is a convenience over the same mechanism.
Initialization and binding failures throw; inference returns `jevt::result`.

## 3. Backend abstraction

The backend boundary carries a backend-neutral `inference_request` and returns
an `inference_response` with scores and a model ID. It does not
contain `Team`, `option<>`, or other application-specific template types.

Required properties:

- safe concurrent inference when the application uses concurrent callers;
- structured error categories rather than parsing messages;
- no ownership of user input beyond the call unless the user opts in;
- a score vector matching the requested option count.

Adapters can include:

- a deterministic fake for acceptance tests;
- an application callback for existing inference services;
- a remote backend owned by the host application;
- the optional ONNX Runtime adapter described below.

Backend errors are separate from a model abstention. Mapping an unknown label,
NaN score or malformed tensor is a technical error; a valid score below the
policy threshold is an abstention.

## 4. Typed decisions and results

`choose(input)` returns an expected-like result. Success contains either an
enum value plus evidence metadata, or abstention. Failure contains a stable
error code and safe diagnostic message. `evaluate(input)` follows the same
shape for true/false/abstain.

`match` is the exhaustive alternative to a `switch`: every declared option
and abstention must be handled exactly once. Consumers may still use a normal
`switch` when that better fits their code.

Bound decisions record diagnostics after normalization, so metrics reflect the
same result the caller receives. The registry is also available from
`app::diagnostics()` for direct snapshots and export.

## 5. Diagnostics and dashboard

Each bound decision records bounded, low-cardinality measurements in a
shared registry. The supported public operation is a consistent snapshot,
which keeps exporters independent from inference. Percentiles come from a
bounded histogram; request threads never sort a global
list and raw latency samples do not grow without limit.

Counters and histogram buckets cover every recorded call. Optional
`recent_sample_every` sampling affects only retained recent-call metadata;
`recent_capacity` bounds retention, and either option set to zero disables
recent traces. The default sampling interval is one. Snapshots and resets
remain synchronized with recording.

HTTP is an optional adapter over this API, not the owner of metrics. The
dashboard must be separately enabled, should default to loopback, and must
never make inference availability depend on the web server. See
[OBSERVABILITY.md](OBSERVABILITY.md).

## Optional ONNX adapter

The optional Laya adapter validates model metadata and tensor contracts during
construction and validates output shapes and values on each inference call.
It preserves tokenizer parity with the reference model. Applications can
always supply their own backend adapter, so ONNX is never required to install
or use the core.

CPU execution is the default. `laya_provider::cuda` explicitly selects the
CUDA provider, checks its availability and propagates initialization failure.
TF32 is disabled by default. Optional I/O binding uses reusable host tensors;
ONNX Runtime still performs device transfers. It does not provide zero-copy
or CUDA-graph execution.

Prepared question/option tokens use a bounded cache local to the model
instance. Input state tokens are reused only within the current batch.
Reusable tensor storage is leased exclusively to one call, and retained input
tokens are wiped before storage returns to the pool. Limits on cache entries,
retained payload capacity and pooled buffers are configurable. Warmup runs
representative caller-supplied requests through the same session.

## Optional native ggml adapter

`jevt::laya_native` implements the same backend boundary using pinned
`laya.cpp` and original safetensors weights. It uses the native tokenizer and
raw runtime logits, preserving duplicate options and unrounded calibrated
probabilities rather than passing through upstream JSON output. Its bounded
schema cache and batch-local context reuse do not retain answer results.

The runtime owns resident weights, reusable allocations and shape-dependent
compute graphs. One instance serializes calls to protect this mutable state.
CUDA optimized FP32 is explicit; CPU strict FP32 remains the default. Native
ggml graph replay is independent of ORT I/O binding and CUDA Graph capture.
See [native setup, precision and deployment constraints](NATIVE.md).

## Concurrency and lifetime

Bound decisions are cheap handles to shared backend state. Calls may run
concurrently when the supplied backend supports concurrent `predict()` calls.
Backend-specific session pooling belongs behind the backend boundary.
Snapshotting diagnostics is thread-safe and bounded in time.

`choose_async()` uses `std::async(std::launch::async)` and an owned input
string. A future may block on destruction. The optional `batching_backend`
wrapper supplies bounded workers, queue admission, microbatching and
`submit()` returning a future. Submission owns request text; synchronous
`predict()` and `predict_batch()` wait for queued completion. Batch admission
is atomic, overload returns a structured error, and optional length bucketing
preserves caller result order. Multiple workers require a thread-safe wrapped
backend.

The queue limit counts waiting requests, with up to
`worker_count * max_batch_size` additional requests executing. Shutdown stops
admission and drains accepted work before joining workers. It cannot interrupt
a backend call. Neither async API provides coroutines, Asio integration or
running-inference cancellation. Keep blocking waits off the I/O loop.
See the [concurrency guide](https://wiatrm.github.io/jevtpp/concurrency/).

Stop the optional HTTP server before releasing its diagnostics registry.
Destructors do not throw.

## Packaging boundary

The core exports `jevt::jevt` and requires C++20. Optional components add
dependencies privately where possible. The installed CMake config resolves
only dependencies of features compiled into that package. Conan, vcpkg and a
plain CMake install all expose the same target and headers.

