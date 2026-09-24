# Laya through ONNX Runtime

`jevt::laya` connects the typed decision API to a local Laya model through ONNX
Runtime and the Hugging Face tokenizer implementation in `tokenizers-cpp`.
Python is used only by the optional download and reference-validation tools;
inference runs inside the C++ process.

For model-specific ggml kernels and native CUDA graph reuse, see the optional
[laya.cpp backend](NATIVE.md). Both implement the same typed decision API;
their model artifacts and execution controls differ.

## Build and run

Install a C++20 toolchain, CMake 3.24+, Git, Rust/Cargo and an
[ONNX Runtime C/C++ SDK](https://onnxruntime.ai/docs/get-started/with-cpp.html)
for your platform. `ONNXRUNTIME_ROOT` must point to the extracted SDK containing
`include/` and `lib/`. The tokenizer build fetches a pinned source revision and
uses Cargo, so its first build requires network access.

From the repository root:

```sh
python3 scripts/fetch_laya.py models/laya-multilingual
cmake -S . -B build-laya -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DJEVT_ENABLE_LAYA=ON \
  -DJEVT_BUILD_TESTS=ON \
  -DJEVT_BUILD_EXAMPLES=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-sdk
cmake --build build-laya
ctest --test-dir build-laya --output-on-failure
./build-laya/examples/jevt_laya_demo models/laya-multilingual
```

With the HTTP feature enabled, set `JEVT_DEMO_SERVE=1` to keep the process open
after inference and inspect the loopback diagnostics dashboard. The System One
batch is recorded as one call under its stable decision-model ID.

To compile the real-model parity test and the small labeled quality harness,
also set `-DJEVT_LAYA_MODEL_DIR=$PWD/models/laya-multilingual` and keep
`JEVT_BUILD_TESTS`/`JEVT_BUILD_BENCHMARKS` enabled. Then run:

```sh
ctest --test-dir build-laya -L model-integration --output-on-failure
./build-laya/benchmarks/jevt_laya_quality models/laya-multilingual
```

The integration test validates mixed-cardinality tensors, one batch dispatch,
normalization, and batch-versus-single probability parity. The quality harness
reports category accuracy/NLL, urgent accuracy/Brier score, and p50/p95/p99 on
eight transparent example tickets. It is a smoke benchmark, not evidence of
production quality; replace or extend its labeled cases for your domain.

For an independent tokenizer/tensor check, install Python `tokenizers`,
`onnxruntime`, and `numpy`, then run
`python3 tests/laya_reference.py models/laya-multilingual`. It reconstructs the
upstream sequence and tensor contract without calling the C++ adapter. Compare
its four distributions with the `probabilities:` line printed by
`jevt_laya_integration_tests`; small floating-point differences between ONNX
Runtime builds are expected.

For negative contract tests, install Python `onnx` and generate tiny synthetic
graphs, then supply their directory as the integration executable's second
argument (the first remains the real model bundle):

```sh
python3 tests/laya_contract_fixtures.py /tmp/laya-contract-fixtures
./build-laya/tests/jevt_laya_integration_tests \
  models/laya-multilingual /tmp/laya-contract-fixtures
```

These 20 fixtures exercise wrong input/output dtypes and ranks, exact batch
dimensions, non-finite active logits, and ignored masked padding. Construction
checks the model's tensor contract; each inference validates the exact logits
shape before reading scores. The Linux Laya CI workflow runs both the real
model and these negative cases. Core-only CI is not a model-integration test.

The downloader verifies SHA-256 hashes, reuses matching files and pins the
model revision. Model weights are downloaded separately from the library.
The demo sends a support-ticket state to Choice, Noul, Score and probability
fields and prints the resulting typed decisions and probabilities. See its
[source](../examples/laya_routing_demo.cpp) for command-line options.

ONNX Runtime's shared library must be discoverable by the operating system.
If a Linux runtime reports that `libonnxruntime.so` cannot be found, run with:

```sh
LD_LIBRARY_PATH=/path/to/onnxruntime-sdk/lib:${LD_LIBRARY_PATH:-} \
  ./build-laya/examples/jevt_laya_demo models/laya-multilingual
```

On Windows, make the matching ONNX Runtime DLL available beside the executable
or on `PATH`. Multi-configuration generators place executables under the
selected configuration directory, such as `examples/Release/`.

Consumers link the additional target and include its explicit header:

```cmake
find_package(jevtpp CONFIG REQUIRED)
target_link_libraries(my_service PRIVATE jevt::laya)
```

```cpp
#include <jevt/laya.hpp>
#include <jevt/system_one.hpp>

auto backend = jevt::models::laya_multilingual("models/laya-multilingual", 4);
auto brain = jevt::bind_system_one(ticket_model, backend);
auto answers = brain.evaluate(jevt::json_state(serialized_application_context));
```

The second factory argument controls ONNX Runtime's intra-op thread count;
zero uses the runtime default. For advanced settings, construct
`std::make_shared<jevt::laya_backend>(jevt::laya_options{...})`. Options include
explicit model/config/tokenizer paths, intra- and inter-op thread counts,
graph optimization and a diagnostic model ID. Create the backend once and
reuse it across evaluations. Model loading is synchronous and throws if the
bundle cannot be loaded; inference failures use `jevt::result`.

## Shared context and batching

A System One model defines the decision description, field instructions and
enum criteria. Each evaluation receives one owned state. The state can be a
string, serialized JSON object/array, or an application record converted by
a serializer. The [System One guide](SYSTEM_ONE.md) shows each form and typed
projection into a C++ struct.

All questions see the same serialized context. Laya builds one sequence per
question, pads the sequences into a tensor batch and calls ONNX Runtime once.
Sharing a state at the API level does not mean the encoder computes that state
only once: each question's sequence contains the state, so adding questions
increases work and memory. Custom backends may use the default sequential
`predict_batch()` implementation.

The adapter uses `input_ids`, `attention_mask`, `marker_pos`, `marker_mask` and
`qtype`, requests the graph's `logits` output, and applies the bundle's
temperature calibration. Question types are Choice, Score and Noul;
`probability<...>` uses Noul and returns P(true). The model supplies probability
distributions; JevT++ applies the declared abstention policy and maps answers
to application types. Schema constraints guarantee the shape of a valid
answer, not that a model's judgment is correct.

## CUDA, warmup and bounded memory

CPU is the default. To opt into CUDA, build/link against an ONNX Runtime **GPU**
SDK and install its matching CUDA/cuDNN runtime libraries. The tested local
runtime was ORT 1.29.0 on RTX 4090. A CPU SDK still compiles the adapter, but
requesting CUDA fails explicitly when unavailable. Individual shape operations
may still be assigned to CPU by ORT. See the official
[CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

```cpp
jevt::laya_options options;
options.model_directory = "models/laya-multilingual";
options.provider = jevt::laya_provider::cuda;
options.intra_op_threads = 2;
options.inter_op_threads = 1;
options.use_tf32 = false;
auto native = std::make_shared<jevt::laya_backend>(options);
auto brain = jevt::bind_system_one(ticket, native); // your decision model
auto batch = brain.request(jevt::text_state("Representative warmup input"));
auto views = batch.views();
auto warmed = native->warmup(views.requests(), 3);
if (!warmed) throw std::runtime_error(warmed.error_value().message);
auto routing_only = brain.select<"category">(); // computes one field, typed result
auto answer = routing_only.evaluate("A new customer message");
```

Keep `native` and bindings alive across requests. `parallel_execution` explicitly
enables ORT's inter-op execution mode; merely setting `inter_op_threads` does not.
`allow_spinning=false` is available for shared hosts; benchmark the latency tradeoff.
`context_token_limit=0` retains the bundle budget; a positive value limits state
tokens without trimming the schema. Projection or truncation can change quality.

Defaults cache up to 256 schema heads and 4 MiB of retained payload capacity
(container/allocator overhead excluded), evicting FIFO. Set `schema_cache_entries=0`
to disable. Cache identity includes exact question/criteria and kind, scoped to
one model instance; state is never persisted in this cache. Identical state is
tokenized once per batch, though each field still has its own transformer row.

`reusable_buffers=2` retains at most two isolated host-buffer sets, each no larger
than `reusable_buffer_bytes` (8 MiB by default). Input-token storage is cleared
before reuse. Larger working buffers are released after their call. This is not
a total memory or request-size limit. `statistics()` exposes cache hits/misses,
retained payload, run count and buffer reuse. Input bytes still need a service limit.

`use_io_binding=true` selects ORT I/O binding with host inputs and a host logits
output. Device transfers still occur; this is not zero-copy, persistent device
allocation or CUDA Graph replay. It is off by default because it did not improve
our short latency probe. [Cross-request batching](BATCHING.md) and
[precision experiments](PRECISION.md) are separate opt-in paths.

Run `jevt_laya_performance_tests MODEL cuda` to compare against CPU at a `1e-4`
probability tolerance, including Unicode, empty/long input, mixed kinds, cache
eviction, warmup, host-buffer reuse and concurrent calls. The ordinary model CI
runs its CPU version; hosted CI does not validate CUDA hardware.

## Model provenance and resources

The download script pins the `multilingual/` export in
[`codenamev/laya-onnx`](https://huggingface.co/codenamev/laya-onnx/tree/1bc2622b5a4e4ceb46aadf709d7a360eb7d3d1f4/multilingual)
at revision `1bc2622b5a4e4ceb46aadf709d7a360eb7d3d1f4`. Its model card identifies
the source as Convai Innovations' multilingual checkpoint in
[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya/tree/1c5edc17a7acd8701df6fc341c0d179f1c62c982/multilingual),
revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`. `onnx_config.json` is kept
with the downloaded bundle as export provenance.

The ONNX model is about 647 MB and its tokenizer about 34 MB. Plan for several
GB of RAM while loading and evaluating the model; actual peak memory depends
on context length, question count, ONNX Runtime settings and concurrency.
The bundled configuration limits each sequence to 1,024 tokens and allocates
256 tokens to its question/option header. The remaining space holds state.
Long state and criterion text can be truncated, so keep routing context
focused and test the largest inputs used by your application. The adapter's
`max_context_tokens()` and `max_question_tokens()` expose the configured
total-sequence and header limits; the former is not a promise of 1,024 tokens
of state in addition to the question.

The library is [MIT licensed](../LICENSE). The downloaded Laya weights and ONNX
export are [Apache-2.0 licensed](https://huggingface.co/codenamev/laya-onnx/blob/1bc2622b5a4e4ceb46aadf709d7a360eb7d3d1f4/README.md)
and carry their own redistribution obligations. The sequence implementation
was developed against the [Receptron Laya reference](https://github.com/receptron/laya).
ONNX Runtime and tokenizer dependencies retain their respective licenses.

Measure warm inference separately from model loading and report hardware,
thread count, input length and number of fields with latency percentiles.
Backend parity checks and domain-quality checks answer different questions:
matching reference tensors/probabilities validates the adapter, while labeled
application cases are needed to choose confidence thresholds and assess
routing accuracy. No model-quality claim follows from the deterministic
acceptance tests or the core keyword-backend benchmark.
