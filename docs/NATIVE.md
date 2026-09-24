# Native Laya backend

`jevt::laya_native` runs Laya through the native
[`laya.cpp`](https://github.com/lkarlslund/laya.cpp) runtime and ggml. It is an
optional alternative to the [ONNX Runtime backend](LAYA.md), using the same
JevT++ schemas, shared context, typed results and backend interface. It loads
the original checkpoint's safetensors files instead of an ONNX export.

The wrapper supports CPU and CUDA. CPU with strict FP32 is the default.
CUDA optimized FP32 is an explicit selection; it uses upstream compensated
Tensor Core projections and FP32 fused attention where the kernel supports
the sequence shape. This is not a generic FP16 conversion of the model.

## Build and download

Requirements are C++20, CMake 3.24+, Git, ICU development libraries and,
for CUDA, a compatible CUDA toolkit with `nvcc`, cuBLAS and cuBLASLt.
The build obtains nlohmann-json if it is not already available. On Debian or
Ubuntu, ICU development files are provided by `libicu-dev`.
The current Linux x86 build uses ggml's AVX2/FMA/F16C/BMI2 baseline, not generic
x86-64. Validate CPU capabilities before distributing binaries. Linux CPU and
CUDA are the initial validation targets; Windows and macOS native packages are
not yet release-qualified.

```sh
python3 scripts/fetch_laya_native.py models/laya-native-multilingual
cmake -S . -B build-native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DJEVT_ENABLE_LAYA_NATIVE=ON \
  -DJEVT_LAYA_NATIVE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DJEVT_LAYA_NATIVE_MODEL_DIR="$PWD/models/laya-native-multilingual"
cmake --build build-native
ctest --test-dir build-native --output-on-failure
```

Architecture `89` targets an RTX 4090; choose the architecture matching your
deployment GPU. For CPU-only builds, set `JEVT_LAYA_NATIVE_CUDA=OFF` and omit
`CMAKE_CUDA_ARCHITECTURES`. ONNX Runtime and the Rust tokenizer dependency
are not required by this backend.

By default CMake fetches the pinned native runtime and its ggml dependency.
To use an already fetched checkout, supply
`-DJEVT_LAYA_NATIVE_SOURCE_DIR=/path/to/laya.cpp`. Keep that checkout and its
submodules at the documented revision for reproducible comparisons. The
native runtime and ggml are embedded in the shared JevT++ adapter. ICU,
compiler/OpenMP runtime libraries and, for CUDA builds, CUDA runtime libraries
still need to be available on the deployment host.

The native model directory contains `encoder/`, `tokenizer/`, decision-head
weights and `rl_agent_config.json`. It is not interchangeable with the
`model.onnx` directory produced by `scripts/fetch_laya.py`.

## Select the backend

Link `jevt::laya_native` and include `<jevt/laya_native.hpp>`:

```cpp
#include <jevt/laya_native.hpp>
#include <jevt/system_one.hpp>

auto backend = std::make_shared<jevt::laya_native_backend>(
    jevt::laya_native_options{
        .model_directory = "models/laya-native-multilingual",
        .provider = jevt::laya_native_provider::cuda,
        .precision = jevt::laya_native_precision::optimized_fp32,
    });
auto brain = jevt::bind_system_one(ticket_model, backend);
auto decision = brain.evaluate(jevt::json_state(serialized_ticket_context));
```

`ticket_model` is a regular [System One definition](SYSTEM_ONE.md). The native
backend accepts the same Choice, Noul and Score requests, including probability
fields represented by Noul. It calibrates raw logits into probabilities and
returns them through the existing backend response; it does not round answers
through the upstream JSON CLI. JevT++ then applies the declared confidence and
abstention policies. Backend failures remain technical errors.

The native provider currently uses CUDA device 0. `device_name()` identifies
the selected device. Selecting CUDA requires a CUDA-enabled build and a usable
device; there is no automatic retry on CPU after a CUDA failure. Call
`warmup(requests, iterations)` with representative request spans before
measuring steady-state latency. Warmup runs actual inference and propagates
errors.

Construct native CUDA backends before initializing other CUDA consumers in
the process: the pinned runtime configures process-wide CUDA math policy
during initialization. If a service embeds multiple GPU libraries, validate
their initialization order together.

`context_token_limit` optionally caps state tokens without changing question
metadata. `schema_cache_entries` and `schema_cache_bytes` bound retained
question/option heads; zero entries disable that cache. Shared state is
tokenized once within a batch and is not retained as a result cache. Apply
request byte limits before calling the backend; a token budget does not bound
the size of the incoming serialized object.

## Precision and graph reuse

| Native precision | Selection and scope |
|---|---|
| `strict_fp32` | Default numerical baseline; CPU or CUDA |
| `optimized_fp32` | CUDA compensated projection products with FP32 accumulation and supported FP32 fused attention |
| `bf16` | Explicit CUDA experiment; requires the upstream validated compiler/library profile |

At the pinned upstream revision, the specialized FP32 attention kernel covers
sequences up to 128 tokens; longer sequences use the upstream cuBLAS attention
path. This is a kernel-selection boundary, not a 128-token context limit.
Checkpoint budgets still determine accepted question and context lengths.

Weights remain resident. Upstream retains encoder/action graphs for their
current shapes and can reuse allocations and ggml CUDA Graphs for repeated
shapes. Batch size, padded sequence length, option count and applicable
low-precision padding changes can trigger rebuilding. Warming one shape
does not eliminate cold work when subsequent requests use other shapes.

The BF16 path has stricter build requirements than optimized FP32. An
unsupported compiler/library profile is rejected; requesting BF16 does not
silently select another precision. Treat precision as part of the deployment
identity and validate categories, probability differences and domain quality
against a matching baseline before selecting it.

## Concurrency and batching

One native backend owns mutable graph state, so it serializes its inference
calls internally. Sharing the backend between threads is supported, but it
does not create simultaneous native executions. A `predict_batch()` call
evaluates its question rows together.

Wrap the backend in [bounded batching](BATCHING.md) to combine independent
requests and apply admission limits. A single batching worker is a useful
starting point for one native instance. More workers aimed at that same
instance still contend for its execution lock. Separate backend instances
have separate model/graph allocations and require explicit memory and
contention measurements.

Calls remain synchronous. Application deadlines do not interrupt active
native inference. Drain accepted work before destroying the backend; do not
move or destroy it concurrently with callers.

## Measurement and provenance

Compare native and ONNX backends using identical model weights, serialized
state, question metadata, field counts, precision, hardware and timing
boundaries. Report probability and category agreement beside latency and
throughput. Upstream measurements on other hardware are reference results,
not measurements of this adapter. No best-in-class claim follows from adding
the backend. See [the measurement guide](PERFORMANCE.md).

The local Linux/WSL2 validation includes native-only CPU build/install consumers,
mixed-request and concurrent-call tests, and native/ONNX probability checks.
The RTX 4090 optimized-FP32 comparison passes all eight one/four-field and
short/long-context cases. The guide includes full samples and the observed
latencies; CUDA optimized mode remains an explicit deployment choice.

The native runtime is pinned to
[`lkarlslund/laya.cpp` at `e1c6e7832189d36903e90c6fe3b8b1fece7f6f17`](https://github.com/lkarlslund/laya.cpp/tree/e1c6e7832189d36903e90c6fe3b8b1fece7f6f17),
which is MIT licensed; preserve its copyright and license when redistributing.
Its dependencies retain their own licenses. The multilingual checkpoint comes
from Convai Innovations'
[`laya` revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`](https://huggingface.co/convaiinnovations/laya/tree/1c5edc17a7acd8701df6fc341c0d179f1c62c982/multilingual)
under Apache-2.0. The model revision matches the source of the pinned ONNX
export; the on-disk formats differ.

Upstream also implements Vulkan, Core ML and HTTP serving. Those features are
not automatically exposed by `jevt::laya_native`; this integration exposes
the in-process CPU/CUDA decision backend. Native ggml graph replay is separate
from ONNX Runtime's CUDA Graph feature and its provider-placement constraints.
