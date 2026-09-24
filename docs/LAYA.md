# Native Laya inference

`jevt::laya` connects the typed decision API to a local Laya model through ONNX
Runtime and the Hugging Face tokenizer implementation in `tokenizers-cpp`.
Python is used only by the optional download and reference-validation tools;
inference runs inside the C++ process.

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
