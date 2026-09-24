# Opt-in precision experiments

The downloaded Laya export stores weights in FP16 and casts them to FP32 for
computation. Smaller weight files alone do not establish FP16 execution.
`scripts/optimize_laya.py` creates separate candidate bundles; it never changes
the original bundle or the application's default backend.

FP16 conversion preserves public input/output types with `keep_io_types=True`.
In particular, Laya's integer/bool inputs stay unchanged and `logits` remains
FP32. Dynamic INT8 targets constant-weight MatMul/Gemm operations, using signed
INT8 weights and runtime activation quantization. It is a separate CPU
experiment, not an assertion that CUDA will execute INT8 operators faster.
Both transformations can change model decisions.

## Generate candidate bundles

Use an isolated Python environment with `onnx`, `onnxruntime-gpu`, `numpy`
and `tokenizers`. Conversion uses ONNX Runtime's transformer FP16 converter.
A CPU-only environment can use
`onnxruntime` for INT8 experiments. Do not install both ONNX Runtime packages
in one environment. CUDA experiments also require compatible CUDA/cuDNN
libraries; the benchmark calls `onnxruntime.preload_dlls()`.

```sh
python scripts/optimize_laya.py models/laya-multilingual models/laya-fp16 \
  --precision fp16 --disable-shape-infer
python scripts/optimize_laya.py models/laya-multilingual models/laya-int8 \
  --precision int8 --per-channel
```

Output directories must not exist and must be outside the source bundle.
Each receives its own tokenizer/configuration and a `precision-manifest.json`
with source/output hashes, tool versions, transformation settings and graph
statistics. A successful conversion is marked `converted_unvalidated`: this
does not mean it loads on a particular execution provider or meets a quality
gate. Failed runs retain their manifest and any partial output for diagnosis;
retry with a new directory.

For graphs whose existing types conflict with converter shape inference,
`--disable-shape-infer` uses the graph's existing type information. Validate
the resulting graph and its answers before using it. `--block-op Softmax` or
`--block-node NODE_NAME` can keep selected computation in FP32; these are
explicit experiment settings recorded in the manifest. The converter's
standard unsupported-operation block list is retained.

INT8 conversion first folds only constant initializer-to-FP32 Casts so that
the quantizer sees the actual weight matrices. It fails if no MatMulInteger
nodes are produced. It recomputes intermediate shape annotations after folding;
public input/output declarations are preserved. Because unquantized embeddings
remain FP32, an INT8 bundle can be larger than the original FP16-storage bundle.
Conversion can temporarily require several GB of RAM.

## Measure parity and labeled smoke quality

Run each precision/provider combination serially on an otherwise idle host.
The reference defaults to the original bundle's `model.onnx`. CUDA uses
`use_tf32=0` to avoid confounding precision changes with TF32, and refuses a
session that silently falls back to CPU. Individual unsupported operators
may still execute on CPU; inspect a provider profile before claiming all-GPU
execution.

```sh
python benchmarks/laya_precision.py models/laya-multilingual \
  --candidate models/laya-fp16/model.onnx --provider cuda \
  --iterations 20 --warmup 5 \
  --max-probability-delta 0.01 --max-argmax-disagreements 0 \
  --max-accuracy-drop 0 --output fp16-report.json

python benchmarks/laya_precision.py models/laya-multilingual \
  --candidate models/laya-int8/model.onnx --provider cpu \
  --iterations 20 --warmup 5 \
  --max-probability-delta 0.03 --max-argmax-disagreements 0 \
  --max-accuracy-drop 0 --output int8-report.json
```

These example thresholds are experiment settings, not recommended production
SLOs. Supply your own tolerances. A violated gate, invalid output, missing
provider or candidate loading failure produces a nonzero exit status and is
recorded in JSON. Other candidates are still tested after one candidate
fails. Without gates, successful candidates are marked `measured_no_gates`.
The tool never selects or deploys a candidate automatically.

Reports include maximum/mean probability differences, argmax disagreements,
the actual probabilities, category/urgency accuracy, category negative log
likelihood, urgency Brier score, model hash, provider settings, tensor shapes
and warm p50/p95/p99 latency with raw samples. Timing uses prepared tensors
and synchronous `session.run` output copies; it excludes tokenizer execution
and is not a C++ end-to-end measurement. The suite mixes short two-field
tickets with four-field English/Polish contexts. Model/session loading and
the first suite are reported separately from warm latency.

The default labeled set is the same eight hand-written cases as
`benchmarks/laya_quality.cpp`. It is a smoke/regression check, not a business
benchmark or evidence of production accuracy. Extra four-field examples
test parity but have no accuracy labels. To use a reviewed dataset, pass
`--labels tickets.json` with this format:

```json
[
  {"state":"Please refund the duplicate payment.", "category":0, "urgent":false},
  {"state":"All staff are locked out of production.", "category":1, "urgent":true}
]
```

Category indices are billing=0, technical=1, sales=2, spam=3. `state` is the
exact text sent to the model; serialize structured context inside that string
when needed. The custom set replaces all eight smoke labels and its hash is
recorded. Optional `--min-category-accuracy` and `--min-urgent-accuracy` gates
can require absolute accuracy in addition to bounded regression from FP32.

Tool regression tests run without model weights:

```sh
python tests/precision_tools_test.py
```

## Initial pinned-model experiment

On 2026-09-24, the FP16 candidate was evaluated on an RTX 4090 with ONNX Runtime
GPU 1.29.0, ONNX 1.23.0, two CPU threads, TF32 disabled, two warmup rounds per
case and five measured rounds. The source graph SHA-256 was
`da6a0f87380597f679b12ce539e0a187e923fcffcf8dfc2f035728388dceacb0`;
the FP16 candidate graph SHA-256 was
`a5bb78127212e5eaf633e44ed64a962eb19385c93fc397efc01a6dc59bef76da`.

| Metric | FP32 compute | FP16 candidate |
|---|---:|---:|
| Mixed-suite warm p50 | 6.174 ms | 6.148 ms |
| Mixed-suite warm p95 | 10.398 ms | 12.162 ms |
| Category smoke accuracy | 6/8 | 6/8 |
| Urgency smoke accuracy | 5/8 | 5/8 |
| Maximum probability difference | Reference | 0.016309 |
| Argmax disagreements | Reference | 0/24 fields |

The FP16 experiment **failed** its `--max-probability-delta 0.01` gate. It
passed the zero-argmax-disagreement and zero-accuracy-drop gates. This small,
mixed-shape run does not show a useful latency improvement and does not
justify selecting FP16 by default. It is distinct from the C++ four-field
end-to-end latency benchmark.

Dynamic INT8 with per-channel weights was also evaluated on the same host's
CPU provider with two threads, one warmup round and three measured rounds.
Its graph hash was
`c9d7e6834d032c17795edb1208de0e4509aa7ca9ace6a4d73ab84164d4fb402d`
(the graph references a separate external-weight file). The transformation
folded 104 constant casts and produced 100 `MatMulInteger` nodes.

| Metric | FP32 compute, CPU | INT8 candidate, CPU |
|---|---:|---:|
| Mixed-suite warm p50 | 199.779 ms | 154.200 ms |
| Mixed-suite warm p95 | 1,038.738 ms | 749.840 ms |
| Category smoke accuracy | 6/8 | 1/8 |
| Urgency smoke accuracy | 5/8 | 5/8 |
| Maximum probability difference | Reference | 0.748507 |
| Argmax disagreements | Reference | 15/24 fields |

INT8 **failed all three parity/regression gates**: maximum probability
difference 0.03, zero argmax changes and zero accuracy drop. Its lower measured
latency came with unacceptable regression on even this small fixture, so the
candidate is not selected. The complete INT8 model files total approximately
915 MB versus 647 MB for the original export. GPU and CPU table latencies are
separate experiments and must not be used as a precision-only comparison.

The pinned export's converter shape inference required
`--disable-shape-infer`. The ONNX Runtime transformer converter also appends
boundary casts; the script restores dependency order before checking and
saving the graph. Conversion with onnxconverter-common 1.16 was tested during
development but produced incompatible intermediate types for this export;
the tool uses the ONNX Runtime converter instead.

The conversion follows ONNX Runtime's official
[FP16/mixed-precision guide](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html)
and [quantization guide](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
The FP16 converter implementation is maintained in
[ONNX Runtime's transformer tools](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/float16.py).
