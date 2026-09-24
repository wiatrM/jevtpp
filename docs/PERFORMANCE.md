# Performance and comparisons

JevT++ is a typed application library with a model adapter. Most inference work
is performed by ONNX Runtime. Python and C++ can call the same native engine,
so changing the application language does not imply a large model speedup.

## What the original 567 ms measured

The original support demo measured one evaluation of four fields on CPU,
including tokenization and inference, after loading the model. There was no
warmup or repeated latency distribution. Its request contained a formatted
JSON ticket, customer metadata, service metadata and four different questions.
It used ONNX Runtime's default thread selection.

The model processes a separate sequence for each question. Sharing JSON at
the API level does not reuse the encoded state across questions. A four-field
batch therefore does more work than a single-field request. Model loading,
first inference, warm latency, field count and input length need separate
measurements.

## Local CPU results

Measured 2026-09-24 on an Intel Core i7-10700K (8 cores / 16 logical CPUs),
Ubuntu 24.04 under WSL2. This was a shared development workstation, not an
isolated benchmark host. Background activity was not controlled. Each row
has 30 measured requests after the first call and five additional warmups;
each language ran in a separate process, sequentially. Values are milliseconds.

| Intra-op threads | Fields | C++ p50 | C++ p95 | Python p50 | Python p95 |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 449.3 | 503.8 | 495.5 | 543.2 |
| 1 | 4 | 1948.1 | 2192.1 | 1824.3 | 2091.0 |
| 2 | 1 | 263.0 | 287.1 | 257.3 | 305.2 |
| 2 | 4 | 998.5 | 1138.1 | 995.9 | 1060.2 |
| 4 | 1 | 177.8 | 194.7 | 175.4 | 200.2 |
| 4 | 4 | 732.5 | 839.9 | 763.9 | 881.8 |
| ORT default (`0`) | 1 | 138.9 | 150.6 | 138.2 | 145.2 |
| ORT default (`0`) | 4 | 565.0 | 605.1 | 584.3 | 630.6 |

The default-thread four-field warm result is close to the original 567 ms.
Warmup alone does not fix that latency. Both languages spend their time in
the same native inference engine; this run does not establish a meaningful
C++ language advantage. Thread selection had a much larger effect, and
restricting this host to two threads made the four-field request slower.

Both processes loaded ONNX Runtime **1.29.0** for this paired comparison.
The C++ adapter used the 1.29.1 SDK headers with the 1.29.0 shared library;
this was a benchmark environment override, not a dependency change.
C++ tokenization used tokenizers-cpp's pinned Rust tokenizers 0.21.2;
Python used tokenizers 0.22.1. Inputs were padded to `[1, 166]` or `[4, 166]`.
C++/Python output probabilities differed by at most `5.94e-8` across the
paired rows. This is numerical parity on this fixture, not a quality benchmark.
Model: `codenamev/laya-onnx`, multilingual export, revision
`1bc2622b5a4e4ceb46aadf709d7a360eb7d3d1f4`, SHA-256
`da6a0f87380597f679b12ce539e0a187e923fcffcf8dfc2f035728388dceacb0`.

[Raw CPU observations](../benchmarks/results/2026-09-24-cpu.jsonl) include
chronological samples, first-call latency, loading time and output probabilities.

A separate C++ check using the shipped **1.29.1** runtime and default threads
gave four-field p50 **577.1 ms**, p95 **725.5 ms** (30 samples). It is not mixed
into the paired table above. See the
[raw 1.29.1 check](../benchmarks/results/2026-09-24-cpu-ort-1.29.1.jsonl).

### Where the CPU time goes

A separate Python/ORT 1.29.0 probe with two threads and four fields measured:

| Boundary | p50 | p95 | Samples |
|---|---:|---:|---:|
| Tokenization and tensor setup | 0.93 ms | 1.13 ms | 200 |
| `session.run()` with prepared tensors | 1049.1 ms | 1252.5 ms | 30 |
| Full backend request | 1019.2 ms | 1166.4 ms | 30 |

These were separate sequential trials, not additive timing spans: the prepared
trial being slower than the full trial illustrates host/run variability.
Do not subtract these rows to estimate wrapper overhead. The useful conclusion
is the order-of-magnitude difference: model execution dominates, not JSON or
tokenization. [Raw breakdown](../benchmarks/results/2026-09-24-cpu-breakdown.json).

## Reproduce the local CPU comparison

Build with `JEVT_ENABLE_LAYA=ON` and `JEVT_BUILD_BENCHMARKS=ON`, following
[Laya setup](LAYA.md). The latency target does not require a model path at
configure time. Install `onnxruntime`, `tokenizers` and `numpy` in an isolated
Python environment. Match the Python ONNX Runtime version to the C++ SDK and
ensure both processes load that version's native library.

```sh
# MODEL_DIRECTORY, ORT library loading and Python environment are configured first.
./build-laya/benchmarks/jevt_laya_latency MODEL_DIRECTORY 2 4 50
python benchmarks/laya_latency.py MODEL_DIRECTORY 2 4 50
# Separate preparation / inference probe (200 preparation samples).
python benchmarks/laya_latency.py MODEL_DIRECTORY 2 4 30 --breakdown
```

Arguments are model directory, intra-op threads, question count (`1` or `4`),
and measured iterations. `0` threads delegates selection to ONNX Runtime.
Run processes sequentially, with no competing inference. Both harnesses use
the exact demo state and field descriptions, tokenize on every request and
return probabilities. They separate loading and the first inference, then
perform five additional warmups before measuring. Percentiles use nearest
rank. The C++ harness measures the backend entry point; typed result mapping
and application diagnostics are outside this timing boundary.

This Python harness is a paired ONNX baseline, not the upstream Laya PyTorch
package. It isolates the language boundary while keeping the model and
inference engine fixed. It cannot establish a speedup over all Python
implementations.

These are sequential, single-client latency tests, not concurrent load tests.
Thirty observations give only a coarse estimate of p95; production sizing
needs longer runs, representative input lengths, concurrent clients and a
controlled host. Do not convert reciprocal latency into a service throughput
claim without measuring queuing and concurrency.

## Local GPU headroom experiment

This is **Python + ONNX Runtime CUDA**, not the released C++ adapter.
The RTX 4090 probe used the same unchanged model and fixture on the same host,
with CPU tests finished before GPU timing started. Tokenization, host/device
copies and probability conversion are included. Model/session loading is
excluded. After the first call for each shape, it ran 10 warmups and 50 measured
requests; percentiles use nearest rank.

| Fields | Tensor shape | p50 | p95 |
|---|---|---:|---:|
| 1 | `[1, 166]` | 8.58 ms | 11.34 ms |
| 4 | `[4, 166]` | 14.14 ms | 15.65 ms |

CUDA was explicitly selected and verified, with TF32 disabled. The export has
mixed stored floating-point weights (104 FLOAT16 and 74 FLOAT initializers);
profiled MatMul/Gemm-family inputs were FP32. No quantization or model rewrite
was performed for this test. A separate profiling session recorded 1120 CUDA
node events and 133 CPU node events: not every graph operation ran on GPU.
The CPU events were slicing/concatenation/shape-related operations.

Maximum absolute probability difference from the CPU Python fixture was
`6.14e-6`, with the same winning options. This checks one fixture only, not
accuracy or calibration on real workloads. The result supports investigating
a CUDA provider for JevT++, but does not establish superiority over Jev's
different model or remove the need for production quality tests.

The environment used ORT GPU 1.29.0, tokenizers 0.23.2, NumPy 2.5.3, ONNX
1.23.0, CUDA runtime 12.9.79, cuBLAS 12.9.2.10, cuDNN 9.26.0.51 and NVRTC
12.9.86. Despite the tokenizer version difference, all five input arrays were
byte-identical between CPU and GPU environments for both fixture sizes
(verified with SHA-256 over dtype, shape and data).
Reproduce in a separate environment from the CPU-only wheel:

```sh
python -m venv .venv-gpu
. .venv-gpu/bin/activate
pip install 'onnxruntime-gpu==1.29.0' 'tokenizers==0.23.2' \
  'numpy==2.5.3' 'onnx==1.23.0' \
  'nvidia-cuda-runtime-cu12==12.9.79' 'nvidia-cublas-cu12==12.9.2.10' \
  'nvidia-cudnn-cu12==9.26.0.51' 'nvidia-cuda-nvrtc-cu12==12.9.86'
python benchmarks/laya_gpu_latency.py MODEL_DIRECTORY
```

An appropriate NVIDIA driver is required. The script refuses silent
CPU-provider fallback and profiles a separate session after timing.
[Raw GPU observations](../benchmarks/results/2026-09-24-gpu.jsonl).

## Published measurements from other projects

Retrieved 2026-09-24. These are attributed measurements with different
hardware, requests and timing boundaries; they are not a ranking against the
local CPU harness.

| Source | Reported latency / speed | Conditions |
|---|---|---|
| [TypeSafe launch report](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | 70–500 ms end-to-end | Vendor range for hosted Jev; evaluations generally run from US West Coast laptops near the service |
| [jev-measured](https://github.com/WallerChen/jev-measured#5-the-gateway-is-not-free-and-its-tail-is-worse-than-its-median) | Direct Jev API p50 313 ms, p90 423 ms | Author's alternating direct/gateway experiment; 12 rounds; includes network |
| [Laya upstream](https://github.com/NandhaKishorM/laya/blob/main/BENCHMARKS.md#speed-tesla-t4) | Multilingual: 32.8 ms for 1 question, 40.1 ms for 5 | Author's Tesla T4 measurements |
| [laya-mlx](https://github.com/mizorewww/laya-mlx/blob/main/BENCHMARKS.md) | Multilingual MLX FP32: p50 10.73 ms for 1 question, 21.94 ms for 5 | Apple M3 Max, 40-core GPU; short inputs; includes preprocessing and formatting |
| [laya.cpp](https://github.com/lkarlslund/laya.cpp#performance) | CUDA optimized FP32: 1.35–2.48× Python throughput | Author's paired tests on RTX PRO 6000 Blackwell; depends on model and batch size |

TypeSafe's range is not a p95 guarantee for a client in Europe. Its hosted
model is also different from Laya. A direct comparison needs the same inputs,
question count, quality target, location, connection reuse and repeated
measurements. We have not run a paid Jev API benchmark for this report.

## What to optimize

Keep the backend loaded and warm it using representative request shapes.
Sweep thread counts on the deployment machine; the highest thread count can
be slower. Keep context focused and measure both single-field and multi-field
requests. For GPU latency targets, a GPU execution provider or a specialized
backend is the next relevant layer. Release v0.2.0 selects CPU only; current
`main` additionally supports explicit CUDA and the controls below.

Any precision reduction, quantization, truncation or backend change needs
probability parity and labeled quality checks. The eight-ticket quality smoke
set is useful for exercising the harness, but cannot establish domain accuracy
or calibration. Core keyword-backend timings measure application overhead,
not neural inference.

## Native CUDA and execution controls (unreleased)

Local exploratory measurements on 2026-09-24: RTX 4090, ORT GPU 1.29.0,
unchanged pinned multilingual ONNX model, TF32 disabled, two intra-op threads.
The latency fixture remains four fields, maximum sequence length 166, five
warmups and 50 timed calls after a separately reported cold call. These are
short shared-workstation runs, not a same-hardware comparison with other projects.
See [native CUDA observations](../benchmarks/results/2026-09-24-native-cuda.jsonl).

| Path | p50 | p95 |
|---|---:|---:|
| Native C++ CUDA, ordinary Run | 13.206 ms | 14.240 ms |
| Native C++ CUDA, host I/O binding | 13.513 ms | 19.725 ms |

CPU/CUDA probability parity over the separate empty/Unicode/long-input test
observed maximum absolute delta `7.15256e-7`, below its `1e-4` gate. This does not
establish domain quality. Host I/O binding remains **off** by default: this probe
shows no benefit. Both paths use the same session reuse, token cache and host pool.

A separate closed-loop load smoke test with four clients, four fields each,
25 calls/client, one worker, 16-row microbatches and a 1 ms accumulation delay
completed 100/100 calls in 0.820 s: 122.0 calls/s (488.0 fields/s), end-to-end
p50 33.00 ms and p95 39.37 ms. No errors/rejections occurred. These numbers show
the latency/throughput tradeoff, not the capacity limit or an open-loop SLA.

```sh
jevt_laya_latency models/laya-multilingual 2 4 50 cuda 0
jevt_laya_latency models/laya-multilingual 2 4 50 cuda 1
jevt_laya_load models/laya-multilingual cuda 2 4 4 25 16 1000 0
jevt_diagnostics_contention 4 200000 256 16
```

Thread tuning is a deployment experiment: sweep ORT threads (1/2/4/default),
clients and batch rows together, keeping the model/input/precision fixed. The
load harness records all attempts, errors/rejections, throughput and successful
request percentiles. Do not derive throughput from reciprocal p50.

Diagnostics retain exact counters/histograms with optional sampled traces. Two
four-writer stress runs (200k calls/writer, continuous snapshots) measured
default record p95 16.86/19.19 microseconds, versus 1.20/1.00 microseconds with
`recent_sample_every=16`. Results are noisy; no default speedup is claimed.

## Why laya.cpp's published numbers differ

Source inspected: [`laya.cpp` e1c6e78](https://github.com/lkarlslund/laya.cpp/tree/e1c6e7832189d36903e90c6fe3b8b1fece7f6f17).
This is a different inference engine (ggml and model-specific GPU paths), not
another wrapper around the same ONNX graph. Both projects reuse loaded models.

Its [optimized arithmetic](https://github.com/lkarlslund/laya.cpp/blob/e1c6e7832189d36903e90c6fe3b8b1fece7f6f17/docs/architecture.md)
uses compensated FP16 Tensor Core products for its FP32 mode, fused QKV/rotary
and GELU operations, plus reusable GPU graphs. Its FP32 fused-attention path is
limited to sequences through 128 tokens; longer sequences use cuBLAS attention.
JevT++ does not implement these custom kernels or graph replay. Our TF32-off ORT
path is not equivalent to their `--tensor-core-fp32 --flash-fp32` flags.

Their [published measurements](https://github.com/lkarlslund/laya.cpp/blob/e1c6e7832189d36903e90c6fe3b8b1fece7f6f17/docs/measurements/readme-performance.json)
use RTX PRO 6000 Blackwell 96 GB, a varied 250-question corpus and matching-
precision Python/PyTorch baselines. Multilingual FP32 batch four reports
p50 2.944 ms and p95 25.046 ms; the table's 673.4 figure is **questions/second**,
not requests/second or milliseconds. Our repeated four-field fixture on RTX4090
is not a matched workload. Model source revision matches our export's provenance,
but the artifact, output work and implementation differ.

Their [methodology](https://github.com/lkarlslund/laya.cpp/blob/e1c6e7832189d36903e90c6fe3b8b1fece7f6f17/docs/benchmarking.md)
validates precision against a same-precision baseline; BF16 agreement is not
FP32 agreement. These architectural differences are plausible contributors, not
a measured breakdown of the gap. A fair comparison needs identical hardware,
corpus, shape groups, precision, output contract and timing boundaries. We have
not executed that head-to-head comparison or integrated their backend.
