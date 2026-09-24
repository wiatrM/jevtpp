# Changelog

## Unreleased

### Fixed

- Default tests, benchmarks and examples to off when embedded in another CMake
  project, while preserving explicitly supplied options.
- Reject negative/non-finite predicate scores and overflowing choice totals.
- Validate binding thresholds: `-1` inherits the context threshold; all other
  values must be finite and between zero and one.
- Interrupt active diagnostic HTTP connections during shutdown; apply socket
  idle timeouts, enforce complete bounded headers, and return 405 for non-GET
  methods.
- Use observed maximum latency as a conservative histogram-overflow percentile
  bound, instead of the mean, preserving percentile ordering.
- Validate Laya input/output tensor types and ranks, exact logits dimensions,
  and finite active logits before producing probabilities.
- Export the core's thread dependency and enable Windows shared-library exports.

### Added

- Optional remote backend with injected HTTP transports, default libcurl
  transport, bounded retries/timeouts and local mock/loopback integration tests.
- Optional Boost.Asio completion-token and coroutine adapter over bounded
  workers; request stop tokens and deadlines; scripted/recording test backend.
- Request/model metadata and token-usage propagation through typed evaluations.
- Optional pinned `laya.cpp`/ggml backend: CPU strict FP32 and explicit CUDA
  optimized FP32, native graph reuse, bounded schema cache and shared-context
  tokenization under the existing typed API. Safetensors downloads verify SHA-256.
- Same-process ONNX/native comparison with alternating execution order, retained
  samples, exact category agreement and a `1e-4` probability gate for speedups.
- GitHub Pages documentation with local search, portable MDX and a generated
  Mintlify content bundle.
- CPU/Python/CUDA comparison harnesses and attributed performance measurements.
- Compile-negative schema checks, installed-consumer tests, sanitizers, shared
  builds, HTTP protocol regressions and 20 synthetic ONNX contract fixtures.

- Explicit CUDA selection, representative warmup, schema token cache, shared-state
  tokenization, bounded host-buffer reuse and opt-in ORT I/O binding.
- Typed field projection, context token budgets, ORT parallel/spinning controls.
- Bounded backend workers, owned future submission, cross-request microbatching,
  length buckets, overload rejection and graceful draining shutdown.
- Exact metrics with sampled recent traces; diagnostics contention and concurrent
  Laya load benchmarks; opt-in FP16/INT8 conversion and quality/parity gates.

Running local model inference cannot be forcibly cancelled. The ONNX adapter
does not provide a persistent CUDA device-buffer pool or CUDA Graph replay; the optional native
backend uses ggml's graph/allocation reuse. Precision candidates are not enabled
by default.

## 0.2.0

- Typed System One fields and shared text/JSON/serialized-object context.
- Native CPU Laya adapter with pinned model download and reference tests.
- Multi-field support-ticket example and a small labeled quality smoke set.
