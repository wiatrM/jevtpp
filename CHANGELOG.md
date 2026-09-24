# Changelog

## Unreleased

### Fixed

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

- GitHub Pages documentation with local search, portable MDX and a generated
  Mintlify content bundle.
- CPU/Python/CUDA comparison harnesses and attributed performance measurements.
- Compile-negative schema checks, installed-consumer tests, sanitizers, shared
  builds, HTTP protocol regressions and 20 synthetic ONNX contract fixtures.

No CUDA execution provider, Asio adapter or coroutine API has been added to
the C++ library in these changes.

## 0.2.0

- Typed System One fields and shared text/JSON/serialized-object context.
- Native CPU Laya adapter with pinned model download and reference tests.
- Multi-field support-ticket example and a small labeled quality smoke set.
