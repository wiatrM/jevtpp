#pragma once

#include "jevt/core.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace jevt {

enum class CallOutcome { success, error, abstain };

struct DiagnosticsOptions {
  std::size_t recent_capacity = 256;
  std::size_t max_decisions = 1024;
  std::size_t max_tag_length = 128;
  std::vector<double> histogram_bounds_ms{
      0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5,
      5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0};
  // Retain calls 1, 1 + N, 1 + 2N, ... in each reset window. Zero disables
  // recent traces; counters and histograms always include every call.
  // recent_capacity limits retained samples, independently of this interval.
  std::size_t recent_sample_every = 1;
};

struct HistogramSnapshot {
  // Non-cumulative counts. counts.size() == bounds_ms.size() + 1; the final
  // bucket is +Inf.
  std::vector<double> bounds_ms;
  std::vector<std::uint64_t> counts;
};

struct CounterSnapshot {
  std::uint64_t calls = 0;
  std::uint64_t successes = 0;
  std::uint64_t errors = 0;
  std::uint64_t abstains = 0;
  double latency_mean_ms = 0.0;
  double latency_p50_ms = 0.0;
  double latency_p95_ms = 0.0;
  double latency_p99_ms = 0.0;
  HistogramSnapshot latency_histogram;
  // Observed per-evaluation usage, not provider billing totals. A provider's
  // report shared across separate evaluations can be observed more than once.
  std::uint64_t input_tokens = 0;
  std::uint64_t output_tokens = 0;
};

struct DecisionSnapshot {
  std::string decision;
  CounterSnapshot stats;
};

struct RecentCall {
  // Call ordinal within the reset window; sampled traces can have gaps.
  std::uint64_t sequence = 0;
  std::int64_t unix_time_ms = 0;
  std::string decision;
  CallOutcome outcome = CallOutcome::success;
  double latency_ms = 0.0;
  // Caller-controlled, bounded metadata. Use an opaque identifier or hash;
  // never pass complete user input here.
  std::optional<std::string> tag;
};

struct DiagnosticsSnapshot {
  std::int64_t generated_at_unix_ms = 0;
  CounterSnapshot total;
  std::vector<DecisionSnapshot> decisions;
  std::vector<RecentCall> recent_calls;  // newest first
  std::uint64_t dropped_decisions = 0;
};

class Diagnostics {
 public:
  explicit Diagnostics(DiagnosticsOptions options = {});
  ~Diagnostics();

  Diagnostics(const Diagnostics&) = delete;
  Diagnostics& operator=(const Diagnostics&) = delete;
  Diagnostics(Diagnostics&&) noexcept;
  Diagnostics& operator=(Diagnostics&&) noexcept;

  // `decision` is a stable, bounded-cardinality operation/schema name, not
  // user input. `tag`, when present, should be an opaque identifier or hash.
  void record_call(std::string_view decision, CallOutcome outcome,
                   std::chrono::nanoseconds latency,
                   std::optional<std::string_view> tag = std::nullopt,
                   token_usage usage = {});

  [[nodiscard]] DiagnosticsSnapshot snapshot() const;
  [[nodiscard]] std::string to_json() const;
  [[nodiscard]] std::string to_prometheus() const;
  void reset();

 private:
  struct Impl;
  Impl* impl_;
};

[[nodiscard]] std::string_view to_string(CallOutcome outcome) noexcept;
[[nodiscard]] std::string to_json(const DiagnosticsSnapshot& snapshot);

}  // namespace jevt
