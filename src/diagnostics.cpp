#include "jevt/diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace jevt {
namespace {

std::int64_t now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

std::string json_escape(std::string_view value) {
  std::ostringstream out;
  for (const unsigned char c : value) {
    switch (c) {
      case '"': out << "\\\""; break;
      case '\\': out << "\\\\"; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<unsigned>(c) << std::dec;
        } else {
          out << static_cast<char>(c);
        }
    }
  }
  return out.str();
}

std::string prometheus_escape(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  for (const char c : value) {
    if (c == '\\' || c == '"') result.push_back('\\');
    if (c == '\n') {
      result += "\\n";
    } else {
      result.push_back(c);
    }
  }
  return result;
}

struct Counters {
  std::uint64_t calls = 0;
  std::uint64_t successes = 0;
  std::uint64_t errors = 0;
  std::uint64_t abstains = 0;
  long double latency_sum_ms = 0.0;
  std::vector<std::uint64_t> buckets;
};

double percentile(const Counters& counters, const std::vector<double>& bounds,
                  double quantile) {
  if (counters.calls == 0) return 0.0;
  const auto rank = static_cast<std::uint64_t>(
      std::ceil(quantile * static_cast<double>(counters.calls)));
  std::uint64_t cumulative = 0;
  for (std::size_t i = 0; i < counters.buckets.size(); ++i) {
    cumulative += counters.buckets[i];
    if (cumulative >= rank) {
      if (i < bounds.size()) return bounds[i];
      // The overflow bucket has no upper bound; mean is a stable, finite
      // fallback and avoids fabricating an infinity in JSON.
      return static_cast<double>(counters.latency_sum_ms / counters.calls);
    }
  }
  return 0.0;
}

CounterSnapshot make_snapshot(const Counters& source,
                              const std::vector<double>& bounds) {
  CounterSnapshot result;
  result.calls = source.calls;
  result.successes = source.successes;
  result.errors = source.errors;
  result.abstains = source.abstains;
  if (source.calls != 0) {
    result.latency_mean_ms =
        static_cast<double>(source.latency_sum_ms / source.calls);
  }
  result.latency_p50_ms = percentile(source, bounds, 0.50);
  result.latency_p95_ms = percentile(source, bounds, 0.95);
  result.latency_p99_ms = percentile(source, bounds, 0.99);
  result.latency_histogram.bounds_ms = bounds;
  result.latency_histogram.counts = source.buckets;
  return result;
}

void append_stats_json(std::ostringstream& out, const CounterSnapshot& s) {
  out << "{\"calls\":" << s.calls << ",\"successes\":" << s.successes
      << ",\"errors\":" << s.errors << ",\"abstains\":" << s.abstains
      << ",\"latency_ms\":{\"mean\":" << s.latency_mean_ms
      << ",\"p50\":" << s.latency_p50_ms << ",\"p95\":"
      << s.latency_p95_ms << ",\"p99\":" << s.latency_p99_ms
      << ",\"histogram\":{\"bounds\":[";
  for (std::size_t i = 0; i < s.latency_histogram.bounds_ms.size(); ++i) {
    if (i) out << ',';
    out << s.latency_histogram.bounds_ms[i];
  }
  out << "],\"counts\":[";
  for (std::size_t i = 0; i < s.latency_histogram.counts.size(); ++i) {
    if (i) out << ',';
    out << s.latency_histogram.counts[i];
  }
  out << "]}}}";
}

}  // namespace

struct Diagnostics::Impl {
  explicit Impl(DiagnosticsOptions value) : options(std::move(value)) {
    auto& bounds = options.histogram_bounds_ms;
    if (bounds.empty() || !std::is_sorted(bounds.begin(), bounds.end()) ||
        std::any_of(bounds.begin(), bounds.end(),
                    [](double v) { return !std::isfinite(v) || v <= 0.0; }) ||
        std::adjacent_find(bounds.begin(), bounds.end()) != bounds.end()) {
      throw std::invalid_argument(
          "histogram bounds must be finite, positive and strictly increasing");
    }
    total.buckets.resize(bounds.size() + 1);
  }

  DiagnosticsOptions options;
  mutable std::mutex mutex;
  Counters total;
  std::map<std::string, Counters, std::less<>> decisions;
  std::deque<RecentCall> recent;
  std::uint64_t next_sequence = 1;
  std::uint64_t dropped_decisions = 0;
};

Diagnostics::Diagnostics(DiagnosticsOptions options)
    : impl_(new Impl(std::move(options))) {}

Diagnostics::~Diagnostics() { delete impl_; }

Diagnostics::Diagnostics(Diagnostics&& other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)) {}

Diagnostics& Diagnostics::operator=(Diagnostics&& other) noexcept {
  if (this != &other) {
    delete impl_;
    impl_ = std::exchange(other.impl_, nullptr);
  }
  return *this;
}

void Diagnostics::record_call(std::string_view decision, CallOutcome outcome,
                              std::chrono::nanoseconds latency,
                              std::optional<std::string_view> tag) {
  if (!impl_) return;
  const double latency_ms =
      std::max(0.0, std::chrono::duration<double, std::milli>(latency).count());
  std::lock_guard lock(impl_->mutex);

  auto update = [&](Counters& counters) {
    ++counters.calls;
    if (outcome == CallOutcome::success) ++counters.successes;
    if (outcome == CallOutcome::error) ++counters.errors;
    if (outcome == CallOutcome::abstain) ++counters.abstains;
    counters.latency_sum_ms += latency_ms;
    const auto bucket = std::lower_bound(impl_->options.histogram_bounds_ms.begin(),
                                         impl_->options.histogram_bounds_ms.end(),
                                         latency_ms);
    ++counters.buckets[static_cast<std::size_t>(
        bucket - impl_->options.histogram_bounds_ms.begin())];
  };
  update(impl_->total);

  auto found = impl_->decisions.find(decision);
  if (found == impl_->decisions.end()) {
    if (impl_->decisions.size() < impl_->options.max_decisions) {
      Counters counters;
      counters.buckets.resize(impl_->options.histogram_bounds_ms.size() + 1);
      found = impl_->decisions.emplace(std::string(decision), std::move(counters)).first;
    } else {
      ++impl_->dropped_decisions;
    }
  }
  if (found != impl_->decisions.end()) update(found->second);

  if (impl_->options.recent_capacity != 0) {
    RecentCall call;
    call.sequence = impl_->next_sequence++;
    call.unix_time_ms = now_ms();
    call.decision = std::string(decision);
    call.outcome = outcome;
    call.latency_ms = latency_ms;
    if (tag) call.tag = std::string(tag->substr(0, impl_->options.max_tag_length));
    impl_->recent.push_front(std::move(call));
    if (impl_->recent.size() > impl_->options.recent_capacity) {
      impl_->recent.pop_back();
    }
  }
}

DiagnosticsSnapshot Diagnostics::snapshot() const {
  DiagnosticsSnapshot result;
  result.generated_at_unix_ms = now_ms();
  if (!impl_) return result;
  std::lock_guard lock(impl_->mutex);
  result.total = make_snapshot(impl_->total, impl_->options.histogram_bounds_ms);
  result.decisions.reserve(impl_->decisions.size());
  for (const auto& [name, counters] : impl_->decisions) {
    result.decisions.push_back(
        {name, make_snapshot(counters, impl_->options.histogram_bounds_ms)});
  }
  result.recent_calls.assign(impl_->recent.begin(), impl_->recent.end());
  result.dropped_decisions = impl_->dropped_decisions;
  return result;
}

std::string Diagnostics::to_json() const { return jevt::to_json(snapshot()); }

std::string Diagnostics::to_prometheus() const {
  const auto snap = snapshot();
  std::ostringstream out;
  out << std::setprecision(12)
      << "# HELP jevt_calls_total Total evaluated calls.\n"
      << "# TYPE jevt_calls_total counter\n"
      << "jevt_calls_total " << snap.total.calls << "\n"
      << "# HELP jevt_outcomes_total Evaluated calls by outcome.\n"
      << "# TYPE jevt_outcomes_total counter\n"
      << "jevt_outcomes_total{outcome=\"success\"} " << snap.total.successes << "\n"
      << "jevt_outcomes_total{outcome=\"error\"} " << snap.total.errors << "\n"
      << "jevt_outcomes_total{outcome=\"abstain\"} " << snap.total.abstains << "\n"
      << "# HELP jevt_latency_milliseconds Approximate latency quantiles.\n"
      << "# TYPE jevt_latency_milliseconds gauge\n"
      << "jevt_latency_milliseconds{quantile=\"0.5\"} " << snap.total.latency_p50_ms << "\n"
      << "jevt_latency_milliseconds{quantile=\"0.95\"} " << snap.total.latency_p95_ms << "\n"
      << "jevt_latency_milliseconds{quantile=\"0.99\"} " << snap.total.latency_p99_ms << "\n"
      << "# HELP jevt_latency_milliseconds_histogram Call latency distribution.\n"
      << "# TYPE jevt_latency_milliseconds_histogram histogram\n";
  std::uint64_t cumulative = 0;
  for (std::size_t i = 0; i < snap.total.latency_histogram.counts.size(); ++i) {
    cumulative += snap.total.latency_histogram.counts[i];
    out << "jevt_latency_milliseconds_histogram_bucket{le=\"";
    if (i < snap.total.latency_histogram.bounds_ms.size()) {
      out << snap.total.latency_histogram.bounds_ms[i];
    } else {
      out << "+Inf";
    }
    out << "\"} " << cumulative << "\n";
  }
  out << "jevt_latency_milliseconds_histogram_sum "
      << snap.total.latency_mean_ms * static_cast<double>(snap.total.calls) << "\n"
      << "jevt_latency_milliseconds_histogram_count " << snap.total.calls << "\n";
  for (const auto& d : snap.decisions) {
    const auto name = prometheus_escape(d.decision);
    out << "jevt_decision_calls_total{decision=\"" << name << "\"} "
        << d.stats.calls << "\n"
        << "jevt_decision_outcomes_total{decision=\"" << name
        << "\",outcome=\"success\"} " << d.stats.successes << "\n"
        << "jevt_decision_outcomes_total{decision=\"" << name
        << "\",outcome=\"error\"} " << d.stats.errors << "\n"
        << "jevt_decision_outcomes_total{decision=\"" << name
        << "\",outcome=\"abstain\"} " << d.stats.abstains << "\n"
        << "jevt_decision_latency_milliseconds{decision=\"" << name
        << "\",quantile=\"0.95\"} " << d.stats.latency_p95_ms << "\n";
  }
  out << "jevt_dropped_decisions_total " << snap.dropped_decisions << "\n";
  return out.str();
}

void Diagnostics::reset() {
  if (!impl_) return;
  std::lock_guard lock(impl_->mutex);
  impl_->total = Counters{};
  impl_->total.buckets.resize(impl_->options.histogram_bounds_ms.size() + 1);
  impl_->decisions.clear();
  impl_->recent.clear();
  impl_->next_sequence = 1;
  impl_->dropped_decisions = 0;
}

std::string_view to_string(CallOutcome outcome) noexcept {
  switch (outcome) {
    case CallOutcome::success: return "success";
    case CallOutcome::error: return "error";
    case CallOutcome::abstain: return "abstain";
  }
  return "unknown";
}

std::string to_json(const DiagnosticsSnapshot& snapshot) {
  std::ostringstream out;
  out << std::setprecision(12) << "{\"generated_at_unix_ms\":"
      << snapshot.generated_at_unix_ms << ",\"total\":";
  append_stats_json(out, snapshot.total);
  out << ",\"decisions\":[";
  for (std::size_t i = 0; i < snapshot.decisions.size(); ++i) {
    if (i) out << ',';
    out << "{\"decision\":\"" << json_escape(snapshot.decisions[i].decision)
        << "\",\"stats\":";
    append_stats_json(out, snapshot.decisions[i].stats);
    out << '}';
  }
  out << "],\"recent_calls\":[";
  for (std::size_t i = 0; i < snapshot.recent_calls.size(); ++i) {
    if (i) out << ',';
    const auto& call = snapshot.recent_calls[i];
    out << "{\"sequence\":" << call.sequence << ",\"unix_time_ms\":"
        << call.unix_time_ms << ",\"decision\":\""
        << json_escape(call.decision) << "\",\"outcome\":\""
        << to_string(call.outcome) << "\",\"latency_ms\":" << call.latency_ms;
    if (call.tag) out << ",\"tag\":\"" << json_escape(*call.tag) << '"';
    out << '}';
  }
  out << "],\"dropped_decisions\":" << snapshot.dropped_decisions << '}';
  return out.str();
}

}  // namespace jevt
