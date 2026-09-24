// The support-ticket fixture matches laya_latency.cpp and laya_latency.py.
#include <jevt/batching.hpp>
#include <jevt/laya.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <thread>

namespace {
using clock_type = std::chrono::steady_clock;
int integer(const char* text) {
    const std::string_view value(text);
    int number = 0;
    const auto result = std::from_chars(value.data(), value.data() + value.size(), number);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size())
        throw std::invalid_argument("invalid integer argument");
    return number;
}
struct client_result {
    std::vector<double> success_ms;
    std::size_t errors = 0, rejections = 0, exceptions = 0;
};
}

int main(int argc, char** argv) try {
    if (argc != 10) throw std::invalid_argument(
        "usage: laya_load MODEL PROVIDER(cpu|cuda) ORT_THREADS CLIENTS FIELDS(1|4) ITERATIONS_PER_CLIENT MAX_BATCH DELAY_US IO_BINDING(0|1)");
    const std::string provider = argv[2];
    const int ort_threads = integer(argv[3]), clients = integer(argv[4]), fields = integer(argv[5]);
    const int iterations = integer(argv[6]), max_batch = integer(argv[7]), delay = integer(argv[8]), binding = integer(argv[9]);
    if ((provider != "cpu" && provider != "cuda") || ort_threads < 0 || clients < 1 || clients == std::numeric_limits<int>::max() ||
        (fields != 1 && fields != 4) || iterations < 1 || max_batch < 0 || delay < 0 || (binding != 0 && binding != 1))
        throw std::invalid_argument("invalid arguments");
    if (static_cast<std::size_t>(clients) > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(iterations))
        throw std::invalid_argument("attempt count overflow");
    const auto load_start = clock_type::now();
    jevt::laya_options native_options;
    native_options.model_directory = argv[1];
    native_options.intra_op_threads = ort_threads;
    native_options.provider = provider == "cuda" ? jevt::laya_provider::cuda : jevt::laya_provider::cpu;
    native_options.use_io_binding = binding != 0;
    auto native = std::make_shared<jevt::laya_backend>(std::move(native_options));
    const auto load_ms = std::chrono::duration<double, std::milli>(clock_type::now() - load_start).count();
    const std::string context = R"json({
  "ticket": {
    "subject": "Production login is down",
    "message": "All our users are locked out. We have no workaround. Please fix this immediately!"
  },
  "customer": {"plan": "enterprise", "affected_users": 240},
  "service": {"status": "complete_outage", "sla_response_minutes": 30}
})json";
    const std::string prefix = "Evaluate the incoming customer support ticket using the ticket, customer, and service context.\n\n";
    const std::array<std::string, 4> questions{
        prefix + "Which team should own this request?",
        prefix + "Does this require attention within one hour?",
        prefix + "Rate the frustration level against the rubric.",
        prefix + "The customer is angry."};
    const std::array<std::string_view, 4> categories{"Invoices, payments, or refund requests.", "Software bugs, crashes, or login failures.", "Upgrades, enterprise pricing, or new purchases.", "Unsolicited marketing or automated noise."};
    const std::array<std::string_view, 3> urgency{"Customer is patient and calm.", "Issue blocks work but has a workaround.", "Complete outage or severe frustration."};
    using kind = jevt::inference_request::kind;
    const std::array<jevt::inference_request, 4> requests{{
        {"category", questions[0], context, categories, kind::choice},
        {"is_urgent", questions[1], context, {}, kind::noul},
        {"urgency_score", questions[2], context, urgency, kind::score},
        {"sentiment_probability", questions[3], context, {}, kind::noul}}};
    const auto selected = std::span{requests}.first(static_cast<std::size_t>(fields));
    for (int i = 0; i < 6; ++i) {
        const auto warmup = native->predict_batch(selected);
        if (!warmup) throw std::runtime_error("warmup failed: " + warmup.error_value().message);
    }
    std::shared_ptr<jevt::batching_backend> pool;
    std::shared_ptr<jevt::backend> target = native;
    if (max_batch > 0) {
        pool = std::make_shared<jevt::batching_backend>(native, jevt::batching_options{
            .worker_count = 1, .queue_capacity = 256, .max_batch_size = static_cast<std::size_t>(max_batch),
            .max_delay = std::chrono::microseconds{delay}});
        target = pool;
    }
    std::vector<client_result> results(static_cast<std::size_t>(clients));
    for (auto& result : results) result.success_ms.reserve(static_cast<std::size_t>(iterations));
    std::barrier start(clients + 1);
    std::vector<std::jthread> threads;
    threads.reserve(static_cast<std::size_t>(clients));
    std::atomic<bool> abort_start{false};
    // No think time: each client holds at most one synchronous call outstanding.
    try {
      for (int client = 0; client < clients; ++client) threads.emplace_back([&, client] {
        auto& output = results[static_cast<std::size_t>(client)];
        start.arrive_and_wait();
        if (abort_start.load()) return;
        for (int i = 0; i < iterations; ++i) {
            const auto begin = clock_type::now();
            try {
                auto response = target->predict_batch(selected);
                const auto ms = std::chrono::duration<double, std::milli>(clock_type::now() - begin).count();
                if (response) output.success_ms.push_back(ms);
                else {
                    ++output.errors;
                    if (response.error_value().code == jevt::error_code::overloaded) ++output.rejections;
                }
            } catch (...) { ++output.errors; ++output.exceptions; }
        }
      });
    } catch (...) {
        // A thread-creation failure must not strand already-created clients at
        // the start barrier while their jthread destructors try to join them.
        abort_start.store(true);
        for (auto missing = threads.size(); missing < static_cast<std::size_t>(clients); ++missing)
            start.arrive_and_drop();
        start.arrive_and_drop();
        for (auto& thread : threads) thread.join();
        throw;
    }
    const auto begin = clock_type::now();
    start.arrive_and_wait();
    for (auto& thread : threads) thread.join();
    const auto seconds = std::chrono::duration<double>(clock_type::now() - begin).count();
    if (pool) pool->shutdown();
    std::vector<double> samples;
    std::size_t errors = 0, rejections = 0, exceptions = 0;
    for (const auto& result : results) {
        samples.insert(samples.end(), result.success_ms.begin(), result.success_ms.end());
        errors += result.errors; rejections += result.rejections; exceptions += result.exceptions;
    }
    std::sort(samples.begin(), samples.end());
    const auto percentile = [&](double q) {
        if (samples.empty()) std::cout << "null";
        else std::cout << samples[static_cast<std::size_t>(std::ceil(q * static_cast<double>(samples.size()))) - 1];
    };
    const auto attempts = static_cast<std::size_t>(clients) * static_cast<std::size_t>(iterations);
    std::cout << std::setprecision(9) << "{\"provider\":\"" << provider << "\",\"ort_threads\":" << ort_threads
        << ",\"clients\":" << clients << ",\"fields\":" << fields << ",\"iterations_per_client\":" << iterations
        << ",\"max_batch_rows\":" << max_batch << ",\"delay_us\":" << delay << ",\"io_binding\":" << binding
        << ",\"queue_capacity_rows\":" << (pool ? 256 : 0) << ",\"pool_workers\":" << (pool ? 1 : 0)
        << ",\"warmup_calls\":6,\"load_ms\":" << load_ms << ",\"wall_seconds\":" << seconds
        << ",\"attempts\":" << attempts << ",\"successes\":" << samples.size() << ",\"errors\":" << errors
        << ",\"overload_rejections\":" << rejections << ",\"exceptions\":" << exceptions
        << ",\"error_rate\":" << static_cast<double>(errors) / static_cast<double>(attempts)
        << ",\"success_calls_per_second\":" << static_cast<double>(samples.size()) / seconds
        << ",\"attempts_per_second\":" << static_cast<double>(attempts) / seconds
        << ",\"success_fields_per_second\":" << static_cast<double>(samples.size()) * fields / seconds
        << ",\"success_e2e_p50_ms\":";
    percentile(.50); std::cout << ",\"success_e2e_p95_ms\":"; percentile(.95);
    std::cout << ",\"success_e2e_p99_ms\":"; percentile(.99);
    if (pool) {
        const auto stats = pool->stats();
        std::cout << ",\"backend_batches\":" << stats.batches << ",\"accepted_rows\":" << stats.accepted
            << ",\"rejected_rows\":" << stats.rejected << ",\"completed_rows\":" << stats.completed
            << ",\"mean_batch_rows\":" << (stats.batches ? static_cast<double>(stats.completed) / static_cast<double>(stats.batches) : 0.0);
    }
    std::cout << "}\n";
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
