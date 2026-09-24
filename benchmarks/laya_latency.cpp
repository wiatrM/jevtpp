// Pair with laya_latency.py: identical backend requests, tokenization included.
#include <jevt/laya.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using clock_type = std::chrono::steady_clock;
double elapsed(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}
int main(int argc, char** argv) try {
    if (argc != 5 && argc != 7) throw std::runtime_error("usage: laya_latency MODEL THREADS FIELDS ITERATIONS [cpu|cuda IO_BINDING(0|1)]");
    int threads = std::stoi(argv[2]), fields = std::stoi(argv[3]), iterations = std::stoi(argv[4]);
    if ((fields != 1 && fields != 4) || iterations < 1 || threads < 0) throw std::runtime_error("invalid arguments");
    const auto load_start = clock_type::now();
    const std::string provider = argc == 7 ? argv[5] : "cpu";
    if (provider != "cpu" && provider != "cuda") throw std::runtime_error("unknown provider");
    const bool binding = argc == 7 && std::stoi(argv[6]) == 1;
    jevt::laya_backend backend({.model_directory = argv[1], .intra_op_threads = threads,
        .provider = provider == "cuda" ? jevt::laya_provider::cuda : jevt::laya_provider::cpu,
        .use_io_binding = binding});
    const double load_ms = elapsed(load_start);
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
    const auto run = [&] {
        auto result = backend.predict_batch(std::span{requests}.first(fields));
        if (!result) throw std::runtime_error(result.error_value().message);
        return std::move(*result);
    };
    const auto first_start = clock_type::now();
    auto answers = run();
    const double first_ms = elapsed(first_start);
    for (int i = 0; i < 5; ++i) run();
    std::vector<double> times;
    for (int i = 0; i < iterations; ++i) {
        const auto start = clock_type::now();
        answers = run();
        times.push_back(elapsed(start));
    }
    const auto samples = times;
    std::sort(times.begin(), times.end());
    const auto percentile = [&](double q) { return times[static_cast<std::size_t>(std::ceil(q * times.size())) - 1]; };
    std::cout << std::setprecision(9) << "{\"language\":\"cpp\",\"provider\":\"" << provider
        << "\",\"io_binding\":" << (binding ? "true" : "false") << ",\"threads\":" << threads
        << ",\"fields\":" << fields << ",\"iterations\":" << iterations
        << ",\"warmup\":5,\"load_ms\":" << load_ms << ",\"first_ms\":" << first_ms
        << ",\"p50_ms\":" << percentile(.5) << ",\"p95_ms\":" << percentile(.95) << ",\"probabilities\":[";
    for (std::size_t row = 0; row < answers.size(); ++row) {
        if (row) std::cout << ',';
        std::cout << '[';
        for (std::size_t col = 0; col < answers[row].scores.size(); ++col) {
            if (col) std::cout << ',';
            std::cout << answers[row].scores[col];
        }
        std::cout << ']';
    }
    std::cout << "],\"samples_ms\":[";
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << samples[i];
    }
    std::cout << "]}\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
