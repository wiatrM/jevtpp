#include <jevt/laya.hpp>
#include <array>
#include <cmath>
#include <future>
#include <iostream>
#include <stdexcept>

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void same(const jevt::result<std::vector<jevt::inference_response>>& a,
          const jevt::result<std::vector<jevt::inference_response>>& b) {
    if (!a) throw std::runtime_error(a.error_value().message);
    if (!b) throw std::runtime_error(b.error_value().message);
    require(a->size() == b->size(), "batch cardinality changed");
    for (std::size_t row = 0; row < a->size(); ++row) {
        require((*a)[row].scores.size() == (*b)[row].scores.size(), "option cardinality changed");
        for (std::size_t i = 0; i < (*a)[row].scores.size(); ++i)
            require(std::abs((*a)[row].scores[i] - (*b)[row].scores[i]) < 1e-4F,
                    "optimized path probability differs by >= 1e-4");
    }
}
int main(int argc, char** argv) try {
    if (argc < 2 || argc > 3) throw std::runtime_error("usage: laya_performance_tests MODEL [cpu|cuda]");
    const std::string provider = argc == 3 ? argv[2] : "cpu";
    require(provider == "cpu" || provider == "cuda", "unknown provider");
    jevt::laya_options config{.model_directory = argv[1], .intra_op_threads = 2,
        .schema_cache_entries = 0, .reusable_buffers = 0};
    jevt::laya_backend reference(config);
    config.provider = provider == "cuda" ? jevt::laya_provider::cuda : jevt::laya_provider::cpu;
    config.use_io_binding = true;
    config.schema_cache_entries = 2;
    config.reusable_buffers = 2;
    jevt::laya_backend optimized(config);
    const std::array<std::string_view, 3> options{"Billing and refunds", "Login failures", "Pricing"};
    const std::array<std::string_view, 3> levels{"Calm", "Blocked with workaround", "Complete outage"};
    using kind = jevt::inference_request::kind;
    double maximum_delta = 0;
    for (const std::string input : {std::string{}, std::string{"Nie mogę się zalogować — cała firma zablokowana."},
                                   std::string(5000, 'x')}) {
        const std::array<jevt::inference_request, 3> requests{{
            {"route", "Which team?", input, options, kind::choice},
            {"urgent", "Does this require attention now?", input, {}, kind::noul},
            {"level", "How severe?", input, levels, kind::score}}};
        const auto expected = reference.predict_batch(requests);
        const auto actual = optimized.predict_batch(requests);
        same(expected, actual);
        for (std::size_t r = 0; r < expected->size(); ++r)
            for (std::size_t c = 0; c < (*expected)[r].scores.size(); ++c)
                maximum_delta = std::max(maximum_delta, static_cast<double>(
                    std::abs((*expected)[r].scores[c] - (*actual)[r].scores[c])));
        same(expected, optimized.predict_batch(requests));
    }
    const std::array<jevt::inference_request, 1> repeat{{{"route", "Which team?", "Refund please", options}}};
    const auto warmed = optimized.warmup(repeat, 2);
    require(warmed && *warmed == 2, "warmup failed");
    const auto stats = optimized.statistics();
    require(stats.cached_schemas <= 2 && stats.cached_schema_bytes <= config.schema_cache_bytes,
            "schema cache exceeded limits");
    require(stats.schema_cache_hits > 0 && stats.buffer_reuses > 0 && stats.runs == 8,
            "warmup/cache/buffer reuse not observed");
    require(!optimized.warmup({}), "empty warmup accepted");
    auto first = std::async(std::launch::async, [&] { return optimized.predict_batch(repeat); });
    auto second = std::async(std::launch::async, [&] { return optimized.predict_batch(repeat); });
    same(first.get(), second.get());
    // Changed metadata/options must not reuse a head keyed only by decision id.
    const std::array<jevt::inference_request, 1> changed{{{"route", "Select severity", "Refund please", levels}}};
    same(reference.predict_batch(changed), optimized.predict_batch(changed));
    config.context_token_limit = 1;
    jevt::laya_backend limited(config);
    require(static_cast<bool>(limited.predict_batch(repeat)), "explicit context limit failed");
    std::cout << "provider=" << provider << " cache/io-binding/concurrent-buffer parity passed; max_probability_delta="
              << maximum_delta << '\n';
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
