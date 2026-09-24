#include <jevt/jevt.hpp>

#include "statistics.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

enum class team { billing, technical, sales };
inline constexpr auto routing = jevt::schema<team, "benchmark.routing">(
    jevt::option<team::billing>("billing"),
    jevt::option<team::technical>("technical"),
    jevt::option<team::sales>("sales"));

struct options {
    std::uint64_t iterations = 200'000;
    std::uint64_t warmup = 20'000;
    std::size_t batch_size = 1;
    std::size_t threads = 1;
    std::optional<std::string> json_path;
    std::optional<double> min_throughput;
    std::optional<double> max_p95_us;
};

template <class T>
T parse_number(std::string_view value, std::string_view flag) {
    T parsed{};
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::invalid_argument("invalid value for " + std::string{flag});
    }
    return parsed;
}

double parse_double(std::string_view value, std::string_view flag) {
    std::string owned{value};
    char* end = nullptr;
    const double parsed = std::strtod(owned.c_str(), &end);
    if (end != owned.c_str() + owned.size()) {
        throw std::invalid_argument("invalid value for " + std::string{flag});
    }
    return parsed;
}

options parse_options(int argc, char** argv) {
    options parsed;
    for (int index = 1; index < argc; ++index) {
        const std::string_view flag{argv[index]};
        if (flag == "--help") {
            std::cout << "Usage: jevt_benchmark [--iterations N] [--warmup N] "
                         "[--batch-size N] [--threads N] [--json PATH|-] "
                         "[--min-throughput OPS_PER_SEC] [--max-p95-us MICROSECONDS]\n";
            std::exit(0);
        }
        if (index + 1 == argc) {
            throw std::invalid_argument("missing value for " + std::string{flag});
        }
        const std::string_view value{argv[++index]};
        if (flag == "--iterations") parsed.iterations = parse_number<std::uint64_t>(value, flag);
        else if (flag == "--warmup") parsed.warmup = parse_number<std::uint64_t>(value, flag);
        else if (flag == "--batch-size") parsed.batch_size = parse_number<std::size_t>(value, flag);
        else if (flag == "--threads") parsed.threads = parse_number<std::size_t>(value, flag);
        else if (flag == "--json") parsed.json_path = std::string{value};
        else if (flag == "--min-throughput") parsed.min_throughput = parse_double(value, flag);
        else if (flag == "--max-p95-us") parsed.max_p95_us = parse_double(value, flag);
        else throw std::invalid_argument("unknown option " + std::string{flag});
    }
    if (parsed.iterations == 0 || parsed.batch_size == 0 || parsed.threads == 0) {
        throw std::invalid_argument("iterations, batch-size and threads must be positive");
    }
    return parsed;
}

std::string json_for(const options& config, const jevt::benchmark::summary& result) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(3)
        << "{\n"
        << "  \"benchmark\": \"keyword-routing\",\n"
        << "  \"iterations\": " << config.iterations << ",\n"
        << "  \"warmup\": " << config.warmup << ",\n"
        << "  \"batch_size\": " << config.batch_size << ",\n"
        << "  \"threads\": " << config.threads << ",\n"
        << "  \"operations\": " << result.operations << ",\n"
        << "  \"elapsed_seconds\": " << result.elapsed_seconds << ",\n"
        << "  \"throughput_ops_per_second\": " << result.throughput_per_second << ",\n"
        << "  \"latency_us\": {\n"
        << "    \"mean\": " << result.mean_us << ",\n"
        << "    \"p50\": " << result.p50_us << ",\n"
        << "    \"p95\": " << result.p95_us << ",\n"
        << "    \"p99\": " << result.p99_us << ",\n"
        << "    \"max\": " << result.max_us << "\n"
        << "  }\n"
        << "}\n";
    return out.str();
}

} // namespace

int main(int argc, char** argv) try {
    const options config = parse_options(argc, argv);
    auto backend = std::make_shared<jevt::keyword_backend>(
        jevt::keyword_backend::keyword_table{{"invoice", "refund"},
                                              {"crash", "incident"},
                                              {"pricing", "quote"}},
        "benchmark-rules-v1");
    const jevt::context runtime{{.inference_backend = std::move(backend),
                                 .abstain_threshold = 0.55F,
                                 .diagnostics = nullptr}};
    const auto router = runtime.bind(routing);
    constexpr std::string_view messages[]{
        "refund this invoice", "production crash incident", "pricing quote"};
    std::atomic<std::uint64_t> checksum{0};

    for (std::uint64_t index = 0; index < config.warmup; ++index) {
        const auto result = router.choose(messages[index % 3]);
        if (!result || !result->has_value()) throw std::runtime_error("warmup decision failed");
        checksum.fetch_add(static_cast<std::uint64_t>(result->value()), std::memory_order_relaxed);
    }

    std::barrier start_line{static_cast<std::ptrdiff_t>(config.threads + 1)};
    std::vector<std::vector<double>> samples(config.threads);
    std::vector<std::thread> workers;
    workers.reserve(config.threads);
    const std::uint64_t base = config.iterations / config.threads;
    const std::uint64_t remainder = config.iterations % config.threads;
    for (std::size_t thread = 0; thread < config.threads; ++thread) {
        const std::uint64_t thread_operations = base + (thread < remainder ? 1 : 0);
        workers.emplace_back([&, thread, thread_operations] {
            auto& local = samples[thread];
            local.reserve(static_cast<std::size_t>((thread_operations + config.batch_size - 1) /
                                                   config.batch_size));
            start_line.arrive_and_wait();
            for (std::uint64_t offset = 0; offset < thread_operations;) {
                const std::uint64_t batch = std::min<std::uint64_t>(
                    config.batch_size, thread_operations - offset);
                const auto started = clock_type::now();
                for (std::uint64_t item = 0; item < batch; ++item) {
                    const auto result = router.choose(messages[(thread + offset + item) % 3]);
                    if (!result || !result->has_value()) std::terminate();
                    checksum.fetch_add(static_cast<std::uint64_t>(result->value()),
                                       std::memory_order_relaxed);
                }
                const auto elapsed = std::chrono::duration<double, std::micro>(
                    clock_type::now() - started).count();
                local.push_back(elapsed / static_cast<double>(batch));
                offset += batch;
            }
        });
    }
    start_line.arrive_and_wait();
    const auto started = clock_type::now();
    for (auto& worker : workers) worker.join();
    const double elapsed_seconds = std::chrono::duration<double>(clock_type::now() - started).count();

    std::vector<double> combined;
    for (auto& local : samples) {
        combined.insert(combined.end(), local.begin(), local.end());
    }
    const auto result = jevt::benchmark::summarize(
        std::move(combined), config.iterations, elapsed_seconds);
    std::cout << std::fixed << std::setprecision(3)
              << "operations=" << result.operations
              << " threads=" << config.threads
              << " batch=" << config.batch_size
              << " throughput=" << result.throughput_per_second << " ops/s"
              << " p50=" << result.p50_us << " us"
              << " p95=" << result.p95_us << " us"
              << " p99=" << result.p99_us << " us\n";

    if (config.json_path) {
        const auto json = json_for(config, result);
        if (*config.json_path == "-") {
            std::cout << json;
        } else {
            std::ofstream output{*config.json_path};
            if (!output) throw std::runtime_error("cannot open JSON output file");
            output << json;
        }
    }

    bool thresholds_met = true;
    if (config.min_throughput && result.throughput_per_second < *config.min_throughput) {
        std::cerr << "throughput threshold failed: " << result.throughput_per_second
                  << " < " << *config.min_throughput << '\n';
        thresholds_met = false;
    }
    if (config.max_p95_us && result.p95_us > *config.max_p95_us) {
        std::cerr << "p95 threshold failed: " << result.p95_us
                  << " > " << *config.max_p95_us << " us\n";
        thresholds_met = false;
    }
    return thresholds_met && checksum.load(std::memory_order_relaxed) > 0 ? 0 : 2;
} catch (const std::exception& error) {
    std::cerr << "jevt_benchmark: " << error.what() << '\n';
    return 2;
}
