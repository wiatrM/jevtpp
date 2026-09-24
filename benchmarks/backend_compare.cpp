// Same-process, alternating-order native-vs-ORT comparison. No CUDA APIs are
// needed here: both backend contracts return CPU-readable probabilities.
#include <jevt/laya.hpp>
#include <jevt/laya_native.hpp>
#include "compare_statistics.hpp"

#include <array>
#include <charconv>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

namespace {
using clock_type = std::chrono::steady_clock;
using probabilities = std::vector<std::vector<float>>;
namespace stats = jevt::benchmark;

std::string json(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 0x20) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<unsigned>(c) << std::dec;
        else out << c;
    }
    out << '"';
    return out.str();
}
void number(std::optional<double> value) { if (value && std::isfinite(*value)) std::cout << *value; else std::cout << "null"; }
void print_probabilities(const probabilities& value) {
    std::cout << '[';
    for (std::size_t row = 0; row < value.size(); ++row) {
        if (row) std::cout << ',';
        std::cout << '[';
        for (std::size_t col = 0; col < value[row].size(); ++col) {
            if (col) std::cout << ',';
            number(static_cast<double>(value[row][col]));
        }
        std::cout << ']';
    }
    std::cout << ']';
}

struct arguments {
    std::filesystem::path onnx_model, native_model;
    std::string provider = "cpu", mode = "strict_fp32", weights_provenance;
    int threads = 1, warmup = 5, repetitions = 30;
};
int parse_integer(std::string_view value) {
    int number;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), number);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) throw std::invalid_argument("invalid integer argument");
    return number;
}
arguments parse(int argc, char** argv) {
    const std::string usage = "usage: backend_compare --onnx-model DIR --native-model DIR [--provider cpu|cuda] "
        "[--native-mode strict_fp32|optimized_fp32] [--threads N] [--warmup N] [--repetitions N] "
        "[--weights-provenance TEXT] (device 0; ORT TF32 disabled; iterations are per case)";
    if (argc < 5 || argc % 2 != 1) throw std::invalid_argument(usage);
    arguments out;
    std::map<std::string, bool> seen;
    for (int i = 1; i < argc; i += 2) {
        const std::string key = argv[i], value = argv[i + 1];
        if (!seen.emplace(key, true).second) throw std::invalid_argument("duplicate argument " + key);
        if (key == "--onnx-model") out.onnx_model = value;
        else if (key == "--native-model") out.native_model = value;
        else if (key == "--provider") out.provider = value;
        else if (key == "--native-mode") out.mode = value;
        else if (key == "--threads") out.threads = parse_integer(value);
        else if (key == "--warmup") out.warmup = parse_integer(value);
        else if (key == "--repetitions") out.repetitions = parse_integer(value);
        else if (key == "--weights-provenance") out.weights_provenance = value;
        else throw std::invalid_argument("unknown argument " + key + "; " + usage);
    }
    if (out.onnx_model.empty() || out.native_model.empty() || (out.provider != "cpu" && out.provider != "cuda") ||
        (out.mode != "strict_fp32" && out.mode != "optimized_fp32") || out.threads < 0 || out.warmup < 0 || out.repetitions < 1)
        throw std::invalid_argument(usage);
    return out;
}

struct fixture {
    const std::string full = R"json({
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
        prefix + "Rate the frustration level against the rubric.", prefix + "The customer is angry."};
    const std::array<std::string_view, 4> categories{"Invoices, payments, or refund requests.", "Software bugs, crashes, or login failures.", "Upgrades, enterprise pricing, or new purchases.", "Unsolicited marketing or automated noise."};
    const std::array<std::string_view, 3> urgency{"Customer is patient and calm.", "Issue blocks work but has a workaround.", "Complete outage or severe frustration."};
    std::array<jevt::inference_request, 4> requests(std::string_view context) const {
        using kind = jevt::inference_request::kind;
        return {{{"category", questions[0], context, categories, kind::choice},
                 {"is_urgent", questions[1], context, {}, kind::noul},
                 {"urgency_score", questions[2], context, urgency, kind::score},
                 {"sentiment_probability", questions[3], context, {}, kind::noul}}};
    }
};

struct sample {
    double ms = 0;
    probabilities values;
    std::string error;
    bool succeeded = false;
};
sample run(jevt::backend& backend, std::span<const jevt::inference_request> requests) {
    sample out;
    const auto begin = clock_type::now();
    try {
        auto result = backend.predict_batch(requests);
        // The native backend must copy scores to CPU before return. Reading all
        // returned values here makes that API requirement explicit in timing.
        if (result) {
            for (const auto& row : *result) out.values.push_back(row.scores);
            out.succeeded = out.values.size() == requests.size();
            if (!out.succeeded) out.error = "response row count mismatch";
        } else out.error = result.error_value().message;
    } catch (const std::exception& error) { out.error = error.what(); }
    catch (...) { out.error = "unknown backend exception"; }
    out.ms = std::chrono::duration<double, std::milli>(clock_type::now() - begin).count();
    return out;
}

struct case_result {
    std::string name;
    int fields = 1;
    std::size_t input_bytes = 0, repeated_words = 0;
    std::array<sample, 2> preflight;
    std::array<std::vector<sample>, 2> warmup;
    std::array<std::vector<sample>, 2> measured;
    std::size_t warmup_errors = 0, parity_failures = 0;
    double max_delta = 0;
    std::size_t argmax_mismatches = 0, compared_probabilities = 0;
    std::string first_parity_error;
    void check(const sample& a, const sample& b) {
        const auto parity = stats::compare_probabilities(a.values, b.values);
        max_delta = std::max(max_delta, parity.max_probability_delta);
        argmax_mismatches += parity.argmax_mismatches;
        compared_probabilities += parity.compared_probabilities;
        if (!a.succeeded || !b.succeeded || !parity.passed) {
            ++parity_failures;
            if (first_parity_error.empty()) first_parity_error = !a.succeeded || !b.succeeded ? "backend call failed" : parity.reason;
        }
    }
};
void print_sample(const sample& s) {
    std::cout << "{\"ms\":" << s.ms << ",\"success\":" << (s.succeeded ? "true" : "false")
        << ",\"error\":" << json(s.error) << ",\"probabilities\":";
    print_probabilities(s.values);
    std::cout << '}';
}
void print_samples(const std::vector<sample>& samples) {
    std::cout << '[';
    for (std::size_t i = 0; i < samples.size(); ++i) { if (i) std::cout << ','; print_sample(samples[i]); }
    std::cout << ']';
}
struct summary { double seconds = 0; std::size_t successes = 0, failures = 0; };
summary summarize(const std::vector<sample>& samples) {
    summary out;
    for (const auto& s : samples) { out.seconds += s.ms / 1000; if (s.succeeded) ++out.successes; else ++out.failures; }
    return out;
}
void print_engine(const std::vector<sample>& samples, int fields) {
    const auto totals = summarize(samples);
    std::vector<double> latencies;
    for (const auto& s : samples) if (s.succeeded) latencies.push_back(s.ms);
    std::cout << "{\"attempts\":" << samples.size() << ",\"successes\":" << totals.successes << ",\"errors\":" << totals.failures
        << ",\"error_rate\":" << (samples.empty() ? 0.0 : static_cast<double>(totals.failures) / static_cast<double>(samples.size()))
        << ",\"call_time_wall_seconds\":" << totals.seconds << ",\"success_requests_per_second\":";
    number(stats::throughput(totals.successes, totals.seconds));
    std::cout << ",\"success_fields_per_second\":";
    number(stats::throughput(totals.successes * static_cast<std::size_t>(fields), totals.seconds));
    std::cout << ",\"success_p50_ms\":"; number(stats::percentile(latencies, .50));
    std::cout << ",\"success_p95_ms\":"; number(stats::percentile(latencies, .95));
    std::cout << ",\"success_p99_ms\":"; number(stats::percentile(latencies, .99));
    std::cout << ",\"samples\":";
    print_samples(samples);
    std::cout << '}';
}
void print_configs(const std::filesystem::path& directory, const std::vector<std::string>& names) {
    std::cout << '{';
    bool first = true;
    for (const auto& name : names) {
        std::ifstream input(directory / name);
        if (!input) continue;
        const std::string content((std::istreambuf_iterator<char>(input)), {});
        if (!first) std::cout << ',';
        first = false;
        std::cout << json(name) << ':' << json(content);
    }
    std::cout << '}';
}
}

int main(int argc, char** argv) try {
    const auto args = parse(argc, argv);
    jevt::laya_options ort_options;
    ort_options.model_directory = args.onnx_model;
    ort_options.provider = args.provider == "cpu" ? jevt::laya_provider::cpu : jevt::laya_provider::cuda;
    ort_options.intra_op_threads = args.threads;
    ort_options.use_tf32 = false;
    ort_options.device_id = 0;
    jevt::laya_native_options native_options;
    native_options.model_directory = args.native_model;
    native_options.provider = args.provider == "cpu" ? jevt::laya_native_provider::cpu : jevt::laya_native_provider::cuda;
    native_options.precision = args.mode == "strict_fp32" ? jevt::laya_native_precision::strict_fp32 : jevt::laya_native_precision::optimized_fp32;
    // Native construction sets process-wide CUDA policy; initialize it before
    // ORT creates CUDA state, as required by the native adapter's contract.
    const auto native_load_start = clock_type::now();
    jevt::laya_native_backend native(native_options);
    const auto native_load_ms = std::chrono::duration<double, std::milli>(clock_type::now() - native_load_start).count();
    const auto ort_load_start = clock_type::now();
    jevt::laya_backend ort(ort_options);
    const auto ort_load_ms = std::chrono::duration<double, std::milli>(clock_type::now() - ort_load_start).count();
    fixture input;
    struct context_case { std::string name, text; std::size_t words; };
    std::vector<context_case> contexts{{"short", "Production login is down. All users are locked out without a workaround.", 0},
                                       {"full", input.full, 0}};
    for (const std::size_t words : {1024u, 4096u}) {
        std::string text = input.full + "\nAdditional context: ";
        for (std::size_t word = 0; word < words; ++word) text += "token ";
        contexts.push_back({"repeated_token_words_" + std::to_string(words), std::move(text), words});
    }
    std::vector<case_result> results;
    const auto experiment_start = clock_type::now();
    for (const auto& context : contexts) for (int fields : {1, 4}) {
        std::cerr << "[compare] " << context.name << ", " << fields << " field(s)\n";
        case_result result;
        result.name = context.name;
        result.fields = fields; result.input_bytes = context.text.size(); result.repeated_words = context.words;
        const auto requests = input.requests(context.text);
        const auto selected = std::span{requests}.first(static_cast<std::size_t>(fields));
        const auto pair = [&](std::size_t ordinal) {
            std::array<sample, 2> values;
            if (ordinal % 2 == 0) { values[0] = run(ort, selected); values[1] = run(native, selected); }
            else { values[1] = run(native, selected); values[0] = run(ort, selected); }
            return values;
        };
        result.preflight = pair(results.size());
        result.check(result.preflight[0], result.preflight[1]);
        for (int i = 0; i < args.warmup; ++i) {
            auto values = pair(results.size() + static_cast<std::size_t>(i));
            for (const auto& value : values) if (!value.succeeded) ++result.warmup_errors;
            result.check(values[0], values[1]);
            for (std::size_t engine = 0; engine < 2; ++engine) result.warmup[engine].push_back(std::move(values[engine]));
        }
        for (int i = 0; i < args.repetitions; ++i) {
            auto values = pair(results.size() + static_cast<std::size_t>(i));
            result.check(values[0], values[1]);
            for (std::size_t engine = 0; engine < 2; ++engine) result.measured[engine].push_back(std::move(values[engine]));
        }
        results.push_back(std::move(result));
    }
    const auto experiment_seconds = std::chrono::duration<double>(clock_type::now() - experiment_start).count();
    const bool all_passed = std::all_of(results.begin(), results.end(), [](const auto& r) { return !r.parity_failures && !r.warmup_errors; });
    std::cout << std::setprecision(10) << "{\"schema_version\":1,\"provider\":" << json(args.provider)
        << ",\"device_id\":0,\"native_device\":" << json(native.device_name())
        << ",\"ort_precision\":\"fp32_tf32_disabled\",\"native_precision\":" << json(args.mode)
        << ",\"native_compute_policy\":" << json(args.mode == "strict_fp32" ? "fp32_tf32_disabled" :
            "fp16_stored_weights_high_low_fp16_activation_split_fp32_accumulation_tf32_disabled")
        << ",\"native_cpu_threads\":\"upstream_default_unmanaged\",\"model_residency\":\"both_backends_loaded_in_same_process\""
        << ",\"ort_threads\":" << args.threads << ",\"warmup_per_case\":" << args.warmup
        << ",\"repetitions_per_case\":" << args.repetitions << ",\"onnx_model\":" << json(args.onnx_model.string())
        << ",\"native_model\":" << json(args.native_model.string())
        << ",\"weights_provenance_user_supplied\":" << json(args.weights_provenance)
        << ",\"weights_independently_verified_by_benchmark\":false,\"onnx_config_files\":";
    print_configs(args.onnx_model, {"config.json", "encoder/config.json", "agent_config.json", "laya_config.json",
        "rl_agent_config.json", "tokenizer_config.json", "tokenizer/tokenizer_config.json"});
    std::cout << ",\"native_config_files\":";
    print_configs(args.native_model, {"encoder/config.json", "rl_agent_config.json", "tokenizer/tokenizer_config.json"});
    std::cout << ",\"ort_load_ms\":" << ort_load_ms << ",\"native_load_ms\":" << native_load_ms
        << ",\"experiment_wall_seconds\":" << experiment_seconds
        << ",\"ort_max_context_tokens\":" << ort.max_context_tokens()
        << ",\"long_case_lengths_are_input_words_not_verified_tokens\":true"
        << ",\"call_timing_includes_cpu_ready_probability_copy\":true,\"order\":\"alternating_by_case_and_iteration\""
        << ",\"probability_tolerance\":0.0001,\"require_exact_argmax\":true,\"all_parity_passed\":" << (all_passed ? "true" : "false")
        << ",\"throughput_basis\":\"successful calls divided by sum of all measured call wall times; excludes other engine and warmup\",\"cases\":[";
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (i) std::cout << ',';
        const auto& r = results[i];
        const auto a = summarize(r.measured[0]), b = summarize(r.measured[1]);
        std::cout << "{\"context\":" << json(r.name) << ",\"fields\":" << r.fields << ",\"input_bytes\":" << r.input_bytes
            << ",\"added_repeated_words\":" << r.repeated_words << ",\"parity_passed\":" << (!r.parity_failures ? "true" : "false")
            << ",\"parity_failures\":" << r.parity_failures << ",\"argmax_mismatches\":" << r.argmax_mismatches
            << ",\"compared_probabilities\":" << r.compared_probabilities << ",\"max_probability_delta\":";
        number(r.compared_probabilities ? std::optional<double>{r.max_delta} : std::nullopt);
        std::cout << ",\"first_parity_error\":" << json(r.first_parity_error)
            << ",\"warmup_errors\":" << r.warmup_errors << ",\"preflight_ort\":";
        print_sample(r.preflight[0]); std::cout << ",\"preflight_native\":"; print_sample(r.preflight[1]);
        std::cout << ",\"warmup_ort\":"; print_samples(r.warmup[0]);
        std::cout << ",\"warmup_native\":"; print_samples(r.warmup[1]);
        std::cout << ",\"ort\":"; print_engine(r.measured[0], r.fields);
        std::cout << ",\"native\":"; print_engine(r.measured[1], r.fields);
        std::cout << ",\"accepted_speedup_ratio\":";
        number(stats::accepted_speedup(all_passed, a.failures + b.failures + r.warmup_errors, a.seconds, b.seconds));
        std::cout << '}';
    }
    std::cout << "]}\n";
    return all_passed ? 0 : 2;
} catch (const std::exception& error) {
    std::cout << "{\"schema_version\":1,\"all_parity_passed\":false,\"accepted_speedup_ratio\":null,\"initialization_or_configuration_error\":"
        << json(error.what()) << "}\n";
    return 1;
}
