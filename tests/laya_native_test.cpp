#include <jevt/laya_native.hpp>
#ifdef JEVT_NATIVE_TEST_WITH_ORT
#include <jevt/laya.hpp>
#endif
#include "test_harness.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>

namespace {
jevt::laya_native_options test_options;
std::filesystem::path onnx_directory;
jevt::laya_native_backend& native() {
    static jevt::laya_native_backend model(test_options);
    return model;
}
const std::array<std::string_view, 2> labels{"billing question", "technical support"};
const jevt::inference_request choice{"test.route", "Which team should respond?", "I was charged twice.", labels};
void check_probabilities(const jevt::inference_response& response, std::size_t count) {
    JEVT_REQUIRE_EQ(response.scores.size(), count);
    float sum = 0;
    for (auto value : response.scores) {
        JEVT_REQUIRE(std::isfinite(value));
        JEVT_REQUIRE(value >= 0 && value <= 1);
        sum += value;
    }
    JEVT_REQUIRE(std::abs(sum - 1) < 1e-5F);
}
}

JEVT_TEST("native CPU rejects GPU-only precision before loading a model") {
    for (const auto mode : {jevt::laya_native_precision::optimized_fp32, jevt::laya_native_precision::bf16}) {
        bool rejected = false;
        try {
            jevt::laya_native_backend model({.model_directory = "does-not-exist", .precision = mode});
        } catch (const std::invalid_argument&) { rejected = true; }
        JEVT_REQUIRE(rejected);
    }
}

JEVT_TEST("native backend rejects an empty model directory") {
    bool rejected = false;
    try { jevt::laya_native_backend model({}); }
    catch (const std::invalid_argument&) { rejected = true; }
    JEVT_REQUIRE(rejected);
}

JEVT_TEST("native rejects unknown provider and precision before model loading") {
    for (bool invalid_provider : {true, false}) {
        jevt::laya_native_options options;
        options.model_directory = "does-not-exist";
        if (invalid_provider) options.provider = static_cast<jevt::laya_native_provider>(99);
        else options.precision = static_cast<jevt::laya_native_precision>(99);
        bool rejected = false;
        try { jevt::laya_native_backend model(options); }
        catch (const std::invalid_argument&) { rejected = true; }
        JEVT_REQUIRE(rejected);
    }
}

JEVT_TEST("native validates requests and preserves one-option results without inference") {
    if (test_options.model_directory.empty()) return;
    auto& model = native();
    const auto empty_batch = model.predict_batch({});
    JEVT_REQUIRE(empty_batch && empty_batch->empty());
    JEVT_REQUIRE(!model.warmup({}));
    const auto invalid = model.predict({"invalid", "question", "state", {}});
    JEVT_REQUIRE(!invalid);
    JEVT_REQUIRE(invalid.error_value().code == jevt::error_code::invalid_request);
    const std::array<std::string_view, 1> single{"only"};
    const auto one = model.predict({"one", "question", "state", single});
    JEVT_REQUIRE(one);
    JEVT_REQUIRE_EQ(one->scores, std::vector<float>{1.0F});
    JEVT_REQUIRE_EQ(one->model_id, test_options.model_id);
    const auto invalid_score = model.predict({"score", "question", "state", single, jevt::inference_request::kind::score});
    JEVT_REQUIRE(!invalid_score);
    const auto bad_kind = model.predict({"kind", "question", "state", labels, static_cast<jevt::inference_request::kind>(99)});
    JEVT_REQUIRE(!bad_kind);
}

JEVT_TEST("native batches preserve row order duplicates and mixed question kinds") {
    if (test_options.model_directory.empty()) return;
    const std::array<std::string_view, 1> one{"only"};
    const std::array<std::string_view, 3> duplicates{"same", "same", ""};
    const std::array<std::string_view, 4> levels{"low", "medium", "high", "urgent"};
    const std::array requests{
        jevt::inference_request{"one", "question", "state", one}, choice,
        jevt::inference_request{"duplicates", "Choose one", "Unicode: Zażółć gęślą jaźń. 中文", duplicates},
        jevt::inference_request{"score", "How urgent?", "Server unavailable", levels, jevt::inference_request::kind::score},
        jevt::inference_request{"noul", "Needs escalation?", "", {}, jevt::inference_request::kind::noul},
        jevt::inference_request{"last", "question", "state", one}};
    const auto batched = native().predict_batch(requests);
    JEVT_REQUIRE(batched);
    JEVT_REQUIRE_EQ(batched->size(), requests.size());
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto individual = native().predict(requests[i]);
        JEVT_REQUIRE(individual);
        check_probabilities((*batched)[i], individual->scores.size());
        for (std::size_t j = 0; j < individual->scores.size(); ++j)
            JEVT_REQUIRE(std::abs((*batched)[i].scores[j] - individual->scores[j]) < 1e-4F);
    }
}

JEVT_TEST("native graph reuse serializes concurrent calls and warmup") {
    if (test_options.model_directory.empty()) return;
    JEVT_REQUIRE(native().warmup(std::span{&choice, std::size_t{1}}, 1));
    const auto baseline = native().predict(choice);
    JEVT_REQUIRE(baseline);
    std::atomic<bool> valid{true};
    std::array<std::thread, 2> workers;
    for (auto& worker : workers) worker = std::thread([&] {
        const auto result = native().predict(choice);
        if (!result || result->scores.size() != baseline->scores.size()) { valid = false; return; }
        for (std::size_t i = 0; i < result->scores.size(); ++i)
            if (std::abs(result->scores[i] - baseline->scores[i]) > 1e-6F) valid = false;
    });
    for (auto& worker : workers) worker.join();
    JEVT_REQUIRE(valid.load());
}

#ifdef JEVT_NATIVE_TEST_WITH_ORT
JEVT_TEST("native and ONNX return calibrated probabilities for the same model") {
    if (test_options.model_directory.empty() || onnx_directory.empty()) return;
    jevt::laya_options onnx_options;
    onnx_options.model_directory = onnx_directory;
    onnx_options.intra_op_threads = 2;
    jevt::laya_backend onnx(onnx_options);
    const std::array<std::string_view, 3> duplicates{"same", "same", ""};
    const std::array<std::string_view, 3> levels{"low", "medium", "high"};
    const std::string long_state(8000, 'x');
    const std::array requests{choice,
        jevt::inference_request{"dupes", "Choose one", "Zażółć gęślą jaźń. 中文 <mask> [MASK]", duplicates},
        jevt::inference_request{"score", "How urgent?", "Server unavailable", levels, jevt::inference_request::kind::score},
        jevt::inference_request{"noul", "Needs escalation?", "", {}, jevt::inference_request::kind::noul},
        jevt::inference_request{"long", "Which team?", long_state, labels}};
    const auto expected = onnx.predict_batch(requests);
    const auto actual = native().predict_batch(requests);
    JEVT_REQUIRE(expected && actual);
    JEVT_REQUIRE_EQ(expected->size(), actual->size());
    float maximum_delta = 0;
    for (std::size_t i = 0; i < actual->size(); ++i) {
        JEVT_REQUIRE_EQ((*actual)[i].scores.size(), (*expected)[i].scores.size());
        for (std::size_t j = 0; j < (*actual)[i].scores.size(); ++j) {
            const auto delta = std::abs((*actual)[i].scores[j] - (*expected)[i].scores[j]);
            maximum_delta = std::max(maximum_delta, delta);
            JEVT_REQUIRE(delta < 1e-4F);
        }
    }
    std::cerr << "native/ONNX maximum probability delta: " << maximum_delta << '\n';
}
#endif

int main(int argc, char** argv) {
    if (argc > 1) test_options.model_directory = argv[1];
    if (argc > 2) {
        const std::string mode = argv[2];
        if (mode == "cuda-strict" || mode == "cuda-optimized" || mode == "cuda-bf16")
            test_options.provider = jevt::laya_native_provider::cuda;
        else if (mode != "cpu") { std::cerr << "unknown native test mode\n"; return 2; }
        if (mode == "cuda-optimized") test_options.precision = jevt::laya_native_precision::optimized_fp32;
        if (mode == "cuda-bf16") test_options.precision = jevt::laya_native_precision::bf16;
    }
    if (argc > 3) onnx_directory = argv[3];
    if (test_options.model_directory.empty())
        std::cerr << "[ SKIPPED ] native model integration; pass MODEL_DIRECTORY [cpu|cuda-strict|cuda-optimized|cuda-bf16] [ONNX_DIRECTORY]\n";
#ifdef JEVT_NATIVE_TEST_WITH_ORT
    if (onnx_directory.empty()) std::cerr << "[ SKIPPED ] native/ONNX parity; no ONNX_DIRECTORY supplied\n";
#endif
    return jevt::test::run_all("laya native");
}
