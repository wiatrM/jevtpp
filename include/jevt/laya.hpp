#pragma once

#include "jevt/core.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace jevt {

struct laya_options {
    std::filesystem::path model_directory;
    std::filesystem::path model_file;
    std::filesystem::path tokenizer_file;
    std::filesystem::path tokenizer_config_file;
    std::filesystem::path agent_config_file;
    int intra_op_threads = 0;
    int inter_op_threads = 0;
    bool enable_graph_optimizations = true;
    std::string model_id = "laya-onnx";
};

// Native Laya inference backend. It reproduces the upstream sequence layout,
// option markers, cardinality-aware temperature calibration and tensor
// contract while keeping ONNX Runtime and tokenizer details out of public ABI.
class laya_backend final : public backend {
public:
    explicit laya_backend(laya_options options);
    ~laya_backend() override;

    laya_backend(const laya_backend&) = delete;
    laya_backend& operator=(const laya_backend&) = delete;
    laya_backend(laya_backend&&) noexcept;
    laya_backend& operator=(laya_backend&&) noexcept;

    [[nodiscard]] result<inference_response> predict(const inference_request&) override;
    [[nodiscard]] result<std::vector<inference_response>> predict_batch(
        std::span<const inference_request>) override;
    [[nodiscard]] std::string_view name() const noexcept override;

    [[nodiscard]] std::size_t max_context_tokens() const noexcept;
    [[nodiscard]] std::size_t max_question_tokens() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

namespace models {
[[nodiscard]] std::shared_ptr<backend> laya_multilingual(
    std::filesystem::path model_directory, int compute_threads = 0);
}

} // namespace jevt
