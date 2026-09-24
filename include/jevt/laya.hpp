#pragma once

#include "jevt/core.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace jevt {

enum class laya_provider { cpu, cuda };

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
    laya_provider provider = laya_provider::cpu;
    int device_id = 0;
    bool use_tf32 = false;
    // ORT I/O binding with reusable host tensors. ORT still performs device
    // transfers; this is not a zero-copy or CUDA-graph execution mode.
    bool use_io_binding = false;
    bool parallel_execution = false;
    bool allow_spinning = true;
    // Zero preserves the bundle's context budget; never changes head tokens.
    std::size_t context_token_limit = 0;
    std::size_t schema_cache_entries = 256;
    std::size_t schema_cache_bytes = 4 * 1024 * 1024;
    std::size_t reusable_buffers = 2;
    std::size_t reusable_buffer_bytes = 8 * 1024 * 1024;
};

struct laya_statistics {
    std::uint64_t schema_cache_hits{}, schema_cache_misses{}, buffer_reuses{}, runs{};
    std::size_t cached_schemas{}, cached_schema_bytes{}, pooled_buffers{};
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
    // Reuses this session and validates representative requests, propagating
    // errors. Warmup never invents input or excludes itself from run counters.
    [[nodiscard]] result<std::size_t> warmup(std::span<const inference_request>, std::size_t iterations = 1);
    [[nodiscard]] laya_statistics statistics() const;
    [[nodiscard]] laya_provider provider() const noexcept;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

namespace models {
[[nodiscard]] std::shared_ptr<backend> laya_multilingual(
    std::filesystem::path model_directory, int compute_threads = 0);
}

} // namespace jevt
