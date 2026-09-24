#pragma once

#include <jevt/core.hpp>
#include <filesystem>
#include <memory>
#include <string>

namespace jevt {

enum class laya_native_provider { cpu, cuda };
enum class laya_native_precision { strict_fp32, optimized_fp32, bf16 };

struct laya_native_options {
    // Native safetensors bundle, not an ONNX export. CUDA uses device 0.
    std::filesystem::path model_directory;
    std::string model_id = "laya-native";
    laya_native_provider provider = laya_native_provider::cpu;
    laya_native_precision precision = laya_native_precision::strict_fp32;
    std::size_t context_token_limit = 0;
    std::size_t schema_cache_entries = 256;
    // Retained string/vector capacity, excluding allocator/container metadata.
    std::size_t schema_cache_bytes = 4 * 1024 * 1024;
};

// Optional in-process laya.cpp adapter. Mutable native graph state is protected
// by an instance mutex. Requests are consumed synchronously; only schema heads
// are cached. Native model construction may configure process-wide CUDA math
// policy: initialize before other CUDA users, as required by upstream.
class laya_native_backend final : public backend {
public:
    explicit laya_native_backend(laya_native_options);
    ~laya_native_backend() override;
    laya_native_backend(const laya_native_backend&) = delete;
    laya_native_backend& operator=(const laya_native_backend&) = delete;
    [[nodiscard]] result<inference_response> predict(const inference_request&) override;
    [[nodiscard]] result<std::vector<inference_response>> predict_batch(
        std::span<const inference_request>) override;
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] std::string device_name() const;
    [[nodiscard]] result<std::size_t> warmup(std::span<const inference_request>,
                                           std::size_t iterations = 1);
private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

} // namespace jevt
