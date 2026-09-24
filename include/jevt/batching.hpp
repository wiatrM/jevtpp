#pragma once

#include <jevt/core.hpp>
#include <chrono>
#include <future>

namespace jevt {

struct batching_options {
    std::size_t worker_count = 1;
    // Waiting requests only; at most worker_count * max_batch_size also execute.
    std::size_t queue_capacity = 256;
    std::size_t max_batch_size = 16;
    std::chrono::microseconds max_delay{1000};
    // Zero disables bucketing. Otherwise group by total request text bytes / width.
    std::size_t length_bucket_width = 0;
};

struct batching_stats {
    std::size_t queued = 0;
    std::size_t in_flight = 0;
    std::uint64_t accepted = 0;
    std::uint64_t rejected = 0;
    std::uint64_t completed = 0;
    std::uint64_t batches = 0;
};

// Thread-safe bounded executor. All request text (including options) is copied
// before submit returns. More than one worker requires a thread-safe wrapped
// backend. Batch results remain in caller order, even with length bucketing.
// shutdown rejects new work and drains accepted work; concurrent shutdown calls
// are supported. It waits for the wrapped backend, which must eventually return.
// Do not destroy this object from a wrapped-backend callback or while callers
// are still accessing it. Recursive synchronous calls/shutdown from its workers
// are rejected to prevent a worker waiting on itself.
class batching_backend final : public backend {
public:
    explicit batching_backend(std::shared_ptr<backend>, batching_options = {});
    ~batching_backend() override;
    batching_backend(const batching_backend&) = delete;
    batching_backend& operator=(const batching_backend&) = delete;

    [[nodiscard]] std::future<result<inference_response>> submit(const inference_request&);
    [[nodiscard]] result<inference_response> predict(const inference_request&) override;
    // Admission is atomic: an oversized/full-queue batch rejects every request.
    [[nodiscard]] result<std::vector<inference_response>> predict_batch(
        std::span<const inference_request>) override;
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] batching_stats stats() const;
    void shutdown();

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

} // namespace jevt
