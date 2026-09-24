#pragma once

#include <jevt/core.hpp>
#include <chrono>
#include <future>
#include <stop_token>

namespace jevt {

namespace detail {
// Shared ownership primitive for asynchronous adapters. Every view produced by
// view() remains valid for this object's lifetime (unless assigned to).
class owned_inference_request {
public:
    explicit owned_inference_request(const inference_request& request)
        : decision_id_(request.decision_id), question_(request.question), input_(request.input),
          instructions_(request.instructions.content), kind_(request.question_kind),
          input_kind_(request.input_kind), instruction_kind_(request.instructions.kind),
          deadline_(request.deadline), cancellation_(request.cancellation) {
        options_.reserve(request.options.size());
        for (auto option : request.options) options_.emplace_back(option);
        for (const auto& option : options_) option_views_.push_back(option);
        criteria_.reserve(request.criteria_metadata.size());
        for (const auto& criterion : request.criteria_metadata) criteria_.emplace_back(criterion.content);
        for (std::size_t i = 0; i < criteria_.size(); ++i)
            criterion_views_.push_back({criteria_[i], request.criteria_metadata[i].kind});
    }
    owned_inference_request(const owned_inference_request& other) : owned_inference_request(other.view()) {}
    owned_inference_request(owned_inference_request&&) noexcept = default;
    owned_inference_request& operator=(owned_inference_request&&) noexcept = default;
    owned_inference_request& operator=(const owned_inference_request& other) {
        if (this != &other) *this = owned_inference_request(other);
        return *this;
    }
    [[nodiscard]] inference_request view() const {
        inference_request request{decision_id_, question_, input_, option_views_, kind_};
        request.input_kind = input_kind_;
        request.instructions = {instructions_, instruction_kind_};
        request.criteria_metadata = criterion_views_;
        request.deadline = deadline_;
        request.cancellation = cancellation_;
        return request;
    }
    [[nodiscard]] std::size_t bytes() const noexcept {
        std::size_t size = decision_id_.size() + question_.size() + input_.size() + instructions_.size();
        for (const auto& option : options_) size += option.size();
        for (const auto& criterion : criteria_) size += criterion.size();
        return size;
    }
private:
    std::string decision_id_, question_, input_, instructions_;
    std::vector<std::string> options_, criteria_;
    std::vector<std::string_view> option_views_;
    std::vector<content_view> criterion_views_;
    inference_request::kind kind_;
    content_kind input_kind_, instruction_kind_;
    std::optional<std::chrono::steady_clock::time_point> deadline_;
    std::stop_token cancellation_;
};

[[nodiscard]] inline std::optional<error> request_preflight(const inference_request& request) {
    return execution_error(request);
}
} // namespace detail

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
    // The callback runs on a worker, or inline for admission rejection. It must
    // not block, destroy the pool, or throw. Callback exceptions are contained.
    // This low-level hook lets event-loop adapters post without waiter threads.
    void submit_callback(const inference_request&, std::function<void(result<inference_response>)>);
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
