#pragma once

#include <jevt/batching.hpp>
#include <deque>
#include <mutex>
#include <thread>

namespace jevt::testing {

using recorded_request = detail::owned_inference_request;

struct scripted_step {
    result<inference_response> response = inference_response{};
    std::chrono::milliseconds delay{0};
};

struct scripted_snapshot {
    std::vector<std::vector<recorded_request>> batches;
    std::size_t remaining_steps = 0;
    std::size_t consumed_steps = 0;
};

// One script step per request row. Each batch atomically reserves its steps in
// arrival order. An undersized script reports invalid_backend_output without
// consuming steps; every attempted batch is still recorded. Delays occur outside
// the mutex. Reset affects future calls/history, not already-reserved steps.
class scripted_backend final : public backend {
public:
    explicit scripted_backend(std::vector<scripted_step> script = {}, std::string name = "scripted")
        : script_(script.begin(), script.end()), name_(std::move(name)) {
        validate(script);
    }

    void push(scripted_step step) {
        if (step.delay.count() < 0) throw std::invalid_argument("scripted delay cannot be negative");
        std::lock_guard lock(mutex_);
        script_.push_back(std::move(step));
    }

    void reset(std::vector<scripted_step> script = {}) {
        validate(script);
        std::deque<scripted_step> replacement(script.begin(), script.end());
        std::lock_guard lock(mutex_);
        script_.swap(replacement);
        batches_.clear();
        consumed_ = 0;
    }

    [[nodiscard]] scripted_snapshot snapshot() const {
        std::lock_guard lock(mutex_);
        return {batches_, script_.size(), consumed_};
    }

    [[nodiscard]] result<inference_response> predict(const inference_request& request) override {
        auto responses = predict_batch(std::span(&request, 1));
        if (!responses) return responses.error_value();
        return std::move(responses->front());
    }

    [[nodiscard]] result<std::vector<inference_response>> predict_batch(
        std::span<const inference_request> requests) override {
        std::vector<scripted_step> selected;
        selected.reserve(requests.size());
        {
            std::lock_guard lock(mutex_);
            std::vector<recorded_request> recorded;
            recorded.reserve(requests.size());
            for (const auto& request : requests) recorded.emplace_back(request);
            batches_.push_back(std::move(recorded));
            if (script_.size() < requests.size())
                return error{error_code::invalid_backend_output, "scripted backend exhausted"};
            for (std::size_t i = 0; i < requests.size(); ++i) {
                selected.push_back(std::move(script_.front()));
                script_.pop_front();
            }
            consumed_ += requests.size();
        }
        std::vector<inference_response> responses;
        responses.reserve(requests.size());
        std::optional<error> first_error;
        for (std::size_t i = 0; i < selected.size(); ++i) {
            auto& step = selected[i];
            if (const auto failure = detail::request_preflight(requests[i])) {
                if (!first_error) first_error = *failure;
                continue;
            }
            if (step.delay.count()) std::this_thread::sleep_for(step.delay);
            if (const auto failure = execution_error(requests[i])) {
                if (!first_error) first_error = *failure;
                continue;
            }
            if (!step.response) { if (!first_error) first_error = step.response.error_value(); }
            else responses.push_back(std::move(step.response).value());
        }
        if (first_error) return *first_error;
        return responses;
    }

    [[nodiscard]] std::string_view name() const noexcept override { return name_; }

private:
    static void validate(const std::vector<scripted_step>& script) {
        for (const auto& step : script) if (step.delay.count() < 0)
            throw std::invalid_argument("scripted delay cannot be negative");
    }
    mutable std::mutex mutex_;
    std::deque<scripted_step> script_;
    std::vector<std::vector<recorded_request>> batches_;
    std::size_t consumed_ = 0;
    const std::string name_;
};

} // namespace jevt::testing
