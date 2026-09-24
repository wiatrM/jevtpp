#include <jevt/batching.hpp>

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace jevt {
namespace {
thread_local const void* executing_pool = nullptr;
using response_future = std::future<result<inference_response>>;
}

struct batching_backend::impl {
    struct job {
        std::string decision_id, question, input;
        std::vector<std::string> options;
        std::vector<std::string_view> option_views;
        inference_request::kind kind;
        std::size_t bytes;
        std::chrono::steady_clock::time_point arrival = std::chrono::steady_clock::now();
        std::promise<result<inference_response>> promise;

        explicit job(const inference_request& r)
            : decision_id(r.decision_id), question(r.question), input(r.input),
              kind(r.question_kind), bytes(decision_id.size() + question.size() + input.size()) {
            options.reserve(r.options.size());
            for (auto option : r.options) { options.emplace_back(option); bytes += option.size(); }
            option_views.reserve(options.size());
            for (const auto& option : options) option_views.push_back(option);
        }
        inference_request view() const { return {decision_id, question, input, option_views, kind}; }
    };

    std::shared_ptr<backend> wrapped;
    batching_options options;
    std::string label;
    mutable std::mutex mutex;
    std::mutex join_mutex;
    std::condition_variable changed;
    std::deque<std::unique_ptr<job>> queue;
    std::vector<std::thread> workers;
    batching_stats counters;
    bool stopping = false;

    impl(std::shared_ptr<backend> b, batching_options o) : wrapped(std::move(b)), options(o) {
        if (!wrapped || !o.worker_count || !o.queue_capacity || !o.max_batch_size || o.max_delay.count() < 0)
            throw std::invalid_argument("invalid batching backend configuration");
        label = "batching(" + std::string(wrapped->name()) + ")";
        try {
            workers.reserve(o.worker_count);
            for (std::size_t i = 0; i < o.worker_count; ++i) workers.emplace_back([this] { run(); });
        } catch (...) { shutdown(); throw; }
    }

    std::vector<response_future> enqueue(std::span<const inference_request> requests) {
        std::vector<response_future> futures;
        futures.reserve(requests.size());
        std::unique_lock lock(mutex);
        if (stopping || requests.size() > options.queue_capacity - queue.size()) {
            counters.rejected += requests.size();
            for (std::size_t i = 0; i < requests.size(); ++i) {
                std::promise<result<inference_response>> promise;
                futures.push_back(promise.get_future());
                promise.set_value(error{stopping ? error_code::shutting_down : error_code::overloaded,
                    stopping ? "batching backend is shutting down" : "batching backend queue is full"});
            }
            return futures;
        }
        // Construct everything before admission; failures leave no partially admitted batch.
        std::vector<std::unique_ptr<job>> pending;
        pending.reserve(requests.size());
        for (const auto& request : requests) {
            pending.push_back(std::make_unique<job>(request));
            futures.push_back(pending.back()->promise.get_future());
        }
        const auto old_size = queue.size();
        try { for (auto& item : pending) queue.push_back(std::move(item)); }
        catch (...) { while (queue.size() > old_size) queue.pop_back(); throw; }
        counters.accepted += requests.size();
        lock.unlock();
        changed.notify_all();
        return futures;
    }

    std::size_t bucket(const job& item) const {
        return options.length_bucket_width ? item.bytes / options.length_bucket_width : 0;
    }

    void run() {
        executing_pool = this;
        for (;;) {
            std::vector<std::unique_ptr<job>> batch;
            std::vector<inference_request> views;
            std::unique_lock lock(mutex);
            changed.wait(lock, [&] { return stopping || !queue.empty(); });
            if (queue.empty()) break;
            // Always anchor on the oldest item, so buckets cannot starve requests.
            auto ready = [&] {
                if (stopping || queue.empty()) return true;
                const auto key = bucket(*queue.front());
                return static_cast<std::size_t>(std::count_if(queue.begin(), queue.end(),
                    [&](const auto& item) { return bucket(*item) == key; })) >= options.max_batch_size;
            };
            const auto deadline = queue.front()->arrival + options.max_delay;
            changed.wait_until(lock, deadline, ready);
            // Another worker may have consumed our anchor while we waited.
            if (queue.empty()) continue;
            const auto key = bucket(*queue.front());
            batch.reserve(options.max_batch_size);
            views.reserve(options.max_batch_size);
            for (auto it = queue.begin(); it != queue.end() && batch.size() < options.max_batch_size;) {
                if (bucket(**it) == key) {
                    batch.push_back(std::move(*it));
                    it = queue.erase(it);
                } else ++it;
            }
            counters.in_flight += batch.size();
            ++counters.batches;
            lock.unlock();
            changed.notify_all();
            try {
                for (const auto& item : batch) views.push_back(item->view());
                auto responses = wrapped->predict_batch(views);
                if (!responses) {
                    for (auto& item : batch) item->promise.set_value(responses.error_value());
                } else if (responses->size() != batch.size()) {
                    for (auto& item : batch) item->promise.set_value(error{
                        error_code::invalid_backend_output, "batching backend response count mismatch"});
                } else {
                    for (std::size_t i = 0; i < batch.size(); ++i)
                        batch[i]->promise.set_value(std::move((*responses)[i]));
                }
            } catch (const std::exception& e) {
                for (auto& item : batch) item->promise.set_value(error{error_code::backend_failure, e.what()});
            } catch (...) {
                for (auto& item : batch) item->promise.set_value(error{
                    error_code::backend_failure, "wrapped backend threw an unknown exception"});
            }
            lock.lock();
            counters.in_flight -= batch.size();
            counters.completed += batch.size();
        }
        executing_pool = nullptr;
    }

    void shutdown() {
        if (executing_pool == this) throw std::logic_error("cannot shut down batching backend from its worker");
        { std::lock_guard lock(mutex); stopping = true; }
        changed.notify_all();
        std::lock_guard join_lock(join_mutex);
        for (auto& worker : workers) if (worker.joinable()) worker.join();
    }
};

batching_backend::batching_backend(std::shared_ptr<backend> b, batching_options options)
    : impl_(std::make_unique<impl>(std::move(b), options)) {}
batching_backend::~batching_backend() { impl_->shutdown(); }
std::future<result<inference_response>> batching_backend::submit(const inference_request& r) {
    auto futures = impl_->enqueue(std::span(&r, 1));
    return std::move(futures.front());
}
result<inference_response> batching_backend::predict(const inference_request& r) {
    if (executing_pool == impl_.get()) return error{error_code::invalid_request, "recursive batching predict"};
    return submit(r).get();
}
result<std::vector<inference_response>> batching_backend::predict_batch(std::span<const inference_request> r) {
    if (executing_pool == impl_.get()) return error{error_code::invalid_request, "recursive batching predict_batch"};
    auto futures = impl_->enqueue(r);
    std::vector<inference_response> responses;
    responses.reserve(r.size());
    std::optional<error> failure;
    for (auto& future : futures) {
        auto response = future.get();
        if (!response) { if (!failure) failure = response.error_value(); }
        else responses.push_back(std::move(response).value());
    }
    if (failure) return *failure;
    return responses;
}
std::string_view batching_backend::name() const noexcept { return impl_->label; }
batching_stats batching_backend::stats() const {
    std::lock_guard lock(impl_->mutex);
    auto snapshot = impl_->counters;
    snapshot.queued = impl_->queue.size();
    return snapshot;
}
void batching_backend::shutdown() { impl_->shutdown(); }

} // namespace jevt
