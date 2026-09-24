#pragma once

// utility precedes Boost 1.74 awaitable.hpp, which uses std::exchange.
#include <utility>
#include <jevt/batching.hpp>
#include <boost/asio.hpp>

namespace jevt {

// Completion signature: void(result<inference_response>). Always posts to the
// handler's associated executor (or the executor supplied to this constructor).
// The execution_context must outlive pending operations, as with other Asio I/O.
class asio_backend {
public:
    explicit asio_backend(boost::asio::any_io_executor executor, std::shared_ptr<backend> backend,
                          batching_options options = {})
        : executor_(std::move(executor)), pool_(std::make_shared<batching_backend>(std::move(backend), options)) {}

    template <class CompletionToken>
    auto async_predict(const inference_request& request, CompletionToken&& token) const {
        // Copy now, not when a deferred token (including use_awaitable) initiates
        // the operation. The caller may release every original view immediately.
        auto owned = std::make_shared<detail::owned_inference_request>(request);
        return boost::asio::async_initiate<CompletionToken, void(result<inference_response>)>(
            [pool = pool_, executor = executor_, owned = std::move(owned)](auto&& handler) mutable {
                using handler_type = std::decay_t<decltype(handler)>;
                auto associated = boost::asio::get_associated_executor(handler, executor);
                using executor_type = decltype(associated);
                struct operation {
                    handler_type handler;
                    executor_type executor;
                    boost::asio::executor_work_guard<executor_type> work;
                    std::shared_ptr<batching_backend> pool;
                    operation(handler_type h, executor_type ex, std::shared_ptr<batching_backend> p)
                        : handler(std::move(h)), executor(ex), work(ex), pool(std::move(p)) {}
                };
                auto state = std::make_shared<operation>(std::forward<decltype(handler)>(handler), associated, pool);
                pool->submit_callback(owned->view(), [state = std::move(state)](result<inference_response> response) mutable {
                    // Transfer the LAST callback-owned reference into the posted
                    // handler. Keeping a worker-side copy could destroy/join the
                    // pool on its own worker if completion executes immediately.
                    auto executor = state->executor;
                    boost::asio::post(executor,
                        [state = std::move(state), response = std::move(response)]() mutable {
                            auto handler = std::move(state->handler);
                            state->work.reset();
                            handler(std::move(response));
                        });
                });
            }, token);
    }

    [[nodiscard]] batching_stats stats() const { return pool_->stats(); }
    // Explicitly blocking drain: do not call on the I/O thread while responses
    // are outstanding. Destroying the handle itself need not drain pending work;
    // each operation retains the pool until its posted completion is delivered.
    void shutdown() { pool_->shutdown(); }

private:
    boost::asio::any_io_executor executor_;
    std::shared_ptr<batching_backend> pool_;
};

} // namespace jevt
