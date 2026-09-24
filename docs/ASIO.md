# Boost.Asio inference

`jevt::asio_backend` provides a completion-token API over a bounded inference
pool. Include `<jevt/asio.hpp>` and link `jevt::asio`. Configure with
`JEVT_ENABLE_ASIO=ON`; Boost 1.74 or newer and C++20 are required. No Boost
dependency is introduced by the core library or `<jevt/jevt.hpp>`.

```cpp
#include <jevt/asio.hpp>
#include <jevt/testing.hpp>

boost::asio::io_context io;
auto model = std::make_shared<jevt::testing::scripted_backend>(
    std::vector<jevt::testing::scripted_step>{
        {jevt::inference_response{{0.1f, 0.9f}, "fixture"}}
    });
jevt::asio_backend inference(io.get_executor(), model,
    jevt::batching_options{.worker_count = 1, .queue_capacity = 128,
                          .max_batch_size = 8});
jevt::inference_request request{"routing", "Which team?", "Login failed", {}};

inference.async_predict(request, [](jevt::result<jevt::inference_response> result) {
    // Runs on io's executor. Inspect result before reading result->scores.
});
io.run();
```

The completion signature is `void(result<inference_response>)`. Failures are
returned in `result`, including `overloaded`, `timeout`, and `cancelled`; they are
not converted to `boost::system::error_code`. Callback completion is always
posted, including queue rejection. A handler bound with
`boost::asio::bind_executor` completes on that associated executor. Otherwise the
constructor's executor is used. An executor work guard keeps its `run()` alive
until completion is ready. Move-only completion handlers are supported.

## Coroutines

`use_awaitable` returns an awaitable of the same result type:

```cpp
boost::asio::awaitable<void> classify(jevt::asio_backend& inference) {
    jevt::inference_request request{"routing", "Which team?", "Login failed", {}};
    auto result = co_await inference.async_predict(request, boost::asio::use_awaitable);
    if (!result) co_return;
    // Use result->scores and result->metadata here.
}
```

The adapter copies all request data when `async_predict` is called, even when a
token defers initiation. Decision ID, question, input, options, instructions, and
criterion metadata can therefore be released immediately after that call.
Metadata kinds, deadline, and stop token are preserved. Discarding an unstarted
awaitable does not submit inference.

Backend execution occurs only on the bounded pool's workers. The adapter creates
no waiter thread or `std::async` job per request. Queue capacity counts request
rows. More than one worker requires a wrapped backend that supports concurrent
calls. The low-level `batching_backend::submit_callback` hook invokes callbacks
on workers, or inline for rejection; the Asio adapter uses this hook only to post
the actual completion to the caller's executor.

## Cancellation, deadlines, and lifetime

Set `request.cancellation` to a `std::stop_source` token and optionally set
`request.deadline` to a `std::chrono::steady_clock::time_point`. Workers check both
before executing a row. A cancelled or expired queued row is completed with the
corresponding error and is excluded from the backend batch.

These checks do not provide a hard interrupt. An already running local
inference call may run to completion; its result is discarded if cancellation or
deadline expiry is observed when it returns. The adapter reports `cancelled` or
`timeout` without claiming that the underlying computation was aborted. A
backend with its own cooperative controls receives the same request controls
and may honor them during execution. Queued cancellation is observed when a
worker dispatches the row, so completion is not guaranteed at the deadline.
Boost cancellation slots are not registered by this adapter; the portable
control is `std::stop_token`, including on Boost 1.74.

Each pending operation retains the pool and wrapped backend. Destroying the
`asio_backend` handle while requests are queued or running is supported; their
completion handlers still run. Keep each associated execution context alive
until its operations finish. Stopping an `io_context` delays delivery until it is
restarted and run again. This is the usual Asio context-lifetime requirement.

`shutdown()` is an explicit blocking drain and rejects subsequent submissions.
Call it from a management thread, not an event-loop callback while inference is
outstanding. Backend calls must eventually return. Completion-handler exceptions
propagate from `io_context::run()` as normal Asio handler exceptions.

The current API operates on `inference_request` and returns backend scores and
metadata. Existing typed `SystemOne` evaluation stays synchronous: merely
putting its synchronous `evaluate` call inside a coroutine would still block
that coroutine's event loop. Use this low-level async API where nonblocking
integration is needed; there is no typed async `SystemOne` wrapper in this API.
