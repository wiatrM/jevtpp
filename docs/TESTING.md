# Model-free application tests

`<jevt/testing.hpp>` provides `jevt::testing::scripted_backend`. It implements the
ordinary backend interface and needs no model, provider credentials, network, or
inference runtime. Link the core `jevt::jevt` target.

```cpp
#include <jevt/testing.hpp>

auto backend = std::make_shared<jevt::testing::scripted_backend>(
    std::vector<jevt::testing::scripted_step>{
        {jevt::inference_response{{0.05f, 0.95f}, "fixture"}},
        {jevt::error{jevt::error_code::backend_failure, "injected failure"}}
    });

jevt::inference_request request{"routing", "Which team?", "Login failed", {}};
auto success = backend->predict(request);
auto failure = backend->predict(request);
auto snapshot = backend->snapshot();
// snapshot.batches.size() == 2
// snapshot.batches[0][0].view().input == "Login failed"
```

Use the same shared backend anywhere a typed engine or `SystemOne` builder accepts
a `shared_ptr<jevt::backend>`. Response metadata is retained, so fixtures can
exercise provider/model revision, usage, confidence, score, and attempt-count
handling as well as decision scores. Errors retain their code, message, and
optional status code.

There is one script step per request row. A batch atomically reserves all its
steps in arrival order. If the script is too short, the call returns
`invalid_backend_output` with `scripted backend exhausted`, consumes no steps,
and still records the attempted batch. A failed row makes the batch return the
first row error; all reserved steps are consumed. Empty batches consume no steps.

`push(step)` appends a response or error. `reset(new_script)` atomically replaces
the remaining script, clears recorded batches, and resets the consumed counter.
`snapshot()` returns an independent owned copy containing `batches`,
`remaining_steps`, and `consumed_steps`. Recorded request objects expose `view()`;
those views remain valid while the snapshot object remains alive and unchanged.
All request fields are owned, including options, instructions, and structured
criterion metadata. Input/metadata kinds, deadline, and cancellation token are
preserved. A recorded token reflects subsequent stop requests on its source.

Each `scripted_step` can specify a nonnegative `std::chrono::milliseconds` delay.
The delay occurs outside the script mutex. It is useful for testing event-loop
responsiveness; use a promise/latch-controlled `function_backend` when a test
needs precise ordering instead of a timing assumption. Cancellation and deadline
are checked before and after a scripted row's delay. The delay is not
interruptible; late or cancelled responses are discarded after it finishes.

Calls, script updates, and snapshots are thread safe. Concurrent calls reserve
their rows according to mutex acquisition order, not thread creation order.
Reset does not cancel calls that already reserved their responses: they finish
with those responses, while future calls use the new script. Join active callers
before reset if a test needs a clean, deterministic phase boundary.

The `testing`, `batching`, and optional `asio` test suites exercise script
consumption, snapshot ownership, concurrent callers, queue admission, request
metadata lifetime, cancellation/deadline checks, and callback/coroutine behavior
without downloading models.
