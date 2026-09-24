#include <jevt/asio.hpp>
#include <jevt/testing.hpp>
#include "test_harness.hpp"
#include <atomic>

using namespace std::chrono_literals;
namespace {
jevt::inference_request request(std::string_view input = "input") { return {"id", "question", input, {}}; }
jevt::testing::scripted_step answer(std::string id, std::chrono::milliseconds delay = 0ms) {
    return {jevt::inference_response{{.1f, .9f}, std::move(id)}, delay};
}
struct gate {
    std::promise<void> entered, release;
    std::shared_future<void> released = release.get_future().share();
    std::atomic<int> calls = 0;
    std::shared_ptr<jevt::backend> backend() {
        return std::make_shared<jevt::function_backend>([this](const auto&) -> jevt::result<jevt::inference_response> {
            if (calls.fetch_add(1) == 0) { entered.set_value(); released.wait(); }
            return jevt::inference_response{{1}, "gate"};
        });
    }
};
struct release_guard {
    gate& value;
    ~release_guard() { try { value.release.set_value(); } catch (...) {} }
};
}

JEVT_TEST("Asio callback runs on its associated executor and supports move-only handlers") {
    boost::asio::io_context fallback, associated;
    auto script = std::make_shared<jevt::testing::scripted_backend>(
        std::vector<jevt::testing::scripted_step>{answer("response", 1ms)});
    jevt::asio_backend adapter(fallback.get_executor(), script, {.max_delay = 0us});
    bool completed = false;
    adapter.async_predict(request(), boost::asio::bind_executor(associated.get_executor(),
        [owned = std::make_unique<int>(7), &completed, &associated](auto response) {
            JEVT_REQUIRE(response);
            JEVT_REQUIRE_EQ(response->model_id, "response");
            JEVT_REQUIRE_EQ(*owned, 7);
            JEVT_REQUIRE(associated.get_executor().running_in_this_thread());
            completed = true;
        }));
    JEVT_REQUIRE(!completed);
    fallback.poll();
    JEVT_REQUIRE(!completed);
    // The operation's guard keeps run() alive while inference has not posted yet.
    associated.run();
    JEVT_REQUIRE(completed);
}

JEVT_TEST("bounded inference does not block event-loop callbacks") {
    boost::asio::io_context io;
    gate blocked;
    jevt::asio_backend adapter(io.get_executor(), blocked.backend(), {.max_delay = 0us});
    release_guard cleanup{blocked};
    bool responsive = false, completed = false;
    adapter.async_predict(request(), [&](auto output) { JEVT_REQUIRE(output); completed = true; });
    blocked.entered.get_future().wait();
    boost::asio::post(io, [&] { responsive = true; blocked.release.set_value(); });
    io.run();
    JEVT_REQUIRE(responsive);
    JEVT_REQUIRE(completed);
}

JEVT_TEST("Asio overload is posted and queued cancellation and deadline skip inference") {
    boost::asio::io_context io;
    gate blocked;
    jevt::asio_backend adapter(io.get_executor(), blocked.backend(),
        {.queue_capacity = 2, .max_batch_size = 1, .max_delay = 0us});
    release_guard cleanup{blocked};
    int completed = 0;
    adapter.async_predict(request(), [&](auto response) { JEVT_REQUIRE(response); ++completed; });
    blocked.entered.get_future().wait();
    std::stop_source stop;
    auto cancelled = request();
    cancelled.cancellation = stop.get_token();
    adapter.async_predict(cancelled, [&](auto response) {
        JEVT_REQUIRE_EQ(response.error_value().code, jevt::error_code::cancelled); ++completed;
    });
    auto expired = request();
    expired.deadline = std::chrono::steady_clock::now() - 1ms;
    adapter.async_predict(expired, [&](auto response) {
        JEVT_REQUIRE_EQ(response.error_value().code, jevt::error_code::timeout); ++completed;
    });
    adapter.async_predict(request(), [&](auto response) {
        JEVT_REQUIRE_EQ(response.error_value().code, jevt::error_code::overloaded); ++completed;
    });
    JEVT_REQUIRE_EQ(completed, 0);
    stop.request_stop();
    blocked.release.set_value();
    io.run();
    JEVT_REQUIRE_EQ(completed, 4);
    JEVT_REQUIRE_EQ(blocked.calls.load(), 1);
}

JEVT_TEST("destroyed Asio handle retains backend until late completion is dispatched") {
    boost::asio::io_context io;
    gate blocked;
    auto backend = blocked.backend();
    std::weak_ptr<jevt::backend> weak = backend;
    auto adapter = std::make_unique<jevt::asio_backend>(io.get_executor(), backend,
        jevt::batching_options{.max_delay = 0us});
    release_guard cleanup{blocked};
    bool completed = false;
    adapter->async_predict(request(), [&](auto result) { JEVT_REQUIRE(result); completed = true; });
    blocked.entered.get_future().wait();
    backend.reset();
    adapter.reset(); // Must not synchronously wait for the blocked model.
    JEVT_REQUIRE(!weak.expired());
    io.stop();
    blocked.release.set_value();
    io.restart();
    io.run();
    JEVT_REQUIRE(completed);
    JEVT_REQUIRE(weak.expired());
}

JEVT_TEST("use_awaitable owns views before deferred initiation and preserves metadata") {
    boost::asio::io_context io;
    auto script = std::make_shared<jevt::testing::scripted_backend>(
        std::vector<jevt::testing::scripted_step>{answer("coroutine")});
    auto adapter = std::make_unique<jevt::asio_backend>(io.get_executor(), script,
        jevt::batching_options{.max_delay = 0us});
    std::optional<boost::asio::awaitable<jevt::result<jevt::inference_response>>> operation;
    {
        std::string input = "owned input", instruction = "owned instruction", criterion = "owned criterion", option = "owned option";
        std::array<std::string_view, 1> options{option};
        std::array<jevt::content_view, 1> criteria{jevt::json_metadata(criterion)};
        auto value = request(input);
        value.options = options;
        value.input_kind = jevt::content_kind::json;
        value.instructions = jevt::json_metadata(instruction);
        value.criteria_metadata = criteria;
        operation.emplace(adapter->async_predict(value, boost::asio::use_awaitable));
    }
    adapter.reset(); // The awaitable's initiation state must retain its executor/pool.
    bool completed = false;
    boost::asio::co_spawn(io, [pending = std::move(*operation), &completed]() mutable -> boost::asio::awaitable<void> {
        auto response = co_await std::move(pending);
        JEVT_REQUIRE(response);
        JEVT_REQUIRE_EQ(response->model_id, "coroutine");
        completed = true;
    }, [](std::exception_ptr error) { if (error) std::rethrow_exception(error); });
    operation.reset();
    io.run();
    JEVT_REQUIRE(completed);
    const auto snapshot = script->snapshot();
    const auto recorded = snapshot.batches[0][0].view();
    JEVT_REQUIRE_EQ(recorded.input, "owned input");
    JEVT_REQUIRE_EQ(recorded.options[0], "owned option");
    JEVT_REQUIRE_EQ(recorded.instructions.content, "owned instruction");
    JEVT_REQUIRE_EQ(recorded.criteria_metadata[0].content, "owned criterion");
    JEVT_REQUIRE_EQ(recorded.input_kind, jevt::content_kind::json);
    JEVT_REQUIRE_EQ(recorded.instructions.kind, jevt::content_kind::json);
    JEVT_REQUIRE_EQ(recorded.criteria_metadata[0].kind, jevt::content_kind::json);
}

JEVT_TEST("running local inference is not interrupted but its cancelled result is discarded") {
    boost::asio::io_context io;
    gate blocked;
    jevt::asio_backend adapter(io.get_executor(), blocked.backend(), {.max_delay = 0us});
    release_guard cleanup{blocked};
    std::stop_source source;
    auto value = request(); value.cancellation = source.get_token();
    bool completed = false;
    adapter.async_predict(value, [&](auto response) {
        JEVT_REQUIRE(!response);
        JEVT_REQUIRE_EQ(response.error_value().code, jevt::error_code::cancelled);
        completed = true;
    });
    blocked.entered.get_future().wait();
    source.request_stop();
    // Cancellation alone cannot finish the blocked local call or dispatch its
    // completion. Only releasing the model lets the cancelled result arrive.
    io.poll();
    JEVT_REQUIRE(!completed);
    blocked.release.set_value();
    io.run();
    JEVT_REQUIRE(completed);
}

int main() { return jevt::test::run_all("asio"); }
