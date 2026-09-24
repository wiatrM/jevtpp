#include <jevt/testing.hpp>
#include "test_harness.hpp"
#include <atomic>

using namespace std::chrono_literals;
namespace {
jevt::inference_request request(std::string_view input = "input") { return {"id", "question", input, {}}; }
jevt::testing::scripted_step answer(std::string id, std::chrono::milliseconds delay = 0ms) {
    return {jevt::inference_response{{.2f, .8f}, std::move(id)}, delay};
}
}

JEVT_TEST("scripted backend consumes rows atomically and records exhausted calls") {
    jevt::testing::scripted_backend backend({answer("first"), answer("second")});
    std::array requests{request("one"), request("two"), request("three")};
    auto exhausted = backend.predict_batch(requests);
    JEVT_REQUIRE(!exhausted);
    JEVT_REQUIRE_EQ(exhausted.error_value().code, jevt::error_code::invalid_backend_output);
    JEVT_REQUIRE_EQ(backend.snapshot().remaining_steps, 2u);
    auto output = backend.predict_batch(std::span(requests).first(2));
    JEVT_REQUIRE(output);
    JEVT_REQUIRE_EQ((*output)[0].model_id, "first");
    JEVT_REQUIRE_EQ((*output)[1].model_id, "second");
    const auto recorded = backend.snapshot();
    JEVT_REQUIRE_EQ(recorded.batches.size(), 2u);
    JEVT_REQUIRE_EQ(recorded.consumed_steps, 2u);
    JEVT_REQUIRE_EQ(recorded.remaining_steps, 0u);
    JEVT_REQUIRE_EQ(recorded.batches[1][1].view().input, "two");
}

JEVT_TEST("recorded requests own structured metadata through copies and moves") {
    jevt::testing::scripted_backend backend({answer("answer")});
    std::stop_source source;
    const auto deadline = std::chrono::steady_clock::now() + 1h;
    {
        std::string input = "{\"ticket\":1}", instructions = "{\"policy\":2}", criterion = "{\"criterion\":3}", option = "option";
        const std::array<std::string_view, 1> options{option};
        const std::array<jevt::content_view, 1> criteria{jevt::json_metadata(criterion)};
        auto value = request(input);
        value.options = options;
        value.input_kind = jevt::content_kind::json;
        value.instructions = jevt::json_metadata(instructions);
        value.criteria_metadata = criteria;
        value.deadline = deadline;
        value.cancellation = source.get_token();
        JEVT_REQUIRE(backend.predict(value));
        input.clear(); instructions.clear(); criterion.clear(); option.clear();
    }
    auto snapshot = backend.snapshot();
    auto copied = snapshot;
    snapshot.batches.clear();
    auto moved = std::move(copied);
    source.request_stop();
    const auto recorded = moved.batches[0][0].view();
    JEVT_REQUIRE_EQ(recorded.input, "{\"ticket\":1}");
    JEVT_REQUIRE_EQ(recorded.instructions.content, "{\"policy\":2}");
    JEVT_REQUIRE_EQ(recorded.criteria_metadata[0].content, "{\"criterion\":3}");
    JEVT_REQUIRE_EQ(recorded.options[0], "option");
    JEVT_REQUIRE_EQ(recorded.input_kind, jevt::content_kind::json);
    JEVT_REQUIRE_EQ(recorded.instructions.kind, jevt::content_kind::json);
    JEVT_REQUIRE_EQ(recorded.criteria_metadata[0].kind, jevt::content_kind::json);
    JEVT_REQUIRE_EQ(recorded.deadline, deadline);
    JEVT_REQUIRE(recorded.cancellation.stop_requested());
}

JEVT_TEST("scripted errors preserve response metadata and reset is deterministic") {
    jevt::testing::scripted_backend backend({{jevt::error{jevt::error_code::rate_limited, "try later", 429}, 0ms}});
    const auto failure = backend.predict(request());
    JEVT_REQUIRE_EQ(failure.error_value().status_code, 429);
    auto step = answer("new");
    step.response->metadata.provider = "fixture";
    step.response->metadata.usage = std::make_shared<const jevt::token_usage>(jevt::token_usage{10, 2});
    backend.reset({step});
    JEVT_REQUIRE(backend.snapshot().batches.empty());
    JEVT_REQUIRE_EQ(backend.snapshot().consumed_steps, 0u);
    auto result = backend.predict(request());
    JEVT_REQUIRE_EQ(result->model_id, "new");
    JEVT_REQUIRE_EQ(result->metadata.provider, "fixture");
    JEVT_REQUIRE_EQ(result->metadata.usage->input_tokens, 10u);
    backend.push(answer("pushed"));
    JEVT_REQUIRE_EQ(backend.predict(request())->model_id, "pushed");
}

JEVT_TEST("scripted concurrent calls consume every step once") {
    std::vector<jevt::testing::scripted_step> steps;
    for (int i = 0; i < 32; ++i) steps.push_back(answer(std::to_string(i)));
    jevt::testing::scripted_backend backend(std::move(steps));
    std::vector<std::future<std::string>> calls;
    for (int i = 0; i < 32; ++i) calls.push_back(std::async(std::launch::async, [&] {
        return backend.predict(request())->model_id;
    }));
    std::vector<std::string> outputs;
    for (auto& call : calls) outputs.push_back(call.get());
    std::sort(outputs.begin(), outputs.end());
    JEVT_REQUIRE_EQ(std::unique(outputs.begin(), outputs.end()), outputs.end());
    JEVT_REQUIRE_EQ(backend.snapshot().batches.size(), 32u);
    JEVT_REQUIRE_EQ(backend.snapshot().consumed_steps, 32u);
}

JEVT_TEST("bounded worker owns new metadata and rejects expired or cancelled queued rows") {
    std::promise<void> entered, release;
    auto released = release.get_future().share();
    auto script = std::make_shared<jevt::testing::scripted_backend>(
        std::vector<jevt::testing::scripted_step>{answer("owned")});
    std::atomic<int> calls = 0;
    auto wrapped = std::make_shared<jevt::function_backend>([&](const auto& value) -> jevt::result<jevt::inference_response> {
        if (calls.fetch_add(1) == 0) { entered.set_value(); released.wait(); return jevt::inference_response{{1}, "gate"}; }
        return script->predict(value);
    });
    jevt::batching_backend pool(wrapped, {.max_batch_size = 1, .max_delay = 0us});
    struct guard { std::promise<void>& release; ~guard() { try { release.set_value(); } catch (...) {} } } cleanup{release};
    auto gate = pool.submit(request());
    entered.get_future().wait();
    std::future<jevt::result<jevt::inference_response>> owned;
    {
        std::string instruction = "json instruction", criterion = "json criterion";
        std::array<jevt::content_view, 1> criteria{jevt::json_metadata(criterion)};
        auto value = request();
        value.input_kind = jevt::content_kind::json;
        value.instructions = jevt::json_metadata(instruction);
        value.criteria_metadata = criteria;
        owned = pool.submit(value);
    }
    std::stop_source cancellation;
    auto cancelled_request = request();
    cancelled_request.cancellation = cancellation.get_token();
    auto cancelled = pool.submit(cancelled_request);
    cancellation.request_stop();
    auto expired_request = request();
    expired_request.deadline = std::chrono::steady_clock::now() - 1ms;
    auto expired = pool.submit(expired_request);
    release.set_value();
    JEVT_REQUIRE(gate.get());
    JEVT_REQUIRE(owned.get());
    JEVT_REQUIRE_EQ(cancelled.get().error_value().code, jevt::error_code::cancelled);
    JEVT_REQUIRE_EQ(expired.get().error_value().code, jevt::error_code::timeout);
    pool.shutdown();
    JEVT_REQUIRE_EQ(calls.load(), 2);
    const auto recorded = script->snapshot();
    const auto view = recorded.batches[0][0].view();
    JEVT_REQUIRE_EQ(view.instructions.content, "json instruction");
    JEVT_REQUIRE_EQ(view.criteria_metadata[0].content, "json criterion");
    JEVT_REQUIRE_EQ(view.input_kind, jevt::content_kind::json);
    JEVT_REQUIRE_EQ(pool.stats().completed, 4u);
    JEVT_REQUIRE_EQ(pool.stats().batches, 2u);
}

JEVT_TEST("callback completion exceptions do not kill bounded workers") {
    auto backend = std::make_shared<jevt::testing::scripted_backend>(
        std::vector<jevt::testing::scripted_step>{answer("first"), answer("next")});
    jevt::batching_backend pool(backend, {.max_batch_size = 1, .max_delay = 0us});
    pool.submit_callback(request(), [](auto) { throw std::runtime_error("callback violation"); });
    JEVT_REQUIRE_EQ(pool.predict(request())->model_id, "next");
}

JEVT_TEST("bounded completion discards cancelled results from an uncooperative backend") {
    struct uncooperative final : jevt::backend {
        std::promise<void> entered, release;
        std::shared_future<void> released = release.get_future().share();
        jevt::result<jevt::inference_response> predict(const jevt::inference_request&) override {
            return jevt::inference_response{{1}, "uncooperative"};
        }
        jevt::result<std::vector<jevt::inference_response>> predict_batch(std::span<const jevt::inference_request> values) override {
            entered.set_value(); released.wait();
            return std::vector<jevt::inference_response>(values.size(), {{1}, "uncooperative"});
        }
        std::string_view name() const noexcept override { return "uncooperative"; }
    };
    auto backend = std::make_shared<uncooperative>();
    jevt::batching_backend pool(backend, {.max_delay = 0us});
    struct guard { std::promise<void>& release; ~guard() { try { release.set_value(); } catch (...) {} } } cleanup{backend->release};
    std::stop_source source;
    auto value = request(); value.cancellation = source.get_token();
    auto pending = pool.submit(value);
    backend->entered.get_future().wait();
    source.request_stop();
    JEVT_REQUIRE_EQ(pending.wait_for(0ms), std::future_status::timeout);
    backend->release.set_value();
    JEVT_REQUIRE_EQ(pending.get().error_value().code, jevt::error_code::cancelled);
}

int main() { return jevt::test::run_all("testing"); }
