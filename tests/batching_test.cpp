#include <jevt/batching.hpp>
#include "test_harness.hpp"
#include <atomic>
#include <mutex>

using namespace std::chrono_literals;
namespace {
struct fake_backend final : jevt::backend {
    std::mutex mutex;
    std::vector<std::vector<std::string>> batches;
    std::promise<void> entered, release;
    std::shared_future<void> released = release.get_future().share();
    std::atomic<bool> block_first{true};
    std::atomic<int> behavior{0};
    jevt::result<jevt::inference_response> predict(const jevt::inference_request& r) override {
        return jevt::inference_response{{}, std::string(r.input)};
    }
    jevt::result<std::vector<jevt::inference_response>> predict_batch(
        std::span<const jevt::inference_request> requests) override {
        if (block_first.exchange(false)) { entered.set_value(); released.wait(); }
        if (behavior == 1) throw std::runtime_error("injected exception");
        if (behavior == 2) throw 42;
        if (behavior == 3) return std::vector<jevt::inference_response>{};
        if (behavior == 4) return jevt::error{jevt::error_code::invalid_request, "injected error"};
        std::vector<std::string> inputs;
        std::vector<jevt::inference_response> outputs;
        for (const auto& r : requests) {
            inputs.emplace_back(r.input);
            auto text = std::string(r.decision_id) + ":" + std::string(r.question) + ":" + std::string(r.input);
            for (auto option : r.options) text += ":" + std::string(option);
            outputs.push_back({{static_cast<float>(r.question_kind)}, std::move(text)});
        }
        std::lock_guard lock(mutex);
        batches.push_back(std::move(inputs));
        return outputs;
    }
    std::string_view name() const noexcept override { return "fake"; }
};
struct release_guard {
    std::shared_ptr<fake_backend> fake;
    ~release_guard() { try { fake->release.set_value(); } catch (...) {} }
    void now() { fake->release.set_value(); }
};
jevt::inference_request request(std::string_view input) { return {"id", "q", input, {}}; }
}

JEVT_TEST("queued requests own all text and options") {
    auto fake = std::make_shared<fake_backend>();
    jevt::batching_backend pool(fake, {.max_batch_size = 1, .max_delay = 0us});
    release_guard release{fake};
    auto first = pool.submit(request("first"));
    fake->entered.get_future().wait();
    std::future<jevt::result<jevt::inference_response>> pending;
    {
        std::string id = "owned-id", question = "owned-question", input = "owned-input", option = "owned-option";
        std::vector<std::string_view> options{option};
        pending = pool.submit({id, question, input, options, jevt::inference_request::kind::score});
        id.assign(100, 'x'); question.clear(); input.clear(); option.clear(); options.clear();
    }
    release.now();
    JEVT_REQUIRE(first.get());
    auto output = pending.get();
    JEVT_REQUIRE(output);
    JEVT_REQUIRE_EQ(output->model_id, "owned-id:owned-question:owned-input:owned-option");
    JEVT_REQUIRE_EQ(output->scores[0], 2.0f);
}

JEVT_TEST("bounded queue rejects overload and shutdown drains accepted jobs") {
    auto fake = std::make_shared<fake_backend>();
    jevt::batching_backend pool(fake, {.queue_capacity = 2, .max_batch_size = 1, .max_delay = 0us});
    release_guard release{fake};
    auto first = pool.submit(request("first"));
    fake->entered.get_future().wait();
    auto second = pool.submit(request("second"));
    auto third = pool.submit(request("third"));
    auto rejected = pool.submit(request("fourth")).get();
    JEVT_REQUIRE(!rejected);
    JEVT_REQUIRE_EQ(rejected.error_value().code, jevt::error_code::overloaded);
    JEVT_REQUIRE_EQ(pool.stats().queued, 2u);
    JEVT_REQUIRE_EQ(pool.stats().in_flight, 1u);
    release.now();
    auto stop1 = std::async(std::launch::async, [&] { pool.shutdown(); });
    auto stop2 = std::async(std::launch::async, [&] { pool.shutdown(); });
    stop1.get(); stop2.get();
    JEVT_REQUIRE(first.get()); JEVT_REQUIRE(second.get()); JEVT_REQUIRE(third.get());
    auto stopped = pool.submit(request("late")).get();
    JEVT_REQUIRE_EQ(stopped.error_value().code, jevt::error_code::shutting_down);
    const auto stats = pool.stats();
    JEVT_REQUIRE_EQ(stats.accepted, 3u);
    JEVT_REQUIRE_EQ(stats.completed, 3u);
    JEVT_REQUIRE_EQ(stats.rejected, 2u);
    JEVT_REQUIRE_EQ(stats.queued, 0u);
    JEVT_REQUIRE_EQ(stats.in_flight, 0u);
}

JEVT_TEST("cross-request microbatches bucket lengths and map results correctly") {
    auto fake = std::make_shared<fake_backend>();
    jevt::batching_backend pool(fake, {.max_batch_size = 8, .max_delay = 0us, .length_bucket_width = 16});
    release_guard release{fake};
    auto first = pool.submit(request("first"));
    fake->entered.get_future().wait();
    const std::vector<std::string> inputs{"a", std::string(40, 'b'), "c", std::string(40, 'd')};
    std::vector<std::future<jevt::result<jevt::inference_response>>> futures;
    for (const auto& input : inputs) futures.push_back(pool.submit(request(input)));
    release.now();
    pool.shutdown();
    JEVT_REQUIRE(first.get());
    for (std::size_t i = 0; i < inputs.size(); ++i)
        JEVT_REQUIRE_EQ(futures[i].get()->model_id, "id:q:" + inputs[i]);
    JEVT_REQUIRE_EQ(fake->batches.size(), 3u);
    JEVT_REQUIRE_EQ(fake->batches[1], (std::vector<std::string>{"a", "c"}));
    JEVT_REQUIRE_EQ(fake->batches[2], (std::vector<std::string>{inputs[1], inputs[3]}));
}

JEVT_TEST("sync batches preserve ordering and reject oversized admission atomically") {
    auto fake = std::make_shared<fake_backend>();
    fake->block_first = false;
    jevt::batching_backend pool(fake, {.queue_capacity = 4, .max_batch_size = 4, .max_delay = 0us,
                                     .length_bucket_width = 16});
    const std::string long_input(40, 'x');
    std::vector<jevt::inference_request> requests{request("a"), request(long_input), request("c")};
    auto result = pool.predict_batch(requests);
    JEVT_REQUIRE(result);
    JEVT_REQUIRE_EQ((*result)[0].model_id, "id:q:a");
    JEVT_REQUIRE_EQ((*result)[1].model_id, "id:q:" + long_input);
    JEVT_REQUIRE_EQ((*result)[2].model_id, "id:q:c");
    requests.resize(5, request("extra"));
    auto overloaded = pool.predict_batch(requests);
    JEVT_REQUIRE_EQ(overloaded.error_value().code, jevt::error_code::overloaded);
    JEVT_REQUIRE_EQ(pool.stats().accepted, 3u);
    JEVT_REQUIRE(pool.predict_batch({}));
}

JEVT_TEST("exceptions errors and wrong batch cardinality complete every promise") {
    for (int mode = 1; mode <= 4; ++mode) {
        auto fake = std::make_shared<fake_backend>();
        fake->block_first = false;
        fake->behavior = mode;
        jevt::batching_backend pool(fake, {.max_batch_size = 2, .max_delay = 1s});
        auto one = pool.submit(request("one"));
        auto two = pool.submit(request("two"));
        JEVT_REQUIRE_EQ(one.wait_for(2s), std::future_status::ready);
        JEVT_REQUIRE_EQ(two.wait_for(2s), std::future_status::ready);
        auto a = one.get(), b = two.get();
        JEVT_REQUIRE(!a); JEVT_REQUIRE(!b);
        const auto code = mode <= 2 ? jevt::error_code::backend_failure : mode == 3
            ? jevt::error_code::invalid_backend_output : jevt::error_code::invalid_request;
        JEVT_REQUIRE_EQ(a.error_value().code, code);
        JEVT_REQUIRE_EQ(b.error_value().code, code);
        fake->behavior = 0;
        auto recovered = pool.submit(request("recovered"));
        pool.shutdown();
        JEVT_REQUIRE(recovered.get());
    }
}

JEVT_TEST("multiple workers execute independently within the configured bound") {
    auto fake = std::make_shared<fake_backend>();
    jevt::batching_backend pool(fake, {.worker_count = 2, .max_batch_size = 1, .max_delay = 0us});
    release_guard release{fake};
    auto blocked = pool.submit(request("blocked"));
    fake->entered.get_future().wait();
    auto independent = pool.submit(request("independent"));
    JEVT_REQUIRE_EQ(independent.wait_for(2s), std::future_status::ready);
    JEVT_REQUIRE(independent.get());
    release.now();
    JEVT_REQUIRE(blocked.get());
}

JEVT_TEST("worker reentrancy rejects synchronous self-waits") {
    jevt::batching_backend* pointer = nullptr;
    auto fake = std::make_shared<jevt::function_backend>([&](const jevt::inference_request& r)
        -> jevt::result<jevt::inference_response> {
        auto recursive = pointer->predict(r);
        if (recursive || recursive.error_value().code != jevt::error_code::invalid_request)
            throw std::runtime_error("recursive request was not rejected");
        try { pointer->shutdown(); } catch (const std::logic_error&) {
            return jevt::inference_response{{}, "safe"};
        }
        throw std::runtime_error("recursive shutdown was not rejected");
    });
    jevt::batching_backend pool(fake, {.max_delay = 0us});
    pointer = &pool;
    JEVT_REQUIRE_EQ(pool.predict(request("input"))->model_id, "safe");
}

JEVT_TEST("delay deadline dispatches a partial batch and destructor drains") {
    auto fake = std::make_shared<fake_backend>();
    fake->block_first = false;
    std::future<jevt::result<jevt::inference_response>> future;
    {
        jevt::batching_backend pool(fake, {.max_delay = 100us});
        future = pool.submit(request("partial"));
        JEVT_REQUIRE_EQ(future.wait_for(2s), std::future_status::ready);
        JEVT_REQUIRE(future.get());
        future = pool.submit(request("drained"));
    }
    JEVT_REQUIRE_EQ(future.get()->model_id, "id:q:drained");
}

JEVT_TEST("concurrent producers racing shutdown never lose admitted promises") {
    auto fake = std::make_shared<fake_backend>();
    fake->block_first = false;
    jevt::batching_backend pool(fake, {.worker_count = 3, .queue_capacity = 8,
                                     .max_batch_size = 4, .max_delay = 100us});
    std::promise<void> start;
    auto ready = start.get_future().share();
    std::vector<std::future<void>> callers;
    for (int i = 0; i < 4; ++i) callers.push_back(std::async(std::launch::async, [&] {
        ready.wait();
        std::vector<std::future<jevt::result<jevt::inference_response>>> pending;
        for (int j = 0; j < 64; ++j) pending.push_back(pool.submit(request("race")));
        for (auto& future : pending) {
            JEVT_REQUIRE_EQ(future.wait_for(2s), std::future_status::ready);
            auto result = future.get();
            if (!result) JEVT_REQUIRE(result.error_value().code == jevt::error_code::overloaded ||
                                     result.error_value().code == jevt::error_code::shutting_down);
        }
    }));
    auto stop = std::async(std::launch::async, [&] { ready.wait(); pool.shutdown(); });
    start.set_value();
    for (auto& caller : callers) caller.get();
    stop.get();
    const auto stats = pool.stats();
    JEVT_REQUIRE_EQ(stats.accepted + stats.rejected, 256u);
    JEVT_REQUIRE_EQ(stats.accepted, stats.completed);
    JEVT_REQUIRE_EQ(stats.queued, 0u);
    JEVT_REQUIRE_EQ(stats.in_flight, 0u);
}

JEVT_TEST("invalid configuration is rejected before workers start") {
    auto fake = std::make_shared<fake_backend>();
    fake->block_first = false;
    for (const auto options : {jevt::batching_options{.worker_count = 0},
         jevt::batching_options{.queue_capacity = 0}, jevt::batching_options{.max_batch_size = 0},
         jevt::batching_options{.max_delay = -1us}}) {
        bool threw = false;
        try { jevt::batching_backend pool(fake, options); }
        catch (const std::invalid_argument&) { threw = true; }
        JEVT_REQUIRE(threw);
    }
    bool threw = false;
    try { jevt::batching_backend pool(nullptr); }
    catch (const std::invalid_argument&) { threw = true; }
    JEVT_REQUIRE(threw);
}

int main() { return jevt::test::run_all("batching"); }
