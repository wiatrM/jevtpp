#include <jevt/jevt.hpp>

#include "test_harness.hpp"

#include <atomic>
#include <memory>
#include <limits>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

enum class support_team { billing = 10, technical = 40, sales = 90 };

inline constexpr auto support_routing = jevt::schema<support_team, "support.routing">(
    jevt::option<support_team::billing>("Invoices, payments and refunds"),
    jevt::option<support_team::technical>("Failures, defects and incidents"),
    jevt::option<support_team::sales>("Plans, pricing and purchasing"));

inline constexpr auto needs_human = jevt::predicate<"support.needs_human">(
    "Does this message require intervention by a person?");

std::shared_ptr<jevt::backend> routing_backend() {
    return std::make_shared<jevt::keyword_backend>(
        jevt::keyword_backend::keyword_table{
            {"invoice", "payment", "refund"},
            {"failure", "crash", "broken", "incident"},
            {"pricing", "quote", "purchase", "plan"}},
        "acceptance-rules-v1");
}

} // namespace

JEVT_TEST("a support ticket is routed to its business owner") {
    const jevt::context runtime{{.inference_backend = routing_backend(),
                                 .abstain_threshold = 0.60F,
                                 .diagnostics = nullptr}};
    const auto router = runtime.bind(support_routing, {
        .question = "Which team should own this ticket?",
        .abstain_threshold = -1.0F,
        .backend_override = nullptr});

    const auto billing = router.choose("Please send a corrected invoice and refund the payment");
    const auto technical = router.choose("Production is broken after a crash");
    const auto sales = router.choose("We need a pricing quote for the enterprise plan");

    JEVT_REQUIRE(billing && billing->has_value());
    JEVT_REQUIRE_EQ(billing->value(), support_team::billing);
    JEVT_REQUIRE(technical && technical->has_value());
    JEVT_REQUIRE_EQ(technical->value(), support_team::technical);
    JEVT_REQUIRE(sales && sales->has_value());
    JEVT_REQUIRE_EQ(sales->value(), support_team::sales);
    JEVT_REQUIRE_EQ(sales->metadata().model_id, "acceptance-rules-v1");
}

JEVT_TEST("needs-human remains a typed three-way predicate") {
    auto backend = std::make_shared<jevt::keyword_backend>(
        jevt::keyword_backend::keyword_table{{"resolved", "automatic"},
                                              {"human", "agent", "complaint"}},
        "escalation-rules-v1");
    const jevt::context runtime{{.inference_backend = std::move(backend),
                                 .abstain_threshold = 0.60F,
                                 .diagnostics = nullptr}};
    const auto escalation = runtime.bind(needs_human);

    const auto human = escalation.evaluate("I need a human agent for this complaint");
    const auto automatic = escalation.evaluate("This was resolved by the automatic workflow");
    const auto uncertain = escalation.evaluate("Could you take a look?");

    JEVT_REQUIRE(human && human->is_true());
    JEVT_REQUIRE(automatic && automatic->is_false());
    JEVT_REQUIRE(uncertain && uncertain->abstained());
    static_assert(!std::is_convertible_v<decltype(*human), bool>,
                  "predicate decisions must not silently collapse to bool");
}

JEVT_TEST("low confidence routes to manual review instead of a random team") {
    const jevt::context runtime{{.inference_backend = routing_backend(),
                                 .abstain_threshold = 0.70F,
                                 .diagnostics = nullptr}};
    const auto router = runtime.bind(support_routing);
    const auto decision = router.choose("Hello, I have a general question");

    JEVT_REQUIRE(decision);
    JEVT_REQUIRE(decision->abstained());
    JEVT_REQUIRE(!decision->has_value());
    JEVT_REQUIRE(decision->confidence() < 0.70F);

    bool manual_review = false;
    jevt::match(support_routing, *decision,
                jevt::on<support_team::billing>([] {}),
                jevt::on<support_team::technical>([] {}),
                jevt::on<support_team::sales>([] {}),
                jevt::on_abstain([&] { manual_review = true; }));
    JEVT_REQUIRE(manual_review);
}

JEVT_TEST("complete typed match executes exactly one business action") {
    const jevt::context runtime{{.inference_backend = routing_backend(),
                                 .abstain_threshold = 0.60F,
                                 .diagnostics = nullptr}};
    const auto decision = runtime.bind(support_routing).choose("The application has a crash");
    JEVT_REQUIRE(decision);

    int action = 0;
    jevt::match(support_routing, *decision,
                jevt::on<support_team::billing>([&] { action = 10; }),
                jevt::on<support_team::technical>([&] { action = 40; }),
                jevt::on<support_team::sales>([&] { action = 90; }),
                jevt::on_abstain([&] { action = -1; }));
    JEVT_REQUIRE_EQ(action, 40);
}

JEVT_TEST("one bound decision can serve concurrent callers") {
    constexpr std::size_t thread_count = 8;
    constexpr std::size_t calls_per_thread = 2'000;
    const jevt::context runtime{{.inference_backend = routing_backend(),
                                 .abstain_threshold = 0.60F,
                                 .diagnostics = nullptr}};
    const auto router = runtime.bind(support_routing);
    std::atomic<std::size_t> correct{0};
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (std::size_t thread = 0; thread < thread_count; ++thread) {
        workers.emplace_back([&, thread] {
            for (std::size_t call = 0; call < calls_per_thread; ++call) {
                const bool use_billing = (thread + call) % 2 == 0;
                const auto result = router.choose(use_billing ? "refund my payment"
                                                               : "production crash incident");
                if (result && result->has_value() &&
                    result->value() == (use_billing ? support_team::billing
                                                     : support_team::technical)) {
                    correct.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    JEVT_REQUIRE_EQ(correct.load(), thread_count * calls_per_thread);
}

JEVT_TEST("malformed backend output is a technical error, not abstention") {
    auto malformed = std::make_shared<jevt::function_backend>(
        [](const jevt::inference_request&) -> jevt::result<jevt::inference_response> {
            return jevt::inference_response{{1.0F}, "malformed-test"};
        });
    const jevt::context runtime{{.inference_backend = std::move(malformed),
                                 .abstain_threshold = 0.55F,
                                 .diagnostics = nullptr}};
    const auto result = runtime.bind(support_routing).choose("anything");

    JEVT_REQUIRE(!result);
    JEVT_REQUIRE_EQ(result.error_value().code, jevt::error_code::invalid_backend_output);
}

JEVT_TEST("runtime decisions feed success, abstention, errors and tags into diagnostics") {
    auto diagnostics = std::make_shared<jevt::Diagnostics>();
    auto scripted = std::make_shared<jevt::function_backend>(
        [](const jevt::inference_request& request) -> jevt::result<jevt::inference_response> {
            if (request.input == "matched") {
                return jevt::inference_response{{10.0F, 1.0F, 1.0F}, "diagnostic-test"};
            }
            if (request.input == "uncertain") {
                return jevt::inference_response{{1.0F, 1.0F, 1.0F}, "diagnostic-test"};
            }
            return jevt::inference_response{{1.0F}, "diagnostic-test"};
        });
    const jevt::context runtime{{.inference_backend = std::move(scripted),
                                 .abstain_threshold = 0.70F,
                                 .diagnostics = diagnostics}};
    const auto router = runtime.bind(support_routing);

    JEVT_REQUIRE(router.choose("matched", "ticket-success"));
    JEVT_REQUIRE(router.choose("uncertain", "ticket-review"));
    JEVT_REQUIRE(!router.choose("malformed", "ticket-error"));

    const auto snapshot = diagnostics->snapshot();
    JEVT_REQUIRE_EQ(snapshot.total.calls, 3u);
    JEVT_REQUIRE_EQ(snapshot.total.successes, 1u);
    JEVT_REQUIRE_EQ(snapshot.total.abstains, 1u);
    JEVT_REQUIRE_EQ(snapshot.total.errors, 1u);
    JEVT_REQUIRE_EQ(snapshot.decisions.size(), 1u);
    JEVT_REQUIRE_EQ(snapshot.decisions.front().decision, "support.routing");
    JEVT_REQUIRE_EQ(snapshot.decisions.front().stats.calls, 3u);
    JEVT_REQUIRE_EQ(snapshot.recent_calls.size(), 3u);
    JEVT_REQUIRE(snapshot.recent_calls.front().tag.has_value());
    JEVT_REQUIRE_EQ(*snapshot.recent_calls.front().tag, "ticket-error");
}

JEVT_TEST("invalid predicate scores and overflowing choice totals are errors") {
    const auto maximum = std::numeric_limits<float>::max();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    const auto infinity = std::numeric_limits<float>::infinity();
    for (const auto& scores : std::vector<std::vector<float>>{
             {-1.0F, 2.0F}, {2.0F, -1.0F}, {nan, 1.0F}, {infinity, 1.0F},
             {0.0F, 0.0F}, {maximum, maximum}}) {
        auto backend = std::make_shared<jevt::function_backend>(
            [scores](const jevt::inference_request&) -> jevt::result<jevt::inference_response> {
                return jevt::inference_response{scores, "invalid-scores"};
            });
        const jevt::context runtime{{.inference_backend = backend}};
        const auto answer = runtime.bind(needs_human).evaluate("ticket");
        JEVT_REQUIRE(!answer);
        JEVT_REQUIRE_EQ(answer.error_value().code, jevt::error_code::invalid_backend_output);
        JEVT_REQUIRE_EQ(runtime.diagnostics()->snapshot().total.errors, 1u);
    }
    auto backend = std::make_shared<jevt::function_backend>(
        [maximum](const jevt::inference_request&) -> jevt::result<jevt::inference_response> {
            return jevt::inference_response{{maximum, maximum, maximum}, "overflow"};
        });
    const jevt::context runtime{{.inference_backend = backend}};
    const auto answer = runtime.bind(support_routing).choose("ticket");
    JEVT_REQUIRE(!answer);
    JEVT_REQUIRE_EQ(answer.error_value().code, jevt::error_code::invalid_backend_output);
}

JEVT_TEST("binding thresholds reject invalid overrides for choices and predicates") {
    const jevt::context runtime{{.inference_backend = routing_backend(), .abstain_threshold = 0.6F}};
    for (float threshold : {-2.0F, -0.1F, 1.1F, std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()}) {
        bool choice_rejected = false, predicate_rejected = false;
        try { (void)runtime.bind(support_routing, {.abstain_threshold = threshold}); }
        catch (const std::invalid_argument&) { choice_rejected = true; }
        try { (void)runtime.bind(needs_human, {.abstain_threshold = threshold}); }
        catch (const std::invalid_argument&) { predicate_rejected = true; }
        JEVT_REQUIRE(choice_rejected && predicate_rejected);
    }
    for (float threshold : {-1.0F, 0.0F, 1.0F}) {
        (void)runtime.bind(support_routing, {.abstain_threshold = threshold});
        (void)runtime.bind(needs_human, {.abstain_threshold = threshold});
    }
}

int main() {
    return jevt::test::run_all("acceptance");
}
