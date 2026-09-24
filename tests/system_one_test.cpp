#include <jevt/runtime.hpp>
#include "test_harness.hpp"

#include <limits>

namespace {
enum class category { billing = 7, technical = 12, sales = 18, spam = 23 };
enum class urgency { low = 0, medium = 1, high = 2 };
constexpr auto categories = jevt::schema<category, "categories">(
    jevt::option<category::billing>("Invoices, payments, or refund requests."),
    jevt::option<category::technical>("Software bugs, crashes, or login failures."),
    jevt::option<category::sales>("Upgrades, enterprise pricing, or new purchases."),
    jevt::option<category::spam>("Unsolicited marketing or automated noise."));
constexpr auto levels = jevt::schema<urgency, "urgency">(
    jevt::option<urgency::low>("Customer is patient and calm."),
    jevt::option<urgency::medium>("Issue blocks work but has a workaround."),
    jevt::option<urgency::high>("Complete outage or severe frustration."));
constexpr auto model = jevt::decision_model<"support.ticket">(
    "Evaluate the incoming customer support ticket.",
    jevt::choice<"category">("Which team should own this request?", categories, 0.3F),
    jevt::noul<"is_urgent">("Does this require attention within one hour?", {0.2F, 0.8F}),
    jevt::score<"urgency_score">("Rate the frustration level against the rubric.", levels),
    jevt::probability<"sentiment_probability">("The customer is angry."));

class fake_batch_backend : public jevt::backend {
public:
    std::size_t batches{};
    std::vector<jevt::inference_response> output{
        {{0.05F, 0.9F, 0.03F, 0.02F}, "fixture"},
        {{0.1F, 0.9F}, "fixture"},
        {{0.0F, 0.57F, 0.43F}, "fixture"},
        {{0.25F, 0.75F}, "fixture"}};
    jevt::result<jevt::inference_response> predict(const jevt::inference_request&) override {
        throw std::logic_error("System One must call batch entry point");
    }
    jevt::result<std::vector<jevt::inference_response>> predict_batch(
        std::span<const jevt::inference_request> requests) override {
        ++batches;
        JEVT_REQUIRE_EQ(requests.size(), 4U);
        for (std::size_t i = 0; i < requests.size(); ++i) {
            JEVT_REQUIRE_EQ(requests[i].input.data(), requests[0].input.data());
            JEVT_REQUIRE(requests[i].question.starts_with(model.description));
        }
        JEVT_REQUIRE_EQ(requests[0].decision_id, "support.ticket.category");
        JEVT_REQUIRE_EQ(requests[0].options[1], categories.descriptions()[1]);
        JEVT_REQUIRE_EQ(requests[1].question_kind, jevt::inference_request::kind::noul);
        JEVT_REQUIRE_EQ(requests[2].question_kind, jevt::inference_request::kind::score);
        JEVT_REQUIRE_EQ(requests[3].question_kind, jevt::inference_request::kind::noul);
        return output;
    }
    std::string_view name() const noexcept override { return "fixture"; }
};

struct TicketDecision {
    category category_value;
    bool is_urgent;
    urgency urgency_score;
    float sentiment_probability;
};
struct ticket { std::string text; };
jevt::state_value to_jevt_state(const ticket& value) { return jevt::text_state(value.text); }
} // namespace

JEVT_TEST("metadata and one shared state reach a single batch") {
    auto backend = std::make_shared<fake_batch_backend>();
    auto runner = jevt::bind_system_one(model, backend);
    auto answer = runner.evaluate(jevt::json_state(R"({"ticket":"Nobody can log in","policy":"Outages need immediate action"})"));
    JEVT_REQUIRE(answer);
    JEVT_REQUIRE_EQ(backend->batches, 1U);
    JEVT_REQUIRE_EQ(answer->get<"category">().value(), category::technical);
    JEVT_REQUIRE_NEAR(answer->get<"category">().probabilities()[1], 0.9, 1e-6);
    JEVT_REQUIRE(answer->get<"category">().confidence() < 0.9F);
    JEVT_REQUIRE_NEAR(answer->get<"is_urgent">().probability_true(), 0.9, 1e-6);
    JEVT_REQUIRE(answer->get<"is_urgent">().value());
    JEVT_REQUIRE_NEAR(answer->get<"urgency_score">().score(), 1.43, 1e-6);
    JEVT_REQUIRE_EQ(answer->get<"urgency_score">().value(), urgency::medium);
    JEVT_REQUIRE_EQ(answer->get<"urgency_score">().legend()[2], levels.descriptions()[2]);
    JEVT_REQUIRE_NEAR(answer->get<"sentiment_probability">().value(), 0.75, 1e-6);
    JEVT_REQUIRE(!answer->abstained());
    const auto mapped = answer->map([](const auto& c, const auto& u, const auto& s, const auto& p) {
        return TicketDecision{c.value(), u.value(), s.value(), p.value()};
    });
    JEVT_REQUIRE_EQ(mapped.category_value, category::technical);
    JEVT_REQUIRE_EQ(mapped.urgency_score, urgency::medium);
    JEVT_REQUIRE_NEAR(mapped.sentiment_probability, 0.75, 1e-6);
}

JEVT_TEST("field projection computes only requested typed answers with explicit context") {
    std::size_t calls = 0;
    auto backend = std::make_shared<jevt::function_backend>([&](const jevt::inference_request& r)
        -> jevt::result<jevt::inference_response> {
        ++calls;
        JEVT_REQUIRE_EQ(r.input, "focused ticket context");
        if (calls == 1) {
            JEVT_REQUIRE_EQ(r.decision_id, "support.ticket.is_urgent");
            return jevt::inference_response{{0.1F, 0.9F}, "fixture"};
        }
        JEVT_REQUIRE_EQ(r.decision_id, "support.ticket.category");
        return jevt::inference_response{{0.01F, 0.97F, 0.01F, 0.01F}, "fixture"};
    });
    constexpr auto projection = jevt::select_fields<"is_urgent", "category">(model);
    static_assert(decltype(projection)::size == 2);
    const auto runner = jevt::bind_system_one(model, backend).select<"is_urgent", "category">();
    const auto answer = runner.evaluate(ticket{"focused ticket context"});
    JEVT_REQUIRE(answer);
    JEVT_REQUIRE_EQ(calls, 2U);
    JEVT_REQUIRE(answer->get<"is_urgent">().value());
    JEVT_REQUIRE_EQ(answer->get<"category">().value(), category::technical);
}

JEVT_TEST("uncertainty remains inspectable when a field abstains") {
    auto backend = std::make_shared<fake_batch_backend>();
    backend->output[0].scores = {1, 1, 1, 1};
    backend->output[1].scores = {0.5F, 0.5F};
    auto answer = jevt::bind_system_one(model, backend).evaluate("An ambiguous ticket");
    JEVT_REQUIRE(answer);
    JEVT_REQUIRE(answer->abstained());
    JEVT_REQUIRE(answer->get<"category">().abstained());
    JEVT_REQUIRE_NEAR(answer->get<"category">().confidence(), 0, 1e-6);
    JEVT_REQUIRE_NEAR(answer->get<"category">().probabilities()[0], 0.25, 1e-6);
    JEVT_REQUIRE(answer->get<"is_urgent">().abstained());
    JEVT_REQUIRE(!answer->get<"is_urgent">().selected().has_value());
}

JEVT_TEST("binary thresholds include edges and retain P true") {
    auto backend = std::make_shared<fake_batch_backend>();
    auto runner = jevt::bind_system_one(model, backend);
    backend->output[1].scores = {0.8F, 0.2F};
    auto low = runner.evaluate("ticket");
    JEVT_REQUIRE(low);
    JEVT_REQUIRE(!low->get<"is_urgent">().value());
    backend->output[1].scores = {0.2F, 0.8F};
    auto high = runner.evaluate("ticket");
    JEVT_REQUIRE(high);
    JEVT_REQUIRE(high->get<"is_urgent">().value());
}

JEVT_TEST("serialized state ownership survives original mutation and request moves") {
    auto backend = std::make_shared<fake_batch_backend>();
    auto runner = jevt::bind_system_one(model, backend);
    auto request = runner.request(jevt::json_state(R"({"ticket":"broken"})"));
    auto copied = request;
    request.state.content = "modified";
    auto moved = std::move(copied);
    const auto views = moved.views();
    JEVT_REQUIRE_EQ(views.requests()[0].input, R"({"ticket":"broken"})");
    JEVT_REQUIRE_EQ(moved.state.kind, jevt::state_value::content_kind::json);
    JEVT_REQUIRE(runner.evaluate(ticket{"broken"}));
    int serializations = 0;
    JEVT_REQUIRE(runner.evaluate(ticket{"broken"}, [&](const ticket& value) {
        ++serializations;
        return jevt::text_state(value.text);
    }));
    JEVT_REQUIRE_EQ(serializations, 1);
    bool rejected = false;
    try { (void)jevt::json_state(" \n\t"); } catch (const std::invalid_argument&) { rejected = true; }
    JEVT_REQUIRE(rejected);
}

JEVT_TEST("malformed outputs fail the complete decision atomically") {
    auto backend = std::make_shared<fake_batch_backend>();
    const auto valid = backend->output;
    auto runner = jevt::bind_system_one(model, backend);
    for (const auto& invalid : std::vector<std::vector<float>>{
             {}, {1}, {0, 0}, {-1, 2}, {std::numeric_limits<float>::quiet_NaN(), 1},
             {std::numeric_limits<float>::infinity(), 1}}) {
        backend->output = valid;
        backend->output[1].scores = invalid;
        auto answer = runner.evaluate("ticket");
        JEVT_REQUIRE(!answer);
        JEVT_REQUIRE_EQ(answer.error_value().code, jevt::error_code::invalid_backend_output);
    }
    backend->output = valid;
    backend->output.pop_back();
    JEVT_REQUIRE(!runner.evaluate("ticket"));
}

JEVT_TEST("normalization uses wide accumulator and entropy has expected endpoints") {
    auto backend = std::make_shared<fake_batch_backend>();
    const auto maximum = std::numeric_limits<float>::max();
    backend->output[0].scores = {maximum, maximum, maximum, maximum};
    auto answer = jevt::bind_system_one(model, backend).evaluate("ticket");
    JEVT_REQUIRE(answer);
    JEVT_REQUIRE_NEAR(answer->get<"category">().probabilities()[0], 0.25, 1e-6);
    JEVT_REQUIRE_NEAR(jevt::entropy_confidence(std::array{0.0F, 1.0F}), 1, 1e-6);
    JEVT_REQUIRE_NEAR(jevt::entropy_confidence(std::array{0.5F, 0.5F}), 0, 1e-6);
}

JEVT_TEST("Score positions follow criterion order rather than enum storage values") {
    constexpr auto ordered = jevt::schema<category, "ordered">(
        jevt::option<category::billing>("No operational impact."),
        jevt::option<category::technical>("Service degraded."));
    constexpr auto definition = jevt::decision_model<"ordered.score">("Evaluate impact.",
        jevt::score<"impact">("Rate the operational impact.", ordered, 0.5F));
    auto backend = std::make_shared<jevt::function_backend>([](const jevt::inference_request& request)
        -> jevt::result<jevt::inference_response> {
        JEVT_REQUIRE_EQ(request.question_kind, jevt::inference_request::kind::score);
        return jevt::inference_response{{0.5F, 0.5F}, "fallback-batch"};
    });
    auto answer = jevt::bind_system_one(definition, backend).evaluate("ticket");
    JEVT_REQUIRE(answer);
    JEVT_REQUIRE(answer->get<"impact">().abstained());
    JEVT_REQUIRE_NEAR(answer->get<"impact">().score(), 0.5, 1e-6);
    JEVT_REQUIRE_EQ(answer->get<"impact">().values()[1], category::technical);
}

JEVT_TEST("backend failures propagate without constructing partial answers") {
    auto backend = std::make_shared<jevt::function_backend>([](const jevt::inference_request&)
        -> jevt::result<jevt::inference_response> {
        return jevt::error{jevt::error_code::backend_failure, "model unavailable"};
    });
    auto answer = jevt::bind_system_one(model, backend).evaluate("ticket");
    JEVT_REQUIRE(!answer);
    JEVT_REQUIRE_EQ(answer.error_value().message, "model unavailable");
}

JEVT_TEST("runtime binding records one diagnostic event per batched decision") {
    auto backend = std::make_shared<fake_batch_backend>();
    auto application = jevt::init({.inference_backend = backend});
    auto answer = jevt::bind_system_one(model).evaluate("ticket");
    JEVT_REQUIRE(answer);
    const auto snapshot = application.diagnostics()->snapshot();
    JEVT_REQUIRE_EQ(snapshot.total.calls, 1U);
    JEVT_REQUIRE_EQ(snapshot.decisions.size(), 1U);
    JEVT_REQUIRE_EQ(snapshot.decisions[0].decision, "support.ticket");
    JEVT_REQUIRE_EQ(snapshot.decisions[0].stats.successes, 1U);
}

// These opt-in translation units must fail to compile. They exercise diagnostics
// without making intentionally invalid declarations part of the normal suite.
#if defined(JEVT_TEST_DUPLICATE_FIELDS)
constexpr auto duplicate = jevt::decision_model<"duplicate">("Evaluate.",
    jevt::noul<"field">("A?"), jevt::probability<"field">("B?"));
#elif defined(JEVT_TEST_SCORE_TOO_SMALL)
constexpr auto too_small = jevt::score<"score">("Rate.",
    jevt::schema<urgency, "one">(jevt::option<urgency::low>("Calm.")));
#elif defined(JEVT_TEST_SCORE_UNORDERED)
constexpr auto unordered = jevt::score<"score">("Rate.",
    jevt::schema<urgency, "unordered">(jevt::option<urgency::high>("Outage."),
                                       jevt::option<urgency::low>("Calm.")));
#elif defined(JEVT_TEST_SCORE_TOO_LARGE)
constexpr auto too_large = []<std::size_t... I>(std::index_sequence<I...>) {
    return jevt::score<"score">("Rate.", jevt::schema<urgency, "eleven">(
        jevt::option<static_cast<urgency>(I)>("Level.")...));
}(std::make_index_sequence<11>{});
#elif defined(JEVT_TEST_INVALID_THRESHOLD)
constexpr auto bad_threshold = jevt::noul<"urgent">("Urgent?", {0.9F, 0.1F});
#elif defined(JEVT_TEST_INVALID_FIELD_TYPE)
constexpr auto bad_field = jevt::decision_model<"bad">("Evaluate.", 42);
#endif

int main() { return jevt::test::run_all("system_one"); }
