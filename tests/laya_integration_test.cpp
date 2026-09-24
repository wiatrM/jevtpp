#include <jevt/laya.hpp>
#include <jevt/system_one.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

enum class category { billing, technical, sales, spam };
enum class urgency { low, medium, high };
constexpr auto categories = jevt::schema<category, "integration.categories">(
    jevt::option<category::billing>("Invoices, payments, or refund requests."),
    jevt::option<category::technical>("Software bugs, crashes, or login failures."),
    jevt::option<category::sales>("Upgrades, enterprise pricing, or new purchases."),
    jevt::option<category::spam>("Unsolicited marketing or automated noise."));
constexpr auto levels = jevt::schema<urgency, "integration.urgency">(
    jevt::option<urgency::low>("Customer is patient and calm."),
    jevt::option<urgency::medium>("Issue blocks work but has a workaround."),
    jevt::option<urgency::high>("Complete outage or severe frustration."));
constexpr auto model = jevt::decision_model<"integration.ticket">(
    "Evaluate the customer support ticket using the supplied application context.",
    jevt::choice<"category">("Which team should own this request?", categories),
    jevt::noul<"is_urgent">("Does this require attention within one hour?"),
    jevt::score<"urgency_score">("Rate the frustration level against the rubric.", levels),
    jevt::probability<"sentiment_probability">("The customer is angry."));

class observed_backend final : public jevt::backend {
public:
    explicit observed_backend(std::shared_ptr<jevt::backend> implementation)
        : implementation_(std::move(implementation)) {}
    jevt::result<jevt::inference_response> predict(const jevt::inference_request& request) override {
        ++single_calls;
        return implementation_->predict(request);
    }
    jevt::result<std::vector<jevt::inference_response>> predict_batch(
        std::span<const jevt::inference_request> requests) override {
        ++batch_calls;
        require(requests.size() == 4, "expected all four fields in one batch");
        for (const auto& request : requests)
            require(request.input == expected_context, "a field lost the shared application context");
        require(requests[0].question_kind == jevt::inference_request::kind::choice, "choice kind lost");
        require(requests[1].question_kind == jevt::inference_request::kind::noul, "Noul kind lost");
        require(requests[2].question_kind == jevt::inference_request::kind::score, "Score kind lost");
        require(requests[3].question_kind == jevt::inference_request::kind::noul, "probability kind lost");
        return implementation_->predict_batch(requests);
    }
    std::string_view name() const noexcept override { return implementation_->name(); }
    std::string expected_context;
    unsigned batch_calls = 0;
    unsigned single_calls = 0;
private:
    std::shared_ptr<jevt::backend> implementation_;
};

void check_distribution(std::span<const float> probabilities, std::size_t size) {
    require(probabilities.size() == size, "wrong distribution cardinality");
    for (const auto probability : probabilities)
        require(std::isfinite(probability) && probability >= 0.0F && probability <= 1.0F,
                "probability outside [0, 1]");
    require(std::abs(std::accumulate(probabilities.begin(), probabilities.end(), 0.0) - 1.0) < 1e-5,
            "distribution does not sum to one");
}
void check_unit(float value) {
    require(std::isfinite(value) && value >= 0.0F && value <= 1.0F, "value outside [0, 1]");
}
} // namespace

int main(int argc, char** argv) try {
    if (argc != 2) {
        std::cerr << "usage: jevt_laya_integration_tests MODEL_DIRECTORY\n";
        return 2;
    }
    auto real = jevt::models::laya_multilingual(std::filesystem::path{argv[1]}, 2);
    auto observed = std::make_shared<observed_backend>(real);
    const auto brain = jevt::bind_system_one(model, observed);
    observed->expected_context = R"json({"ticket":{"message":"Nie mogę się zalogować; cała firma jest zablokowana."},"customer":{"plan":"enterprise","affected_users":240},"service":{"sla_minutes":30,"workaround":false}})json";
    const auto result = brain.evaluate(jevt::json_state(observed->expected_context));
    if (!result) throw std::runtime_error(result.error_value().message);
    require(observed->batch_calls == 1 && observed->single_calls == 0,
            "typed evaluation must dispatch exactly one backend batch");
    require(!result->abstained(), "default policies should produce typed values");
    check_distribution(result->get<"category">().probabilities(), 4);
    check_distribution(result->get<"is_urgent">().probabilities(), 2);
    check_distribution(result->get<"urgency_score">().probabilities(), 3);
    check_distribution(result->get<"sentiment_probability">().probabilities(), 2);
    check_unit(result->get<"category">().confidence());
    check_unit(result->get<"urgency_score">().confidence());
    check_unit(result->get<"is_urgent">().probability_true());
    check_unit(result->get<"sentiment_probability">().value());
    const float expected_position = result->get<"urgency_score">().score();
    require(std::isfinite(expected_position) && expected_position >= 0.0F && expected_position <= 2.0F,
            "Score expected position outside rubric");
    require(!result->get<"category">().model_id().empty(), "model provenance missing");

    // Fixture produced independently by tests/laya_reference.py with the pinned
    // tokenizer/model bundle. This catches sequence, marker, qtype and tokenizer drift.
    constexpr std::array reference{
        std::array{0.0014115741F, 0.9712805152F, 0.0049303751F, 0.0223775748F},
        std::array{0.0093023069F, 0.9906976819F, 0.0F, 0.0F},
        std::array{0.2663171589F, 0.4509098530F, 0.2827730477F, 0.0F},
        std::array{0.8110032678F, 0.1889967620F, 0.0F, 0.0F}};
    const std::array actual{
        result->get<"category">().probabilities(), result->get<"is_urgent">().probabilities(),
        result->get<"urgency_score">().probabilities(),
        result->get<"sentiment_probability">().probabilities()};
    constexpr std::array sizes{4U, 2U, 3U, 2U};
    for (std::size_t row = 0; row < actual.size(); ++row)
        for (std::size_t column = 0; column < sizes[row]; ++column)
            require(std::abs(actual[row][column] - reference[row][column]) < 1e-4F,
                    "C++ output drifted from the independent tokenizer/tensor fixture");

    // Variable-length padding in a mixed-cardinality batch must not alter the
    // first question's probabilities compared with independent inference.
    const auto request = brain.request(jevt::json_state(observed->expected_context));
    const auto views = request.views();
    const auto independent = real->predict(views.requests().front());
    if (!independent) throw std::runtime_error(independent.error_value().message);
    require(independent->scores.size() == 4, "single inference cardinality mismatch");
    for (std::size_t i = 0; i < 4; ++i)
        require(std::abs(independent->scores[i] - result->get<"category">().probabilities()[i]) < 1e-4F,
                "batched and independent inference differ");
    std::cout << "Laya integration: shared JSON, mixed batch, typed results, and batch parity passed\n"
              << "probabilities:";
    const auto print = [](std::span<const float> probabilities) {
        std::cout << " [";
        for (std::size_t i = 0; i < probabilities.size(); ++i)
            std::cout << (i ? "," : "") << probabilities[i];
        std::cout << ']';
    };
    print(result->get<"category">().probabilities());
    print(result->get<"is_urgent">().probabilities());
    print(result->get<"urgency_score">().probabilities());
    print(result->get<"sentiment_probability">().probabilities());
    std::cout << '\n';
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "Laya integration failed: " << exception.what() << '\n';
    return 1;
}
