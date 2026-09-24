#include <jevt/laya.hpp>
#include <jevt/runtime.hpp>
#if JEVT_DEMO_HTTP
#include <jevt/http_server.hpp>
#endif

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

enum class support_category { billing, technical, sales, spam };
enum class urgency_level { low = 0, medium = 1, high = 2 };

inline constexpr auto categories = jevt::schema<support_category, "support.category">(
    jevt::option<support_category::billing>("Invoices, payments, or refund requests."),
    jevt::option<support_category::technical>("Software bugs, crashes, or login failures."),
    jevt::option<support_category::sales>("Upgrades, enterprise pricing, or new purchases."),
    jevt::option<support_category::spam>("Unsolicited marketing or automated noise."));

inline constexpr auto urgency_levels = jevt::schema<urgency_level, "support.urgency">(
    jevt::option<urgency_level::low>("Customer is patient and calm."),
    jevt::option<urgency_level::medium>("Issue blocks work but has a workaround."),
    jevt::option<urgency_level::high>("Complete outage or severe frustration."));

inline constexpr auto ticket_model = jevt::decision_model<"support.ticket">(
    "Evaluate the incoming customer support ticket using the ticket, customer, and service context.",
    jevt::choice<"category">("Which team should own this request?", categories),
    jevt::noul<"is_urgent">("Does this require attention within one hour?"),
    jevt::score<"urgency_score">("Rate the frustration level against the rubric.", urgency_levels),
    jevt::probability<"sentiment_probability">("The customer is angry."));

struct ticket_decision {
    support_category category;
    bool is_urgent;
    urgency_level urgency_score;
    float sentiment_probability;
};

int main(int argc, char** argv) try {
    if (argc < 2) {
        std::cerr << "usage: jevt_laya_demo MODEL_DIRECTORY [JSON_CONTEXT]\n";
        return 2;
    }
    // Applications can pass their serializer's JSON output without depending on
    // a particular JSON library. The same state is supplied to all four fields.
    const std::string context = argc >= 3 ? argv[2] : R"json({
  "ticket": {
    "subject": "Production login is down",
    "message": "All our users are locked out. We have no workaround. Please fix this immediately!"
  },
  "customer": {"plan": "enterprise", "affected_users": 240},
  "service": {"status": "complete_outage", "sla_response_minutes": 30}
})json";

    auto application = jevt::init({
        .inference_backend = jevt::models::laya_multilingual(std::filesystem::path{argv[1]})});
    const auto brain = jevt::bind_system_one(ticket_model);
    const auto started = std::chrono::steady_clock::now();
    const auto result = brain.evaluate(jevt::json_state(context));
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    if (!result) throw std::runtime_error(result.error_value().message);

    const auto decision = result->map([](const auto& category, const auto& urgent,
                                        const auto& urgency, const auto& sentiment) {
        return ticket_decision{category.value(), urgent.value(), urgency.value(), sentiment.value()};
    });
    constexpr std::array category_names{"billing", "technical", "sales", "spam"};
    constexpr std::array urgency_names{"low", "medium", "high"};
    const auto& category = result->get<"category">();
    const auto& urgency = result->get<"urgency_score">();
    std::cout << "\nJevT++ / Laya: one JSON state, four typed answers\n"
              << "context:\n" << context << "\n\n"
              << std::boolalpha << std::fixed << std::setprecision(4)
              << "category: " << category_names[static_cast<std::size_t>(decision.category)]
              << "  entropy_confidence=" << category.confidence() << '\n'
              << "is_urgent: " << decision.is_urgent
              << "  P(true)=" << result->get<"is_urgent">().probability_true() << '\n'
              << "urgency_score: " << urgency_names[static_cast<std::size_t>(decision.urgency_score)]
              << "  expected_position=" << urgency.score()
              << "  entropy_confidence=" << urgency.confidence() << '\n'
              << "sentiment_probability: " << decision.sentiment_probability << '\n'
              << "category distribution:\n";
    for (std::size_t i = 0; i < category_names.size(); ++i)
        std::cout << "  " << std::setw(10) << category_names[i] << "  " << category.probabilities()[i] << '\n';
    std::cout << "urgency distribution:\n";
    for (std::size_t i = 0; i < urgency_names.size(); ++i)
        std::cout << "  " << std::setw(10) << urgency_names[i] << "  " << urgency.probabilities()[i] << '\n';
    std::cout << "batch latency_ms: " << elapsed << " (excludes model loading; includes tokenization)\n";
#if JEVT_DEMO_HTTP
    if (const auto* serve = std::getenv("JEVT_DEMO_SERVE"); serve && std::string_view{serve} == "1") {
        jevt::HttpServer dashboard{*application.diagnostics()};
        dashboard.start();
        std::cout << "dashboard: " << dashboard.url() << " (press Enter to stop)\n";
        std::cin.get();
    }
#endif
    return 0;
} catch (const std::exception& exception) {
    std::cerr << "jevt_laya_demo: " << exception.what() << '\n';
    return 1;
}
