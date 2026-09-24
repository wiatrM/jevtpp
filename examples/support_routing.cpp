#include <jevt/jevt.hpp>

#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

enum class Team {
    billing = 10,
    technical = 40,
    sales = 90,
};

inline constexpr auto support = jevt::schema<Team, "support.routing">(
    jevt::option<Team::billing>("Invoices, refunds and payments"),
    jevt::option<Team::technical>("Errors, outages and integrations"),
    jevt::option<Team::sales>("Pricing and licence purchases"));

inline constexpr auto needs_human =
    jevt::predicate<"support.needs_human">(
        "Does this message require intervention by a person?");

static std::string_view name(Team team) {
    switch (team) {
    case Team::billing: return "billing";
    case Team::technical: return "technical";
    case Team::sales: return "sales";
    }
    return "unknown";
}

int main() {
    // This deterministic backend keeps the example runnable. A production
    // adapter implements the same jevt::backend interface.
    auto backend = std::make_shared<jevt::function_backend>(
        [](const jevt::inference_request& request)
            -> jevt::result<jevt::inference_response> {
            if (request.decision_id == "support.routing") {
                return jevt::inference_response{{10.0F, 1.0F, 1.0F},
                                                "support-demo-v1"};
            }
            return jevt::inference_response{{0.1F, 0.9F}, "support-demo-v1"};
        },
        "support-demo");

    auto app = jevt::init({.inference_backend = backend});
    auto routing = jevt::bind(support);
    auto escalation = jevt::bind(needs_human);

    constexpr std::string_view ticket =
        "Our card was charged twice and we need a refund.";
    auto routed = routing.choose(ticket);

    if (!routed) {
        std::cerr << "routing failed: " << routed.error_value().message << '\n';
        return 1;
    }
    if (routed->abstained()) {
        std::cout << "route: manual review\n";
    } else {
        std::cout << "route: " << name(routed->value()) << '\n';
    }

    auto human = escalation.evaluate(ticket);
    if (!human) {
        std::cerr << "escalation failed: " << human.error_value().message << '\n';
        return 1;
    }
    if (human->is_true()) {
        std::cout << "escalation: required\n";
    } else if (human->is_false()) {
        std::cout << "escalation: automatic processing is allowed\n";
    } else {
        std::cout << "escalation: uncertain; manual review\n";
    }

    const auto snapshot = app.diagnostics()->snapshot();
    std::cout << "diagnostics: calls=" << snapshot.total.calls
              << ", p95=" << snapshot.total.latency_p95_ms << "ms\n";
}

