#include <jevt/remote.hpp>
#include <jevt/system_one.hpp>

#include <cstdlib>
#include <iostream>

enum class Team { billing, technical };
enum class Urgency { low, medium, high };
constexpr auto teams = jevt::schema<Team, "remote.support.teams">(
    jevt::option<Team::billing>("Invoices, payments and refund requests"),
    jevt::option<Team::technical>("Software bugs, crashes and login failures"));
constexpr auto urgency = jevt::schema<Urgency, "remote.support.urgency">(
    jevt::option<Urgency::low>("Customer is patient; work can continue"),
    jevt::option<Urgency::medium>("Work is blocked but a workaround exists"),
    jevt::option<Urgency::high>("Complete outage with no workaround"));
constexpr auto ticket = jevt::decision_model<"remote.support.ticket">(
    jevt::json_metadata(R"({"role":"support triage","policy":"Use customer context and SLA"})"),
    jevt::choice<"team">("Which team owns this request?", teams, 0.4F),
    jevt::noul<"urgent">("Does this require attention within one hour?",
        "This can wait until the next business day",
        "The outage or SLA requires attention within one hour"),
    jevt::score<"urgency">("Rate the urgency against the rubric", urgency),
    jevt::probability<"angry">("The customer is angry"));

int main(int argc, char** argv) {
    if (argc != 2 || std::string_view(argv[1]) != "--send") {
        std::cout << "No request sent. To send the synthetic ticket (may be billable),\n"
                     "set TYPESAFE_API_KEY in your environment and run with --send.\n"
                     "Optional TYPESAFE_MODEL selects a provider model version.\n";
        return 0;
    }
    // Credential loading is an explicit application decision, not library behavior.
    const char* key = std::getenv("TYPESAFE_API_KEY");
    if (!key || !*key) {
        std::cerr << "TYPESAFE_API_KEY is required. No request sent.\n";
        return 2;
    }
    try {
        jevt::remote_options options;
        options.api_key = key;
        options.max_retries = 0; // Never repeat a potentially billable demo request.
        if (const char* model = std::getenv("TYPESAFE_MODEL"); model && *model) options.model = model;
        auto backend = std::make_shared<jevt::typesafe_backend>(options);
        auto brain = jevt::bind_system_one(ticket, backend);
        auto answer = brain.evaluate(jevt::json_state(R"({
          "ticket":{"body":"Nobody can log in. The entire team is blocked."},
          "customer":{"plan":"enterprise","active_users":120},
          "context":{"workaround_available":false,"sla_minutes":60}
        })"), jevt::evaluation_options{
            .deadline = std::chrono::steady_clock::now() + std::chrono::seconds{15}});
        if (!answer) {
            std::cerr << "Inference failed: " << answer.error_value().message << '\n';
            return 1;
        }
        const auto& team = answer->get<"team">();
        std::cout << "team: " << (team.abstained() ? "manual review" :
            team.value() == Team::billing ? "billing" : "technical") << '\n';
        const auto& urgent = answer->get<"urgent">();
        std::cout << "urgent: " << (urgent.abstained() ? "uncertain" : urgent.value() ? "yes" : "no") << '\n';
        std::cout << "urgency expectation: " << answer->get<"urgency">().score() << '\n';
        std::cout << "P(angry): " << answer->get<"angry">().value() << '\n';
        std::cout << "provider: " << team.metadata().provider
                  << ", model: " << team.metadata().model_revision
                  << ", attempts: " << team.metadata().attempts << '\n';
    } catch (const std::exception& failure) {
        std::cerr << "Configuration failed: " << failure.what() << '\n';
        return 2;
    }
}
