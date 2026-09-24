#include <jevt/jevt.hpp>
#include <memory>

enum class Team { billing, technical };
constexpr auto teams = jevt::schema<Team, "consumer.team">(
    jevt::option<Team::billing>("Billing"), jevt::option<Team::technical>("Technical"));

int main() {
    auto backend = std::make_shared<jevt::keyword_backend>(
        jevt::keyword_backend::keyword_table{{"invoice"}, {"outage"}});
    const jevt::context runtime{{.inference_backend = backend}};
    auto pending = runtime.bind(teams).choose_async("invoice");
    auto result = pending.get();
    return result && result->has_value() && result->value() == Team::billing ? 0 : 1;
}
