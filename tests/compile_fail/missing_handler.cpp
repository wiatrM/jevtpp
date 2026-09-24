#include <jevt/core.hpp>
enum class Team { billing, technical };
constexpr auto schema = jevt::schema<Team, "missing">(
    jevt::option<Team::billing>("Billing"), jevt::option<Team::technical>("Technical"));
void dispatch(const jevt::decision_result<std::remove_cv_t<decltype(schema)>>& result) {
    jevt::match(schema, result, jevt::on<Team::billing>([] {}), jevt::on_abstain([] {}));
}
