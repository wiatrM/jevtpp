#include <jevt/core.hpp>
enum class Team { billing };
constexpr auto invalid = jevt::schema<Team, "duplicate">(
    jevt::option<Team::billing>("First"), jevt::option<Team::billing>("Second"));
