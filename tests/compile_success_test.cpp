#include <jevt/jevt.hpp>

#include <type_traits>

namespace {

enum class state { accepted = 3, rejected = 8 };
inline constexpr auto review = jevt::schema<state, "compile.review">(
    jevt::option<state::accepted>("accepted"),
    jevt::option<state::rejected>("rejected"));

using schema_type = std::remove_cvref_t<decltype(review)>;
static_assert(std::same_as<typename schema_type::enum_type, state>);
static_assert(schema_type::size == 2);
static_assert(schema_type::values()[0] == state::accepted);
static_assert(schema_type::values()[1] == state::rejected);

} // namespace

int main() {
    const jevt::decision_result<schema_type> result{
        state::accepted, {0.9F, 0.1F}, {"compile-test", 0.9F, false}};
    int selected = 0;
    jevt::match(review, result,
                jevt::on<state::accepted>([&] { selected = 1; }),
                jevt::on<state::rejected>([&] { selected = 2; }),
                jevt::on_abstain([&] { selected = 3; }));
    return selected == 1 ? 0 : 1;
}
