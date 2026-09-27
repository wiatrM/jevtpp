#include <jevt/cache.hpp>
#include "test_harness.hpp"

#include <limits>

JEVT_TEST("revisioned cache fences late writes and invalidates on revision changes") {
    jevt::revisioned_cache<std::string> cache({4, 64});
    auto ticket = cache.advance({.model="m1", .schema="s1"});
    auto value = std::make_shared<const std::string>("prepared");
    JEVT_REQUIRE(cache.put("input", ticket, value, value->size()));
    JEVT_REQUIRE_EQ(*cache.get("input", ticket), "prepared");

    const auto next = cache.advance({.model="m2", .schema="s1"});
    JEVT_REQUIRE(next != ticket);
    JEVT_REQUIRE(!cache.get("input", ticket));
    JEVT_REQUIRE(!cache.put("late", ticket, value, value->size()));
    JEVT_REQUIRE_EQ(cache.stats().revision_invalidations, 1u);
    JEVT_REQUIRE_EQ(cache.stats().stale_writes, 1u);
}

JEVT_TEST("revisioned cache enforces byte, entry and TTL limits") {
    using cache_type = jevt::revisioned_cache<int>;
    cache_type cache({2, 8});
    const auto ticket = cache.advance({.prompt="p1"});
    const auto first = std::make_shared<const int>(1);
    const auto second = std::make_shared<const int>(2);
    const auto start = cache_type::clock::time_point{} + std::chrono::seconds(10);
    // Three retained bytes plus the one-byte key keeps each entry exactly four bytes.
    JEVT_REQUIRE(cache.put("a", ticket, first, 3, std::chrono::seconds(5), start));
    JEVT_REQUIRE(!cache.get("a", ticket, start + std::chrono::seconds(5)));
    JEVT_REQUIRE(cache.put("a", ticket, first, 3, cache_type::clock::duration::max(), start));
    JEVT_REQUIRE(cache.put("b", ticket, second, 3, cache_type::clock::duration::max(), start));
    JEVT_REQUIRE(cache.put("c", ticket, first, 3, cache_type::clock::duration::max(), start));
    JEVT_REQUIRE_EQ(cache.stats().entries, 2u);
    JEVT_REQUIRE_EQ(cache.stats().evictions, 1u);
    JEVT_REQUIRE(!cache.put("oversized", ticket, first, std::numeric_limits<std::size_t>::max()));
    JEVT_REQUIRE(cache.stats().retained_bytes <= 8u);
}

JEVT_TEST("stable revisions retain entries and explicit invalidation fences writes") {
    jevt::revisioned_cache<int> cache({2, 64});
    const auto issued = cache.advance({.knowledge="k1"});
    const auto value = std::make_shared<const int>(7);
    JEVT_REQUIRE(cache.put("key", issued, value, sizeof(int)));
    JEVT_REQUIRE_EQ(cache.advance({.knowledge="k1"}), issued);
    const auto held = cache.get("key", issued);
    cache.invalidate();
    JEVT_REQUIRE(!cache.get("key", issued));
    JEVT_REQUIRE(!cache.put("key", issued, value, sizeof(int)));
    JEVT_REQUIRE_EQ(*held, 7);
    JEVT_REQUIRE_EQ(cache.stats().retained_bytes, 0u);
    const auto next = cache.advance({.knowledge="k1"});
    JEVT_REQUIRE(next != issued);
    JEVT_REQUIRE(cache.put("key", next, value, sizeof(int)));
}

int main() { return jevt::test::run_all("revisioned cache"); }
