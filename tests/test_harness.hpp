#pragma once

#include <chrono>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace jevt::test {

class failure final : public std::runtime_error {
public:
    failure(std::string expression, std::string file, int line, std::string detail = {})
        : std::runtime_error(format(std::move(expression), std::move(file), line,
                                    std::move(detail))) {}

private:
    static std::string format(std::string expression, std::string file, int line,
                              std::string detail) {
        std::ostringstream out;
        out << file << ':' << line << ": assertion `" << expression << "` failed";
        if (!detail.empty()) {
            out << " (" << detail << ')';
        }
        return out.str();
    }
};

using test_function = void (*)();

struct test_case {
    std::string_view name;
    test_function function;
};

inline auto registry() -> std::vector<test_case>& {
    static std::vector<test_case> tests;
    return tests;
}

class registrar {
public:
    registrar(std::string_view name, test_function function) {
        registry().push_back({name, function});
    }
};

template <class Actual, class Expected>
void require_equal(Actual&& actual, Expected&& expected, const char* actual_text,
                   const char* expected_text, const char* file, int line) {
    if (!(actual == expected)) {
        std::ostringstream detail;
        detail << actual_text << " != " << expected_text;
        throw failure(actual_text, file, line, detail.str());
    }
}

inline void require_near(double actual, double expected, double tolerance,
                         const char* actual_text, const char* file, int line) {
    if (std::abs(actual - expected) > tolerance) {
        std::ostringstream detail;
        detail << actual << " is not within " << tolerance << " of " << expected;
        throw failure(actual_text, file, line, detail.str());
    }
}

template <class Predicate>
void eventually(Predicate&& predicate, std::chrono::milliseconds timeout,
                std::chrono::milliseconds poll = std::chrono::milliseconds{1}) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (std::invoke(predicate)) {
            return;
        }
        std::this_thread::sleep_for(poll);
    }
    if (!std::invoke(predicate)) {
        throw failure("eventually(predicate)", __FILE__, __LINE__, "timeout");
    }
}

inline int run_all(std::string_view suite) {
    std::size_t passed = 0;
    std::cerr << "[==========] " << registry().size() << " tests from " << suite << '\n';
    for (const auto& test : registry()) {
        const auto started = std::chrono::steady_clock::now();
        try {
            test.function();
            ++passed;
            const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - started);
            std::cerr << "[       OK ] " << test.name << " (" << elapsed.count() << " us)\n";
        } catch (const std::exception& error) {
            std::cerr << "[  FAILED  ] " << test.name << ": " << error.what() << '\n';
        } catch (...) {
            std::cerr << "[  FAILED  ] " << test.name << ": unknown exception\n";
        }
    }
    std::cerr << "[==========] " << passed << " passed, "
              << registry().size() - passed << " failed\n";
    return passed == registry().size() ? 0 : 1;
}

} // namespace jevt::test

#define JEVT_TEST_DETAIL_JOIN_IMPL(a, b) a##b
#define JEVT_TEST_DETAIL_JOIN(a, b) JEVT_TEST_DETAIL_JOIN_IMPL(a, b)
#define JEVT_TEST(name)                                                                    \
    static void JEVT_TEST_DETAIL_JOIN(jevt_test_, __LINE__)();                             \
    static ::jevt::test::registrar JEVT_TEST_DETAIL_JOIN(jevt_registrar_, __LINE__){       \
        name, &JEVT_TEST_DETAIL_JOIN(jevt_test_, __LINE__)};                               \
    static void JEVT_TEST_DETAIL_JOIN(jevt_test_, __LINE__)()

#define JEVT_REQUIRE(expression)                                                            \
    do {                                                                                    \
        if (!(expression)) {                                                                \
            throw ::jevt::test::failure(#expression, __FILE__, __LINE__);                  \
        }                                                                                   \
    } while (false)

#define JEVT_REQUIRE_EQ(actual, expected)                                                    \
    ::jevt::test::require_equal((actual), (expected), #actual, #expected, __FILE__, __LINE__)

#define JEVT_REQUIRE_NEAR(actual, expected, tolerance)                                      \
    ::jevt::test::require_near((actual), (expected), (tolerance), #actual, __FILE__, __LINE__)

