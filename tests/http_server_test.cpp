#include <jevt/diagnostics.hpp>
#include <jevt/http_server.hpp>

#include "test_harness.hpp"

JEVT_TEST("HTTP diagnostics server supports an ephemeral-port lifecycle") {
    jevt::Diagnostics diagnostics;
    diagnostics.record_call("support.routing", jevt::CallOutcome::success,
                            std::chrono::microseconds{250});
    jevt::HttpServer server{diagnostics};

    JEVT_REQUIRE(!server.running());
    JEVT_REQUIRE_EQ(server.port(), 0u);
    server.start({.bind_address = "127.0.0.1", .port = 0});
    JEVT_REQUIRE(server.running());
    JEVT_REQUIRE(server.port() != 0);
    JEVT_REQUIRE(server.url().find("127.0.0.1") != std::string::npos);
    JEVT_REQUIRE(server.url().find(std::to_string(server.port())) != std::string::npos);

    server.stop();
    JEVT_REQUIRE(!server.running());
    server.stop(); // idempotent shutdown is important for RAII/error paths.
}

int main() {
    return jevt::test::run_all("diagnostics HTTP server");
}
