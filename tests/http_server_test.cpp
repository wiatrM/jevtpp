#include <jevt/diagnostics.hpp>
#include <jevt/http_server.hpp>

#include "test_harness.hpp"

#include <future>
#include <thread>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#ifdef _MSC_VER
#pragma comment(lib, "ws2_32.lib")
#endif
#else
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {
struct Client {
#ifdef _WIN32
    SOCKET socket = INVALID_SOCKET;
    ~Client() { if (socket != INVALID_SOCKET) closesocket(socket); }
#else
    int socket = -1;
    ~Client() { if (socket != -1) ::close(socket); }
#endif
    void disconnect() {
#ifdef _WIN32
        ::shutdown(socket, SD_BOTH);
#else
        ::shutdown(socket, SHUT_RDWR);
#endif
    }
};

std::string exchange_request(std::uint16_t port, std::string_view request, bool finish_writing = true) {
    Client client{::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)};
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    JEVT_REQUIRE_EQ(::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr), 1);
    JEVT_REQUIRE_EQ(::connect(client.socket, reinterpret_cast<sockaddr*>(&address), sizeof(address)), 0);
#ifdef _WIN32
    const DWORD timeout_ms = 4000;
    JEVT_REQUIRE_EQ(::setsockopt(client.socket, SOL_SOCKET, SO_RCVTIMEO,
        reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms)), 0);
#else
    const timeval timeout{4, 0};
    JEVT_REQUIRE_EQ(::setsockopt(client.socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)), 0);
#endif
    while (!request.empty()) {
        const auto sent = ::send(client.socket, request.data(), static_cast<int>(request.size()), 0);
        JEVT_REQUIRE(sent > 0);
        request.remove_prefix(static_cast<std::size_t>(sent));
    }
    if (finish_writing) {
#ifdef _WIN32
        ::shutdown(client.socket, SD_SEND);
#else
        ::shutdown(client.socket, SHUT_WR);
#endif
    }
    std::string response;
    char buffer[1024];
    for (;;) {
        const auto received = ::recv(client.socket, buffer, sizeof(buffer), 0);
        if (received == 0) break;
        JEVT_REQUIRE(received > 0);
        response.append(buffer, static_cast<std::size_t>(received));
    }
    return response;
}
}

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

JEVT_TEST("shutdown interrupts a client waiting to finish its headers") {
    jevt::Diagnostics diagnostics;
    jevt::HttpServer server{diagnostics};
    server.start({.bind_address = "127.0.0.1", .port = 0});
    Client client{::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)};
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(server.port());
    JEVT_REQUIRE_EQ(::inet_pton(AF_INET, "127.0.0.1", &address.sin_addr), 1);
    JEVT_REQUIRE_EQ(::connect(client.socket, reinterpret_cast<sockaddr*>(&address), sizeof(address)), 0);
    constexpr char partial[] = "GET /healthz HTTP/1.1\r\nHost: localhost\r\n";
    JEVT_REQUIRE(::send(client.socket, partial, sizeof(partial) - 1, 0) > 0);
    // Give the accept worker time to wait for the unfinished headers.
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
    auto stopped = std::async(std::launch::async, [&] { server.stop(); });
    const auto status = stopped.wait_for(std::chrono::milliseconds{500});
    // Release the peer even on regression, so the test can report a failure.
    client.disconnect();
    stopped.get();
    JEVT_REQUIRE(status == std::future_status::ready);
    JEVT_REQUIRE(!server.running());
    JEVT_REQUIRE_EQ(server.port(), 0u);
    server.start({.bind_address = "127.0.0.1", .port = 0});
    server.stop();
}

JEVT_TEST("HTTP routes expose diagnostics and reject unsupported methods") {
    jevt::Diagnostics diagnostics;
    diagnostics.record_call("test.route", jevt::CallOutcome::success, std::chrono::milliseconds{1});
    jevt::HttpServer server{diagnostics};
    server.start();
    const auto get = [&](std::string_view path) {
        return exchange_request(server.port(), "GET " + std::string(path) + " HTTP/1.1\r\nHost: localhost\r\n\r\n");
    };
    const auto health = get("/healthz?probe=1");
    JEVT_REQUIRE(health.starts_with("HTTP/1.1 200 "));
    JEVT_REQUIRE(health.ends_with("\r\n\r\nok\n"));
    const auto stats = get("/api/stats");
    JEVT_REQUIRE(stats.starts_with("HTTP/1.1 200 "));
    JEVT_REQUIRE(stats.find("application/json") != std::string::npos);
    JEVT_REQUIRE(stats.find("\"calls\":1") != std::string::npos);
    const auto metrics = get("/metrics");
    JEVT_REQUIRE(metrics.starts_with("HTTP/1.1 200 "));
    JEVT_REQUIRE(metrics.find("jevt_calls_total 1") != std::string::npos);
    const auto dashboard = get("/");
    JEVT_REQUIRE(dashboard.starts_with("HTTP/1.1 200 "));
    JEVT_REQUIRE(dashboard.find("<!doctype html>") != std::string::npos);
    JEVT_REQUIRE(get("/missing").starts_with("HTTP/1.1 404 "));
    const auto post = exchange_request(server.port(), "POST /healthz HTTP/1.1\r\nHost: localhost\r\n\r\n");
    JEVT_REQUIRE(post.starts_with("HTTP/1.1 405 "));
    JEVT_REQUIRE(post.find("\r\nAllow: GET\r\n") != std::string::npos);
}

JEVT_TEST("HTTP requires complete headers within the inclusive byte limit") {
    jevt::Diagnostics diagnostics;
    jevt::HttpServer server{diagnostics};
    server.start({.max_request_bytes = 64});
    const std::string prefix = "GET /healthz HTTP/1.1\r\nX-Padding: ";
    const auto exact = prefix + std::string(64 - prefix.size() - 4, 'x') + "\r\n\r\n";
    JEVT_REQUIRE_EQ(exact.size(), 64u);
    JEVT_REQUIRE(exchange_request(server.port(), exact).starts_with("HTTP/1.1 200 "));
    // Fill the entire limit without completing the header terminator.
    const auto too_large = prefix + std::string(64 - prefix.size(), 'x');
    JEVT_REQUIRE(exchange_request(server.port(), too_large).starts_with("HTTP/1.1 431 "));
    JEVT_REQUIRE(exchange_request(server.port(), "GET /healthz HTTP/1.1\r\n").starts_with("HTTP/1.1 400 "));
    JEVT_REQUIRE(exchange_request(server.port(), "GET /healthz INVALID\r\n\r\n").starts_with("HTTP/1.1 400 "));
    // An idle peer with an unfinished header must receive an error, not health.
    JEVT_REQUIRE(exchange_request(server.port(), "GET /healthz HTTP/1.1\r\n", false).starts_with("HTTP/1.1 400 "));
}

int main() {
    return jevt::test::run_all("diagnostics HTTP server");
}
