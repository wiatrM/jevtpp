#include <jevt/remote.hpp>
#include "test_harness.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <mutex>
#include <sstream>

using namespace std::chrono_literals;
namespace {
struct observed_request {
    std::string method, target, body;
    std::map<std::string, std::string> headers;
};
struct server_response {
    int status = 200;
    std::string body = "ok";
    std::map<std::string, std::string> headers;
    std::chrono::milliseconds delay{0};
};

// A private ephemeral-port server. No DNS or external socket is used. Its
// listener and connection loops are interruptible, including delayed replies.
class loopback_server {
public:
    explicit loopback_server(std::vector<server_response> responses) : responses_(std::move(responses)) {
        listener_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listener_ < 0) throw std::runtime_error("loopback socket failed");
        try {
            sockaddr_in address{};
            address.sin_family = AF_INET;
            address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            address.sin_port = 0;
            if (::bind(listener_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0 ||
                ::listen(listener_, 8) != 0) throw std::runtime_error("loopback bind/listen failed");
            socklen_t size = sizeof(address);
            if (::getsockname(listener_, reinterpret_cast<sockaddr*>(&address), &size) != 0)
                throw std::runtime_error("loopback getsockname failed");
            port_ = ntohs(address.sin_port);
            thread_ = std::thread([this] {
                try { serve(); }
                catch (...) { std::lock_guard lock(mutex_); failure_ = std::current_exception(); changed_.notify_all(); }
            });
        } catch (...) { ::close(listener_); throw; }
    }
    ~loopback_server() {
        stopping_.store(true);
        changed_.notify_all();
        ::shutdown(listener_, SHUT_RDWR);
        if (thread_.joinable()) thread_.join();
        ::close(listener_);
    }
    std::string url(std::string_view path = "/test") const {
        return "http://127.0.0.1:" + std::to_string(port_) + std::string(path);
    }
    std::vector<observed_request> requests() const {
        std::lock_guard lock(mutex_);
        if (failure_) std::rethrow_exception(failure_);
        return requests_;
    }
    void wait_for_requests(std::size_t count) {
        std::unique_lock lock(mutex_);
        const bool ready = changed_.wait_for(lock, 2s, [&] { return failure_ || requests_.size() >= count; });
        if (failure_) std::rethrow_exception(failure_);
        if (!ready) throw std::runtime_error("loopback request did not arrive");
    }
    int accepts() const { return accepts_.load(); }

private:
    static std::string lowercase(std::string value) {
        for (auto& c : value) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
        return value;
    }
    bool read_more(int socket, std::string& buffer) const {
        while (!stopping_.load()) {
            pollfd descriptor{socket, POLLIN, 0};
            const int ready = ::poll(&descriptor, 1, 20);
            if (ready < 0) { if (errno == EINTR) continue; throw std::runtime_error("loopback poll failed"); }
            if (!ready) continue;
            char data[4096];
            const auto count = ::recv(socket, data, sizeof(data), 0);
            if (count <= 0) return false;
            buffer.append(data, static_cast<std::size_t>(count));
            if (buffer.size() > 1024 * 1024) throw std::runtime_error("test request too large");
            return true;
        }
        return false;
    }
    bool read_request(int socket, std::string& buffer, observed_request& out) const {
        auto end = buffer.find("\r\n\r\n");
        while (end == std::string::npos) {
            if (!read_more(socket, buffer)) return false;
            end = buffer.find("\r\n\r\n");
        }
        std::istringstream headers(buffer.substr(0, end));
        std::string line;
        std::getline(headers, line);
        std::istringstream start(line);
        start >> out.method >> out.target;
        std::size_t content_length = 0;
        while (std::getline(headers, line)) {
            const auto colon = line.find(':');
            if (colon == std::string::npos) continue;
            auto key = lowercase(line.substr(0, colon));
            auto value = line.substr(colon + 1);
            while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.erase(0, 1);
            while (!value.empty() && (value.back() == '\r' || value.back() == '\n')) value.pop_back();
            if (key == "content-length") content_length = std::stoul(value);
            out.headers[std::move(key)] = std::move(value);
        }
        if (content_length > 1024 * 1024) throw std::runtime_error("test body too large");
        const auto consumed = end + 4 + content_length;
        while (buffer.size() < consumed) if (!read_more(socket, buffer)) return false;
        out.body = buffer.substr(end + 4, content_length);
        buffer.erase(0, consumed);
        return true;
    }
    static bool send_all(int socket, std::string_view bytes) {
        while (!bytes.empty()) {
#ifdef MSG_NOSIGNAL
            const auto sent = ::send(socket, bytes.data(), bytes.size(), MSG_NOSIGNAL);
#else
            const auto sent = ::send(socket, bytes.data(), bytes.size(), 0);
#endif
            if (sent < 0 && errno == EINTR) continue;
            if (sent <= 0) return false;
            bytes.remove_prefix(static_cast<std::size_t>(sent));
        }
        return true;
    }
    void serve_connection(int socket) {
        std::string buffer;
        while (!stopping_.load()) {
            observed_request request;
            if (!read_request(socket, buffer, request)) return;
            server_response response;
            {
                std::unique_lock lock(mutex_);
                const auto index = requests_.size();
                requests_.push_back(std::move(request));
                if (index < responses_.size()) response = responses_[index];
                changed_.notify_all();
                if (response.delay.count() && changed_.wait_for(lock, response.delay, [&] { return stopping_.load(); })) return;
            }
            std::string wire = "HTTP/1.1 " + std::to_string(response.status) + " Test\r\nContent-Length: " +
                std::to_string(response.body.size()) + "\r\nConnection: keep-alive\r\n";
            for (const auto& [key, value] : response.headers) wire += key + ": " + value + "\r\n";
            wire += "\r\n" + response.body;
            if (!send_all(socket, wire)) return;
        }
    }
    void serve() {
        while (!stopping_.load()) {
            pollfd descriptor{listener_, POLLIN, 0};
            const auto ready = ::poll(&descriptor, 1, 20);
            if (ready <= 0) continue;
            const int client = ::accept(listener_, nullptr, nullptr);
            if (client < 0) { if (stopping_.load()) return; throw std::runtime_error("loopback accept failed"); }
            ++accepts_;
#ifdef SO_NOSIGPIPE
            const int enabled = 1;
            ::setsockopt(client, SOL_SOCKET, SO_NOSIGPIPE, &enabled, sizeof(enabled));
#endif
            try { serve_connection(client); }
            catch (...) { ::close(client); throw; }
            ::close(client);
        }
    }
    int listener_ = -1;
    unsigned short port_ = 0;
    std::atomic<bool> stopping_{false};
    std::atomic<int> accepts_{0};
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::vector<server_response> responses_;
    std::vector<observed_request> requests_;
    std::exception_ptr failure_;
    std::thread thread_;
};

jevt::http_request request(const loopback_server& server, std::string method = "GET") {
    return {std::move(method), server.url(), {}, {}, std::chrono::steady_clock::now() + 2s, {}};
}

// Curl honors environment proxy settings; force only the literal loopback host
// to bypass them so these tests cannot send traffic to an external proxy.
struct loopback_proxy_bypass {
    std::optional<std::string> upper, lower;
    loopback_proxy_bypass() {
        if (const char* value = std::getenv("NO_PROXY")) upper = value;
        if (const char* value = std::getenv("no_proxy")) lower = value;
        ::setenv("NO_PROXY", "127.0.0.1", 1);
        ::setenv("no_proxy", "127.0.0.1", 1);
    }
    ~loopback_proxy_bypass() {
        if (upper) ::setenv("NO_PROXY", upper->c_str(), 1); else ::unsetenv("NO_PROXY");
        if (lower) ::setenv("no_proxy", lower->c_str(), 1); else ::unsetenv("no_proxy");
    }
};
}

JEVT_TEST("curl sends POST body and headers then reuses the connection for GET") {
    loopback_server server({{201, "created", {{"X-Reply", " first "}}}, {200, "listed", {{"X-Reply", "second"}}}});
    auto transport = jevt::make_curl_transport(1);
    auto post = request(server, "POST");
    post.url = server.url("/submit");
    post.headers = {{"Content-Type", "application/octet-stream"}, {"X-Request", "custom"}};
    post.body = std::string("binary\0body", 11);
    auto first = transport->perform(post);
    JEVT_REQUIRE(first);
    JEVT_REQUIRE_EQ(first->status, 201);
    JEVT_REQUIRE_EQ(first->body, "created");
    JEVT_REQUIRE_EQ(first->headers.at("x-reply"), "first");
    auto get = request(server);
    get.url = server.url("/models?limit=1");
    auto second = transport->perform(get);
    JEVT_REQUIRE(second);
    JEVT_REQUIRE_EQ(second->body, "listed");
    JEVT_REQUIRE_EQ(second->headers.at("x-reply"), "second");
    const auto recorded = server.requests();
    JEVT_REQUIRE_EQ(recorded.size(), 2u);
    JEVT_REQUIRE_EQ(recorded[0].method, "POST");
    JEVT_REQUIRE_EQ(recorded[0].target, "/submit");
    JEVT_REQUIRE_EQ(recorded[0].body, post.body);
    JEVT_REQUIRE_EQ(recorded[0].headers.at("x-request"), "custom");
    JEVT_REQUIRE_EQ(recorded[1].method, "GET");
    JEVT_REQUIRE_EQ(recorded[1].target, "/models?limit=1");
    JEVT_REQUIRE(recorded[1].body.empty());
    JEVT_REQUIRE(!recorded[1].headers.contains("x-request"));
    JEVT_REQUIRE_EQ(server.accepts(), 1);
}

JEVT_TEST("curl deadline terminates a server that leaves the response idle") {
    loopback_server server({{200, "late", {}, 1h}});
    auto transport = jevt::make_curl_transport();
    auto value = request(server);
    value.deadline = std::chrono::steady_clock::now() + 80ms;
    const auto started = std::chrono::steady_clock::now();
    const auto response = transport->perform(value);
    JEVT_REQUIRE(!response);
    JEVT_REQUIRE_EQ(response.error_value().code, jevt::error_code::timeout);
    JEVT_REQUIRE(std::chrono::steady_clock::now() - started < 2s);
    JEVT_REQUIRE_EQ(server.requests().size(), 1u);
}

JEVT_TEST("curl stop token cancels while waiting for response bytes") {
    loopback_server server({{200, "late", {}, 1h}});
    auto transport = jevt::make_curl_transport();
    std::stop_source cancellation;
    auto value = request(server);
    value.deadline = std::chrono::steady_clock::now() + 5s;
    value.cancellation = cancellation.get_token();
    auto pending = std::async(std::launch::async, [&] { return transport->perform(value); });
    server.wait_for_requests(1);
    cancellation.request_stop();
    JEVT_REQUIRE_EQ(pending.wait_for(1s), std::future_status::ready);
    auto response = pending.get();
    JEVT_REQUIRE(!response);
    JEVT_REQUIRE_EQ(response.error_value().code, jevt::error_code::cancelled);
}

JEVT_TEST("curl rejects oversized response bodies") {
    loopback_server server({{200, std::string(1024, 'x'), {}}});
    auto transport = jevt::make_curl_transport();
    auto value = request(server);
    value.max_response_bytes = 16;
    const auto response = transport->perform(value);
    JEVT_REQUIRE(!response);
    JEVT_REQUIRE_EQ(response.error_value().code, jevt::error_code::invalid_backend_output);
}

JEVT_TEST("curl returns redirects without following Location") {
    loopback_server server({{302, "moved", {{"Location", "/redirect-target"}}}, {200, "must not follow", {}}});
    auto transport = jevt::make_curl_transport();
    auto response = transport->perform(request(server));
    JEVT_REQUIRE(response);
    JEVT_REQUIRE_EQ(response->status, 302);
    JEVT_REQUIRE_EQ(response->body, "moved");
    JEVT_REQUIRE_EQ(response->headers.at("location"), "/redirect-target");
    JEVT_REQUIRE_EQ(server.requests().size(), 1u);
}

JEVT_TEST("curl rejects pre-cancelled expired and malformed requests before network IO") {
    loopback_server server({});
    auto transport = jevt::make_curl_transport();
    auto expired = request(server);
    expired.deadline = std::chrono::steady_clock::now() - 1ms;
    JEVT_REQUIRE_EQ(transport->perform(expired).error_value().code, jevt::error_code::timeout);
    std::stop_source stop; stop.request_stop();
    auto cancelled = request(server); cancelled.cancellation = stop.get_token();
    JEVT_REQUIRE_EQ(transport->perform(cancelled).error_value().code, jevt::error_code::cancelled);
    auto invalid = request(server); invalid.headers["Bad\r\nInjected"] = "value";
    JEVT_REQUIRE_EQ(transport->perform(invalid).error_value().code, jevt::error_code::invalid_request);
    invalid = request(server, "DELETE");
    JEVT_REQUIRE_EQ(transport->perform(invalid).error_value().code, jevt::error_code::invalid_request);
    invalid = request(server);
    invalid.url += std::string("\0/ignored", 9);
    JEVT_REQUIRE_EQ(transport->perform(invalid).error_value().code, jevt::error_code::invalid_request);
    JEVT_REQUIRE_EQ(server.accepts(), 0);
}

int main() {
    loopback_proxy_bypass bypass;
    return jevt::test::run_all("curl_transport");
}
