#pragma once

#include <jevt/core.hpp>
#include <chrono>
#include <map>

namespace jevt {

// Optional remote module. Construction is explicit; local backends never fall
// back to this module or read credentials from the environment.
enum class remote_provider { typesafe, openrouter };
struct remote_options {
    std::string api_key;
    std::string base_url = "https://api.typesafe.ai";
    std::string model = "jev-latest";
    remote_provider provider = remote_provider::typesafe;
    std::chrono::milliseconds call_timeout{30000};
    std::chrono::milliseconds attempt_timeout{10000};
    std::size_t max_retries = 2;
    std::chrono::milliseconds initial_backoff{500};
    std::chrono::milliseconds max_backoff{5000};
    std::size_t max_request_bytes = 4 * 1024 * 1024;
    std::size_t max_response_bytes = 4 * 1024 * 1024;
    // Only literal localhost/127.0.0.1/[::1] HTTP endpoints, for local tests.
    bool allow_insecure_loopback = false;
};
[[nodiscard]] remote_options openrouter_options(std::string api_key);

struct http_request {
    std::string method;
    std::string url;
    std::map<std::string, std::string> headers;
    std::string body;
    std::chrono::steady_clock::time_point deadline;
    std::stop_token cancellation;
    std::size_t max_response_bytes = 4 * 1024 * 1024;
};
struct http_response {
    int status = 0;
    std::string body;
    std::map<std::string, std::string> headers;
};
class http_transport {
public:
    virtual ~http_transport() = default;
    // Implementations must be thread safe and honor the absolute deadline and
    // stop token. Late responses are discarded, but arbitrary injected blocking
    // code cannot be forcibly interrupted by the caller.
    [[nodiscard]] virtual result<http_response> perform(const http_request&) = 0;
};
// A bounded retained pool of exclusive libcurl easy/multi pairs. Connections
// survive calls and can be reused across worker threads. No redirects allowed.
[[nodiscard]] std::shared_ptr<http_transport> make_curl_transport(std::size_t retained_connections = 8);

struct remote_model {
    std::string name;
    std::string description;
    std::string release_date;
};
class typesafe_backend final : public backend {
public:
    explicit typesafe_backend(remote_options, std::shared_ptr<http_transport> = {});
    ~typesafe_backend() override;
    typesafe_backend(const typesafe_backend&) = delete;
    typesafe_backend& operator=(const typesafe_backend&) = delete;
    [[nodiscard]] result<inference_response> predict(const inference_request&) override;
    // Different input states are never merged. Each same-state HTTP group is
    // atomic: cancellation of one constituent cancels the enclosing batch.
    [[nodiscard]] result<std::vector<inference_response>> predict_batch(std::span<const inference_request>) override;
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] result<std::vector<remote_model>> list_models(evaluation_options = {}) const;
private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

} // namespace jevt
