#include <jevt/remote.hpp>
#include <curl/curl.h>

#include <array>
#include <climits>
#include <mutex>
#include <stdexcept>

namespace jevt {
namespace {
using clock_type = std::chrono::steady_clock;
struct connection {
    CURL* easy = curl_easy_init();
    CURLM* multi = curl_multi_init();
    connection() {
        if (!easy || !multi) {
            if (easy) curl_easy_cleanup(easy);
            if (multi) curl_multi_cleanup(multi);
            throw std::runtime_error("cannot initialize HTTP connection");
        }
    }
    ~connection() { curl_easy_cleanup(easy); curl_multi_cleanup(multi); }
};
struct receive_state {
    http_response response;
    std::size_t limit;
    std::size_t header_bytes = 0;
    bool overflow = false;
    bool allocation_failed = false;
};
std::size_t receive_body(char* data, std::size_t size, std::size_t count, void* opaque) noexcept {
    auto& state = *static_cast<receive_state*>(opaque);
    if (size && count > SIZE_MAX / size) { state.overflow = true; return 0; }
    const auto bytes = size * count;
    if (bytes > state.limit - state.response.body.size()) { state.overflow = true; return 0; }
    try { state.response.body.append(data, bytes); }
    catch (...) { state.allocation_failed = true; return 0; }
    return bytes;
}
std::size_t receive_header(char* data, std::size_t size, std::size_t count, void* opaque) noexcept {
    auto& state = *static_cast<receive_state*>(opaque);
    if (size && count > SIZE_MAX / size) { state.overflow = true; return 0; }
    const auto bytes = size * count;
    if (bytes > 65536 - state.header_bytes) { state.overflow = true; return 0; }
    state.header_bytes += bytes;
    try {
        const std::string_view line(data, bytes);
        if (line.starts_with("HTTP/")) state.response.headers.clear();
        else if (auto colon = line.find(':'); colon != std::string_view::npos) {
            auto key = std::string(line.substr(0, colon));
            for (auto& c : key) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
            auto value = line.substr(colon + 1);
            while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) value.remove_prefix(1);
            while (!value.empty() && (value.back() == '\r' || value.back() == '\n' || value.back() == ' ' || value.back() == '\t')) value.remove_suffix(1);
            state.response.headers[std::move(key)] = value;
        }
    } catch (...) { state.allocation_failed = true; return 0; }
    return bytes;
}
class curl_transport final : public http_transport {
    std::mutex mutex_;
    std::vector<std::unique_ptr<connection>> idle_;
    std::size_t retained_;
    struct lease {
        curl_transport& owner;
        std::unique_ptr<connection> value;
        ~lease() {
            if (!value) return;
            std::lock_guard lock(owner.mutex_);
            if (owner.idle_.size() < owner.retained_) owner.idle_.push_back(std::move(value));
        }
    };
public:
    explicit curl_transport(std::size_t retained) : retained_(retained) {
        // libcurl must have an asynchronous resolver for DNS to respect the
        // multi-loop deadline; never silently promise cancellable blocking DNS.
        static const auto initialized = curl_global_init(CURL_GLOBAL_DEFAULT);
        if (initialized != CURLE_OK || !(curl_version_info(CURLVERSION_NOW)->features & CURL_VERSION_ASYNCHDNS))
            throw std::runtime_error("HTTP transport requires libcurl with asynchronous DNS");
        idle_.reserve(retained_);
    }
    result<http_response> perform(const http_request& request) override {
        if (auto e = execution_error(evaluation_options{request.deadline, request.cancellation})) return *e;
        if ((request.method != "GET" && request.method != "POST") || !request.max_response_bytes ||
            (!request.url.starts_with("https://") && !request.url.starts_with("http://")))
            return error{error_code::invalid_request, "unsupported HTTP transport request"};
        for (unsigned char c : request.url)
            if (c <= 0x20 || c == 0x7f || c == '\\')
                return error{error_code::invalid_request, "invalid HTTP URL character"};
        lease slot{*this, {}};
        {
            std::lock_guard lock(mutex_);
            if (!idle_.empty()) { slot.value = std::move(idle_.back()); idle_.pop_back(); }
        }
        if (!slot.value) slot.value = std::make_unique<connection>();
        auto* easy = slot.value->easy;
        auto* multi = slot.value->multi;
        curl_easy_reset(easy);
        receive_state receive{{}, request.max_response_bytes};
        struct header_list {
            curl_slist* value = nullptr;
            ~header_list() { curl_slist_free_all(value); }
        } headers;
        for (const auto& [key, value] : request.headers) {
            if (key.empty() || key.find_first_of(":\r\n") != std::string::npos ||
                value.find_first_of("\r\n") != std::string::npos || key.find('\0') != std::string::npos || value.find('\0') != std::string::npos)
                return error{error_code::invalid_request, "invalid HTTP header"};
            auto* added = curl_slist_append(headers.value, (key + ": " + value).c_str());
            if (!added) return error{error_code::connection_failure, "cannot allocate HTTP headers"};
            headers.value = added;
        }
        const auto remaining = std::chrono::ceil<std::chrono::milliseconds>(request.deadline - clock_type::now()).count();
        if (remaining <= 0) return error{error_code::timeout, "HTTP deadline exceeded"};
        CURLcode setup = CURLE_OK;
        auto set = [&](auto option, auto value) {
            if (setup == CURLE_OK) setup = curl_easy_setopt(easy, option, value);
        };
        set(CURLOPT_URL, request.url.c_str());
        set(CURLOPT_HTTPHEADER, headers.value);
        set(CURLOPT_NOSIGNAL, 1L);
        set(CURLOPT_TIMEOUT_MS, static_cast<long>(std::min<long long>(remaining, LONG_MAX)));
        set(CURLOPT_FOLLOWLOCATION, 0L);
        set(CURLOPT_SSL_VERIFYPEER, 1L);
        set(CURLOPT_SSL_VERIFYHOST, 2L);
#if LIBCURL_VERSION_NUM >= 0x075500
        set(CURLOPT_PROTOCOLS_STR, "http,https");
#else
        set(CURLOPT_PROTOCOLS, static_cast<long>(CURLPROTO_HTTP | CURLPROTO_HTTPS));
#endif
        set(CURLOPT_WRITEFUNCTION, &receive_body);
        set(CURLOPT_WRITEDATA, &receive);
        set(CURLOPT_HEADERFUNCTION, &receive_header);
        set(CURLOPT_HEADERDATA, &receive);
        if (request.method == "POST") {
            set(CURLOPT_POST, 1L);
            set(CURLOPT_POSTFIELDS, request.body.data());
            set(CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(request.body.size()));
        } else set(CURLOPT_HTTPGET, 1L);
        if (setup != CURLE_OK || curl_multi_add_handle(multi, easy) != CURLM_OK)
            return error{error_code::connection_failure, "cannot configure HTTP transfer"};
        struct attached {
            CURLM* multi; CURL* easy;
            ~attached() { curl_multi_remove_handle(multi, easy); }
        } transfer{multi, easy};
        int running = 0;
        do {
            if (auto e = execution_error(evaluation_options{request.deadline, request.cancellation})) return *e;
            if (curl_multi_perform(multi, &running) != CURLM_OK)
                return error{error_code::connection_failure, "HTTP event loop failed"};
            if (!running) break;
            const auto left = std::chrono::ceil<std::chrono::milliseconds>(request.deadline - clock_type::now()).count();
            const auto wait = static_cast<int>(std::clamp<long long>(left, 0, 20));
            if (curl_multi_poll(multi, nullptr, 0, wait, nullptr) != CURLM_OK)
                return error{error_code::connection_failure, "HTTP polling failed"};
        } while (running);
        if (auto e = execution_error(evaluation_options{request.deadline, request.cancellation})) return *e;
        if (receive.overflow) return error{error_code::invalid_backend_output, "HTTP response exceeds byte limit"};
        if (receive.allocation_failed) return error{error_code::connection_failure, "cannot allocate HTTP response"};
        int messages = 0;
        auto code = CURLE_FAILED_INIT;
        while (auto* message = curl_multi_info_read(multi, &messages))
            if (message->msg == CURLMSG_DONE && message->easy_handle == easy) code = message->data.result;
        if (code != CURLE_OK)
            return error{code == CURLE_OPERATION_TIMEDOUT ? error_code::timeout : error_code::connection_failure,
                         "HTTP transfer failed"};
        long status = 0;
        if (curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &status) != CURLE_OK)
            return error{error_code::connection_failure, "HTTP status unavailable"};
        receive.response.status = static_cast<int>(status);
        return std::move(receive.response);
    }
};
} // namespace
std::shared_ptr<http_transport> make_curl_transport(std::size_t retained_connections) {
    if (!retained_connections || retained_connections > 1024)
        throw std::invalid_argument("retained HTTP connection count must be 1..1024");
    return std::make_shared<curl_transport>(retained_connections);
}
} // namespace jevt
