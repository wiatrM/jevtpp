#include <jevt/remote.hpp>
#include <nlohmann/json.hpp>

#include <charconv>
#include <condition_variable>
#include <ctime>
#include <iomanip>
#include <locale>
#include <mutex>
#include <random>
#include <set>
#include <sstream>

namespace jevt {
namespace {
using json = nlohmann::json;
using clock_type = std::chrono::steady_clock;
using milliseconds = std::chrono::milliseconds;

std::string lower(std::string text) {
    for (auto& c : text) if (c >= 'A' && c <= 'Z') c = static_cast<char>(c + ('a' - 'A'));
    return text;
}
json content(content_view value, bool nullable = false) {
    if (value.kind == content_kind::text) return std::string(value.content);
    if (value.kind != content_kind::json) throw std::invalid_argument("unknown content kind");
    const auto parsed = json::parse(value.content);
    if (!parsed.is_string() && !parsed.is_object() && !parsed.is_array() && !(nullable && parsed.is_null()))
        throw std::invalid_argument("structured content must be text, object or array");
    return parsed;
}
void validate_url(const remote_options& options) {
    const auto& url = options.base_url;
    const bool secure = url.starts_with("https://");
    if (!secure && !url.starts_with("http://")) throw std::invalid_argument("remote endpoint requires HTTPS");
    for (const unsigned char c : url)
        if (c <= 0x20 || c >= 0x7f || c == '\\') throw std::invalid_argument("invalid remote endpoint character");
    if (url.find_first_of("?#") != std::string::npos) throw std::invalid_argument("endpoint must not contain query or fragment");
    const auto start = secure ? 8u : 7u;
    const auto end = url.find('/', start);
    const auto authority = url.substr(start, end - start);
    if (authority.empty() || authority.find('@') != std::string::npos)
        throw std::invalid_argument("endpoint requires a host without user credentials");
    if (!secure) {
        auto host = authority;
        if (host.starts_with('[')) {
            const auto close = host.find(']');
            if (close == std::string::npos || (close + 1 < host.size() && host[close + 1] != ':'))
                throw std::invalid_argument("invalid loopback host");
            host.resize(close + 1);
        } else if (const auto colon = host.find(':'); colon != std::string::npos) host.resize(colon);
        if (!options.allow_insecure_loopback || (host != "localhost" && host != "127.0.0.1" && host != "[::1]"))
            throw std::invalid_argument("plain HTTP is restricted to explicitly enabled loopback tests");
    }
}
float unit(const json& value) {
    if (!value.is_number()) throw std::invalid_argument("probability/confidence must be numeric");
    const double number = value.get<double>();
    if (!std::isfinite(number) || number < 0 || number > 1) throw std::invalid_argument("probability/confidence outside [0,1]");
    return static_cast<float>(number);
}
std::uint64_t tokens(const json& value) {
    if (value.is_number_unsigned()) return value.get<std::uint64_t>();
    if (!value.is_number_integer() || value.get<std::int64_t>() < 0)
        throw std::invalid_argument("token usage must be a nonnegative integer");
    return static_cast<std::uint64_t>(value.get<std::int64_t>());
}
std::optional<milliseconds> retry_after(const http_response& response) {
    std::optional<std::string> seconds, millis;
    for (const auto& [key, value] : response.headers) {
        if (lower(key) == "retry-after") seconds = value;
        if (lower(key) == "retry-after-ms") millis = value;
    }
    auto numeric = [](std::string_view value, double scale) -> std::optional<milliseconds> {
        double amount{};
        const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), amount);
        if (ec != std::errc{} || end != value.data() + value.size() || !std::isfinite(amount) || amount < 0) return {};
        return milliseconds{static_cast<long long>(std::min(86400000.0, std::ceil(amount * scale)))};
    };
    if (millis) if (auto value = numeric(*millis, 1)) return value;
    if (!seconds) return {};
    if (auto value = numeric(*seconds, 1000)) return value;
    std::tm time{};
    std::istringstream stream(*seconds);
    stream.imbue(std::locale::classic());
    stream >> std::get_time(&time, "%a, %d %b %Y %H:%M:%S GMT");
    if (stream.fail()) return {};
    const std::chrono::year_month_day date{std::chrono::year{time.tm_year + 1900},
        std::chrono::month{static_cast<unsigned>(time.tm_mon + 1)}, std::chrono::day{static_cast<unsigned>(time.tm_mday)}};
    if (!date.ok() || time.tm_hour > 23 || time.tm_min > 59 || time.tm_sec > 60) return {};
    const auto point = std::chrono::sys_days{date} + std::chrono::hours{time.tm_hour} +
        std::chrono::minutes{time.tm_min} + std::chrono::seconds{time.tm_sec};
    return std::clamp(std::chrono::duration_cast<milliseconds>(point - std::chrono::system_clock::now()),
                      milliseconds{0}, milliseconds{86400000});
}
error status_error(int status) {
    auto code = error_code::remote_error;
    if (status == 401 || status == 403) code = error_code::authentication;
    if (status == 429) code = error_code::rate_limited;
    if (status == 400 || status == 422) code = error_code::invalid_request;
    // Do not echo provider bodies: they may contain request data or credentials.
    return {code, "remote HTTP status " + std::to_string(status), status};
}
struct sent_response { http_response value; std::size_t attempts; };
}

remote_options openrouter_options(std::string key) {
    remote_options options;
    options.api_key = std::move(key);
    options.base_url = "https://openrouter.ai/api";
    options.model = "typesafe/jev-1.13";
    options.provider = remote_provider::openrouter;
    return options;
}
#ifndef JEVT_REMOTE_CURL
std::shared_ptr<http_transport> make_curl_transport(std::size_t) {
    throw std::invalid_argument("remote CURL support is disabled; inject an HTTP transport");
}
#endif

struct typesafe_backend::impl {
    remote_options options;
    std::shared_ptr<http_transport> transport;
    impl(remote_options configured, std::shared_ptr<http_transport> injected)
        : options(std::move(configured)), transport(std::move(injected)) {
        if (options.api_key.empty()) throw std::invalid_argument("remote API key must be explicitly supplied");
        for (const unsigned char c : options.api_key) if (c < '!' || c > '~') throw std::invalid_argument("invalid API key characters");
        if (options.model.empty() || options.call_timeout <= milliseconds{0} || options.attempt_timeout <= milliseconds{0} ||
            options.call_timeout > std::chrono::hours{24} || options.attempt_timeout > std::chrono::hours{24} ||
            options.max_retries > 100 || options.initial_backoff < milliseconds{0} ||
            options.max_backoff < options.initial_backoff || options.max_backoff > std::chrono::hours{24} ||
            !options.max_request_bytes || !options.max_response_bytes)
            throw std::invalid_argument("invalid remote resource/time limits");
        if (options.provider != remote_provider::typesafe && options.provider != remote_provider::openrouter)
            throw std::invalid_argument("unknown remote provider");
        validate_url(options);
        while (options.base_url.ends_with('/')) options.base_url.pop_back();
        if (!transport) transport = make_curl_transport();
    }
    result<sent_response> send(std::string method, std::string path, std::string body,
                               clock_type::time_point deadline, std::stop_token cancellation) const {
        http_request request{std::move(method), options.base_url + path,
            {{"Authorization", "Bearer " + options.api_key}, {"Content-Type", "application/json"},
             {"Accept", "application/json"}, {"User-Agent", "jevtpp-remote/0.2"}}, std::move(body), deadline,
            cancellation, options.max_response_bytes};
        auto delay = options.initial_backoff;
        for (std::size_t attempt = 0; ; ++attempt) {
            if (auto e = execution_error(evaluation_options{deadline, cancellation})) return *e;
            request.deadline = std::min(deadline, clock_type::now() + options.attempt_timeout);
            result<http_response> response = error{error_code::connection_failure, "HTTP transport failed"};
            try { response = transport->perform(request); }
            catch (...) { response = error{error_code::connection_failure, "HTTP transport threw an exception"}; }
            if (auto e = execution_error(evaluation_options{deadline, cancellation})) return *e;
            // Enforce attempt bounds even for an injected transport returning late.
            if (clock_type::now() >= request.deadline) response = error{error_code::timeout, "HTTP attempt deadline exceeded"};
            if (response && response->body.size() > options.max_response_bytes)
                return error{error_code::invalid_backend_output, "HTTP response exceeds byte limit"};
            if (response && response->status >= 200 && response->status < 300)
                return sent_response{std::move(*response), attempt + 1};
            const auto failure = response ? status_error(response->status) : response.error_value();
            const bool retryable = response ? (response->status == 408 || response->status == 429 ||
                (response->status >= 500 && response->status <= 599)) :
                (failure.code == error_code::connection_failure || failure.code == error_code::timeout);
            if (!retryable || attempt >= options.max_retries) return failure;
            thread_local std::mt19937 random{std::random_device{}()};
            const auto jitter = std::uniform_real_distribution<double>{0.75, 1.0}(random);
            const auto wait = response ? retry_after(*response).value_or(milliseconds{static_cast<long long>(delay.count() * jitter)}) :
                milliseconds{static_cast<long long>(delay.count() * jitter)};
            if (wait >= deadline - clock_type::now()) return failure;
            std::mutex mutex;
            std::condition_variable_any wake;
            std::unique_lock lock(mutex);
            wake.wait_until(lock, cancellation, std::min(deadline, clock_type::now() + wait), [] { return false; });
            delay = std::min(options.max_backoff, delay * 2);
        }
    }
};

typesafe_backend::typesafe_backend(remote_options options, std::shared_ptr<http_transport> transport)
    : impl_(std::make_unique<impl>(std::move(options), std::move(transport))) {}
typesafe_backend::~typesafe_backend() = default;
std::string_view typesafe_backend::name() const noexcept { return impl_->options.model; }
result<inference_response> typesafe_backend::predict(const inference_request& request) {
    auto responses = predict_batch(std::span{&request, std::size_t{1}});
    if (!responses) return responses.error_value();
    return std::move(responses->front());
}

result<std::vector<inference_response>> typesafe_backend::predict_batch(std::span<const inference_request> requests) {
    if (requests.empty()) return std::vector<inference_response>{};
    auto deadline = clock_type::now() + impl_->options.call_timeout;
    std::stop_source stop;
    using callback = std::stop_callback<std::function<void()>>;
    std::vector<std::unique_ptr<callback>> cancellation;
    for (const auto& request : requests) {
        if (request.deadline) deadline = std::min(deadline, *request.deadline);
        cancellation.push_back(std::make_unique<callback>(request.cancellation, [&stop] { stop.request_stop(); }));
    }
    if (auto e = execution_error(evaluation_options{deadline, stop.get_token()})) return *e;
    std::vector<inference_response> output(requests.size());
    std::vector<std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < requests.size(); ++i) {
        auto found = std::find_if(groups.begin(), groups.end(), [&](const auto& group) {
            const auto& first = requests[group.front()];
            return first.input == requests[i].input && first.input_kind == requests[i].input_kind;
        });
        if (found == groups.end()) groups.push_back({i}); else found->push_back(i);
    }
    for (const auto& group : groups) {
        json payload;
        try {
            const auto& first = requests[group.front()];
            if (first.input.size() > impl_->options.max_request_bytes) throw std::invalid_argument("remote state exceeds byte limit");
            payload = {{"model", impl_->options.model}, {"state", content({first.input, first.input_kind})}, {"questions", json::object()}};
            for (std::size_t row = 0; row < group.size(); ++row) {
                const auto& r = requests[group[row]];
                if (r.question.size() > impl_->options.max_request_bytes || r.instructions.content.size() > impl_->options.max_request_bytes)
                    throw std::invalid_argument("remote instructions exceed byte limit");
                const auto count = r.options.size();
                const auto expected_metadata = r.question_kind == inference_request::kind::noul ? 2 : count;
                if (!r.criteria_metadata.empty() && r.criteria_metadata.size() != expected_metadata)
                    throw std::invalid_argument("criteria metadata cardinality mismatch");
                const auto criterion = [&](std::size_t index, bool nullable) -> json {
                    const auto value = r.criteria_metadata.empty() ? content_view{r.options[index], content_kind::text} : r.criteria_metadata[index];
                    if (value.content.size() > impl_->options.max_request_bytes) throw std::invalid_argument("criterion exceeds byte limit");
                    return content(value, nullable);
                };
                json q{{"instructions", content(r.instructions.content.empty() && r.instructions.kind == content_kind::text ?
                    content_view{r.question, content_kind::text} : r.instructions)}};
                switch (r.question_kind) {
                    case inference_request::kind::choice:
                        if (!count || count > 255) throw std::invalid_argument("Choice requires 1..255 criteria");
                        q["type"] = "choice"; q["criteria"] = json::object();
                        // C++ enum values have no portable semantic label. A
                        // null rubric paired with synthetic option_N would lose
                        // its meaning; require explicit meaningful metadata.
                        for (std::size_t i = 0; i < count; ++i) q["criteria"]["option_" + std::to_string(i)] = criterion(i, false);
                        break;
                    case inference_request::kind::score:
                        if (count < 2 || count > 10) throw std::invalid_argument("Score requires 2..10 criteria");
                        q["type"] = "score"; q["criteria"] = json::array();
                        for (std::size_t i = 0; i < count; ++i) q["criteria"].push_back(criterion(i, false));
                        break;
                    case inference_request::kind::noul:
                        q["type"] = "noul";
                        if (!r.criteria_metadata.empty() &&
                            (!r.criteria_metadata[0].content.empty() || !r.criteria_metadata[1].content.empty()))
                            q["criteria"] = {{"false", criterion(0, false)}, {"true", criterion(1, false)}};
                        break;
                    default: throw std::invalid_argument("unsupported question kind");
                }
                payload["questions"]["q" + std::to_string(row)] = std::move(q);
            }
        } catch (const std::exception&) { return error{error_code::invalid_request, "invalid remote state, instructions or criteria"}; }
        std::string body;
        try { body = payload.dump(); }
        catch (...) { return error{error_code::invalid_request, "remote request is not valid UTF-8 JSON"}; }
        if (body.size() > impl_->options.max_request_bytes) return error{error_code::invalid_request, "remote request exceeds byte limit"};
        auto response = impl_->send("POST", "/v1/systemone", std::move(body), deadline, stop.get_token());
        if (!response) return response.error_value();
        try {
            const auto parsed = json::parse(response->value.body);
            const auto model = parsed.at("model").get<std::string>();
            if (model.empty()) throw std::invalid_argument("missing model identity");
            const auto& answers = parsed.at("answers");
            if (!answers.is_object() || answers.size() != group.size()) throw std::invalid_argument("answer count mismatch");
            auto usage = std::make_shared<token_usage>(token_usage{tokens(parsed.at("usage").at("input_tokens")), tokens(parsed.at("usage").at("output_tokens"))});
            for (std::size_t row = 0; row < group.size(); ++row) {
                const auto index = group[row];
                const auto& r = requests[index];
                const auto& answer = answers.at("q" + std::to_string(row));
                const auto type = answer.at("type").get<std::string>();
                auto& out = output[index];
                out.model_id = model;
                out.metadata.provider = impl_->options.provider == remote_provider::typesafe ? "typesafe" : "openrouter";
                out.metadata.model_revision = model;
                out.metadata.usage = usage;
                out.metadata.attempts = response->attempts;
                if (r.question_kind == inference_request::kind::noul) {
                    if (type != "noul") throw std::invalid_argument("Noul answer type mismatch");
                    const auto yes = unit(answer.at("noul"));
                    out.scores = {1.0F - yes, yes};
                    continue;
                }
                const bool choice = r.question_kind == inference_request::kind::choice;
                if (type != (choice ? "choice" : "score")) throw std::invalid_argument("answer type mismatch");
                out.metadata.provider_confidence = unit(answer.at("confidence"));
                const auto& probabilities = answer.at("probabilities");
                if (!probabilities.is_object() || probabilities.size() != r.options.size()) throw std::invalid_argument("probability count mismatch");
                double total = 0, expected = 0;
                for (std::size_t i = 0; i < r.options.size(); ++i) {
                    const auto p = unit(probabilities.at((choice ? "option_" : "") + std::to_string(i)));
                    out.scores.push_back(p); total += p; expected += i * static_cast<double>(p);
                }
                if (std::abs(total - 1.0) > 1e-4) throw std::invalid_argument("probability mass must sum to one");
                if (choice) {
                    const auto selected = answer.at("choice").get<std::string>();
                    std::size_t selected_index = r.options.size();
                    for (std::size_t i = 0; i < r.options.size(); ++i) if (selected == "option_" + std::to_string(i)) selected_index = i;
                    if (selected_index == r.options.size() || out.scores[selected_index] < *std::max_element(out.scores.begin(), out.scores.end()))
                        throw std::invalid_argument("selected Choice is not a maximum-probability option");
                    out.metadata.provider_choice_index = selected_index;
                } else {
                    const auto score = answer.at("score").get<double>();
                    if (!std::isfinite(score) || score < 0 || score > r.options.size() - 1 || std::abs(score - expected / total) > 1e-3)
                        throw std::invalid_argument("Score inconsistent with probabilities");
                    out.metadata.provider_score = static_cast<float>(score);
                    const auto& legend = answer.at("legend");
                    if (!legend.is_object() || legend.size() != r.options.size()) throw std::invalid_argument("Score legend size mismatch");
                    for (std::size_t i = 0; i < r.options.size(); ++i) {
                        const auto& item = legend.at(std::to_string(i));
                        if (!item.is_string() && !item.is_object() && !item.is_array()) throw std::invalid_argument("invalid Score legend");
                    }
                }
            }
        } catch (...) { return error{error_code::invalid_backend_output, "invalid remote response contract"}; }
        if (auto e = execution_error(evaluation_options{deadline, stop.get_token()})) return *e;
    }
    return output;
}

result<std::vector<remote_model>> typesafe_backend::list_models(evaluation_options options) const {
    if (impl_->options.provider == remote_provider::openrouter)
        return error{error_code::unsupported_feature, "OpenRouter model discovery has a different contract"};
    auto deadline = clock_type::now() + impl_->options.call_timeout;
    if (options.deadline) deadline = std::min(deadline, *options.deadline);
    auto response = impl_->send("GET", "/v1/models", {}, deadline, options.cancellation);
    if (!response) return response.error_value();
    try {
        const auto parsed = json::parse(response->value.body);
        if (!parsed.at("models").is_array()) throw std::invalid_argument("invalid model catalog");
        std::vector<remote_model> models;
        for (const auto& model : parsed.at("models")) {
            models.push_back({model.at("name").get<std::string>(), model.at("description").get<std::string>(), model.at("release_date").get<std::string>()});
            if (models.back().name.empty()) throw std::invalid_argument("empty model name");
        }
        if (auto e = execution_error(evaluation_options{deadline, options.cancellation})) return *e;
        return models;
    } catch (...) { return error{error_code::invalid_backend_output, "invalid model catalog response"}; }
}
} // namespace jevt
