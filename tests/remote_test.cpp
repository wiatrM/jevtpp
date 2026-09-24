#include <jevt/remote.hpp>
#include <jevt/system_one.hpp>
#include <nlohmann/json.hpp>
#include "test_harness.hpp"
#include <thread>

using namespace std::chrono_literals;
namespace {
using json = nlohmann::json;
using clock_type = std::chrono::steady_clock;
class transport final : public jevt::http_transport {
public:
    std::vector<jevt::http_request> calls;
    std::function<jevt::result<jevt::http_response>(const jevt::http_request&, std::size_t)> handler;
    jevt::result<jevt::http_response> perform(const jevt::http_request& request) override {
        calls.push_back(request);
        return handler(request, calls.size());
    }
};
jevt::remote_options options() {
    jevt::remote_options value;
    value.api_key = "fixture-key-not-a-secret";
    value.initial_backoff = 0ms;
    value.max_backoff = 0ms;
    return value;
}
json reply(const jevt::http_request& request) {
    auto body = json::parse(request.body);
    json result{{"model", "jev-fixture-1"}, {"usage", {{"input_tokens", 100}, {"output_tokens", 4}}}, {"answers", json::object()}};
    for (const auto& [key, q] : body["questions"].items()) {
        json answer{{"type", q["type"]}};
        if (q["type"] == "noul") answer["noul"] = 0.75;
        else {
            const bool choice = q["type"] == "choice";
            auto size = q["criteria"].size();
            answer["confidence"] = 0.27;
            answer["probabilities"] = json::object();
            for (std::size_t i = 0; i < size; ++i) {
                answer["probabilities"][(choice ? "option_" : "") + std::to_string(i)] = i == size - 1 ? 1.0 : 0.0;
                if (!choice) answer["legend"][std::to_string(i)] = q["criteria"][i];
            }
            if (choice) answer["choice"] = "option_" + std::to_string(size - 1);
            else answer["score"] = size - 1;
        }
        result["answers"][key] = std::move(answer);
    }
    return result;
}
std::shared_ptr<transport> fixture() {
    auto value = std::make_shared<transport>();
    value->handler = [](const auto& request, std::size_t) { return jevt::http_response{200, reply(request).dump(), {}}; };
    return value;
}
const std::array<std::string_view, 2> criteria{"billing", "technical"};
jevt::inference_request request() { return {"ticket", "Select the team", "charged twice", criteria}; }
}

JEVT_TEST("remote Choice preserves JSON state instructions criteria and execution metadata") {
    auto wire = fixture();
    jevt::typesafe_backend backend(options(), wire);
    auto value = request();
    value.input = R"({"ticket":"charged twice","customer":{"plan":"enterprise"}})";
    value.input_kind = jevt::content_kind::json;
    value.instructions = jevt::json_metadata(R"({"rule":"choose owner","priority":1})");
    const std::array metadata{jevt::json_metadata(R"({"meaning":"payments"})"), jevt::json_metadata(R"(["bugs","outages"])")};
    value.criteria_metadata = metadata;
    auto result = backend.predict(value);
    JEVT_REQUIRE(result);
    JEVT_REQUIRE_EQ(result->scores, (std::vector<float>{0, 1}));
    JEVT_REQUIRE_EQ(result->metadata.provider, "typesafe");
    JEVT_REQUIRE_EQ(result->metadata.model_revision, "jev-fixture-1");
    JEVT_REQUIRE_EQ(result->metadata.provider_choice_index, 1u);
    JEVT_REQUIRE_EQ(result->metadata.usage->input_tokens, 100u);
    JEVT_REQUIRE_EQ(result->metadata.provider_confidence, .27f);
    const auto& call = wire->calls.at(0);
    JEVT_REQUIRE_EQ(call.url, "https://api.typesafe.ai/v1/systemone");
    JEVT_REQUIRE_EQ(call.method, "POST");
    JEVT_REQUIRE_EQ(call.headers.at("Authorization"), "Bearer fixture-key-not-a-secret");
    const auto payload = json::parse(call.body);
    JEVT_REQUIRE_EQ(payload["state"]["customer"]["plan"], "enterprise");
    JEVT_REQUIRE_EQ(payload["questions"]["q0"]["instructions"]["priority"], 1);
    JEVT_REQUIRE(payload["questions"]["q0"]["criteria"]["option_1"].is_array());
}

JEVT_TEST("same-state typed questions share HTTP usage and different states remain isolated") {
    auto wire = fixture();
    jevt::typesafe_backend backend(options(), wire);
    std::array values{request(), request(), request(), request()};
    values[1].question_kind = jevt::inference_request::kind::score;
    values[2].question_kind = jevt::inference_request::kind::noul;
    values[3].input = "different context";
    // Repeated decision IDs must never overwrite a question on the wire.
    auto results = backend.predict_batch(values);
    JEVT_REQUIRE(results);
    JEVT_REQUIRE_EQ(wire->calls.size(), 2u);
    JEVT_REQUIRE_EQ(json::parse(wire->calls[0].body)["questions"].size(), 3u);
    JEVT_REQUIRE_EQ((*results)[0].metadata.usage, (*results)[1].metadata.usage);
    JEVT_REQUIRE_EQ((*results)[1].metadata.usage, (*results)[2].metadata.usage);
    JEVT_REQUIRE((*results)[3].metadata.usage != (*results)[0].metadata.usage);
    JEVT_REQUIRE_EQ((*results)[1].metadata.provider_score, 1.f);
    JEVT_REQUIRE_EQ((*results)[2].scores, (std::vector<float>{.25f, .75f}));
    JEVT_REQUIRE(!(*results)[2].metadata.provider_confidence);
}

JEVT_TEST("Noul sends custom false true meanings and omits empty default meanings") {
    auto wire = fixture();
    jevt::typesafe_backend backend(options(), wire);
    auto value = request();
    value.question_kind = jevt::inference_request::kind::noul;
    std::array metadata{jevt::text_metadata("patient"), jevt::json_metadata(R"({"meaning":"one hour SLA"})")};
    value.criteria_metadata = metadata;
    JEVT_REQUIRE(backend.predict(value));
    auto criteria_json = json::parse(wire->calls.back().body)["questions"]["q0"]["criteria"];
    JEVT_REQUIRE_EQ(criteria_json["false"], "patient");
    JEVT_REQUIRE_EQ(criteria_json["true"]["meaning"], "one hour SLA");
    metadata = {jevt::text_metadata(""), jevt::text_metadata("")};
    JEVT_REQUIRE(backend.predict(value));
    JEVT_REQUIRE(!json::parse(wire->calls.back().body)["questions"]["q0"].contains("criteria"));
}

JEVT_TEST("remote rejects malformed or inconsistent provider output without retry") {
    const std::vector<std::function<void(json&)>> corrupt{
        [](auto& j) { j.erase("model"); },
        [](auto& j) { j["usage"]["input_tokens"] = -1; },
        [](auto& j) { j["answers"]["extra"] = j["answers"]["q0"]; },
        [](auto& j) { j["answers"]["q0"]["type"] = "noul"; },
        [](auto& j) { j["answers"]["q0"]["confidence"] = 1.1; },
        [](auto& j) { j["answers"]["q0"]["probabilities"]["option_0"] = .5; },
        [](auto& j) { j["answers"]["q0"]["probabilities"].erase("option_0"); },
        [](auto& j) { j["answers"]["q0"]["choice"] = "option_0"; },
        [](auto& j) { j["answers"]["q0"]["choice"] = "unknown"; },
    };
    for (const auto& mutation : corrupt) {
        auto wire = fixture();
        wire->handler = [&](const auto& r, std::size_t) { auto j = reply(r); mutation(j); return jevt::http_response{200, j.dump(), {}}; };
        jevt::typesafe_backend backend(options(), wire);
        const auto result = backend.predict(request());
        JEVT_REQUIRE(!result);
        JEVT_REQUIRE_EQ(result.error_value().code, jevt::error_code::invalid_backend_output);
        JEVT_REQUIRE_EQ(wire->calls.size(), 1u);
    }
    for (bool legend : {false, true}) {
        auto wire = fixture();
        wire->handler = [&](const auto& r, std::size_t) {
            auto j = reply(r);
            if (legend) j["answers"]["q0"]["legend"].erase("0");
            else j["answers"]["q0"]["score"] = .1;
            return jevt::http_response{200, j.dump(), {}};
        };
        jevt::typesafe_backend backend(options(), wire);
        auto value = request(); value.question_kind = jevt::inference_request::kind::score;
        JEVT_REQUIRE_EQ(backend.predict(value).error_value().code, jevt::error_code::invalid_backend_output);
    }
}

JEVT_TEST("Choice provider tie winner is preserved") {
    auto wire = fixture();
    wire->handler = [](const auto& r, std::size_t) {
        auto j = reply(r);
        j["answers"]["q0"]["probabilities"] = {{"option_0", .5}, {"option_1", .5}};
        return jevt::http_response{200, j.dump(), {}};
    };
    jevt::typesafe_backend backend(options(), wire);
    auto answer = backend.predict(request());
    JEVT_REQUIRE(answer);
    JEVT_REQUIRE_EQ(answer->metadata.provider_choice_index, 1u);
}

JEVT_TEST("retry is bounded respects status classes and records attempts") {
    for (int status : {408, 429, 500, 503}) {
        auto wire = fixture();
        wire->handler = [status](const auto& r, std::size_t n) {
            return n == 1 ? jevt::http_response{status, "sensitive context", {{"Retry-After-ms", "1"}}} : jevt::http_response{200, reply(r).dump(), {}};
        };
        jevt::typesafe_backend backend(options(), wire);
        const auto result = backend.predict(request());
        JEVT_REQUIRE(result);
        JEVT_REQUIRE_EQ(result->metadata.attempts, 2u);
        JEVT_REQUIRE(wire->calls[1].deadline >= wire->calls[0].deadline);
    }
    for (auto [status, expected] : std::array{std::pair{401, jevt::error_code::authentication},
         std::pair{403, jevt::error_code::authentication}, std::pair{400, jevt::error_code::invalid_request},
         std::pair{422, jevt::error_code::invalid_request}, std::pair{302, jevt::error_code::remote_error}}) {
        auto wire = fixture();
        wire->handler = [status](const auto&, std::size_t) { return jevt::http_response{status, "private prompt and key", {}}; };
        jevt::typesafe_backend backend(options(), wire);
        const auto result = backend.predict(request());
        JEVT_REQUIRE_EQ(result.error_value().code, expected);
        JEVT_REQUIRE_EQ(result.error_value().status_code, status);
        JEVT_REQUIRE_EQ(wire->calls.size(), 1u);
        JEVT_REQUIRE(result.error_value().message.find("private") == std::string::npos);
    }
    auto wire = fixture();
    wire->handler = [](const auto&, std::size_t) { return jevt::http_response{429, "", {}}; };
    jevt::typesafe_backend backend(options(), wire);
    JEVT_REQUIRE_EQ(backend.predict(request()).error_value().code, jevt::error_code::rate_limited);
    JEVT_REQUIRE_EQ(wire->calls.size(), 3u);
}

JEVT_TEST("connection errors and exceptions retry without exposing exception details") {
    auto wire = fixture();
    wire->handler = [](const auto& r, std::size_t n) -> jevt::result<jevt::http_response> {
        if (n == 1) throw std::runtime_error("secret-token");
        if (n == 2) return jevt::error{jevt::error_code::connection_failure, "disconnect"};
        return jevt::http_response{200, reply(r).dump(), {}};
    };
    jevt::typesafe_backend backend(options(), wire);
    JEVT_REQUIRE_EQ(backend.predict(request())->metadata.attempts, 3u);
}

JEVT_TEST("absolute budget includes serialization attempts backoff and caller cancellation") {
    auto wire = fixture();
    auto opts = options(); opts.call_timeout = 20ms; opts.attempt_timeout = 10ms; opts.max_retries = 0;
    wire->handler = [](const auto& r, std::size_t) { std::this_thread::sleep_for(25ms); return jevt::http_response{200, reply(r).dump(), {}}; };
    jevt::typesafe_backend backend(opts, wire);
    JEVT_REQUIRE_EQ(backend.predict(request()).error_value().code, jevt::error_code::timeout);
    auto expired = request(); expired.deadline = clock_type::now() - 1ms;
    JEVT_REQUIRE_EQ(backend.predict(expired).error_value().code, jevt::error_code::timeout);
    JEVT_REQUIRE_EQ(wire->calls.size(), 1u);
    std::stop_source stop;
    auto stopped = request(); stopped.cancellation = stop.get_token(); stop.request_stop();
    JEVT_REQUIRE_EQ(backend.predict(stopped).error_value().code, jevt::error_code::cancelled);
    JEVT_REQUIRE_EQ(wire->calls.size(), 1u);

    auto backoff = fixture();
    std::stop_source during_wait;
    backoff->handler = [](const auto&, std::size_t) { return jevt::http_response{429, "", {{"Retry-After", "10"}}}; };
    jevt::typesafe_backend waiting(options(), backoff);
    auto value = request(); value.cancellation = during_wait.get_token();
    std::jthread cancel([&] { std::this_thread::sleep_for(20ms); during_wait.request_stop(); });
    const auto start = clock_type::now();
    JEVT_REQUIRE_EQ(waiting.predict(value).error_value().code, jevt::error_code::cancelled);
    JEVT_REQUIRE(clock_type::now() - start < 1s);
    JEVT_REQUIRE_EQ(backoff->calls.size(), 1u);
}

JEVT_TEST("remote validates input resource caps credentials and endpoints before IO") {
    auto wire = fixture();
    for (const auto& endpoint : {"http://example.com", "https://user:pass@example.com", "https://example.com/?query=1", "http://127.0.0.1.evil"}) {
        auto opts = options(); opts.base_url = endpoint; opts.allow_insecure_loopback = true;
        bool rejected = false;
        try { jevt::typesafe_backend backend(opts, wire); } catch (const std::invalid_argument&) { rejected = true; }
        JEVT_REQUIRE(rejected);
    }
    for (const auto& key : {"", "key\r\nInjected: value"}) {
        auto opts = options(); opts.api_key = key;
        bool rejected = false;
        try { jevt::typesafe_backend backend(opts, wire); } catch (const std::invalid_argument&) { rejected = true; }
        JEVT_REQUIRE(rejected);
    }
    jevt::typesafe_backend backend(options(), wire);
    auto value = request(); value.input = "42"; value.input_kind = jevt::content_kind::json;
    JEVT_REQUIRE_EQ(backend.predict(value).error_value().code, jevt::error_code::invalid_request);
    value.input = "{";
    JEVT_REQUIRE_EQ(backend.predict(value).error_value().code, jevt::error_code::invalid_request);
    value = request(); value.options = {};
    JEVT_REQUIRE_EQ(backend.predict(value).error_value().code, jevt::error_code::invalid_request);
    const std::array<std::string_view, 1> one_option{"invalid"};
    const std::array one_criterion{jevt::text_metadata("invalid")};
    value.question_kind = jevt::inference_request::kind::noul;
    value.options = one_option;
    value.criteria_metadata = one_criterion;
    JEVT_REQUIRE_EQ(backend.predict(value).error_value().code, jevt::error_code::invalid_request);
    value = request();
    const std::array null_criteria{jevt::json_metadata("null"), jevt::text_metadata("technical")};
    value.criteria_metadata = null_criteria;
    JEVT_REQUIRE_EQ(backend.predict(value).error_value().code, jevt::error_code::invalid_request);
    JEVT_REQUIRE(wire->calls.empty());
    auto opts = options(); opts.max_request_bytes = 10;
    jevt::typesafe_backend limited(opts, wire);
    JEVT_REQUIRE_EQ(limited.predict(request()).error_value().code, jevt::error_code::invalid_request);
    JEVT_REQUIRE(wire->calls.empty());
    opts = options(); opts.max_response_bytes = 10;
    jevt::typesafe_backend response_limited(opts, wire);
    JEVT_REQUIRE_EQ(response_limited.predict(request()).error_value().code, jevt::error_code::invalid_backend_output);
}

JEVT_TEST("model discovery validates contract and OpenRouter explicitly changes provider") {
    auto wire = fixture();
    wire->handler = [](const auto&, std::size_t) { return jevt::http_response{200, R"({"models":[{"name":"jev-1","description":"fixture","release_date":"2026-01-01"}]})", {}}; };
    jevt::typesafe_backend backend(options(), wire);
    auto catalog = backend.list_models();
    JEVT_REQUIRE(catalog);
    JEVT_REQUIRE_EQ(catalog->front().name, "jev-1");
    JEVT_REQUIRE_EQ(wire->calls.front().method, "GET");
    JEVT_REQUIRE_EQ(wire->calls.front().url, "https://api.typesafe.ai/v1/models");
    auto router_wire = fixture();
    jevt::typesafe_backend router(jevt::openrouter_options("key"), router_wire);
    JEVT_REQUIRE_EQ(router.list_models().error_value().code, jevt::error_code::unsupported_feature);
    JEVT_REQUIRE(router_wire->calls.empty());
    JEVT_REQUIRE_EQ(router.predict(request())->metadata.provider, "openrouter");
    JEVT_REQUIRE_EQ(router_wire->calls[0].url, "https://openrouter.ai/api/v1/systemone");
}

int main() { return jevt::test::run_all("remote"); }
