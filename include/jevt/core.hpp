#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace jevt {

template <std::size_t N>
struct fixed_string {
    char value[N]{};
    consteval fixed_string(const char (&text)[N]) { std::copy_n(text, N, value); }
    [[nodiscard]] constexpr std::string_view view() const noexcept { return {value, N - 1}; }
    constexpr auto operator<=>(const fixed_string&) const = default;
};

template <auto Value>
struct option_definition {
    static constexpr auto value = Value;
    std::string_view description;
};

template <auto Value>
[[nodiscard]] consteval auto option(std::string_view description) {
    static_assert(std::is_enum_v<decltype(Value)>, "jevt::option requires an enum value");
    return option_definition<Value>{description};
}

template <class Enum, fixed_string Id, class... Options>
class schema_definition {
    static_assert(std::is_enum_v<Enum>, "jevt::schema requires an enum type");
    static_assert(sizeof...(Options) > 0, "a decision schema needs at least one option");
    static_assert((std::same_as<Enum, std::remove_cv_t<decltype(Options::value)>> && ...),
                  "all options must use the schema enum type");

    static consteval bool unique_values() {
        constexpr std::array values{Options::value...};
        for (std::size_t i = 0; i < values.size(); ++i)
            for (std::size_t j = i + 1; j < values.size(); ++j)
                if (values[i] == values[j]) return false;
        return true;
    }

public:
    using enum_type = Enum;
    static constexpr auto id = Id;
    static constexpr std::size_t size = sizeof...(Options);
    static_assert(unique_values(), "schema option values must be unique");

    constexpr explicit schema_definition(Options... options)
        : descriptions_{options.description...} {}

    [[nodiscard]] static constexpr auto values() noexcept {
        return std::array<Enum, size>{Options::value...};
    }
    [[nodiscard]] constexpr const auto& descriptions() const noexcept { return descriptions_; }
    [[nodiscard]] static constexpr std::string_view name() noexcept { return Id.view(); }

private:
    std::array<std::string_view, size> descriptions_;
};

template <class Enum, fixed_string Id, class... Options>
[[nodiscard]] consteval auto schema(Options... options) {
    return schema_definition<Enum, Id, Options...>{options...};
}

enum class truth_value : std::uint8_t { false_value, true_value };

template <fixed_string Id>
struct predicate_definition {
    static constexpr auto id = Id;
    std::string_view question;
    [[nodiscard]] static constexpr std::string_view name() noexcept { return Id.view(); }
};

template <fixed_string Id>
[[nodiscard]] consteval auto predicate(std::string_view question) {
    return predicate_definition<Id>{question};
}

enum class error_code : std::uint8_t {
    invalid_request,
    backend_failure,
    invalid_backend_output,
    not_initialized,
    overloaded,
    shutting_down,
};

struct error {
    error_code code{};
    std::string message;
};

template <class T>
class result {
public:
    result(T value) : value_(std::move(value)) {}
    result(error failure) : failure_(std::move(failure)) {}
    [[nodiscard]] explicit operator bool() const noexcept { return value_.has_value(); }
    [[nodiscard]] T& value() & { if (!value_) throw std::logic_error("jevt::result has no value"); return *value_; }
    [[nodiscard]] const T& value() const& { if (!value_) throw std::logic_error("jevt::result has no value"); return *value_; }
    [[nodiscard]] T&& value() && { if (!value_) throw std::logic_error("jevt::result has no value"); return std::move(*value_); }
    [[nodiscard]] const error& error_value() const { if (!failure_) throw std::logic_error("jevt::result has no error"); return *failure_; }
    [[nodiscard]] T* operator->() { return &value(); }
    [[nodiscard]] const T* operator->() const { return &value(); }
    [[nodiscard]] T& operator*() { return value(); }
    [[nodiscard]] const T& operator*() const { return value(); }
private:
    std::optional<T> value_;
    std::optional<error> failure_;
};

struct inference_request {
    enum class kind : std::uint8_t { choice, noul, score };
    std::string_view decision_id;
    std::string_view question;
    std::string_view input;
    std::span<const std::string_view> options;
    kind question_kind = kind::choice;
};

struct inference_response {
    std::vector<float> scores;
    std::string model_id;
};

class backend {
public:
    virtual ~backend() = default;
    [[nodiscard]] virtual result<inference_response> predict(const inference_request&) = 0;
    [[nodiscard]] virtual result<std::vector<inference_response>> predict_batch(
        std::span<const inference_request> requests) {
        std::vector<inference_response> responses;
        responses.reserve(requests.size());
        for (const auto& request : requests) {
            auto response = predict(request);
            if (!response) return response.error_value();
            responses.push_back(std::move(response).value());
        }
        return responses;
    }
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
};

class function_backend final : public backend {
public:
    using function_type = std::function<result<inference_response>(const inference_request&)>;
    explicit function_backend(function_type function, std::string name = "function")
        : function_(std::move(function)), name_(std::move(name)) {}
    [[nodiscard]] result<inference_response> predict(const inference_request& request) override {
        return function_(request);
    }
    [[nodiscard]] std::string_view name() const noexcept override { return name_; }
private:
    function_type function_;
    std::string name_;
};

struct decision_metadata {
    std::string model_id;
    float confidence{};
    bool abstained{};
};

template <class Schema>
class decision_result {
public:
    using enum_type = typename Schema::enum_type;
    decision_result(std::optional<enum_type> selected, std::vector<float> scores,
                    decision_metadata metadata)
        : selected_(selected), scores_(std::move(scores)), metadata_(std::move(metadata)) {}
    [[nodiscard]] bool has_value() const noexcept { return selected_.has_value(); }
    [[nodiscard]] bool abstained() const noexcept { return !selected_.has_value(); }
    [[nodiscard]] enum_type value() const { if (!selected_) throw std::logic_error("decision abstained"); return *selected_; }
    [[nodiscard]] float confidence() const noexcept { return metadata_.confidence; }
    [[nodiscard]] std::span<const float> scores() const noexcept { return scores_; }
    [[nodiscard]] const decision_metadata& metadata() const noexcept { return metadata_; }
private:
    std::optional<enum_type> selected_;
    std::vector<float> scores_;
    decision_metadata metadata_;
};

template <fixed_string Id>
class predicate_result {
public:
    predicate_result(std::optional<bool> value, float confidence, std::string model_id)
        : value_(value), confidence_(confidence), model_id_(std::move(model_id)) {}
    [[nodiscard]] bool is_true() const noexcept { return value_ == true; }
    [[nodiscard]] bool is_false() const noexcept { return value_ == false; }
    [[nodiscard]] bool abstained() const noexcept { return !value_.has_value(); }
    [[nodiscard]] float confidence() const noexcept { return confidence_; }
    [[nodiscard]] std::string_view model_id() const noexcept { return model_id_; }
    explicit operator bool() const = delete;
private:
    std::optional<bool> value_;
    float confidence_{};
    std::string model_id_;
};

template <auto Value, class Handler>
struct case_handler { static constexpr auto value = Value; Handler handler; };
template <auto Value, class Handler>
[[nodiscard]] constexpr auto on(Handler&& handler) {
    return case_handler<Value, std::decay_t<Handler>>{std::forward<Handler>(handler)};
}
template <class Handler>
struct abstain_handler { Handler handler; };
template <class Handler>
[[nodiscard]] constexpr auto on_abstain(Handler&& handler) {
    return abstain_handler<std::decay_t<Handler>>{std::forward<Handler>(handler)};
}

namespace detail {
template <class T> struct is_case : std::false_type {};
template <auto V, class H> struct is_case<case_handler<V, H>> : std::true_type {};
template <class T> struct is_abstain : std::false_type {};
template <class H> struct is_abstain<abstain_handler<H>> : std::true_type {};
template <auto Wanted, class... Handlers>
consteval std::size_t count_case() { return ((is_case<Handlers>::value && []{
    if constexpr (is_case<Handlers>::value) return Handlers::value == Wanted;
    return false;
}()) + ... + 0); }
template <class Handler, class Enum>
void invoke_case(Handler& h, Enum value) {
    if constexpr (is_case<Handler>::value) if (Handler::value == value) std::invoke(h.handler);
}
}

template <class Schema, class... Handlers>
void match(const Schema&, const decision_result<Schema>& decision, Handlers&&... handlers) {
    constexpr auto values = Schema::values();
    static_assert(((detail::count_case<values[0], std::decay_t<Handlers>...>() >= 0)), "invalid handlers");
    static_assert((detail::is_abstain<std::decay_t<Handlers>>::value + ... + 0) == 1,
                  "match requires exactly one on_abstain handler");
    constexpr bool covered = []<std::size_t... I>(std::index_sequence<I...>) {
        return ((detail::count_case<Schema::values()[I], std::decay_t<Handlers>...>() == 1) && ...);
    }(std::make_index_sequence<Schema::size>{});
    static_assert(covered, "match requires each schema option exactly once");

    if (decision.abstained()) {
        ([&] { if constexpr (detail::is_abstain<std::decay_t<Handlers>>::value) std::invoke(handlers.handler); }(), ...);
        return;
    }
    (detail::invoke_case(handlers, decision.value()), ...);
}

} // namespace jevt
