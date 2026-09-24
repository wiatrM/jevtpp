#pragma once

#include "jevt/core.hpp"
#include "jevt/diagnostics.hpp"

#include <chrono>
#include <numeric>
#include <tuple>

namespace jevt {

// JSON is supplied by the application's serializer. No JSON library is imposed.
// content_kind describes the serialization; it does not claim JSON validation.
struct state_value {
    enum class content_kind { text, json };
    std::string content;
    content_kind kind = content_kind::text;
};

[[nodiscard]] inline state_value text_state(std::string content) {
    return {std::move(content), state_value::content_kind::text};
}
[[nodiscard]] inline state_value json_state(std::string content) {
    if (content.find_first_not_of(" \t\r\n") == std::string::npos)
        throw std::invalid_argument("JSON state must not be empty");
    return {std::move(content), state_value::content_kind::json};
}

// The open interval between thresholds abstains. Equal thresholds implement a
// binary threshold (equality is true). Noul exposes P(true), not confidence.
struct noul_policy {
    float false_threshold = 0.5F;
    float true_threshold = 0.5F;
};

enum class field_kind { choice, noul, score, probability };

template <fixed_string Name, field_kind Kind, class Schema>
struct enum_field_definition {
    static constexpr auto name = Name;
    static constexpr auto kind = Kind;
    using schema_type = Schema;
    using enum_type = typename Schema::enum_type;
    static constexpr std::size_t size = Schema::size;
    std::string_view description;
    Schema criteria;
    float abstain_threshold;
};

template <fixed_string Name, field_kind Kind>
struct binary_field_definition {
    static constexpr auto name = Name;
    static constexpr auto kind = Kind;
    static constexpr std::size_t size = 2;
    std::string_view description;
    noul_policy policy;
};

namespace system_one_detail {
constexpr bool unit_interval(float value) { return value >= 0.0F && value <= 1.0F; }
consteval void validate_description(std::string_view name, std::string_view description) {
    if (name.empty() || description.empty()) throw "field name and description must not be empty";
}
template <class Schema>
consteval bool ordered_levels() {
    constexpr auto values = Schema::values();
    using underlying = std::underlying_type_t<typename Schema::enum_type>;
    for (std::size_t i = 1; i < values.size(); ++i)
        if (static_cast<underlying>(values[i]) <= static_cast<underlying>(values[i - 1])) return false;
    return true;
}
template <class T> struct is_field : std::false_type {};
template <fixed_string N, field_kind K, class S>
struct is_field<enum_field_definition<N, K, S>> : std::true_type {};
template <fixed_string N, field_kind K>
struct is_field<binary_field_definition<N, K>> : std::true_type {};
template <class... Fields>
consteval bool unique_names() {
    constexpr std::array names{Fields::name.view()...};
    for (std::size_t i = 0; i < names.size(); ++i)
        for (std::size_t j = i + 1; j < names.size(); ++j)
            if (names[i] == names[j]) return false;
    return true;
}
template <fixed_string Name, class... Fields>
consteval std::size_t field_index() {
    constexpr std::array names{Fields::name.view()...};
    for (std::size_t i = 0; i < names.size(); ++i) if (names[i] == Name.view()) return i;
    return names.size();
}
} // namespace system_one_detail

template <fixed_string Name, class Schema>
[[nodiscard]] consteval auto choice(std::string_view description, Schema criteria,
                                   float abstain_threshold = 0.0F) {
    static_assert(Schema::size >= 1 && Schema::size <= 255, "Choice requires 1 to 255 options");
    system_one_detail::validate_description(Name.view(), description);
    if (!system_one_detail::unit_interval(abstain_threshold)) throw "invalid Choice confidence threshold";
    for (const auto item : criteria.descriptions()) if (item.empty()) throw "empty Choice criterion";
    return enum_field_definition<Name, field_kind::choice, Schema>{description, criteria, abstain_threshold};
}

template <fixed_string Name, class Schema>
[[nodiscard]] consteval auto score(std::string_view description, Schema criteria,
                                  float abstain_threshold = 0.0F) {
    static_assert(Schema::size >= 2 && Schema::size <= 10, "Score requires 2 to 10 levels");
    static_assert(system_one_detail::ordered_levels<Schema>(), "Score enum levels must be strictly ascending");
    system_one_detail::validate_description(Name.view(), description);
    if (!system_one_detail::unit_interval(abstain_threshold)) throw "invalid Score confidence threshold";
    for (const auto item : criteria.descriptions()) if (item.empty()) throw "empty Score criterion";
    return enum_field_definition<Name, field_kind::score, Schema>{description, criteria, abstain_threshold};
}

template <fixed_string Name>
[[nodiscard]] consteval auto noul(std::string_view description, noul_policy policy = {}) {
    system_one_detail::validate_description(Name.view(), description);
    if (!system_one_detail::unit_interval(policy.false_threshold) ||
        !system_one_detail::unit_interval(policy.true_threshold) ||
        policy.false_threshold > policy.true_threshold) throw "invalid Noul thresholds";
    return binary_field_definition<Name, field_kind::noul>{description, policy};
}

// Explicit probability semantics: the description must be a true/false
// proposition, not an arbitrary floating-point extraction instruction.
template <fixed_string Name>
[[nodiscard]] consteval auto probability(std::string_view proposition) {
    system_one_detail::validate_description(Name.view(), proposition);
    return binary_field_definition<Name, field_kind::probability>{proposition, {}};
}

template <fixed_string Id, class... Fields>
struct decision_model_definition {
    static_assert(sizeof...(Fields) > 0, "decision model needs at least one field");
    static_assert((system_one_detail::is_field<Fields>::value && ...), "invalid decision field type");
    static_assert(system_one_detail::unique_names<Fields...>(), "decision field names must be unique");
    static constexpr auto id = Id;
    static constexpr std::size_t size = sizeof...(Fields);
    std::string_view description;
    std::tuple<Fields...> fields;
};

template <fixed_string Id, class... Fields>
[[nodiscard]] consteval auto decision_model(std::string_view description, Fields... fields) {
    system_one_detail::validate_description(Id.view(), description);
    return decision_model_definition<Id, Fields...>{description, std::tuple{fields...}};
}

// Owns every string. views() deliberately rebuilds views after a copy/move.
// Views are valid until the owner is moved, mutated or destroyed.
struct system_one_request {
    struct field {
        std::string id;
        std::string description;
        std::string question;
        std::vector<std::string> criteria;
        inference_request::kind kind;
    };
    std::string decision_id;
    std::string description;
    state_value state;
    std::vector<field> fields;

    class request_views {
    public:
        explicit request_views(const system_one_request& owner) {
            criteria_.reserve(owner.fields.size());
            requests_.reserve(owner.fields.size());
            for (const auto& field : owner.fields) {
                auto& options = criteria_.emplace_back();
                options.reserve(field.criteria.size());
                for (const auto& criterion : field.criteria) options.emplace_back(criterion);
                requests_.push_back({field.id, field.question, owner.state.content, options, field.kind});
            }
        }
        request_views(const request_views&) = delete;
        request_views& operator=(const request_views&) = delete;
        request_views(request_views&&) noexcept = default;
        request_views& operator=(request_views&&) noexcept = default;
        [[nodiscard]] std::span<const inference_request> requests() const noexcept { return requests_; }
    private:
        std::vector<std::vector<std::string_view>> criteria_;
        std::vector<inference_request> requests_;
    };
    [[nodiscard]] request_views views() const & { return request_views{*this}; }
    request_views views() const && = delete;
};

template <fixed_string Id, class... Fields>
[[nodiscard]] system_one_request make_system_one_request(
    const decision_model_definition<Id, Fields...>& model, state_value state) {
    system_one_request output{std::string{Id.view()}, std::string{model.description}, std::move(state), {}};
    output.fields.reserve(sizeof...(Fields));
    std::apply([&](const auto&... field) {
        ([&] {
            using F = std::remove_cvref_t<decltype(field)>;
            system_one_request::field item;
            item.id = std::string{Id.view()} + "." + std::string{F::name.view()};
            item.description = field.description;
            item.question = std::string{model.description} + "\n\n" + std::string{field.description};
            if constexpr (F::kind == field_kind::choice || F::kind == field_kind::score) {
                item.kind = F::kind == field_kind::score ? inference_request::kind::score : inference_request::kind::choice;
                for (auto criterion : field.criteria.descriptions()) item.criteria.emplace_back(criterion);
            } else {
                item.kind = inference_request::kind::noul;
                item.criteria = {"false", "true"};
            }
            output.fields.push_back(std::move(item));
        }(), ...);
    }, model.fields);
    return output;
}

// jevtpp's explicit metric: 1 - H(p)/log(N). This is not a claim of
// bit-for-bit equivalence to a proprietary model's confidence calculation.
[[nodiscard]] inline float entropy_confidence(std::span<const float> probabilities) {
    if (probabilities.size() <= 1) return 1.0F;
    double entropy = 0.0;
    for (const auto p : probabilities) if (p > 0.0F) entropy -= static_cast<double>(p) * std::log(p);
    return static_cast<float>(std::clamp(1.0 - entropy / std::log(static_cast<double>(probabilities.size())), 0.0, 1.0));
}

template <class Field>
class enum_answer {
public:
    using enum_type = typename Field::enum_type;
    enum_answer(const Field& field, std::vector<float> probabilities, std::string model_id)
        : probabilities_(std::move(probabilities)), model_id_(std::move(model_id)),
          confidence_(entropy_confidence(probabilities_)) {
        const auto index = static_cast<std::size_t>(std::max_element(probabilities_.begin(), probabilities_.end()) - probabilities_.begin());
        if (confidence_ >= field.abstain_threshold) selected_ = Field::schema_type::values()[index];
        for (auto description : field.criteria.descriptions()) legend_.emplace_back(description);
        for (std::size_t i = 0; i < probabilities_.size(); ++i) score_ += static_cast<float>(i) * probabilities_[i];
    }
    [[nodiscard]] bool abstained() const noexcept { return !selected_; }
    [[nodiscard]] const std::optional<enum_type>& selected() const noexcept { return selected_; }
    [[nodiscard]] enum_type value() const {
        if (!selected_) throw std::logic_error("decision field abstained");
        return *selected_;
    }
    [[nodiscard]] float confidence() const noexcept { return confidence_; }
    [[nodiscard]] std::span<const float> probabilities() const noexcept { return probabilities_; }
    [[nodiscard]] static constexpr auto values() noexcept { return Field::schema_type::values(); }
    [[nodiscard]] const std::vector<std::string>& legend() const noexcept { return legend_; }
    [[nodiscard]] std::string_view model_id() const noexcept { return model_id_; }
    [[nodiscard]] float score() const noexcept requires (Field::kind == field_kind::score) { return score_; }
private:
    std::optional<enum_type> selected_;
    std::vector<float> probabilities_;
    std::vector<std::string> legend_;
    std::string model_id_;
    float confidence_{};
    float score_{};
};

template <class Field>
class binary_answer {
public:
    binary_answer(const Field& field, std::vector<float> probabilities, std::string model_id)
        : probabilities_{probabilities[0], probabilities[1]}, model_id_(std::move(model_id)) {
        if (probabilities_[1] >= field.policy.true_threshold) selected_ = true;
        else if (probabilities_[1] <= field.policy.false_threshold) selected_ = false;
    }
    [[nodiscard]] float probability_true() const noexcept { return probabilities_[1]; }
    [[nodiscard]] std::span<const float> probabilities() const noexcept { return probabilities_; }
    [[nodiscard]] bool abstained() const noexcept {
        if constexpr (Field::kind == field_kind::probability) return false;
        return !selected_;
    }
    [[nodiscard]] const std::optional<bool>& selected() const noexcept requires (Field::kind == field_kind::noul) { return selected_; }
    [[nodiscard]] auto value() const {
        if constexpr (Field::kind == field_kind::probability) return probability_true();
        else {
            if (!selected_) throw std::logic_error("Noul field abstained");
            return *selected_;
        }
    }
    [[nodiscard]] std::string_view model_id() const noexcept { return model_id_; }
    explicit operator bool() const = delete;
private:
    std::optional<bool> selected_;
    std::array<float, 2> probabilities_;
    std::string model_id_;
};

template <class Field>
using field_answer = std::conditional_t<Field::kind == field_kind::choice || Field::kind == field_kind::score,
                                       enum_answer<Field>, binary_answer<Field>>;

template <class... Fields>
class system_one_result {
public:
    using tuple_type = std::tuple<field_answer<Fields>...>;
    explicit system_one_result(tuple_type answers) : answers_(std::move(answers)) {}
    template <fixed_string Name>
    [[nodiscard]] const auto& get() const {
        constexpr auto index = system_one_detail::field_index<Name, Fields...>();
        static_assert(index < sizeof...(Fields), "unknown decision field name");
        return std::get<index>(answers_);
    }
    [[nodiscard]] const tuple_type& answers() const noexcept { return answers_; }
    [[nodiscard]] bool abstained() const noexcept {
        return std::apply([](const auto&... answer) { return (answer.abstained() || ...); }, answers_);
    }
    // Mapping is explicit: callers choose how to handle abstentions and whether
    // a Score is mapped to its modal enum or its fractional expected position.
    template <class Mapper>
    [[nodiscard]] decltype(auto) map(Mapper&& mapper) const {
        return std::apply(std::forward<Mapper>(mapper), answers_);
    }
private:
    tuple_type answers_;
};

template <class Model> class bound_system_one;

template <fixed_string Id, class... Fields>
class bound_system_one<decision_model_definition<Id, Fields...>> {
public:
    using model_type = decision_model_definition<Id, Fields...>;
    using answer_type = system_one_result<Fields...>;
    bound_system_one(model_type model, std::shared_ptr<backend> inference_backend,
                     std::shared_ptr<Diagnostics> diagnostics = {})
        : model_(std::move(model)), backend_(std::move(inference_backend)),
          diagnostics_(std::move(diagnostics)) {
        if (!backend_) throw std::invalid_argument("System One requires a backend");
    }
    [[nodiscard]] system_one_request request(state_value state) const {
        return make_system_one_request(model_, std::move(state));
    }
    [[nodiscard]] result<answer_type> evaluate(state_value state) const {
        const auto started = std::chrono::steady_clock::now();
        const auto record = [&](CallOutcome outcome) {
            if (diagnostics_) diagnostics_->record_call(
                Id.view(), outcome, std::chrono::steady_clock::now() - started);
        };
        const auto batch = request(std::move(state));
        const auto views = batch.views();
        auto responses = backend_->predict_batch(views.requests());
        if (!responses) { record(CallOutcome::error); return responses.error_value(); }
        if (responses->size() != sizeof...(Fields)) {
            record(CallOutcome::error);
            return error{error_code::invalid_backend_output, "System One backend returned wrong number of answers"};
        }
        constexpr std::array sizes{Fields::size...};
        for (std::size_t i = 0; i < sizes.size(); ++i) {
            auto& probabilities = (*responses)[i].scores;
            if (probabilities.size() != sizes[i]) {
                record(CallOutcome::error);
                return error{error_code::invalid_backend_output, "System One field has wrong probability count"};
            }
            double total = 0.0;
            for (auto p : probabilities) {
                if (!std::isfinite(p) || p < 0.0F) {
                    record(CallOutcome::error);
                    return error{error_code::invalid_backend_output, "System One probabilities must be finite and nonnegative"};
                }
                total += p;
            }
            if (!(total > 0.0)) {
                record(CallOutcome::error);
                return error{error_code::invalid_backend_output, "System One probabilities have zero mass"};
            }
            for (auto& p : probabilities) p = static_cast<float>(static_cast<double>(p) / total);
        }
        auto answer = [&]<std::size_t... I>(std::index_sequence<I...>) -> answer_type {
            return answer_type{typename answer_type::tuple_type{
                field_answer<Fields>{std::get<I>(model_.fields), std::move((*responses)[I].scores),
                                      std::move((*responses)[I].model_id)}...}};
        }(std::index_sequence_for<Fields...>{});
        record(answer.abstained() ? CallOutcome::abstain : CallOutcome::success);
        return answer;
    }
    [[nodiscard]] result<answer_type> evaluate(std::string_view state) const {
        return evaluate(text_state(std::string{state}));
    }
    template <class T, class Serializer>
    [[nodiscard]] result<answer_type> evaluate(const T& value, Serializer&& serializer) const {
        static_assert(std::same_as<std::remove_cvref_t<std::invoke_result_t<Serializer, const T&>>, state_value>,
                      "state serializer must return jevt::state_value");
        return evaluate(std::invoke(std::forward<Serializer>(serializer), value));
    }
    template <class T>
        requires requires(const T& value) { { to_jevt_state(value) } -> std::same_as<state_value>; }
    [[nodiscard]] result<answer_type> evaluate(const T& value) const {
        return evaluate(to_jevt_state(value));
    }
private:
    model_type model_;
    std::shared_ptr<backend> backend_;
    std::shared_ptr<Diagnostics> diagnostics_;
};

template <class Model>
[[nodiscard]] auto bind_system_one(Model model, std::shared_ptr<backend> inference_backend) {
    return bound_system_one<Model>{std::move(model), std::move(inference_backend)};
}

} // namespace jevt
