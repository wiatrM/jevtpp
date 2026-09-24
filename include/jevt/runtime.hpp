#pragma once

#include "jevt/core.hpp"
#include "jevt/diagnostics.hpp"
#include "jevt/system_one.hpp"

#include <chrono>
#include <future>
#include <mutex>
#include <numeric>

namespace jevt {

struct bind_options {
    std::string question;
    float abstain_threshold = -1.0F;
    std::shared_ptr<backend> backend_override;
};

struct init_options {
    std::shared_ptr<backend> inference_backend;
    float abstain_threshold = 0.55F;
    std::shared_ptr<Diagnostics> diagnostics;
};

class context;

template <class Schema>
class bound_decision {
public:
    bound_decision(Schema schema, std::shared_ptr<backend> backend, float threshold, std::string question,
                   std::shared_ptr<Diagnostics> diagnostics)
        : schema_(std::move(schema)), backend_(std::move(backend)), threshold_(threshold),
          question_(std::move(question)), diagnostics_(std::move(diagnostics)) {}

    [[nodiscard]] result<decision_result<Schema>> choose(
        std::string_view input, std::optional<std::string_view> diagnostic_tag = std::nullopt,
        evaluation_options execution = {}, content_kind input_kind = content_kind::text) const {
        const auto started = std::chrono::steady_clock::now();
        token_usage usage;
        const auto record = [&](CallOutcome outcome) {
            if (diagnostics_) diagnostics_->record_call(
                Schema::name(), outcome, std::chrono::steady_clock::now() - started, diagnostic_tag, usage);
        };
        if (auto failure = execution_error(execution)) { record(CallOutcome::error); return *failure; }
        const inference_request request{Schema::name(), question_, input, schema_.descriptions(),
            inference_request::kind::choice, input_kind, text_metadata(question_), schema_.criteria_metadata(),
            execution.deadline, execution.cancellation};
        auto response = backend_->predict(request);
        if (!response) { record(CallOutcome::error); return response.error_value(); }
        if (response->metadata.usage) usage = *response->metadata.usage;
        if (auto failure = execution_error(execution)) { record(CallOutcome::error); return *failure; }
        auto scores = std::move(response).value().scores;
        if (scores.size() != Schema::size || scores.empty()) {
            record(CallOutcome::error);
            return error{error_code::invalid_backend_output, "backend returned a score vector with the wrong size"};
        }
        if (std::any_of(scores.begin(), scores.end(), [](float v) { return !std::isfinite(v) || v < 0.0F; })) {
            record(CallOutcome::error);
            return error{error_code::invalid_backend_output, "backend returned a negative or non-finite score"};
        }
        auto best = static_cast<std::size_t>(std::distance(scores.begin(), std::max_element(scores.begin(), scores.end())));
        if (response->metadata.provider_choice_index && *response->metadata.provider_choice_index < scores.size() &&
            scores[*response->metadata.provider_choice_index] == scores[best]) best = *response->metadata.provider_choice_index;
        const auto total = std::accumulate(scores.begin(), scores.end(), 0.0F);
        if (!(total > 0.0F) || !std::isfinite(total)) { record(CallOutcome::error); return error{error_code::invalid_backend_output, "backend returned an invalid score total"}; }
        for (auto& score : scores) score /= total;
        const float confidence = scores[best];
        const auto values = Schema::values();
        std::optional<typename Schema::enum_type> selected;
        if (confidence >= threshold_) selected = values[best];
        record(selected ? CallOutcome::success : CallOutcome::abstain);
        return decision_result<Schema>{selected, std::move(scores),
            decision_metadata{response->model_id, confidence, !selected.has_value(), std::move(response->metadata)}};
    }

    [[nodiscard]] result<decision_result<Schema>> choose(
        state_value input, std::optional<std::string_view> tag = std::nullopt, evaluation_options execution = {}) const {
        return choose(input.content, tag, std::move(execution), input.kind);
    }

    [[nodiscard]] std::future<result<decision_result<Schema>>> choose_async(std::string input, evaluation_options execution = {}) const {
        return std::async(std::launch::async, [self = *this, input = std::move(input), execution] { return self.choose(input, std::nullopt, execution); });
    }
private:
    Schema schema_;
    std::shared_ptr<backend> backend_;
    float threshold_;
    std::string question_;
    std::shared_ptr<Diagnostics> diagnostics_;
};

template <fixed_string Id>
class bound_predicate {
public:
    bound_predicate(predicate_definition<Id> definition, std::shared_ptr<backend> backend, float threshold,
                    std::shared_ptr<Diagnostics> diagnostics)
        : definition_(definition), backend_(std::move(backend)), threshold_(threshold), diagnostics_(std::move(diagnostics)) {}
    [[nodiscard]] result<predicate_result<Id>> evaluate(
        std::string_view input, std::optional<std::string_view> diagnostic_tag = std::nullopt,
        evaluation_options execution = {}, content_kind input_kind = content_kind::text) const {
        const auto started = std::chrono::steady_clock::now();
        token_usage usage;
        const auto record = [&](CallOutcome outcome) {
            if (diagnostics_) diagnostics_->record_call(
                predicate_definition<Id>::name(), outcome,
                std::chrono::steady_clock::now() - started, diagnostic_tag, usage);
        };
        if (auto failure = execution_error(execution)) { record(CallOutcome::error); return *failure; }
        constexpr std::array<std::string_view, 2> options{"false", "true"};
        const inference_request request{predicate_definition<Id>::name(), definition_.question, input,
                                        options, inference_request::kind::noul, input_kind,
                                        text_metadata(definition_.question), {}, execution.deadline, execution.cancellation};
        auto response = backend_->predict(request);
        if (!response) { record(CallOutcome::error); return response.error_value(); }
        if (response->metadata.usage) usage = *response->metadata.usage;
        if (auto failure = execution_error(execution)) { record(CallOutcome::error); return *failure; }
        if (response->scores.size() != 2) { record(CallOutcome::error); return error{error_code::invalid_backend_output, "predicate backend must return two scores"}; }
        if (std::any_of(response->scores.begin(), response->scores.end(),
                        [](float value) { return !std::isfinite(value) || value < 0.0F; })) {
            record(CallOutcome::error);
            return error{error_code::invalid_backend_output, "predicate backend returned a negative or non-finite score"};
        }
        const float total = response->scores[0] + response->scores[1];
        if (!(total > 0.0F) || !std::isfinite(total)) { record(CallOutcome::error); return error{error_code::invalid_backend_output, "invalid predicate scores"}; }
        const float false_score = response->scores[0] / total;
        const float true_score = response->scores[1] / total;
        const bool value = true_score >= false_score;
        const float confidence = std::max(true_score, false_score);
        const auto selected = confidence >= threshold_ ? std::optional<bool>{value} : std::nullopt;
        record(selected ? CallOutcome::success : CallOutcome::abstain);
        return predicate_result<Id>{selected,
                                    confidence, response->model_id, std::move(response->metadata)};
    }
    [[nodiscard]] result<predicate_result<Id>> evaluate(
        state_value input, std::optional<std::string_view> tag = std::nullopt, evaluation_options execution = {}) const {
        return evaluate(input.content, tag, std::move(execution), input.kind);
    }
private:
    predicate_definition<Id> definition_;
    std::shared_ptr<backend> backend_;
    float threshold_;
    std::shared_ptr<Diagnostics> diagnostics_;
};

class context {
public:
    explicit context(init_options options);
    [[nodiscard]] std::shared_ptr<Diagnostics> diagnostics() const noexcept { return diagnostics_; }
    template <class Schema>
    [[nodiscard]] bound_decision<Schema> bind(Schema schema, bind_options options = {}) const {
        auto selected_backend = options.backend_override ? std::move(options.backend_override) : backend_;
        if (!selected_backend) throw std::invalid_argument("jevt context has no backend");
        const float threshold = resolve_threshold(options.abstain_threshold);
        return {std::move(schema), std::move(selected_backend), threshold, std::move(options.question), diagnostics_};
    }
    template <fixed_string Id>
    [[nodiscard]] bound_predicate<Id> bind(predicate_definition<Id> definition, bind_options options = {}) const {
        auto selected_backend = options.backend_override ? std::move(options.backend_override) : backend_;
        if (!selected_backend) throw std::invalid_argument("jevt context has no backend");
        const float threshold = resolve_threshold(options.abstain_threshold);
        return {definition, std::move(selected_backend), threshold, diagnostics_};
    }
    template <class Model>
    [[nodiscard]] bound_system_one<Model> bind_system_one(Model model) const {
        if (!backend_) throw std::invalid_argument("jevt context has no backend");
        return {std::move(model), backend_, diagnostics_};
    }
private:
    [[nodiscard]] float resolve_threshold(float value) const {
        if (value == -1.0F) return threshold_;
        if (!std::isfinite(value) || value < 0.0F || value > 1.0F)
            throw std::invalid_argument("binding threshold must be -1 (inherit) or between 0 and 1");
        return value;
    }
    std::shared_ptr<backend> backend_;
    float threshold_;
    std::shared_ptr<Diagnostics> diagnostics_;
};

class app {
public:
    app() = default;
    explicit app(std::shared_ptr<context> owned) : owned_(std::move(owned)) {}
    [[nodiscard]] const context& get_context() const { return *owned_; }
    [[nodiscard]] std::shared_ptr<Diagnostics> diagnostics() const noexcept {
        return owned_ ? owned_->diagnostics() : nullptr;
    }
private:
    std::shared_ptr<context> owned_;
};

[[nodiscard]] app init(init_options options);
[[nodiscard]] std::shared_ptr<context> default_context();

template <class Definition>
[[nodiscard]] auto bind(Definition definition, bind_options options = {}) {
    auto current = default_context();
    if (!current) throw std::runtime_error("jevt::init must be called before jevt::bind");
    return current->bind(std::move(definition), std::move(options));
}

template <class Model>
[[nodiscard]] auto bind_system_one(Model model) {
    auto current = default_context();
    if (!current) throw std::runtime_error("jevt::init must be called before jevt::bind_system_one");
    return current->bind_system_one(std::move(model));
}

class keyword_backend final : public backend {
public:
    using keyword_table = std::vector<std::vector<std::string>>;
    explicit keyword_backend(keyword_table keywords, std::string model_id = "keyword-v1");
    [[nodiscard]] result<inference_response> predict(const inference_request&) override;
    [[nodiscard]] std::string_view name() const noexcept override { return model_id_; }
private:
    keyword_table keywords_;
    std::string model_id_;
};

} // namespace jevt
