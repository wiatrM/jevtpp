#pragma once

#include "jevt/core.hpp"
#include "jevt/diagnostics.hpp"

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
        std::string_view input, std::optional<std::string_view> diagnostic_tag = std::nullopt) const {
        const auto started = std::chrono::steady_clock::now();
        const auto record = [&](CallOutcome outcome) {
            if (diagnostics_) diagnostics_->record_call(
                Schema::name(), outcome, std::chrono::steady_clock::now() - started, diagnostic_tag);
        };
        const inference_request request{Schema::name(), question_, input, schema_.descriptions()};
        auto response = backend_->predict(request);
        if (!response) { record(CallOutcome::error); return response.error_value(); }
        auto scores = std::move(response).value().scores;
        if (scores.size() != Schema::size || scores.empty()) {
            record(CallOutcome::error);
            return error{error_code::invalid_backend_output, "backend returned a score vector with the wrong size"};
        }
        if (std::any_of(scores.begin(), scores.end(), [](float v) { return !std::isfinite(v) || v < 0.0F; })) {
            record(CallOutcome::error);
            return error{error_code::invalid_backend_output, "backend returned a negative or non-finite score"};
        }
        const auto best = static_cast<std::size_t>(std::distance(scores.begin(), std::max_element(scores.begin(), scores.end())));
        const auto total = std::accumulate(scores.begin(), scores.end(), 0.0F);
        if (total <= 0.0F) { record(CallOutcome::error); return error{error_code::invalid_backend_output, "backend returned only zero scores"}; }
        for (auto& score : scores) score /= total;
        const float confidence = scores[best];
        const auto values = Schema::values();
        std::optional<typename Schema::enum_type> selected;
        if (confidence >= threshold_) selected = values[best];
        record(selected ? CallOutcome::success : CallOutcome::abstain);
        return decision_result<Schema>{selected, std::move(scores),
            decision_metadata{response->model_id, confidence, !selected.has_value()}};
    }

    [[nodiscard]] std::future<result<decision_result<Schema>>> choose_async(std::string input) const {
        return std::async(std::launch::async, [self = *this, input = std::move(input)] { return self.choose(input); });
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
        std::string_view input, std::optional<std::string_view> diagnostic_tag = std::nullopt) const {
        const auto started = std::chrono::steady_clock::now();
        const auto record = [&](CallOutcome outcome) {
            if (diagnostics_) diagnostics_->record_call(
                predicate_definition<Id>::name(), outcome,
                std::chrono::steady_clock::now() - started, diagnostic_tag);
        };
        constexpr std::array<std::string_view, 2> options{"false", "true"};
        const inference_request request{predicate_definition<Id>::name(), definition_.question, input, options};
        auto response = backend_->predict(request);
        if (!response) { record(CallOutcome::error); return response.error_value(); }
        if (response->scores.size() != 2) { record(CallOutcome::error); return error{error_code::invalid_backend_output, "predicate backend must return two scores"}; }
        const float total = response->scores[0] + response->scores[1];
        if (!(total > 0.0F) || !std::isfinite(total)) { record(CallOutcome::error); return error{error_code::invalid_backend_output, "invalid predicate scores"}; }
        const float false_score = response->scores[0] / total;
        const float true_score = response->scores[1] / total;
        const bool value = true_score >= false_score;
        const float confidence = std::max(true_score, false_score);
        const auto selected = confidence >= threshold_ ? std::optional<bool>{value} : std::nullopt;
        record(selected ? CallOutcome::success : CallOutcome::abstain);
        return predicate_result<Id>{selected,
                                    confidence, response->model_id};
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
        const float threshold = options.abstain_threshold >= 0.0F ? options.abstain_threshold : threshold_;
        return {std::move(schema), std::move(selected_backend), threshold, std::move(options.question), diagnostics_};
    }
    template <fixed_string Id>
    [[nodiscard]] bound_predicate<Id> bind(predicate_definition<Id> definition, bind_options options = {}) const {
        auto selected_backend = options.backend_override ? std::move(options.backend_override) : backend_;
        if (!selected_backend) throw std::invalid_argument("jevt context has no backend");
        const float threshold = options.abstain_threshold >= 0.0F ? options.abstain_threshold : threshold_;
        return {definition, std::move(selected_backend), threshold, diagnostics_};
    }
private:
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
