#include "jevt/runtime.hpp"

#include <algorithm>
#include <cctype>

namespace jevt {
namespace {
std::mutex default_mutex;
std::weak_ptr<context> global_context;

std::string lowercase(std::string_view input) {
    std::string output(input);
    std::transform(output.begin(), output.end(), output.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return output;
}
}

context::context(init_options options)
    : backend_(std::move(options.inference_backend)), threshold_(options.abstain_threshold),
      diagnostics_(std::move(options.diagnostics)) {
    if (!backend_) throw std::invalid_argument("jevt::init requires an inference backend");
    if (!(threshold_ >= 0.0F && threshold_ <= 1.0F))
        throw std::invalid_argument("abstain threshold must be between 0 and 1");
    if (!diagnostics_) diagnostics_ = std::make_shared<Diagnostics>();
}

app init(init_options options) {
    auto instance = std::make_shared<context>(std::move(options));
    std::scoped_lock lock(default_mutex);
    global_context = instance;
    return app{std::move(instance)};
}

std::shared_ptr<context> default_context() {
    std::scoped_lock lock(default_mutex);
    return global_context.lock();
}

keyword_backend::keyword_backend(keyword_table keywords, std::string model_id)
    : keywords_(std::move(keywords)), model_id_(std::move(model_id)) {}

result<inference_response> keyword_backend::predict(const inference_request& request) {
    if (keywords_.size() != request.options.size())
        return error{error_code::invalid_request, "keyword table and option count differ"};
    const auto normalized = lowercase(request.input);
    std::vector<float> scores(keywords_.size(), 1.0F);
    for (std::size_t option_index = 0; option_index < keywords_.size(); ++option_index) {
        for (const auto& keyword : keywords_[option_index]) {
            if (!keyword.empty() && normalized.find(lowercase(keyword)) != std::string::npos)
                scores[option_index] += 4.0F;
        }
    }
    return inference_response{std::move(scores), model_id_};
}

} // namespace jevt
