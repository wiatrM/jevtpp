#include "jevt/laya.hpp"

#include <onnxruntime_cxx_api.h>
#include <tokenizers_cpp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace jevt {
namespace {

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open " + path.string());
    return {std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
}

std::filesystem::path first_existing(const std::filesystem::path& directory,
                                     std::initializer_list<std::string_view> names) {
    for (const auto name : names) {
        const auto candidate = directory / name;
        if (std::filesystem::is_regular_file(candidate)) return candidate;
    }
    return {};
}

std::size_t json_integer(const std::string& json, std::string_view key) {
    const std::regex expression{"\\\"" + std::string{key} + "\\\"\\s*:\\s*([0-9]+)"};
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        throw std::runtime_error("missing integer config field: " + std::string{key});
    return static_cast<std::size_t>(std::stoull(match[1].str()));
}

std::string json_string(const std::string& json, std::string_view key) {
    const std::regex expression{"\\\"" + std::string{key} + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\""};
    std::smatch match;
    if (!std::regex_search(json, match, expression))
        throw std::runtime_error("missing string config field: " + std::string{key});
    return match[1].str();
}

std::array<float, 3> json_temperatures(const std::string& json) {
    const std::regex expression{"\\\"temperature\\\"\\s*:\\s*\\[([^\\]]+)\\]"};
    std::smatch match;
    if (!std::regex_search(json, match, expression)) return {1.0F, 1.0F, 1.0F};
    std::array<float, 3> values{1.0F, 1.0F, 1.0F};
    std::stringstream stream{match[1].str()};
    std::string item;
    for (std::size_t i = 0; i < values.size() && std::getline(stream, item, ','); ++i)
        values[i] = std::stof(item);
    return values;
}

std::optional<float> json_number(const std::string& json, std::string_view key) {
    const std::regex expression{"\\\"" + std::string{key} +
                                "\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"};
    std::smatch match;
    if (!std::regex_search(json, match, expression)) return std::nullopt;
    return std::stof(match[1].str());
}

std::string replace_all(std::string text, std::string_view needle, std::string_view replacement) {
    if (needle.empty()) return text;
    std::size_t offset = 0;
    while ((offset = text.find(needle, offset)) != std::string::npos) {
        text.replace(offset, needle.size(), replacement);
        offset += replacement.size();
    }
    return text;
}

std::string bucket(std::size_t count) {
    if (count <= 2) return "2";
    if (count <= 5) return "3-5";
    if (count <= 10) return "6-10";
    return "11+";
}

std::vector<float> softmax(std::span<const float> logits, float temperature) {
    if (!(temperature > 0.0F) || !std::isfinite(temperature))
        throw std::runtime_error("Laya temperature must be finite and positive");
    const auto maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<float> probabilities;
    probabilities.reserve(logits.size());
    for (const auto value : logits) probabilities.push_back(std::exp((value - maximum) / temperature));
    const auto total = std::accumulate(probabilities.begin(), probabilities.end(), 0.0F);
    for (auto& value : probabilities) value /= total;
    return probabilities;
}

#ifdef _WIN32
std::wstring ort_path(const std::filesystem::path& path) { return path.wstring(); }
#else
std::string ort_path(const std::filesystem::path& path) { return path.string(); }
#endif

} // namespace

struct laya_backend::impl {
    explicit impl(laya_options requested)
        : options(std::move(requested)), environment(ORT_LOGGING_LEVEL_WARNING, "jevt-laya") {
        resolve_paths();
        const auto agent_json = read_file(options.agent_config_file);
        max_len = json_integer(agent_json, "max_len");
        head_max_len = json_integer(agent_json, "head_max_len");
        temperatures = json_temperatures(agent_json);
        for (const auto prefix : {std::string{"choice:"}, std::string{"noul:"}})
            for (const auto suffix : {std::string{"2"}, std::string{"3-5"}, std::string{"6-10"}, std::string{"11+"}}) {
                const auto key = prefix + suffix;
                if (auto value = json_number(agent_json, key)) bucket_temperatures.emplace(key, *value);
            }

        const auto tokenizer_config = read_file(options.tokenizer_config_file);
        cls_token = json_string(tokenizer_config, "cls_token");
        sep_token = json_string(tokenizer_config, "sep_token");
        mask_token = json_string(tokenizer_config, "mask_token");
        pad_token = json_string(tokenizer_config, "pad_token");
        tokenizer = tokenizers::Tokenizer::FromBlobJSON(read_file(options.tokenizer_file));
        if (!tokenizer) throw std::runtime_error("failed to create Hugging Face tokenizer");
        cls_id = required_token(cls_token);
        sep_id = required_token(sep_token);
        mask_id = required_token(mask_token);
        pad_id = required_token(pad_token);

        Ort::SessionOptions session_options;
        if (options.intra_op_threads > 0) session_options.SetIntraOpNumThreads(options.intra_op_threads);
        if (options.inter_op_threads > 0) session_options.SetInterOpNumThreads(options.inter_op_threads);
        session_options.SetGraphOptimizationLevel(options.enable_graph_optimizations
            ? GraphOptimizationLevel::ORT_ENABLE_ALL : GraphOptimizationLevel::ORT_DISABLE_ALL);
        const auto path = ort_path(options.model_file);
        session = std::make_unique<Ort::Session>(environment, path.c_str(), session_options);
        validate_contract();
    }

    void resolve_paths() {
        if (options.model_directory.empty()) throw std::invalid_argument("Laya model_directory is required");
        const auto& root = options.model_directory;
        if (options.model_file.empty())
            options.model_file = first_existing(root, {"model.onnx", "laya.onnx", "laya-multilingual.onnx"});
        if (options.tokenizer_file.empty()) options.tokenizer_file = root / "tokenizer/tokenizer.json";
        if (options.tokenizer_config_file.empty()) options.tokenizer_config_file = root / "tokenizer/tokenizer_config.json";
        if (options.agent_config_file.empty())
            options.agent_config_file = first_existing(root, {"laya_config.json", "rl_agent_config.json"});
        for (const auto& path : {options.model_file, options.tokenizer_file,
                                options.tokenizer_config_file, options.agent_config_file})
            if (path.empty() || !std::filesystem::is_regular_file(path))
                throw std::runtime_error("incomplete Laya bundle; missing " + path.string());
    }

    std::int64_t required_token(const std::string& token) {
        const auto id = tokenizer->TokenToId(token);
        if (id < 0) throw std::runtime_error("special token is absent from tokenizer: " + token);
        return id;
    }

    void validate_contract() {
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        for (std::size_t i = 0; i < session->GetInputCount(); ++i)
            inputs.emplace_back(session->GetInputNameAllocated(i, allocator).get());
        for (std::size_t i = 0; i < session->GetOutputCount(); ++i)
            outputs.emplace_back(session->GetOutputNameAllocated(i, allocator).get());
        for (const auto required : {"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"})
        {
            const auto found = std::find(inputs.begin(), inputs.end(), required);
            if (found == inputs.end())
                throw std::runtime_error(std::string{"Laya graph is missing input: "} + required);
            const bool mask = std::string_view{required} == "marker_mask";
            const std::size_t rank = std::string_view{required} == "qtype" ? 1 : 2;
            const auto type = session->GetInputTypeInfo(static_cast<std::size_t>(found - inputs.begin()));
            if (type.GetONNXType() != ONNX_TYPE_TENSOR)
                throw std::runtime_error(std::string{"Laya input "} + required + " must be a tensor");
            const auto tensor = type.GetTensorTypeAndShapeInfo();
            if (tensor.GetElementType() != (mask ? ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL
                                               : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) ||
                tensor.GetDimensionsCount() != rank)
                throw std::runtime_error(std::string{"Laya input "} + required + " must be " +
                    (mask ? "BOOL" : "INT64") + " rank " + std::to_string(rank));
        }
        const auto logits = std::find(outputs.begin(), outputs.end(), "logits");
        if (logits == outputs.end())
            throw std::runtime_error("Laya graph is missing logits output");
        const auto type = session->GetOutputTypeInfo(static_cast<std::size_t>(logits - outputs.begin()));
        if (type.GetONNXType() != ONNX_TYPE_TENSOR)
            throw std::runtime_error("Laya logits output must be a tensor");
        const auto tensor = type.GetTensorTypeAndShapeInfo();
        if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            tensor.GetDimensionsCount() != 2)
            throw std::runtime_error("Laya logits output must be FLOAT rank 2 [batch, marker_width]");
    }

    std::vector<std::int64_t> encode(std::string text) {
        std::scoped_lock lock(tokenizer_mutex);
        auto encoded = tokenizer->Encode(text);
        return {encoded.begin(), encoded.end()};
    }

    std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>> build_sequence(
        const inference_request& request, std::span<const std::string> options_text) {
        const auto type = request.question_kind == inference_request::kind::noul ? std::string{"noul"}
            : request.question_kind == inference_request::kind::score ? std::string{"score"}
            : std::string{"choice"};
        auto head = encode(type + " question: " + replace_all(std::string{request.question}, mask_token, " "));
        std::vector<std::vector<std::int64_t>> option_ids;
        option_ids.reserve(options_text.size());
        for (const auto& option : options_text) {
            auto encoded = encode(" " + replace_all(option, mask_token, " "));
            if (encoded.size() > 48) encoded.resize(48);
            encoded.insert(encoded.begin(), mask_id);
            option_ids.push_back(std::move(encoded));
        }
        const auto option_total = std::accumulate(option_ids.begin(), option_ids.end(), std::size_t{0},
            [](std::size_t total, const auto& value) { return total + value.size(); });
        std::size_t total = option_total;
        if (total + 16 > head_max_len) {
            const auto per = std::max<std::size_t>(4, (head_max_len - std::min<std::size_t>(head_max_len, 16)) /
                                                     std::max<std::size_t>(1, option_ids.size()));
            for (auto& value : option_ids) if (value.size() > per) value.resize(per);
            total = std::accumulate(option_ids.begin(), option_ids.end(), std::size_t{0},
                [](std::size_t sum, const auto& value) { return sum + value.size(); });
        }
        const auto head_budget = head_max_len > total ? head_max_len - total : std::size_t{8};
        if (head.size() > std::max<std::size_t>(8, head_budget)) head.resize(std::max<std::size_t>(8, head_budget));

        std::vector<std::int64_t> sequence{cls_id};
        sequence.insert(sequence.end(), head.begin(), head.end());
        sequence.push_back(sep_id);
        std::vector<std::int64_t> markers;
        for (const auto& encoded : option_ids) {
            markers.push_back(static_cast<std::int64_t>(sequence.size()));
            sequence.insert(sequence.end(), encoded.begin(), encoded.end());
        }
        sequence.push_back(sep_id);
        const auto room = max_len > sequence.size() + 1 ? max_len - sequence.size() - 1 : 0;
        auto state = encode(replace_all(std::string{request.input}, mask_token, " "));
        if (state.size() > room) state.resize(room);
        sequence.insert(sequence.end(), state.begin(), state.end());
        sequence.push_back(sep_id);
        if (sequence.size() > max_len) sequence.resize(max_len);
        markers.erase(std::remove_if(markers.begin(), markers.end(), [&](auto value) {
            return static_cast<std::size_t>(value) >= sequence.size();
        }), markers.end());
        return {std::move(sequence), std::move(markers)};
    }

    result<std::vector<inference_response>> predict_batch(std::span<const inference_request> requests) {
        try {
            if (requests.empty()) return std::vector<inference_response>{};
            struct item {
                std::vector<std::string> options;
                std::vector<std::int64_t> ids;
                std::vector<std::int64_t> markers;
                std::size_t qtype{};
                float temperature{1.0F};
            };
            std::vector<item> items;
            items.reserve(requests.size());
            std::size_t length = 8;
            std::size_t marker_width = 2;
            for (const auto& request : requests) {
                item current;
                current.qtype = request.question_kind == inference_request::kind::noul ? 2U
                    : request.question_kind == inference_request::kind::score ? 1U : 0U;
                if (request.question_kind == inference_request::kind::noul) {
                    current.options = {"false: no, the statement does not hold",
                                       "true: yes, the statement holds"};
                } else {
                    if (request.options.empty())
                        return error{error_code::invalid_request, "Laya question requires at least one option"};
                    if (request.question_kind == inference_request::kind::score &&
                        (request.options.size() < 2 || request.options.size() > 10))
                        return error{error_code::invalid_request, "Laya score requires between 2 and 10 levels"};
                    current.options.reserve(request.options.size());
                    for (std::size_t i = 0; i < request.options.size(); ++i) {
                        auto rendered = request.options[i].empty()
                            ? "option " + std::to_string(i) : std::string{request.options[i]};
                        if (request.question_kind == inference_request::kind::score)
                            rendered = "level " + std::to_string(i) + ": " + rendered;
                        current.options.push_back(std::move(rendered));
                    }
                }
                auto sequence = build_sequence(request, current.options);
                current.ids = std::move(sequence.first);
                current.markers = std::move(sequence.second);
                if (current.markers.size() != current.options.size())
                    return error{error_code::invalid_request, "Laya options do not fit inside head_max_len"};
                const auto prefix = current.qtype == 0 ? "choice:" : current.qtype == 1 ? "score:" : "noul:";
                const auto key = std::string{prefix} + bucket(current.options.size());
                const auto found = bucket_temperatures.find(key);
                current.temperature = found == bucket_temperatures.end()
                    ? temperatures[current.qtype] : found->second;
                length = std::max(length, current.ids.size());
                marker_width = std::max(marker_width, current.markers.size());
                items.push_back(std::move(current));
            }

            const std::size_t batch = items.size();
            std::vector<std::int64_t> input_ids(batch * length, pad_id);
            std::vector<std::int64_t> attention(batch * length, 0);
            std::vector<std::int64_t> marker_positions(batch * marker_width, 0);
            std::unique_ptr<bool[]> marker_mask{new bool[batch * marker_width]{}};
            std::vector<std::int64_t> qtypes(batch, 0);
            for (std::size_t row = 0; row < batch; ++row) {
                const auto& current = items[row];
                std::copy(current.ids.begin(), current.ids.end(), input_ids.begin() + row * length);
                std::fill_n(attention.begin() + row * length, current.ids.size(), 1);
                std::copy(current.markers.begin(), current.markers.end(),
                          marker_positions.begin() + row * marker_width);
                for (std::size_t column = 0; column < current.markers.size(); ++column)
                    marker_mask[row * marker_width + column] = true;
                qtypes[row] = static_cast<std::int64_t>(current.qtype);
            }
            std::array<std::int64_t, 2> text_shape{
                static_cast<std::int64_t>(batch), static_cast<std::int64_t>(length)};
            std::array<std::int64_t, 2> marker_shape{
                static_cast<std::int64_t>(batch), static_cast<std::int64_t>(marker_width)};
            std::array<std::int64_t, 1> qtype_shape{static_cast<std::int64_t>(batch)};
            auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::array<Ort::Value, 5> tensors{
                Ort::Value::CreateTensor<std::int64_t>(memory, input_ids.data(), input_ids.size(), text_shape.data(), 2),
                Ort::Value::CreateTensor<std::int64_t>(memory, attention.data(), attention.size(), text_shape.data(), 2),
                Ort::Value::CreateTensor<std::int64_t>(memory, marker_positions.data(), marker_positions.size(), marker_shape.data(), 2),
                Ort::Value::CreateTensor<bool>(memory, marker_mask.get(), batch * marker_width, marker_shape.data(), 2),
                Ort::Value::CreateTensor<std::int64_t>(memory, qtypes.data(), qtypes.size(), qtype_shape.data(), 1)};
            constexpr std::array input_names{"input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"};
            constexpr std::array output_names{"logits"};
            auto outputs = session->Run(Ort::RunOptions{nullptr}, input_names.data(), tensors.data(), tensors.size(),
                                        output_names.data(), output_names.size());
            if (outputs.size() != 1 || !outputs[0].IsTensor())
                return error{error_code::invalid_backend_output, "Laya logits output must be a tensor"};
            const auto info = outputs[0].GetTensorTypeAndShapeInfo();
            if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
                info.GetShape() != std::vector<std::int64_t>{marker_shape.begin(), marker_shape.end()})
                return error{error_code::invalid_backend_output,
                    "Laya logits must be FLOAT with exact shape [" + std::to_string(batch) + ", " +
                    std::to_string(marker_width) + "] (batch, marker_width)"};
            const auto* logits = outputs[0].GetTensorData<float>();
            std::vector<inference_response> responses;
            responses.reserve(batch);
            for (std::size_t row = 0; row < batch; ++row) {
                const auto& current = items[row];
                // Only active options are probabilities; padded markers may carry -infinity.
                for (std::size_t column = 0; column < current.options.size(); ++column)
                    if (!std::isfinite(logits[row * marker_width + column]))
                        return error{error_code::invalid_backend_output,
                            "Laya logits contain a non-finite value at active option [" +
                            std::to_string(row) + ", " + std::to_string(column) + "]"};
                auto probabilities = softmax(
                    {logits + row * marker_width, current.options.size()}, current.temperature);
                responses.push_back(inference_response{std::move(probabilities), options.model_id});
            }
            return responses;
        } catch (const Ort::Exception& exception) {
            return error{error_code::backend_failure, std::string{"ONNX Runtime: "} + exception.what()};
        } catch (const std::exception& exception) {
            return error{error_code::backend_failure, std::string{"Laya backend: "} + exception.what()};
        }
    }

    result<inference_response> predict(const inference_request& request) {
        const std::array requests{request};
        auto responses = predict_batch(requests);
        if (!responses) return responses.error_value();
        return std::move(responses).value().front();
    }

    laya_options options;
    Ort::Env environment;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;
    std::mutex tokenizer_mutex;
    std::size_t max_len{};
    std::size_t head_max_len{};
    std::array<float, 3> temperatures{1.0F, 1.0F, 1.0F};
    std::unordered_map<std::string, float> bucket_temperatures;
    std::string cls_token, sep_token, mask_token, pad_token;
    std::int64_t cls_id{}, sep_id{}, mask_id{}, pad_id{};
};

laya_backend::laya_backend(laya_options options) : impl_(std::make_unique<impl>(std::move(options))) {}
laya_backend::~laya_backend() = default;
laya_backend::laya_backend(laya_backend&&) noexcept = default;
laya_backend& laya_backend::operator=(laya_backend&&) noexcept = default;
result<inference_response> laya_backend::predict(const inference_request& request) { return impl_->predict(request); }
result<std::vector<inference_response>> laya_backend::predict_batch(
    std::span<const inference_request> requests) { return impl_->predict_batch(requests); }
std::string_view laya_backend::name() const noexcept { return impl_->options.model_id; }
std::size_t laya_backend::max_context_tokens() const noexcept { return impl_->max_len; }
std::size_t laya_backend::max_question_tokens() const noexcept { return impl_->head_max_len; }

namespace models {
std::shared_ptr<backend> laya_multilingual(std::filesystem::path directory, int threads) {
    return std::make_shared<laya_backend>(laya_options{
        .model_directory = std::move(directory),
        .intra_op_threads = threads,
        .model_id = "laya-multilingual-onnx"});
}
}

} // namespace jevt
