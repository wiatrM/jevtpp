#include "jevt/laya.hpp"

#include <onnxruntime_cxx_api.h>
#include <tokenizers_cpp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <deque>
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
        if (options.intra_op_threads < 0 || options.inter_op_threads < 0 || options.device_id < 0)
            throw std::invalid_argument("Laya thread counts and device_id must be nonnegative");
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
        session_options.SetExecutionMode(options.parallel_execution ? ORT_PARALLEL : ORT_SEQUENTIAL);
        session_options.AddConfigEntry("session.intra_op.allow_spinning", options.allow_spinning ? "1" : "0");
        session_options.AddConfigEntry("session.inter_op.allow_spinning", options.allow_spinning ? "1" : "0");
        session_options.SetGraphOptimizationLevel(options.enable_graph_optimizations
            ? GraphOptimizationLevel::ORT_ENABLE_ALL : GraphOptimizationLevel::ORT_DISABLE_ALL);
        if (options.provider == laya_provider::cuda) {
            const auto providers = Ort::GetAvailableProviders();
            if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") == providers.end())
                throw std::runtime_error("CUDAExecutionProvider unavailable; install an ONNX Runtime GPU build (no CPU fallback)");
            Ort::CUDAProviderOptions cuda;
            cuda.Update({{"device_id", std::to_string(options.device_id)},
                         {"use_tf32", options.use_tf32 ? "1" : "0"},
                         {"do_copy_in_default_stream", "1"}});
            session_options.AppendExecutionProvider_CUDA_V2(*cuda);
        }
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

    struct prepared_head {
        std::vector<std::int64_t> ids, markers;
    };

    std::shared_ptr<const prepared_head> prepare_head(
        const inference_request& request, std::span<const std::string> options_text) {
        // Length-prefix every component. Identity includes kind and exact text;
        // cache lifetime is one immutable model/tokenizer/options instance.
        std::string key(1, static_cast<char>(request.question_kind));
        const auto append_key = [&](std::string_view text) {
            key += std::to_string(text.size()); key += ':'; key += text;
        };
        append_key(request.question);
        for (const auto& option : options_text) append_key(option);
        const bool cacheable = options.schema_cache_entries && key.size() <= options.schema_cache_bytes;
        if (cacheable) {
            std::lock_guard lock(cache_mutex);
            if (const auto found = head_cache.find(key); found != head_cache.end()) {
                ++cache_hits;
                return found->second;
            }
        }
        ++cache_misses;
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

        auto prepared = std::make_shared<prepared_head>();
        auto& sequence = prepared->ids;
        auto& markers = prepared->markers;
        sequence.push_back(cls_id);
        sequence.insert(sequence.end(), head.begin(), head.end());
        sequence.push_back(sep_id);
        for (const auto& encoded : option_ids) {
            markers.push_back(static_cast<std::int64_t>(sequence.size()));
            sequence.insert(sequence.end(), encoded.begin(), encoded.end());
        }
        sequence.push_back(sep_id);
        if (cacheable) {
            std::string map_key = key;
            // Count retained payload capacity (not allocator/container metadata).
            const auto bytes = key.capacity() + map_key.capacity() +
                (sequence.capacity() + markers.capacity()) * sizeof(std::int64_t);
            std::lock_guard lock(cache_mutex);
            if (bytes <= options.schema_cache_bytes && !head_cache.contains(key)) {
                while (!cache_order.empty() && (head_cache.size() >= options.schema_cache_entries ||
                       cached_bytes > options.schema_cache_bytes - bytes)) {
                    cached_bytes -= cache_order.front().second;
                    head_cache.erase(cache_order.front().first);
                    cache_order.pop_front();
                }
                cache_order.emplace_back(std::move(key), bytes);
                try { head_cache.emplace(std::move(map_key), prepared); }
                catch (...) { cache_order.pop_back(); throw; }
                cached_bytes += bytes;
            }
        }
        return prepared;
    }

    std::pair<std::vector<std::int64_t>, std::vector<std::int64_t>> build_sequence(
        const inference_request& request, std::span<const std::string> options_text,
        const std::vector<std::int64_t>& state) {
        const auto prepared = prepare_head(request, options_text);
        auto sequence = prepared->ids;
        auto markers = prepared->markers;
        const auto room = max_len > sequence.size() + 1 ? max_len - sequence.size() - 1 : 0;
        const auto budget = options.context_token_limit ? std::min(room, options.context_token_limit) : room;
        sequence.insert(sequence.end(), state.begin(), state.begin() + std::min(state.size(), budget));
        sequence.push_back(sep_id);
        if (sequence.size() > max_len) sequence.resize(max_len);
        markers.erase(std::remove_if(markers.begin(), markers.end(), [&](auto value) {
            return static_cast<std::size_t>(value) >= sequence.size();
        }), markers.end());
        return {std::move(sequence), std::move(markers)};
    }

    struct buffers {
        std::vector<std::int64_t> input_ids, attention, positions, qtypes;
        std::unique_ptr<bool[]> mask;
        std::size_t mask_capacity{};
        [[nodiscard]] std::size_t bytes() const {
            return (input_ids.capacity() + attention.capacity() + positions.capacity() + qtypes.capacity()) *
                   sizeof(std::int64_t) + mask_capacity * sizeof(bool);
        }
    };
    struct buffer_lease {
        impl& owner;
        std::unique_ptr<buffers> value;
        ~buffer_lease() {
            if (!value || value->bytes() > owner.options.reusable_buffer_bytes) return;
            // Retained buffers contain input tokens: wipe before pooling.
            std::fill(value->input_ids.begin(), value->input_ids.end(), 0);
            try {
                std::lock_guard lock(owner.pool_mutex);
                if (owner.pool.size() < owner.options.reusable_buffers) owner.pool.push_back(std::move(value));
            } catch (...) { /* caching must not turn a successful call into failure */ }
        }
    };
    buffer_lease acquire_buffers() {
        std::lock_guard lock(pool_mutex);
        if (pool.empty()) return {*this, std::make_unique<buffers>()};
        auto value = std::move(pool.back()); pool.pop_back(); ++buffer_reuses;
        return {*this, std::move(value)};
    }

    std::vector<std::int64_t> encode_state(std::string_view input) {
        auto full = encode(replace_all(std::string{input}, mask_token, " "));
        const auto limit = options.context_token_limit ? std::min(max_len, options.context_token_limit) : max_len;
        // Do not retain a huge unused token-vector capacity across a batch.
        return {full.begin(), full.begin() + std::min(full.size(), limit)};
    }

    result<std::vector<inference_response>> predict_batch(std::span<const inference_request> requests) {
        try {
            if (requests.empty()) return std::vector<inference_response>{};
            for (const auto& request : requests) if (auto failure = execution_error(request)) return *failure;
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
            // Shared state across typed fields is encoded once per batch, never
            // persisted in the schema cache. Views live only during this call.
            std::unordered_map<std::string_view, std::vector<std::int64_t>> states;
            for (const auto& request : requests) {
                item current;
                current.qtype = request.question_kind == inference_request::kind::noul ? 2U
                    : request.question_kind == inference_request::kind::score ? 1U : 0U;
                if (request.question_kind == inference_request::kind::noul) {
                    const auto descriptions = local_noul_criteria(request);
                    current.options.assign(descriptions.begin(), descriptions.end());
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
                auto state_found = states.find(request.input);
                if (state_found == states.end())
                    state_found = states.emplace(request.input, encode_state(request.input)).first;
                auto sequence = build_sequence(request, current.options, state_found->second);
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
            if (batch > std::numeric_limits<std::size_t>::max() / std::max(length, marker_width))
                return error{error_code::invalid_request, "Laya batch dimensions overflow"};
            auto lease = acquire_buffers();
            auto& storage = *lease.value;
            auto& input_ids = storage.input_ids;
            auto& attention = storage.attention;
            auto& marker_positions = storage.positions;
            auto& marker_mask = storage.mask;
            auto& qtypes = storage.qtypes;
            input_ids.assign(batch * length, pad_id);
            attention.assign(batch * length, 0);
            marker_positions.assign(batch * marker_width, 0);
            if (storage.mask_capacity < batch * marker_width) {
                marker_mask = std::make_unique<bool[]>(batch * marker_width);
                storage.mask_capacity = batch * marker_width;
            }
            std::fill_n(marker_mask.get(), batch * marker_width, false);
            qtypes.assign(batch, 0);
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
            std::vector<Ort::Value> outputs;
            for (const auto& request : requests) if (auto failure = execution_error(request)) return *failure;
            ++runs;
            if (options.use_io_binding) {
                Ort::IoBinding binding{*session};
                for (std::size_t i = 0; i < tensors.size(); ++i) binding.BindInput(input_names[i], tensors[i]);
                // Tiny probability output belongs on CPU for validation/softmax.
                // Let ORT allocate it so malformed output shapes remain detectable.
                binding.BindOutput("logits", memory);
                binding.SynchronizeInputs();
                session->Run(Ort::RunOptions{nullptr}, binding);
                binding.SynchronizeOutputs();
                outputs = binding.GetOutputValues();
            } else {
                outputs = session->Run(Ort::RunOptions{nullptr}, input_names.data(), tensors.data(), tensors.size(),
                                       output_names.data(), output_names.size());
            }
            for (const auto& request : requests) if (auto failure = execution_error(request)) return *failure;
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
                execution_metadata metadata;
                metadata.provider = "onnxruntime";
                // Exact per-row usage remains additive when a queue combines
                // requests from independent evaluations into the same batch.
                metadata.usage = std::make_shared<const token_usage>(token_usage{current.ids.size(), 0});
                responses.push_back(inference_response{std::move(probabilities), options.model_id, std::move(metadata)});
            }
            return responses;
        } catch (const Ort::Exception& exception) {
            return error{error_code::backend_failure, std::string{"ONNX Runtime: "} + exception.what()};
        } catch (const std::invalid_argument& exception) {
            return error{error_code::invalid_request, exception.what()};
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
    mutable std::mutex cache_mutex, pool_mutex;
    std::unordered_map<std::string, std::shared_ptr<const prepared_head>> head_cache;
    std::deque<std::pair<std::string, std::size_t>> cache_order;
    std::size_t cached_bytes{};
    std::vector<std::unique_ptr<buffers>> pool;
    std::atomic<std::uint64_t> cache_hits{}, cache_misses{}, buffer_reuses{}, runs{};
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
laya_provider laya_backend::provider() const noexcept { return impl_->options.provider; }
result<std::size_t> laya_backend::warmup(std::span<const inference_request> requests, std::size_t iterations) {
    if (requests.empty()) return error{error_code::invalid_request, "warmup requires representative requests"};
    for (std::size_t i = 0; i < iterations; ++i) {
        auto response = predict_batch(requests);
        if (!response) return response.error_value();
    }
    return iterations;
}
laya_statistics laya_backend::statistics() const {
    std::scoped_lock lock(impl_->cache_mutex, impl_->pool_mutex);
    return {impl_->cache_hits.load(), impl_->cache_misses.load(), impl_->buffer_reuses.load(), impl_->runs.load(),
            impl_->head_cache.size(), impl_->cached_bytes, impl_->pool.size()};
}

namespace models {
std::shared_ptr<backend> laya_multilingual(std::filesystem::path directory, int threads) {
    return std::make_shared<laya_backend>(laya_options{
        .model_directory = std::move(directory),
        .intra_op_threads = threads,
        .model_id = "laya-multilingual-onnx"});
}
}

} // namespace jevt
