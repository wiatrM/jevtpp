#include <jevt/laya_native.hpp>
#include <laya/runtime.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace jevt {
namespace {
laya::backend_type native_provider(const laya_native_options& options) {
    if (options.model_directory.empty()) throw std::invalid_argument("native model_directory is required");
    if (options.precision != laya_native_precision::strict_fp32 &&
        options.precision != laya_native_precision::optimized_fp32 &&
        options.precision != laya_native_precision::bf16)
        throw std::invalid_argument("unsupported native precision");
    if (options.provider == laya_native_provider::cpu) {
        if (options.precision != laya_native_precision::strict_fp32)
            throw std::invalid_argument("native CPU requires strict_fp32 precision");
        return laya::backend_type::cpu;
    }
    if (options.provider != laya_native_provider::cuda)
        throw std::invalid_argument("unsupported native provider");
    return laya::backend_type::cuda;
}
std::string bucket(std::size_t count) {
    return count <= 2 ? "2" : count <= 5 ? "3-5" : count <= 10 ? "6-10" : "11+";
}
int question_type(inference_request::kind kind) {
    switch (kind) {
        case inference_request::kind::choice: return 0;
        case inference_request::kind::score: return 1;
        case inference_request::kind::noul: return 2;
    }
    throw std::invalid_argument("unsupported question kind");
}
constexpr std::array<const char*, 3> type_names{"choice", "score", "noul"};

std::size_t positive_dimension(const laya::json& config, const char* name, int fallback) {
    const auto value = config.value(name, laya::json(fallback));
    if (!value.is_number_integer())
        throw std::invalid_argument(std::string("native ") + name + " must be an integer");
    // Check signedness before conversion: JSON can represent unsigned values
    // larger than INT64_MAX, and negative values must not wrap into size_t.
    const bool valid = value.is_number_unsigned()
        ? value.get<std::uint64_t>() > 0 && value.get<std::uint64_t>() <= static_cast<std::uint64_t>(std::numeric_limits<int>::max())
        : value.get<std::int64_t>() > 0 && value.get<std::int64_t>() <= std::numeric_limits<int>::max();
    if (!valid) throw std::invalid_argument(std::string("native ") + name + " is outside the supported integer range");
    return value.get<std::size_t>();
}
} // namespace

struct laya_native_backend::impl {
    explicit impl(laya_native_options requested)
        : options(std::move(requested)),
          model(options.model_directory, native_provider(options),
                options.precision == laya_native_precision::bf16 ? laya::precision_type::bf16 : laya::precision_type::fp32,
                options.precision != laya_native_precision::strict_fp32,
                options.precision == laya_native_precision::optimized_fp32),
          tokenizer(options.model_directory / "tokenizer/tokenizer.json") {
        const auto& config = model.config();
        max_len = positive_dimension(config, "max_len", 512);
        head_max_len = positive_dimension(config, "head_max_len", 192);
        if ((max_len != 512 && max_len != 1024) || head_max_len >= max_len)
            throw std::invalid_argument("native serving limits must match a supported model profile");
        auto temperature = config.value("temperature", laya::json::array({1, 1, 1}));
        if (!temperature.is_array() || temperature.size() != 3)
            throw std::invalid_argument("native temperature must contain three values");
        for (std::size_t i = 0; i < temperatures.size(); ++i) {
            temperatures[i] = temperature[i].get<float>();
            validate_temperature(temperatures[i]);
        }
        const auto by_options = config.value("temperature_by_options", laya::json::object());
        if (!by_options.is_object())
            throw std::invalid_argument("native temperature_by_options must be an object");
        for (const auto& [key, value] : by_options.items()) {
            const auto t = value.get<float>();
            validate_temperature(t);
            bucket_temperatures.emplace(key, t);
        }
        std::ifstream stream(options.model_directory / "tokenizer/tokenizer_config.json");
        const auto settings = laya::json::parse(stream);
        const auto token_text = [&](const char* name) {
            const auto& value = settings.at(name);
            return value.is_string() ? value.get<std::string>() : value.at("content").get<std::string>();
        };
        mask = token_text("mask_token");
        if (mask.empty()) throw std::invalid_argument("native mask token is empty");
        mask_id = tokenizer.token_id(mask);
        cls_id = tokenizer.token_id(token_text("cls_token"));
        sep_id = tokenizer.token_id(token_text("sep_token"));
        pad_id = tokenizer.token_id(token_text("pad_token"));
        if (std::min({mask_id, cls_id, sep_id, pad_id}) < 0)
            throw std::invalid_argument("native tokenizer is missing a special token");
        device = model.device_name();
    }
    static void validate_temperature(float value) {
        if (!std::isfinite(value) || value <= 0)
            throw std::invalid_argument("native temperature must be finite and positive");
    }
    std::vector<std::int32_t> encode(std::string text) {
        std::size_t pos = 0;
        while ((pos = text.find(mask, pos)) != std::string::npos) {
            text.replace(pos, mask.size(), " ");
            ++pos;
        }
        return tokenizer.encode(text);
    }
    struct head { std::vector<std::int32_t> ids, markers; };
    std::shared_ptr<const head> prepare_head(const inference_request& request, int type,
                                           const std::vector<std::string>& labels) {
        std::string key(1, static_cast<char>(type));
        const auto append = [&](std::string_view text) {
            key += std::to_string(text.size()); key += ':'; key += text;
        };
        append(request.question);
        for (const auto& label : labels) append(label);
        const bool cacheable = options.schema_cache_entries && key.size() <= options.schema_cache_bytes;
        if (cacheable) {
            const auto found = heads.find(key);
            if (found != heads.end()) return found->second;
        }
        auto heading = encode(std::string(type_names[type]) + " question: " + std::string(request.question));
        std::vector<std::vector<std::int32_t>> encoded;
        std::size_t used = 0;
        for (const auto& label : labels) {
            auto ids = encode(" " + label);
            if (ids.size() > 48) ids.resize(48);
            ids.insert(ids.begin(), mask_id);
            used += ids.size();
            encoded.push_back(std::move(ids));
        }
        if (used + 16 > head_max_len) {
            const auto per = std::max<std::size_t>(4, (head_max_len - std::min<std::size_t>(head_max_len, 16)) / labels.size());
            used = 0;
            for (auto& ids : encoded) {
                if (ids.size() > per) ids.resize(per);
                used += ids.size();
            }
        }
        const auto budget = head_max_len > used ? head_max_len - used : std::size_t{8};
        if (heading.size() > std::max<std::size_t>(8, budget)) heading.resize(std::max<std::size_t>(8, budget));
        auto result = std::make_shared<head>();
        result->ids.push_back(cls_id);
        result->ids.insert(result->ids.end(), heading.begin(), heading.end());
        result->ids.push_back(sep_id);
        for (const auto& ids : encoded) {
            result->markers.push_back(static_cast<std::int32_t>(result->ids.size()));
            result->ids.insert(result->ids.end(), ids.begin(), ids.end());
        }
        result->ids.push_back(sep_id);
        if (result->markers.back() >= static_cast<std::int32_t>(max_len))
            throw std::invalid_argument("native options exceed the sequence limit");
        if (cacheable) {
            std::string map_key = key;
            const auto bytes = key.capacity() + map_key.capacity() +
                (result->ids.capacity() + result->markers.capacity()) * sizeof(std::int32_t);
            if (bytes <= options.schema_cache_bytes) {
                while (!order.empty() && (heads.size() >= options.schema_cache_entries ||
                       cached_bytes > options.schema_cache_bytes - bytes)) {
                    cached_bytes -= order.front().second;
                    heads.erase(order.front().first); order.pop_front();
                }
                order.emplace_back(std::move(key), bytes);
                try { heads.emplace(std::move(map_key), result); }
                catch (...) { order.pop_back(); throw; }
                cached_bytes += bytes;
            }
        }
        return result;
    }
    result<std::vector<inference_response>> predict_batch(std::span<const inference_request> requests) {
        if (requests.empty()) return std::vector<inference_response>{};
        if (requests.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) / max_len)
            return error{error_code::invalid_request, "native batch dimensions exceed int limits"};
        std::lock_guard lock(mutex);
        try {
            std::vector<inference_response> responses(requests.size());
            std::vector<std::size_t> rows;
            std::vector<std::vector<std::int32_t>> sequences, markers;
            std::vector<float> calibration;
            std::unordered_map<std::string_view, std::vector<std::int32_t>> states;
            laya::batch batch;
            batch.length = 8;
            batch.options = 2;
            for (std::size_t index = 0; index < requests.size(); ++index) {
                const auto& request = requests[index];
                const int type = question_type(request.question_kind);
                if (type != 2 && (request.options.empty() || request.options.size() > 255))
                    throw std::invalid_argument("native choice/score requires 1 through 255 options");
                if (type == 1 && (request.options.size() < 2 || request.options.size() > 10))
                    throw std::invalid_argument("native score requires 2 through 10 levels");
                responses[index].model_id = options.model_id;
                // A one-option softmax is exactly one; upstream requires two
                // active markers, so omit this row without changing ordering.
                if (type == 0 && request.options.size() == 1) {
                    responses[index].scores = {1.0F};
                    continue;
                }
                std::vector<std::string> labels;
                if (type == 2) labels = {"false: no, the statement does not hold", "true: yes, the statement holds"};
                else for (std::size_t i = 0; i < request.options.size(); ++i) {
                    auto text = request.options[i].empty() ? "option " + std::to_string(i) : std::string(request.options[i]);
                    if (type == 1) text = "level " + std::to_string(i) + ": " + text;
                    labels.push_back(std::move(text));
                }
                const auto prepared = prepare_head(request, type, labels);
                auto sequence = prepared->ids;
                auto state = states.find(request.input);
                if (state == states.end()) {
                    auto full = encode(std::string(request.input));
                    const auto limit = options.context_token_limit ? std::min(max_len, options.context_token_limit) : max_len;
                    std::vector<std::int32_t> bounded(full.begin(), full.begin() + std::min(full.size(), limit));
                    state = states.emplace(request.input, std::move(bounded)).first;
                }
                const auto room = max_len > sequence.size() + 1 ? max_len - sequence.size() - 1 : 0;
                sequence.insert(sequence.end(), state->second.begin(), state->second.begin() + std::min(room, state->second.size()));
                sequence.push_back(sep_id);
                if (sequence.size() > max_len) sequence.resize(max_len);
                batch.length = std::max(batch.length, static_cast<int>(sequence.size()));
                batch.options = std::max(batch.options, static_cast<int>(labels.size()));
                batch.lengths.push_back(static_cast<std::int32_t>(sequence.size()));
                batch.counts.push_back(static_cast<std::int32_t>(labels.size()));
                batch.types.push_back(type);
                sequences.push_back(std::move(sequence)); markers.push_back(prepared->markers); rows.push_back(index);
                const auto found = bucket_temperatures.find(std::string(type_names[type]) + ':' + bucket(labels.size()));
                calibration.push_back(found == bucket_temperatures.end() ? temperatures[type] : found->second);
            }
            if (rows.empty()) return responses;
            batch.size = static_cast<int>(rows.size());
            // Native ggml code uses signed int products for flattened offsets.
            // Keep both text and marker products in range, independently of the
            // earlier conservative request-count bound.
            if (batch.size > std::numeric_limits<int>::max() / std::max(batch.length, batch.options))
                return error{error_code::invalid_request, "native tensor dimensions exceed int limits"};
            batch.ids.assign(static_cast<std::size_t>(batch.size) * batch.length, pad_id);
            batch.markers.resize(static_cast<std::size_t>(batch.size) * batch.options);
            for (int row = 0; row < batch.size; ++row) {
                if (markers[row].size() != static_cast<std::size_t>(batch.counts[row]) ||
                    std::any_of(markers[row].begin(), markers[row].end(), [&](auto marker) {
                        return marker < 0 || marker >= batch.lengths[row];
                    }))
                    return error{error_code::invalid_request, "native option markers exceed their sequence"};
                std::copy(sequences[row].begin(), sequences[row].end(), batch.ids.begin() + row * batch.length);
                for (int col = 0; col < batch.options; ++col)
                    batch.markers[row * batch.options + col] = row * batch.length +
                        (col < batch.counts[row] ? markers[row][col] : 0);
            }
            const auto raw = model.forward(batch);
            if (raw.logits.size() != static_cast<std::size_t>(batch.size) * batch.options)
                return error{error_code::invalid_backend_output, "native logits have unexpected shape"};
            for (int row = 0; row < batch.size; ++row) {
                const auto first = raw.logits.begin() + row * batch.options;
                const auto last = first + batch.counts[row];
                if (!std::all_of(first, last, [](float x) { return std::isfinite(x); }))
                    return error{error_code::invalid_backend_output, "native logits contain a non-finite active value"};
                const auto maximum = *std::max_element(first, last);
                auto& scores = responses[rows[row]].scores;
                for (auto it = first; it != last; ++it) scores.push_back(std::exp((*it - maximum) / calibration[row]));
                const auto sum = std::accumulate(scores.begin(), scores.end(), 0.0F);
                for (auto& score : scores) score /= sum;
            }
            return responses;
        } catch (const std::invalid_argument& e) {
            return error{error_code::invalid_request, e.what()};
        } catch (const std::exception& e) {
            return error{error_code::backend_failure, std::string("native Laya: ") + e.what()};
        }
    }
    laya_native_options options;
    laya::runtime model;
    laya::tokenizer tokenizer;
    std::string mask, device;
    std::int32_t mask_id{}, cls_id{}, sep_id{}, pad_id{};
    std::size_t max_len{}, head_max_len{}, cached_bytes{};
    std::array<float, 3> temperatures{};
    std::unordered_map<std::string, float> bucket_temperatures;
    std::unordered_map<std::string, std::shared_ptr<const head>> heads;
    std::deque<std::pair<std::string, std::size_t>> order;
    std::mutex mutex;
};

laya_native_backend::laya_native_backend(laya_native_options options) : impl_(std::make_unique<impl>(std::move(options))) {}
laya_native_backend::~laya_native_backend() = default;
result<inference_response> laya_native_backend::predict(const inference_request& request) {
    auto responses = predict_batch(std::span{&request, std::size_t{1}});
    if (!responses) return responses.error_value();
    return std::move(responses.value().front());
}
result<std::vector<inference_response>> laya_native_backend::predict_batch(std::span<const inference_request> requests) {
    return impl_->predict_batch(requests);
}
std::string_view laya_native_backend::name() const noexcept { return impl_->options.model_id; }
std::string laya_native_backend::device_name() const { return impl_->device; }
result<std::size_t> laya_native_backend::warmup(std::span<const inference_request> requests, std::size_t iterations) {
    if (requests.empty()) return error{error_code::invalid_request, "native warmup requires representative requests"};
    for (std::size_t i = 0; i < iterations; ++i) {
        const auto result = predict_batch(requests);
        if (!result) return result.error_value();
    }
    return iterations;
}

} // namespace jevt
