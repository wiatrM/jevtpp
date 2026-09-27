#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <list>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace jevt {

// Supply only revisions that affect the cached artifact. For tokenized input,
// weights are intentionally omitted; for inference results include weights.
struct cache_revisions {
    std::string model;
    std::string tokenizer;
    std::string prompt;
    std::string schema;
    std::string knowledge;
    std::string weights;
    std::string environment;
    friend bool operator==(const cache_revisions&, const cache_revisions&) = default;
};

struct cache_limits {
    std::size_t entries = 1024;
    std::size_t bytes = 8 * 1024 * 1024;
};

struct cache_counters {
    std::uint64_t hits = 0, misses = 0, writes = 0, evictions = 0;
    std::uint64_t stale_reads = 0, stale_writes = 0, revision_invalidations = 0;
    std::size_t entries = 0, retained_bytes = 0;
};

// Thread-safe bounded LRU for immutable prepared inputs or deterministic
// inference results. A ticket fences late asynchronous writes: once advance()
// changes a revision, completions carrying an older ticket cannot repopulate
// the cache with stale data. Keys must be exact canonical content, not hashes.
template <class Value>
class revisioned_cache {
public:
    using clock = std::chrono::steady_clock;
    using ticket = std::uint64_t;

    explicit revisioned_cache(cache_limits limits = {}) : limits_(limits) {
        if (limits_.entries == 0 || limits_.bytes == 0)
            throw std::invalid_argument("cache limits must be positive");
    }

    [[nodiscard]] ticket advance(cache_revisions revisions) {
        std::lock_guard lock(mutex_);
        if (!initialized_ || revisions != revisions_) {
            if (initialized_) ++counters_.revision_invalidations;
            clear_locked();
            revisions_ = std::move(revisions);
            initialized_ = true;
            advance_generation_locked();
        }
        return generation_;
    }

    [[nodiscard]] std::shared_ptr<const Value> get(
        std::string_view key, ticket issued, clock::time_point now = clock::now()) {
        std::lock_guard lock(mutex_);
        if (!initialized_ || issued != generation_) {
            ++counters_.stale_reads;
            return {};
        }
        const auto found = entries_.find(std::string(key));
        if (found == entries_.end()) {
            ++counters_.misses;
            return {};
        }
        if (found->second.expires <= now) {
            erase_locked(found);
            ++counters_.misses;
            return {};
        }
        lru_.splice(lru_.begin(), lru_, found->second.position);
        ++counters_.hits;
        return found->second.value;
    }

    [[nodiscard]] bool put(std::string key, ticket issued,
                           std::shared_ptr<const Value> value, std::size_t value_bytes,
                           clock::duration ttl = clock::duration::max(),
                           clock::time_point now = clock::now()) {
        if (key.empty() || !value || ttl <= clock::duration::zero()) return false;
        if (value_bytes > limits_.bytes || key.size() > limits_.bytes - value_bytes) return false;
        const auto retained = key.size() + value_bytes;
        std::lock_guard lock(mutex_);
        if (!initialized_ || issued != generation_) {
            ++counters_.stale_writes;
            return false;
        }
        if (auto found = entries_.find(key); found != entries_.end()) erase_locked(found);
        while (!lru_.empty() &&
               (entries_.size() >= limits_.entries || retained > limits_.bytes - counters_.retained_bytes)) {
            auto found = entries_.find(lru_.back());
            if (found == entries_.end()) { lru_.pop_back(); continue; }
            erase_locked(found);
            ++counters_.evictions;
        }
        lru_.push_front(std::move(key));
        const auto position = lru_.begin();
        const auto expires = ttl == clock::duration::max() || ttl > clock::time_point::max() - now
            ? clock::time_point::max() : now + ttl;
        try {
            entries_.emplace(*position, entry{std::move(value), retained, expires, position});
        } catch (...) {
            lru_.pop_front();
            throw;
        }
        counters_.retained_bytes += retained;
        ++counters_.writes;
        return true;
    }

    void invalidate() {
        std::lock_guard lock(mutex_);
        clear_locked();
        initialized_ = false;
        advance_generation_locked();
        ++counters_.revision_invalidations;
    }

    [[nodiscard]] cache_counters stats() const {
        std::lock_guard lock(mutex_);
        auto result = counters_;
        result.entries = entries_.size();
        return result;
    }

private:
    struct entry {
        std::shared_ptr<const Value> value;
        std::size_t bytes;
        clock::time_point expires;
        typename std::list<std::string>::iterator position;
    };
    using map_type = std::unordered_map<std::string, entry>;

    void clear_locked() {
        entries_.clear();
        lru_.clear();
        counters_.retained_bytes = 0;
    }
    void advance_generation_locked() noexcept {
        generation_ = generation_ == std::numeric_limits<ticket>::max() ? 1 : generation_ + 1;
    }
    void erase_locked(typename map_type::iterator found) {
        counters_.retained_bytes -= found->second.bytes;
        lru_.erase(found->second.position);
        entries_.erase(found);
    }

    cache_limits limits_;
    mutable std::mutex mutex_;
    cache_revisions revisions_;
    bool initialized_ = false;
    ticket generation_ = 0;
    map_type entries_;
    std::list<std::string> lru_;
    cache_counters counters_;
};

} // namespace jevt
