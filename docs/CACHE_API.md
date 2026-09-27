# Revisioned cache API

Include `<jevt/cache.hpp>` for a thread-safe, bounded LRU cache of immutable
prepared inputs or deterministic inference results. No inference backend is required.

```cpp
jevt::revisioned_cache<std::string> cache({1024, 8 * 1024 * 1024});
auto ticket = cache.advance({.model="laya-v1", .prompt="attack-v2"});
auto value = std::make_shared<const std::string>("prepared input");
bool stored = cache.put("exact canonical input", ticket, value, value->size());
auto hit = cache.get("exact canonical input", ticket);
```

Advance revisions when relevant model, tokenizer, prompt, schema, knowledge,
weights or environment changes. Tickets reject late asynchronous writes from an
older revision. `invalidate()` also fences pending writes. Keep exact canonical
input as the key; a digest alone is not a collision-proof identity.

Use a finite TTL for time-sensitive observations. Tokenization caches normally
omit weight revisions; inference-result caches must include them. Do not cache
stochastic decisions unless reusing the sampled result is explicitly intended.

Byte limits account for caller-reported value bytes plus key bytes, not allocator
overhead or values kept alive by external shared pointers. `stats()` exposes hits,
misses, stale reads/writes, evictions and retained bytes. Shared immutable results
remain valid after eviction. Cache reuse does not establish decision freshness:
applications must validate current targets and execution preconditions separately.
