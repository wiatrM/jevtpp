# Empirical experience memory

`mario::KnowledgeMemory` accumulates simulator observations across episodes. It does not fine-tune Open-JEV, invent unseen level geometry, or alter the reactive controller's thresholds.

The bounded store contains at most 512 observed world columns, up to 64 enemy types, signed enemy speed running statistics, completed jump duration/range/height/hold statistics, and deduplicated death/flag counters. The tile map includes only the declared visible viewport; without explicit viewport bounds it uses a conservative 80-pixel corridor. Unknown columns remain unknown. The confidence field describes repeated observation support and is not a calibrated probability.

Enemy speed uses differences in the enemy's world position, not relative motion caused by Mario. Slot reuse, discontinuous observations and implausible displacement do not contribute speed samples. Jump measurements require a continuous ground-to-ground trajectory. Flights with a stomp bounce are counted separately and excluded from the basic jump estimate. These are empirical samples of the actual control policy, not universal physics guarantees.

Call `observe(state, epoch, frame)` for current telemetry and the terminal next-state. `reset_episode(epoch)` clears transient tracking while retaining aggregate evidence. `context()` supplies a small model input summary. `snapshot()` adds the observed map and a graph of the stored evidence sources. `export_json()` and `import_json(value, &error)` provide versioned, environment-scoped persistence; loading validates the whole document before replacing the current store.

The owner should save exports atomically and record the environment/ROM/preprocessing contract used. A changed environment requires a distinct scope string. Persisted RAM geometry is prior experience; fresh telemetry remains authoritative for action execution.

Standalone C++ regressions are in `test_experience_memory.cpp`. They cover visible-only map capture, duplicate frame handling, Koopa type zero and absolute speed, full jump measurement, episode retention, terminal event deduplication, persistence roundtrip/rejection, bounded storage, and rejection of incomplete trajectory samples.
