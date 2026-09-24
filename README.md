# JevT++

**Jev-like decisions. Strong C++ types.**

JevT++ is a C++20 library for embedding model-backed decisions in ordinary
C++ control flow. A decision schema maps free-form input to your own enum or
to an explicit three-state predicate. The schema is checked at compile time;
the model, policy and execution backend can change without changing business
types.

The intended everyday API has four operations:

```cpp
auto app = jevt::init({.inference_backend = backend});    // once
auto routing = jevt::bind(support);                       // once per decision
auto route = routing.choose(ticket);                      // typed choice
auto human = escalation.evaluate(ticket);                // true/false/abstain
```

`init()` owns shared runtime resources, `bind()` prepares and validates a
decision, and the hot-path calls do not re-resolve configuration. The same
API works with a deterministic test backend, an application-defined backend,
or the planned optional ONNX Runtime adapter.

## A typed routing decision

```cpp
#include <jevt/jevt.hpp>

enum class Team { billing = 10, technical = 40, sales = 90 };

inline constexpr auto support = jevt::schema<Team, "support.routing">(
    jevt::option<Team::billing>("Invoices, refunds and payments"),
    jevt::option<Team::technical>("Errors, outages and integrations"),
    jevt::option<Team::sales>("Pricing and licence purchases")
);

int main() {
    auto app = jevt::init({.inference_backend = backend});
    auto routing = jevt::bind(support, {
        .question = "Which team should handle this ticket?"
    });

    auto result = routing.choose("Card charged twice; please refund it");
    if (!result) {
        // Technical failure: backend unavailable or invalid output.
        return 1;
    }

    if (result->abstained()) {
        send_to_manual_review();
    } else {
        switch (result->value()) {
        case Team::billing:   handle_billing(); break;
        case Team::technical: handle_technical(); break;
        case Team::sales:     handle_sales(); break;
        }
    }
}
```

Abstention is not an error and uncertain predicates never silently become
`false`. See [the complete support routing example](examples/support_routing.cpp)
and [the architecture](docs/ARCHITECTURE.md).

## Build and test

Requirements are a C++20 compiler, CMake 3.24+ and Ninja (recommended).

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DJEVT_BUILD_TESTS=ON \
  -DJEVT_BUILD_EXAMPLES=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Build the performance suite separately so benchmark flags do not affect the
library used by tests:

```sh
cmake -S . -B build-bench -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DJEVT_BUILD_BENCHMARKS=ON
cmake --build build-bench
ctest --test-dir build-bench -L benchmark-smoke --output-on-failure
```

Useful options:

| Option | Default | Purpose |
|---|---:|---|
| `JEVT_BUILD_TESTS` | `ON` | Acceptance and unit tests |
| `JEVT_BUILD_EXAMPLES` | `ON` | Runnable examples |
| `JEVT_BUILD_BENCHMARKS` | `ON` | Benchmarks and load scenarios |
| `JEVT_ENABLE_HTTP` | `ON` | Optional HTTP diagnostics service |

## Consume with CMake

Install and use the exported target:

```sh
cmake --install build --prefix "$HOME/.local"
```

```cmake
find_package(jevtpp CONFIG REQUIRED)
target_link_libraries(my_service PRIVATE jevt::jevt)
target_compile_features(my_service PRIVATE cxx_std_20)
```

The package is relocatable; consumers do not need to copy headers or depend
on the JevT++ source tree.

### Conan 2

```sh
conan profile detect --force
conan create . --build=missing
# Optional dashboard:
conan create . -o jevtpp/*:dashboard=True --build=missing
```

The recipe also works as a local editable/package dependency. It exports the
same `jevt::jevt` CMake target.

### vcpkg manifest mode

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
cmake --build build
```

Select optional manifest features with `VCPKG_MANIFEST_FEATURES=dashboard`.
An ONNX Runtime adapter is planned as an optional integration; the core and
current package have no ONNX dependency.

## Diagnostics and HTTP dashboard

Diagnostics are a library API first. Callers can obtain a consistent snapshot
containing request counts, outcome counters and latency percentiles such
as p50, p95 and p99. Metrics are grouped by the stable schema identifier, not
by raw user input, which avoids unbounded labels and accidental sensitive-data
exposure.

The optional dashboard is a thin read-only adapter over those snapshots. It
exposes JSON at `/api/stats`, Prometheus text at `/metrics`, health at
`/healthz`, and an HTML view at `/`. Starting is explicit and the server binds
to loopback by default. Raw inputs and model payloads are not retained.

See [observability and dashboard operations](docs/OBSERVABILITY.md) for the
metric model, endpoint contract and production guidance.

## Design principles

- The type system owns the decision vocabulary; a backend only supplies
  scores or outcomes.
- A stable schema identifier makes diagnostics, policies and model rollout
  auditable.
- Configuration precedence is local binding, then context, then library
  defaults. Existing bindings never change implicitly.
- A technical error, an abstention and a valid negative result are different
  states.
- Model execution is replaceable. The core does not require ONNX, HTTP or a
  specific model family.

More detail is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). JevT++ is
available under the [MIT License](LICENSE).

