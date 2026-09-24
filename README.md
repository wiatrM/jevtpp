# JevT++

**Jev-like decisions. Strong C++ types.**

JevT++ is a C++20 library for embedding model-backed decisions in ordinary
C++ control flow. Inject text, JSON or a serialized application object as shared
context, then ask several independent questions with typed answers. Decision
descriptions, field instructions and enum rubrics are part of the schema and
are sent to the model. C++ checks the output vocabulary at compile time.

The optional native Laya backend runs locally with ONNX Runtime. It evaluates
all fields of a System One request in a single batched forward pass.

## One context, several typed answers

```cpp
#include <jevt/system_one.hpp>
#include <jevt/laya.hpp>

enum class Category { billing, technical, sales, spam };
enum class Urgency { low = 0, medium = 1, high = 2 };

constexpr auto categories = jevt::schema<Category, "support.categories">(
    jevt::option<Category::billing>("Invoices, payments, or refund requests."),
    jevt::option<Category::technical>("Software bugs, crashes, or login failures."),
    jevt::option<Category::sales>("Upgrades, enterprise pricing, or new purchases."),
    jevt::option<Category::spam>("Unsolicited marketing or automated noise."));
constexpr auto urgency = jevt::schema<Urgency, "support.urgency">(
    jevt::option<Urgency::low>("Customer is patient and calm."),
    jevt::option<Urgency::medium>("Issue blocks work but has a workaround."),
    jevt::option<Urgency::high>("Complete outage or severe frustration."));

constexpr auto ticket = jevt::decision_model<"support.ticket">(
    "Evaluate the incoming customer support ticket.",
    jevt::choice<"category">("Which team should own this request?", categories, 0.4F),
    jevt::noul<"is_urgent">("Does this require attention within one hour?", {0.2F, 0.8F}),
    jevt::score<"urgency_score">("Rate the frustration level against the rubric.", urgency),
    jevt::probability<"sentiment_probability">("The customer is angry."));

int main() {
    auto model = jevt::models::laya_multilingual("models/laya-multilingual");
    auto brain = jevt::bind_system_one(ticket, model);
    auto answer = brain.evaluate(jevt::json_state(R"({
        "ticket": {"body": "Nobody can log in. Our entire team is blocked."},
        "customer": {"plan": "enterprise", "active_users": 120},
        "context": {"workaround_available": false, "sla_minutes": 60}
    })"));

    if (!answer) return 1;             // Technical failure.
    if (answer->abstained()) return 2; // Application can request human review.
    Category owner = answer->get<"category">().value();
    bool urgent = answer->get<"is_urgent">().value();
    Urgency level = answer->get<"urgency_score">().value();
    float expected_level = answer->get<"urgency_score">().score();
    float angry = answer->get<"sentiment_probability">().value();
}
```

The input is owned once and shared across the questions. For an application
object, use `brain.evaluate(record, serializer)` with a serializer returning
`jevt::json_state(...)` or `jevt::text_state(...)`; it runs once per evaluation.
An ADL `to_jevt_state(const Record&)` customization is also supported. No JSON
library is required by JevT++, and JSON syntax validation belongs to your
serializer.

Choice returns an enum and its full distribution. Score returns both the modal
enum and the expected rubric position, including fractions. Noul exposes
P(true) and an explicit true/false/abstain policy. A probability field exposes
P(true) for its proposition; it is not arbitrary floating-point extraction.
Choice and Score use entropy-based confidence. Questions are independent: an
answer to one field is not fed into another field.

See [System One API and serializers](docs/SYSTEM_ONE.md),
[the runnable Laya demo](examples/laya_routing_demo.cpp), and
[native inference setup](docs/LAYA.md).

## Single decisions

The single-decision runtime API has four operations:

```cpp
auto app = jevt::init({.inference_backend = backend});    // once
auto routing = jevt::bind(support);                       // once per decision
auto route = routing.choose(ticket);                      // typed choice
auto human = escalation.evaluate(ticket);                // true/false/abstain
```

`init()` owns shared runtime resources, `bind()` prepares and validates a
decision, and the hot-path calls do not re-resolve configuration. The same
API works with a deterministic test backend, an application-defined backend,
or the optional Laya ONNX Runtime adapter. Backends that do not override
`predict_batch()` use the sequential compatibility implementation for System One.

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
| `JEVT_ENABLE_LAYA` | `OFF` | Native Laya inference with ONNX Runtime |
| `JEVT_FETCH_TOKENIZERS_CPP` | `ON` | Fetch pinned tokenizer dependency when building Laya |

To run the real multilingual model, fetch the pinned bundle and build the
optional adapter (C++20, Rust/Cargo and an ONNX Runtime SDK are required):

```sh
python3 scripts/fetch_laya.py models/laya-multilingual
cmake -S . -B build-laya -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DJEVT_ENABLE_LAYA=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime-sdk
cmake --build build-laya
./build-laya/examples/jevt_laya_demo models/laya-multilingual
```

See [Laya setup and operating limits](docs/LAYA.md) for shared-library loading,
model provenance, token budgets and integration testing.

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
The core package has no ONNX dependency. Build the optional `jevt::laya` target
from source using [the Laya instructions](docs/LAYA.md); the Conan and vcpkg
recipes currently cover the core and dashboard.

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

