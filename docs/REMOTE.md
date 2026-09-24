# Remote inference

`jevt::typesafe_backend` is an optional HTTP backend for the typed System One
API. It uses the same Choice, Noul, Score and probability fields as local
backends. Selecting it is explicit: local inference never falls back to a
hosted service and the library never reads API keys from environment variables.

This is an independent client implementation, not an official TypeSafe SDK.
The integration tests use injected responses and a loopback HTTP server; they
do not establish live-provider compatibility or make paid requests. Validate
the chosen provider/model with your account before deploying.

## Build and link

Install a libcurl 7.66+ development package with asynchronous DNS support, then build:

```sh
cmake -S . -B build-remote -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DJEVT_ENABLE_REMOTE=ON
cmake --build build-remote
ctest --test-dir build-remote --output-on-failure
```

Link `jevt::remote` and include `<jevt/remote.hpp>`. The module is a shared
library; deploy it and its libcurl runtime dependencies with your application.
JSON parsing is private to this module. CMake reuses an existing
`nlohmann_json::nlohmann_json` target, finds version 3.11+, or fetches 3.12.0
pinned to `55f93686c01528224f448c19128836e7df245f72`. Dependency install rules are
excluded from JevT++'s install.

Set `JEVT_REMOTE_CURL=OFF` to omit libcurl and inject your own `http_transport`.
Constructing the backend without an injected transport in that configuration
throws; it does not silently disable remote inference. Neither remote nor
Asio is enabled by default.

The [remote routing example](../examples/remote_routing.cpp) builds as
`examples/jevt_remote_routing` when examples and Curl are enabled. Running it
without arguments only prints help and sends nothing. To send its synthetic
four-field support ticket, set `TYPESAFE_API_KEY` in your application's
environment and run with `--send`; that call may be billable. The example reads
the variable explicitly and disables retries. Never commit credentials or put
them in command-line arguments. Optional `TYPESAFE_MODEL` selects a model version.

## Use the existing typed model

Given the `ticket` model from the [System One guide](SYSTEM_ONE.md):

```cpp
#include <jevt/remote.hpp>
#include <jevt/system_one.hpp>

// Obtain this from your application's secret manager, not a committed literal.
jevt::remote_options options;
options.api_key = api_key;
options.model = "jev-latest";
options.max_retries = 0; // Choose an explicit retry/billing policy.

auto remote = std::make_shared<jevt::typesafe_backend>(options);
auto brain = jevt::bind_system_one(ticket, remote);
auto answer = brain.evaluate(jevt::json_state(R"({
  "ticket": {"body": "Our entire team cannot log in."},
  "customer": {"plan": "enterprise"},
  "sla_minutes": 60
})"));
if (!answer) {
    // Inspect answer.error_value().code; this is not an abstention.
}
```

`remote_options` defaults to `https://api.typesafe.ai` and model `jev-latest`.
Model aliases can change provider-side; pin a supported version for reproducible
deployments. `list_models()` queries the configured TypeSafe endpoint and returns
model names, descriptions and release dates. It is a network request, not a
locally cached catalog.

The experimental `openrouter_options(api_key)` configures the OpenRouter System One endpoint and
the adapter's default model identifier. It is not an OpenAI chat-completions
adapter. OpenRouter model discovery has a different contract, so
`list_models()` returns `unsupported_feature` for that provider. Override the
model deliberately and check current availability with the provider. This path
uses the same `/v1/systemone` endpoint family and has not been verified live;
it carries no hosted parity, quality or performance claim.

## Payload and result contract

The backend sends `POST /v1/systemone`. Requests with identical input bytes and
content kind share one state with multiple questions; distinct states are sent
as separate groups, preserving caller order. It does not combine unrelated
application contexts into one prompt. Question instructions and structured
criterion metadata retain their JSON types. The remote module rejects malformed
or unsupported structured content before sending it.

The supported structured values are text strings, objects and arrays. Choice
wire keys are stable synthetic identifiers (`option_0`, `option_1`, ...), not
C++ enum names. Supply meaningful criterion descriptions; nullable label-only
Choice criteria are not supported, because C++ enum reflection cannot supply
the missing semantic label. This is a deliberate subset of the hosted schema.

Choice and Score responses must have the expected number of options, valid
finite probabilities and consistent provider selections/scores. Noul returns
P(true); typed thresholds determine true, false or abstention locally.
Provider confidence and model/usage information are retained as execution
metadata, not substituted for the typed API's local confidence calculation.
See [result metadata](SYSTEM_ONE.md) for field and aggregate access.

Diagnostic token totals report observed usage, not billing totals. A shared
provider usage report is counted once within a typed evaluation; independent
evaluations coalesced into one provider call can each observe that aggregate.
Do not infer per-field billing or a provider invoice from these counters.

Cancellation and deadlines apply to the enclosing backend batch. The earliest
deadline wins, and cancelling one constituent cancels the batch. Keep requests
with incompatible cancellation lifetimes out of the same submission. A batch
failure is returned as an error, not partial success.

## Time, retry and resource limits

| Option | Default | Meaning |
|---|---:|---|
| `call_timeout` | 30 s | Total backend-call budget, including groups and retries |
| `attempt_timeout` | 10 s | Budget for an individual HTTP attempt |
| `max_retries` | 2 | Additional attempts after the first |
| `initial_backoff` / `max_backoff` | 500 ms / 5 s | Exponential jittered backoff bounds |
| `max_request_bytes` / `max_response_bytes` | 4 MiB each | Serialized request and received-body limits |

Caller deadlines can shorten these budgets. Retries cover connection failures,
timeouts, HTTP 408, 429 and 5xx responses. Authentication and ordinary request
errors are not retried. `Retry-After` and `Retry-After-Ms` are observed within
the total deadline. A retry can repeat an already accepted or billable request;
no exactly-once guarantee is made. Disable retries when your application owns
that policy.

Errors distinguish `authentication`, `rate_limited`, `connection_failure`,
`timeout`, `cancelled`, `invalid_request`, `invalid_backend_output` and other
remote failures. HTTP status is available when applicable. Raw remote error
bodies and credentials are not copied into backend-generated error messages.

## Transport and security

The default libcurl transport reuses a bounded pool of exclusive connection
handles (`make_curl_transport(retained_connections = 8)`). The retained count
is not an admission limit; put a [bounded worker pool](BATCHING.md) in front of
the backend to limit concurrency. HTTPS certificate verification is enabled
and redirects are not followed. Plain HTTP is rejected except when
`allow_insecure_loopback` explicitly enables literal loopback test endpoints.

The supplied transport observes stop tokens and absolute deadlines during I/O.
An injected `http_transport` must be thread-safe and honor those controls too.
The backend discards late replies but cannot forcibly interrupt arbitrary
blocking code in a custom transport. Custom transports also own TLS,
connection reuse, body limits and credential-handling behavior.

Context, instructions and criteria leave your process when this backend runs.
Review provider retention terms and your application's data policy before
sending customer data. Do not expose API keys, payloads or sensitive metadata
through logs or public diagnostics. No credentials are required to run the
[scripted application tests](TESTING.md) or the remote mock/loopback CI suite.
