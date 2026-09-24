# Metadata-driven System One decisions

Include `<jevt/system_one.hpp>`. C++20 declarations carry the decision ID and
docstring, named field descriptions, enum option descriptions, and ordered score
rubrics. All fields are evaluated independently against one owned state through
one `backend::predict_batch` call. Laya overrides this with a tensor batch;
other backends may use the default sequential compatibility implementation.

```cpp
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

auto runner = jevt::bind_system_one(ticket, inference_backend);
auto response = runner.evaluate(jevt::json_state(R"({
    "ticket": "Nobody can log in since this morning.",
    "customer": {"plan": "enterprise"},
    "routing_policy": "Outages need immediate engineering attention."
})"));
if (!response) { /* inspect response.error_value() */ }
else if (response->abstained()) { /* route to review */ }
else {
    const auto category = response->get<"category">().value();
    const float impact = response->get<"urgency_score">().score();
    const float angry = response->get<"sentiment_probability">().value();
}
```

Choice has a typed enum value, the complete normalized `probabilities()`,
`values()` in matching order, `legend()`, and entropy-based `confidence()`.
The confidence gate abstains below its threshold. A threshold of zero disables
abstention. A tied distribution selects the first criterion unless the gate
abstains. Fields are identified at compile time with `get<"name">()`.

Score accepts 2–10 distinct, strictly ascending enum levels. `score()` is the
probability-weighted **position** from 0 to N−1, including fractional values;
the enum's underlying storage values do not change the position scale.
`value()` returns the most probable typed enum level, not a rounded mean.
Score also preserves all probabilities, the rubric legend, and confidence.

Noul exposes `probability_true()` and an optional boolean in `selected()`.
With `{0.2F, 0.8F}`, probabilities at or below 0.2 are false, at or above 0.8
are true, and the intervening range abstains. Equal thresholds define a single
binary threshold, with equality interpreted as true. Noul has no separate
confidence statistic. `probability<...>` explicitly maps a Noul proposition
to a floating-point P(true); it does not perform arbitrary float extraction.

`value()` on an abstained Choice, Score or boolean Noul throws; inspect
`abstained()` or `selected()` first. Full distributions remain available after
abstention. Output vectors must have the right cardinality, finite nonnegative
weights and positive mass. An invalid field fails the entire response.

The confidence metric is `1 - H(p)/log(N)`, bounded to [0,1]. It is 0 for a
uniform distribution and 1 for a point mass (also 1 for a one-option Choice).
This documents jevtpp's own metric, not a claim of numerical parity with Jev's
proprietary confidence calculation. Thresholds require domain evaluation.

## Application structs and serializers

`answers()` exposes a typed tuple. `map()` applies a callable to each answer
in declaration order, allowing ordinary C++20 structs without reflection:

```cpp
struct TicketDecision {
    Category category;
    bool is_urgent;
    Urgency urgency_score;
    float sentiment_probability;
};

// Check response and abstentions before this projection.
auto decision = response->map([](const auto& category, const auto& urgent,
                                const auto& score, const auto& probability) {
    return TicketDecision{category.value(), urgent.value(), score.value(),
                          probability.value()};
});
```

`text_state(string)` and `json_state(string)` own their serialization and carry
a content kind. JSON state rejects empty/whitespace-only input; syntax validation
belongs to the application's JSON serializer. JSON objects, arrays and scalars
can be passed without adding a JSON dependency to jevtpp. The serialized text is
passed unchanged to the backend. Metadata and state remain separate.

Use `runner.evaluate(record, serializer)` where the callable returns a
`jevt::state_value`, or provide `to_jevt_state(const Record&)` in the record's
namespace for ADL customization. The serializer is invoked exactly once per
evaluation, and all questions share its result. `runner.evaluate(string_view)`
is shorthand for text state.

`runner.request(state)` exposes the owning batch for inspection. Its `views()`
object provides the inference requests; keep both the owner and views alive
through backend use. Rebuild views after copying or moving the owner. A field's
question joins the model docstring and that field's description. Every field has
the same input state; no other field's answer becomes implicit context.

## Compile-time contract

Names and descriptions must be nonempty, field names must be unique, enum
options must have one enum type and unique values, Score levels must be ordered
and within the 2–10 limit, and probability thresholds must be within [0,1].
Descriptions are string views intended for static string literals.

This interface follows the [Jev primitives](https://docs.typesafe.ai/primitives/score)
and shared-state independent-question model described in the supplied
[project reference](https://gist.github.com/pjburnhill/adf8d28efcad9df037bfdece178ef965).
It does not imply equivalent model quality or calibration.
