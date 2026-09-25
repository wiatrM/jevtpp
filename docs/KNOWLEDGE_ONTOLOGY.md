# Evidence linked knowledge view

The Mario dashboard now has two related live views. The map shows discovered
tiles and unknown space. The ontology shows **entities, action variants,
observed effects and relations**. The executed decision DAG remains a separate
view: it answers which computation and action ran, while the ontology answers
which claims about the world have evidence.

The view protocol is `jevt.knowledge_view.v1`. Each node has a stable `id`,
`kind`, `label`, `status`, `source`, first/last observation frames and bounded
attributes. Each relation has `from`, `to`, `relation`, the same evidence
fields and an event count when repeated independent transitions are observed.
`model_focus` lists current nodes highlighted by an accepted, current model
judgment. A highlight never creates a node or confirms a relation. Both views
share the game run, frame and memory generation.

The Mario adapter currently creates relations from these observations:

- a confirmed flight and its recorded A-button hold duration, travel and
  landing result;
- enemy type and empirical horizontal motion samples;
- a pickup counter increment plus small-to-`tall` transition;
- `tall`-to-small transition with no life lost;
- contact-free completed flights with measured A-button hold and apex height,
  plus confirmed landings on higher observed support;
- the NES `$079F` star timer: active state and zero-to-positive activation,
  reported separately from Mario's size;
- a RAM alive-to-defeated enemy transition near Mario while the star timer is
  active, with unchanged lives and no new stomp: stored as an event association
  with the named enemy type, not proof that every collision is safe;
- RAM-tracked enemy identities and statuses.
- post-step life-counter losses joined to the last pre-step position, including
  a stage-scoped danger zone and an event count across retained runs;
- an executed skill's preceding input sequence, when observed: at least eight
  consecutive airborne `noop` intercept frames shortly before a life loss
  with a recently visible nearby target form an empirical failure-context
  relation. An enemy becoming unobserved on the collision frame is not
  interpreted as elimination or proof that the enemy caused the death.

An observed transition is not automatically a general causal law. A flight with
suspected contact stays uncertain. Adjacent motion samples are dependent. A
disappeared enemy is not considered eliminated. A growth transition without a
pickup counter increment does not establish a mushroom effect. Star-timer
activation is not evidence that Mario touched a particular visible sprite.
The timer's invincibility meaning comes from the
[SMB disassembly](https://6502disassembly.com/nes-smb/SuperMarioBros.html),
not from a learned collision experiment. Different observed jump heights are
correlations until samples control for starting speed, support and collision.
The view has bounded nodes/relations.

## Cross-run experiment

The header's KNOWLEDGE selector defaults to **RESET AT GAME OVER**. Choose
**RETAIN ACROSS RUNS** to keep observed C++ map columns, enemy-type motion
statistics, completed-flight aggregates and the bounded ontology in this
process. Explicit RESTART also preserves them in this mode. Entity instances
remain scoped to their physical run (`enemy:<run>:<id>`), so a tracker ID reused
later is not falsely treated as the same enemy. Switching back to RESET clears
retained memory immediately. Process restart still begins fresh; no
cross-process import or neural weight training occurs. The C++ action-state
learner resets its per-run dynamics and tasks regardless of this setting.
Retention alone therefore does not prove better game control.

Post-step outcomes must be joined explicitly. The observation/decision graph
runs before `environment.step()`; a death is only known afterward. Earlier
versions never inserted that event into the ontology, so a retained graph
could grow with motion samples but still omit the repeated fatal maneuver.
The new `knowledge_outcome` trace record includes the stage, position and
current danger-zone counts. The typed affordance sent to C++ includes
`death_zones`; only a zone linked to the repeated intercept-coasting
sequence can veto that same airborne target skill. Other deaths do not
disable target pursuit merely because they share a location.

Each plan request contains `knowledge_graph` (`jevt.knowledge_context.v1`):
up to twelve named, source-bearing relation statements prioritized for the
current run, stage, mode and active star state. The dashboard exposes the
exact last request under **LAST KNOWLEDGE SENT TO JEV**. Expired star activity,
old-run enemy instances and other-stage location edges are excluded. Open-JEV
may use this context to rank plans; the C++ arbiter still verifies freshness,
feasibility and risk. A model answer cannot create or verify a graph fact.
The model rank is asynchronous; many answers expire or fail a changed-context
gate before affecting the actuator. `model_focus` highlights only accepted
current judgments, not merely model requests.

After at least two independently reported star-contact elimination events for
one enemy type, the adapter exposes a typed `knowledge_affordances` input to
the C++ graph. Its `star_contact` proposal can replace an enemy-avoidance jump
with `right_run` in hunter/score modes only while the timer has a conservative
margin, that named type is currently visible nearby on the same elevation,
Mario is grounded, and live RAM shows no near gap or wall. It is one proposal
to the existing single arbiter, not a post-selection actuator override. The
executed graph shows whether it was eligible and selected. This is a narrow
reuse of observed outcomes, not a general proof of invulnerability.

In the first paired same-seed retention check, both consecutive games still
ended at score 21,000 and world 1-2, progress x472, frame 2960. The earlier
model ranker influenced zero final actions in those runs. This negative result
is why retention, model input and actual controller adaptation are reported
separately. In the first full run with the evidence-gated `star_contact` skill,
the arbiter selected it for 12 frames. Score was 21,400 versus 21,000 in the
preceding same-seed runs and GAME OVER came at frame 2916 versus 2960, but all
ended at world 1-2, progress x472. The next attempt in the same process retained
the ontology, selected the skill for 19 frames, scored 21,950 and ended at
frame 2782, again at x472 in world 1-2. These observations demonstrate that
the evidence changed actual actions and score in two runs; they do not prove
general improvement, fix the repeated death or establish model-weight learning.

## Reusing this beyond Mario

`KnowledgeOntology` in the demo is a Python projection fed by the C++ graph's
trace and RAM adapter. It does not drive the actuator. The protocol and display
can be reused by another adapter that supplies entity identities, relation
types, source observations and a domain verifier. The current browser renderer
groups by Mario oriented `kind` values; a reusable UI package would need a
registered layout and vocabulary for each domain. The generic C++ arbiter and
operator registry are documented in [SKILLS.md](SKILLS.md).

A candidate feedback loop for a new domain is:

```text
observe event → attach source/frame → update provisional entity/relation
→ predict result of a chosen skill → execute through one arbiter
→ verify actual result → revise evidence / uncertainty
→ show the exact relation and result used in the next decision
```

The model may rank present, feasible plans and emphasize the current evidence.
Claims from model prose would require a separate verifier before entering the
knowledge store. The opt-in mode retains evidence, never inferred model claims.

## Review and diagnostics

Click a node, or focus the canvas and use the arrow keys, to inspect its
attributes and linked relations. Filter for learned relations or currently
highlighted facts. The OUTCOME FEEDBACK filter isolates the observed
`skill → danger zone → life lost` chain; the header distinguishes the total
fact count from the bounded number drawn on the canvas. The JSON disclosure
contains the exact versioned view.
The status colors distinguish current RAM, empirical statistics, confirmed
transitions and uncertainty. The model focus color represents attention only.

The paired controller evaluation and the model trace diagnostics are described
in [the Mario README](../examples/mario_dashboard/README.md). Neither a
visually coherent graph nor a lower prediction error proves that the game
policy improved.
The actual post-step feedback gap, x472 precursor and model-input intervention
are documented in [the x472 trace review](../examples/mario_dashboard/TRACE_X472_FEEDBACK.md).
