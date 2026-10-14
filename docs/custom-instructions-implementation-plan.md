# Custom Instructions & Dynamic Knowledge — Implementation Plan

**Status:** design → ready to build
**Author:** engineering design pass, 2026-09-04
**Companion to:** [`custom-instructions.md`](./custom-instructions.md) (the product/design spec)
**Purpose:** ground the spec in the actual codebase, resolve how a plain-English
instruction is classified and compiled, and give a build order that ships value early
without compromising the "reliable, every time" guarantee.

---

## 0. TL;DR

- ~80% of the machinery already exists and is load-bearing. This feature is mostly
  **two new tables + one new author-time LLM compiler + two small runtime edits**,
  plus (for routing) closing one real reliability gap.
- **Classification is LLM-backed, runs once at author time**, and classifies *by
  consequence*, not by phrasing — in a single structured-output call that also compiles
  the artifact.
- **The one genuine gap is the intent vocabulary.** Today `turn.intent` is a free-form
  LLM label. Routing rules that key on it are only as reliable as a fuzzy string match.
  Fix this **before** routing ships (not before everything).
- **Sequencing:** Advisory instructions + Dynamic Knowledge first (no dependency on
  intent reliability). Routing second, gated on the intent-vocabulary artifact.

---

## 1. What already exists (reuse, don't rebuild)

| Capability the spec needs | Where it already lives |
|---|---|
| Rule engine: compile / validate / evaluate | `core/app/engine/rules/schema.py` (`compile_rules`, `validate`), `conditions.py`, `engine.py` (`evaluate_point`) |
| `RuleSet` table — `yaml_source`, `compiled`, `version`, `is_active`, `activated_by/at` | `core/app/db/models/engine_config.py:32` |
| Rule evaluation at the `turn` point (two passes) | `core/app/engine/planner/planner.py:474` (pass 1: customer/state/time), `:609` (pass 2: intent/sentiment) |
| Forced-escalation ride-along path | `planner.py:631` → `proposal.forced_escalation` → `turn_runner.py:3739` |
| Admin CRUD + compile-check + dry-run test | `core/app/api/rules_admin.py` (`PUT /rules/{name}`, `/activate`, `/test`, `/validate`) |
| Audit trail | `AuditEvent` (used at `rules_admin.py:88`) |
| PROJECT FACTS injected into the **responder** (customer-facing) | `core/app/engine/planner/responder.py:364` `_project_facts_block`, wired at `turn_runner.py:3978` |
| Per-turn assembly of `project_facts` | `core/app/engine/planner/turn_context.py:239-267,358` |
| Verdict whitelist (`block/require/escalate/set`) | `schema.py:25` `VERDICT_KINDS` |
| Condition grammar (`all/any/not` + leaf ops) | `conditions.py:13-16` (`OPERATORS`, `COMBINATORS`) |
| Nascent intent vocabulary | `core/app/engine/planner/confidence.py:39` `INTENT_ALIASES` + `normalize_intent` |
| Cacheable-prefix discipline for tenant-static blocks | `prompts.py:409-455` (PERSONA, PROJECT FACTS) |

**Implication:** the routing half is *almost entirely* the existing rule engine. The
advisory half reuses the prompt path. The only genuinely new runtime code is targeted
escalation.

---

## 2. Classification — the LLM step, precisely

### 2.1 Classify by consequence, not phrasing

Every instruction is classified **once, at author time**, into `advisory` or `routing`.
The deciding question (spec §2.1) is *not* how it is worded — it is **what happens if the
model ignores it on 1 turn in 50**:

| If ignoring it produces… | Kind | Backend |
|---|---|---|
| a wrong or missing side effect (transfer that didn't happen, booking made when it shouldn't be, required field skipped) | **routing** | Rule engine (deterministic) |
| a slightly worse *sentence* (tone, emphasis, what to volunteer, framing) | **advisory** | Prompt injection (cacheable) |

### 2.2 One structured-output call: classify **and** compile together

Do **not** ask the LLM a bare "is this advisory or routing?" — it drifts and is
unauditable. Instead run a single `complete_json` call (the exact pattern already used by
`understanding.py:126` and `responder.py:165`: strict schema, every field required) that
classifies *and* produces the compiled artifact in one shot. If the model claims
`routing`, it must emit compilable YAML; if it can't, that is itself the signal it was
advisory or ambiguous.

The call is handed four grounding artifacts so it can never produce something the runtime
cannot honour:

1. **Context field catalog** — the exact namespaces a `when` may reference, from
   `context.py` (`turn.intent`, `turn.sentiment`, `customer.tags`, `customer.on_dnc_list`,
   `session.direction`, `session.purpose`, `now.weekday`, `counters.*`, `proposal.*`, …).
2. **Intent vocabulary** — the closed set of `turn.intent` labels (see §4).
3. **Transfer-target registry** — the tenant's known escalation targets (for
   `escalate.target`).
4. **Verdict whitelist** — `block | require | escalate | set` only (`VERDICT_KINDS`).

**Proposed output schema:**

```json
{
  "kind": "advisory | routing | ambiguous",
  "reasoning": "consequence-based justification (audit trail)",
  "consequence_if_missed": "string used to justify kind",
  "compiled_advisory": "clean imperative snippet, or null",
  "compiled_rule_yaml": "rule in the existing YAML schema, or null",
  "scope": {
    "direction": "inbound|outbound|null",
    "state": "string|null",
    "intent": "vocab-label|null",
    "valid_from": "YYYY-MM-DD|null",
    "valid_until": "YYYY-MM-DD|null"
  },
  "grounding_gaps": ["e.g. 'no transfer target named — which team?'"],
  "test_scenarios": [
    {"caller_says": "...", "expected_behaviour": "..."}
  ]
}
```

- `ambiguous` + `grounding_gaps` is a **first-class result**, not an error — it is how
  the compiler returns "which team should loan enquiries go to?" to the admin (spec Open
  Q5). The instruction stays inactive until resolved.
- The classifier does **not** need to be perfect, because the **review gate** (§3.5)
  catches misclassification: the admin sees the rendered artifact *and* auto-generated
  test transcripts before anything activates. This is the same three-legged safety model
  industry agent platforms use: *deterministic execution for consequences + LLM only for
  authoring + a mandatory human approval gate.*

### 2.3 One LLM call — generation vs. execution

The `test_scenarios` field is part of **this same schema**, so scenario *generation* costs
**zero extra LLM calls** — the compiler emits the 2–3 scenarios in the same response that
produces `kind` and the compiled artifact. Do not add a second "generate tests" call.

Generating scenarios is not the same as *running* them (§3 step [4]):

| Sub-step | Extra LLM calls |
|---|---|
| Generate scenarios (inputs + expected behaviour) | **0** — folded into the one classify+compile call |
| Run a **routing** scenario | **0** — `evaluate_point`, pure deterministic Python (`engine.py`) |
| Run an **advisory** scenario | **1 per scenario** — a real sandbox `responder.generate()` |

So a routing preview is a **single** LLM call end to end. An advisory preview is one call
to generate + N sandbox calls to *demonstrate* behaviour — those N are the point of the
review gate (spec §4.5, "here's how the agent would behave") and should not be dropped to
save latency.

Two caveats:
- **This is the author-time path, not the turn path.** The budget is "admin waits a second
  at a form," not the ~1.3 s-per-turn live-call budget. Collapse generation into one call,
  but never trade running the advisory scenarios for author-time latency.
- **Mild self-grading:** when one call invents both the scenario and its
  `expected_behaviour`, the model partly grades itself. It still catches real failures
  (won't-compile, un-emittable intent, stripped advisory); the admin reading the rendered
  transcript is the true gate, and `expected_behaviour` is an admin hint, not a pass/fail.

---

## 3. Author-time pipeline — every step

```
Admin types plain English
        │
        ▼  POST /custom-instructions/compile-preview
┌─────────────────────────────────────────────────────────────────────┐
│ [1] CLASSIFY + COMPILE   one complete_json call, strict schema,       │
│     artifacts injected (context catalog, intent vocab, targets,       │
│     verdict whitelist)                                                │
│        ├─ advisory ─▶ compiled_advisory (imperative snippet)          │
│        └─ routing  ─▶ compiled_rule_yaml (grounded when + whitelisted │
│                       then)                                           │
├─────────────────────────────────────────────────────────────────────┤
│ [2] VALIDATE                                                          │
│     routing  → schema.compile_rules(yaml)   ← REUSE, unchanged        │
│               (a rule that doesn't compile is NEVER stored)           │
│     advisory → length + per-tenant token-cap check                    │
├─────────────────────────────────────────────────────────────────────┤
│ [3] GROUNDING CHECK                                                   │
│     unknown intent? unknown target? field not in catalog?             │
│     → return ambiguity to admin, stay inactive (spec Open Q5)         │
├─────────────────────────────────────────────────────────────────────┤
│ [4] AUTO-GENERATE + RUN TEST SCENARIOS  (compiler, not admin)         │
│     routing  → rules_admin.py `/test` (evaluate_point dry-run) REUSE  │
│     advisory → sandbox responder.generate() with the block injected;  │
│               assert the nudge appears AND grounding still passes      │
├─────────────────────────────────────────────────────────────────────┤
│ [5] REVIEW PAYLOAD → admin                                           │
│     rendered rule/snippet + test transcripts + grounding gaps         │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼  admin clicks Approve
   is_active = true
   (routing also flips its owned RuleSet is_active, atomically)
```

Steps **[2]** and **[4-routing]** are pure reuse of `compile_rules` and the `/test`
endpoint. The compiler's only novel job is **English → the YAML your engine already
validates.**

---

## 4. The one real reliability gap: `turn.intent` is free-form

This is the most important finding in this analysis.

**Today:** the planner emits intent as *"a short lowercase label in your own words"*
(`understanding.py:63`). There is **no closed set**. `normalize_intent`
(`confidence.py:73`) then maps *some* free-form labels onto ~9 canonical ones via a
hardcoded `INTENT_ALIASES` dict — but only for the handful the engine already cares about
(`request_human`, `book_appointment`, `reschedule`, `cancel`, `dnc_request`, …).

**Consequence for routing:** a rule `when: {field: turn.intent, op: eq, value:
loan_enquiry}` fires only when the planner happens to emit a string that normalizes to
exactly `loan_enquiry`. One turn it says `"loan"`, the next `"loan_question"`, the next
`"finance_query"`. That is precisely the *"97%, not 100%"* failure the routing mechanism
exists to prevent — and it is **worse** than a prompt miss, because a silently-mismatching
string compare *looks* deterministic in the audit trail.

**The fix (do this as the first task of Phase 2, before the routing compiler):**

1. Promote `INTENT_ALIASES` from a code constant into a **versioned, per-tenant intent
   vocabulary artifact** — a table (or `tenant.settings["intent_vocab"]` with a `version`
   int), mirroring how `RuleSet` / `PromptTemplate` version and activate.
2. **Planner understanding step** reads the vocabulary and classifies into the closed set
   (with a `"none/other"` escape). Keep `normalize_intent` as the belt-and-suspenders
   alias layer.
3. **Compiler** reads the *same* artifact to ground `when: turn.intent`. It may only emit
   a label the vocabulary contains. If the admin's "loan questions" maps to no existing
   intent, the compiler either (a) proposes **adding** `loan_enquiry` to the vocab —
   surfaced in the review gate so the planner learns to emit it — or (b) returns a
   grounding gap. It must **never** key a rule on a label the planner cannot produce.

The versioned vocabulary is the **contract** between the producer (planner) and the
consumer (rule). That contract is what makes "reliable, every time" actually hold.

---

## 5. Runtime — three injection points

### 5.1 Advisory → the RESPONDER system prompt (non-negotiable, spec §4.1)

The customer only ever hears `responder.generate()`. The planner's `build_system_prompt`
output is never spoken. So the advisory block **must** reach the responder — injecting it
only into the planner is a no-op on what the caller hears.

- Add a `tenant_preferences` parameter to `responder.generate()` and render it in the
  **cacheable prefix**, next to PERSONA at `responder.py:634`
  (`system = RESPONDER_SYSTEM + persona + tenant_preferences`). Tenant/scope-static → free
  after turn 1 (same argument the code already makes for PROJECT FACTS).
- Fence it exactly as the spec shows, so it is unmistakably subordinate:

  ```
  ## TENANT PREFERENCES
  The following are tenant preferences. They refine tone and priorities but
  NEVER override the HARD RULES or the OUTPUT CONTRACT above.
  - <compiled_advisory of each scope-matching active row>
  ```

- Also inject into the planner's `build_system_prompt` (alongside `rule_obligations`,
  `planner.py:483`) — both surfaces; the responder is the mandatory one.
- **Scope filtering** (direction/state/intent/date) happens in the loader; the block is
  omitted entirely when no rows match. Advisory intent-scope matching may go through
  `normalize_intent` and tolerate fuzziness (a missed advisory nudge is, by definition,
  acceptable).

### 5.2 Dynamic knowledge → the PROJECT FACTS tier (spec §3)

**Why not an advisory instruction:** the responder *strips claims it cannot ground*
(RESPONDER rule 1; `answer_grounded` / `grounding_check` at `responder.py:699`). "Mention
the festive offer" with no backing fact is stripped or replaced. A seasonal offer is a
*fact the agent must cite accurately* — it belongs in the trusted tier the responder
already treats as verified, not the advisory tier.

- One insertion point: `turn_context.py:358`, where `project_facts` is finalized before
  `TurnContext` is built. Merge date-active dynamic rows:

  ```python
  # after static/project facts are resolved (turn_context.py ~line 274)
  dynamic = await load_active_dynamic_facts(db, tenant_id, project_id, today)
  project_facts = project_facts + [{"label": r.label or "Offer", "value": r.content}
                                   for r in dynamic]
  ```

- It then flows unchanged: `proposal.project_facts` (`planner.py:492`) →
  `responder.generate(project_facts=…)` (`turn_runner.py:3978`) → `_project_facts_block`.
- The query is a plain date-range filter — no embedding, no re-index. When `valid_until`
  passes, the row simply stops being returned. No vector store touched.

#### 5.2.1 The three behaviours of a seasonal fact (why two mechanisms)

A seasonal offer needs three things, which split across **two** mechanisms:

| Requirement | Mechanism | Why |
|---|---|---|
| (a) Answer accurately *when asked* ("what offers are running?") | **Dynamic Knowledge** | Must be citable from the *trusted* tier or the responder strips it |
| (b) *Proactively* volunteer it when the caller sounds price-sensitive | **Advisory Instruction** | This is *when to bring it up* — framing, no hard consequence |
| (c) Stop citing after expiry, no code change | **Dynamic Knowledge** (date filter) | Row falls out of the query on `valid_until` |

So "lead with the festive offer when pricing comes up, until Oct 31" is **Advisory +
Dynamic Knowledge together** (spec §3.4): the *fact* lives in `project_facts_dynamic` so
it can be spoken truthfully; the *advisory* row says "volunteer it when price-sensitive."
An advisory-only "mention the offer" fails, because it asks the responder to assert a fact
it has no evidence for — `grounding_check` (`responder.py:699`) sets `answer_grounded:
false` and strips or replaces it. That is the entire reason this is a separate module.

Because the dynamic fact merges into the same `project_facts` the responder treats "as
authoritative as VERIFIED RESULTS," it can also ground a feature/scope answer (RESPONDER
rules 13–14). The base knowledge base is never touched — no embedding, no re-index. When
`valid_until` passes the row stops being returned; no job, no code change. `is_active` is a
separate manual kill-switch, independent of dates.

**Decision:** dynamic facts should also reach the planner prompt (`prompts.py:446` already
renders `project_facts` into the planner) for symmetry — but the responder is the
mandatory surface, same rule as advisory.

### 5.3 Routing → nothing new at runtime

A routing instruction's owned `RuleSet` is picked up by `active_rule_sets`
(`engine.py:42`) and evaluated at both turn passes automatically. Escalations already ride
`forced_escalation`; requirements already surface as POLICY OBLIGATIONS
(`planner.py:476`). The **only** new runtime code on the routing side is **targeted
escalation** (`escalate.target` + a tenant transfer-target registry, spec §4.4) — and it
degrades safely to generic escalation with the reason recorded until built.

---

## 5.5 Lifecycle & deletion (admin-managed)

Nothing is cached: `active_rule_sets` (`engine.py:42`) re-queries the DB every turn
filtered on `is_active == True`, and `turn_context` rebuilds per turn. So **every
activate / deactivate / delete takes effect on the next turn of any call — including calls
already in progress — with zero cache invalidation.**

```
draft ──compile──▶ inactive ──admin approve──▶ ACTIVE ──admin delete──▶ inactive (retained)
                       │                          │                          │
                  (never spoken)          in every turn's loader     out of every turn's
                                          query                      query; audit kept
   update = new version (old retained); routing: RuleSet version bumps in lockstep
```

**Deletion is soft-delete (deactivate + retain), not a hard delete** — the guardrails
(§7) require `raw_text` verbatim + version history + `created_by` for the audit trail.

- **Advisory delete** → `is_active = false`. Gone from the responder prompt next turn (and
  one intentional cache-prefix miss on that turn). Row + versions retained.
- **Routing delete** → `is_active = false` on the `custom_instructions` row **and** its
  owned `RuleSet`, in **one atomic transaction**. The rule stops firing next turn.
  *Failure to avoid:* deactivating the wrapper but leaving the `RuleSet` active orphans a
  live rule that keeps escalating with no admin-facing owner.
- **Dynamic fact delete** → `is_active = false` (or let it expire). Gone next turn.
- **Hard purge** (genuine cleanup only) → separate, permission-gated endpoint; log an
  `AuditEvent`. Not the default "Delete" action.

## 6. Data model

Both tables from the spec are correct. Grounded refinements:

### `custom_instructions`
- Columns per spec §2.3 (incl. `scope.valid_from` / `scope.valid_until`).
- Partial index on `(tenant_id, is_active)`.
- Runtime loader: filter `is_active AND (valid_from IS NULL OR <= today) AND (valid_until
  IS NULL OR >= today)` in SQL; match `direction/state/intent` from the `jsonb` scope in
  Python.
- **Routing rows own a `RuleSet`** (`rule_set_id`). On approve, write/activate both
  atomically; version bumps stay in lockstep (custom_instructions.version ↔
  RuleSet.version).

### `project_facts_dynamic`
- Columns per spec §3.2. Render `content` as the `value` in the existing `{label, value}`
  fact shape used throughout `turn_context.py`.
- Index on `(tenant_id, project_id, is_active)`.
- `is_active` is a manual kill-switch independent of the date range.

### Caps (spec Open Q1)
- Enforce at **write time** in the compile-preview endpoint (advisory row count + summed
  token estimate of `compiled_advisory`), not at runtime — fail the admin fast, keep the
  turn path clean.

---

## 7. Guardrails → existing mechanisms

| Guardrail (spec §2.6 / §8) | Enforced by |
|---|---|
| **Subordinate always** | Fenced block + RESPONDER_SYSTEM rules 1–15 sit above PERSONA/preferences by construction |
| **Whitelisted consequences only** | `VERDICT_KINDS` in `schema.validate` — any other verdict never compiles |
| **Author review gate** | No `is_active=true` without approve; `AuditEvent` records `raw_text` verbatim + compiled artifact + `created_by` + version |
| **Conflict resolution** (Open Q4) | Engine already has `priority` + `terminal` + "a block can never be overridden" (`engine.py`). **Convention:** custom-instruction RuleSets get a priority *band* (e.g. 100–999) strictly below hand-authored compliance rules (0–99), so a tenant instruction can never override DNC/compliance |
| **Per-tenant caps** | Write-time enforcement (§6) |
| **Full audit** | `AuditEvent` + the engine's per-rule evaluation trace (`decision.evaluated`) |

---

## 8. Build order

1. **Intent vocabulary artifact** — versioned table/setting; migrate `INTENT_ALIASES`
   into it; wire the planner understanding step + `normalize_intent` to read it.
   *Prerequisite for reliable routing only — see §4.*
2. **`project_facts_dynamic`** — table + migration + the ~5-line merge at
   `turn_context.py:358`. Lowest risk, immediate value, no LLM.
3. **`custom_instructions` + advisory compiler** — table + classify/compile
   `complete_json` + responder injection (§5.1) + compile-preview/CRUD/approve API
   (clone `rules_admin.py`) + write-time caps.
4. **Routing compiler pass** — grounds against #1, validates via `compile_rules`, owns a
   `RuleSet`; auto-test via the existing `/test` path.
5. **Targeted escalation** — `escalate.target` + tenant transfer-target registry +
   escalation-handler extension.

**Phase 1 = items 2 + 3** (advisory + dynamic knowledge, end to end; no intent
dependency).
**Phase 2 = items 1 + 4 + 5**, with **item 1 as a hard gate** before item 4.

---

## 9. Open questions (spec §6) — engineering positions

| # | Question | Position |
|---|---|---|
| Q1 | Per-tenant advisory caps | Enforce at write time; start with a conservative count + token budget, tune against real usage |
| Q2 | Who can author | Product/tenant call; gate the write endpoints behind a dedicated permission (mirror `require("rules:write")`) |
| Q3 | Review-gate UX | Design/frontend; the compile-preview endpoint already returns everything the UI needs (rendered artifact + test transcripts + gaps) |
| Q4 | Conflicting routing rules | Use the engine's existing priority/terminal/block semantics + the priority-band convention in §7 |
| Q5 | Ungroundable instruction UX | Compiler returns `ambiguous` + `grounding_gaps`; the row stays inactive until the admin resolves it |

---

## 9a. Phase 1 — implemented

Built as a self-contained feature module (`core/app/custom_instructions/`) plus minimal,
best-effort wiring into the turn path. Advisory instructions + Dynamic Knowledge are live;
routing (Phase 2) is accommodated by the data model but not activated.

**Feature module — `core/app/custom_instructions/`**
- `compiler.py` — the single classify+compile LLM call (strict schema, `test_scenarios`
  folded in). Only `advisory` is activatable in Phase 1; `routing`/`ambiguous` return with
  `grounding_gaps` and stay inactive.
- `llm.py` — the compile step's OWN provider + model selector, independent of the
  voice-agent LLMs. `CI_LLM_PROVIDER` (empty = reuse voice agent | openai | openrouter |
  groq | mock), `CI_LLM_API_KEY`, `CI_LLM_BASE_URL`, `CI_COMPILE_MODEL` — change the model
  or vendor for authoring without touching the live call path. All OpenAI-protocol
  compatible, and the chosen model id is passed through unchanged (no planner-style remap).
- `scope.py` — pure, deterministic scope matching (direction/state/intent/date) + a
  built-in self-test (`python -m app.custom_instructions.scope`).
- `loader.py` — per-turn, uncached DB loaders: `load_active_advisory`,
  `load_active_dynamic_facts`.
- `render.py` — the subordinate `## TENANT PREFERENCES` block.
- `caps.py` — per-tenant caps, enforced at activation.
- `service.py` — `compile_preview` + `assert_can_activate` (caps + state checks).

**Data + API**
- `core/app/db/models/custom_instructions.py` — `CustomInstruction` (+ `root_id` version
  lineage, `rule_set_id` reserved for Phase 2) and `ProjectFactDynamic`.
- `migrations/versions/0013_custom_instructions.py` — additive, idempotent.
- `core/app/api/custom_instructions_admin.py` — `/custom-instructions` (compile-preview,
  CRUD, versions, activate/deactivate, soft-delete + `?purge`) and `/dynamic-facts` CRUD.
  Guarded by new `instructions:read` / `instructions:write` permissions (rbac.py).

**Turn-path wiring (all best-effort — a failure leaves the turn byte-for-byte unchanged)**
- `planner.py` step 6b — after understanding (intent known), load + scope-filter + render
  advisory rows onto `Proposal.tenant_preferences`.
- `turn_runner.py` — pass `proposal.tenant_preferences` into `responder.generate(...)`.
- `responder.py` — inject the block into the cacheable system prefix, after PERSONA.
- `turn_context.py` — merge active dynamic facts into `project_facts` (flows to both the
  planner prompt and the responder). `project_id` is None when project is ambiguous, so
  only tenant-wide offers surface then.
- `config.py` — `ci_compile_model`, `ci_advisory_max_rows`, `ci_advisory_max_chars`.

**Apply the migration**

```bash
cd core && alembic upgrade head
```

**Deferred within Phase 1 (documented, not built):** live sandbox *execution* of advisory
test scenarios (generation is done; execution needs a stubbed `responder.generate` run) and
injecting the advisory block into the planner prompt as well (responder is the mandatory
surface and is wired; planner injection is an easy follow-up).

## 9b. Phase 2 — implemented

Routing instructions, the intent-vocabulary gate, and targeted escalation. Built on the
same feature module; routing reuses the existing rule engine end to end.

**Intent vocabulary (the gate — spec §4.3)**
- `core/app/custom_instructions/intents.py` — `BASE_INTENTS` exposes the code defaults
  (from `confidence.INTENT_ALIASES`) as a readable artifact; `load_vocabulary` merges them
  with active tenant rows; `extra_aliases` feeds the planner.
- `confidence.normalize_intent(raw, extra_aliases=None)` — now tenant-extensible; built-in
  aliases are checked first so a tenant entry can never shadow a core safety intent.
- `planner.py` rule-pass-2 loads the tenant vocabulary and normalizes `turn.intent` through
  it, so a routing rule keyed on a tenant intent (e.g. `loan_enquiry`) fires reliably.
- `core/app/db/models/custom_instructions.py::IntentVocabularyEntry` + admin CRUD at
  `/intent-vocabulary` (GET merged view, PUT upsert, DELETE).

**Routing compiler (spec §2.3 / §4)**
- `compiler.py` now emits a rule's `when`/`then` (as JSON) grounded against the vocabulary,
  transfer targets, and the context-field catalog, all injected into the prompt.
- `core/app/custom_instructions/routing.py` — wraps the rule into a RuleSet applied at the
  `turn` point, validates through the SAME `schema.compile_rules`, and GROUNDS it: unknown
  field / off-vocabulary intent / unknown target ⇒ a grounding gap, never a silent compile.
- `service.py::activate_owned_ruleset` — a routing instruction OWNS a generated RuleSet
  (versioned by a stable per-instruction name), re-validated at activation; deactivate /
  delete tear it down in the same transaction so a live rule is never orphaned.

**Targeted escalation (spec §4.4 / §5) — route + record**
- `core/app/custom_instructions/routing.py` whitelists `escalate.target`; the compiler only
  emits a registered target.
- `TransferTarget` model + `/transfer-targets` admin CRUD.
- `planner.py` threads the rule's `escalate.target` onto `forced_escalation`; `turn_runner.py`
  resolves it against the active registry, stamps `Escalation.target` + the
  `escalation.created` event payload, and **degrades to a generic escalation** (target=None,
  reason recorded) when the target is unknown/inactive — never a hard failure.

**Data + migration**
- `migrations/versions/0014_intents_and_transfer_targets.py` — `intent_vocabulary`,
  `transfer_targets`, and `escalations.target`. Additive + idempotent.

**Verified (deps-less):** routing grounding across valid / unknown-intent / unknown-target /
unknown-field / require cases; `normalize_intent` extension (incl. the core-intent-wins
guardrail); compiler routing assembly → activatable; existing `confidence` self-test still
green.

**Apply:** `cd core && alembic upgrade head` (runs 0013 then 0014).

**Deferred within Phase 2:** live sandbox *execution* of routing test scenarios via the
`/rules/{name}/test` dry-run (grounding + generation are done; wiring the dry-run into the
review payload is the remaining step), and a live SIP transfer to a target's destination
(out of scope by decision — escalations are picked up from the queue today).

## 9c. Post-smoke hardening + pending tasks closed

Fixes and pending items completed after the Docker smoke test:

- **Priority-band enforcement (Q4/§7, safety).** Routing rules are clamped to a floor
  (`CI_ROUTING_PRIORITY_MIN`, default 100), reserving 0–99 for compliance. `Verdict` now
  carries its rule's priority and the planner acts on the **lowest-priority escalation**, so
  a compliance escalation always precedes a tenant one. (`routing.py`, `rules/engine.py`,
  `planner.py`, `config.py`.)
- **Routing test-scenario execution (review gate, spec §4.5).** `routing.simulate()`
  deterministically runs the compiled rule against a synthesised trigger (no LLM, no DB) and
  the outcome rides in the compile-preview payload as `dry_run` — e.g. *"turn.intent=
  loan_enquiry → escalate to finance_desk"*.
- **Advisory injection into the planner prompt (spec §4.1).** The planner now also receives
  the subordinate `TENANT PREFERENCES` block (non-intent-scoped rows, in its cacheable
  prefix); the responder remains the mandatory, fully intent-aware surface.
- **Compiler retry.** The compile call retries up to 3× on provider failure — Groq's
  gpt-oss models intermittently fail strict structured-output generation on the larger
  routing schema; retries make authoring reliable (verified 3/3 after, ~1/3 before).
- **Two smoke-test bugs fixed:** a malformed `CI_LLM_BASE_URL` no longer crashes (falls back
  to the provider default — `llm.py::_clean_base_url`); server-generated timestamps no longer
  raise `MissingGreenlet` after commit (`eager_defaults` on the four models).

**Still pending (not code, or deliberately deferred):**
- Advisory test-scenario *execution* (routing execution is done; advisory needs a stubbed
  responder sandbox).
- Full conversation-test datasets via `run_conversation_tests.py` (the smoke script
  `scripts/smoke_custom_instructions.py` is the interim regression check).
- Admin UI / review-gate UX (frontend).
- Live SIP transfer (out of scope by decision).

## 10. Explicitly out of scope (spec §11)

- Vector storage of instructions — instructions are a small, curated, enumerable set
  selected by metadata filter, applied *every* turn; similarity retrieval would be a
  correctness bug for policy. Revisit only if a single tenant's metadata-scoped block
  exceeds the token budget.
- Raw YAML authoring for power users — plain English only.
- Cross-tenant / global instruction libraries.
