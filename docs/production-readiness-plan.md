# Production Readiness — Master Remediation Plan

**Status:** living document · master plan
**Scope:** the full set of changes required to take the agent from a reactively‑patched
prototype to a production‑grade, accurate, reliable, and low‑latency system.
**Companion docs (state/memory detail):**
[production-grade-fixes.md](./production-grade-fixes.md) ·
[booking-task-architecture.md](./booking-task-architecture.md) ·
[entity-slot-memory.md](./entity-slot-memory.md) ·
[task-frame-continuity.md](./task-frame-continuity.md)

---

## 0. TL;DR for reviewers

The **architecture in the PRD is sound and we are keeping it** — modular monolith,
Postgres + pgvector, the 12‑step decision cycle, the tool‑contract / rule‑engine /
state‑machine separation, provider abstraction. **No rewrite, no microservices, no
framework change.**

What has gone wrong is the **implementation has drifted into symptom‑patching.**
[`turn_runner.py`](../core/app/engine/planner/turn_runner.py) is ~3,500 lines, most of it
regex nets that police the LLM *after the fact* (`CLAIM_PATTERNS`, `PROMISE_PATTERNS`,
`_BOOKING_STATUS_RE`, `_fit_params`). Every bug in testing added another pattern. That
layer is not production‑grade and is an unbounded maintenance liability — but it is a
**symptom**, not the disease.

The disease is **three structural gaps**, each with a targeted fix. Closing them lets us
*delete* the majority of the regex layer rather than keep growing it.

| # | Defect | Fix | Deletes |
|---|--------|-----|---------|
| **P0** | Ungrounded reply can bypass the grounded responder | Make grounded responder the only path to the customer | `CLAIM_PATTERNS`, `PROMISE_PATTERNS`, `NO_AVAILABILITY_CLAIM` + guards |
| **P1** | Pre‑model routing is regex because there is no understanding step | Restore the fast understanding call (PRD's two‑tier design) | `needs_knowledge`, `SCHEDULING`, `_is_scheduling_aside`, `_asks_booking_status` regexes |
| **P2** | Conversation state is emergent and scattered | Explicit orchestrator‑owned state (see companion docs) | ad‑hoc session‑key reconstruction |
| **P3** | Weak retrieval + eager "callback" fallback | Structured ingestion, synonym/expansion, honest fallback | — |
| **P4** | Per‑turn latency 5–15 s | Prompt/KV caching, model tiering, streaming, speculative retrieval | — |
| **P5** | Every bug becomes a regex, not a test | Golden‑conversation suite; no new pattern without a failing test | (prevents regrowth) |

**Sequence: P0 → P1 → P4 → P2 → P3 → P5.** P0 alone visibly fixes most of `Bugs.md`.
P4 is pulled forward because current per‑call latency makes a two‑call model unviable
until it is fixed.

---

## 1. Diagnosis — a good architecture, patched into fragility

The core reasoning is **already LLM‑driven**, not regex: intent, entities, action, tool,
and params all come from one structured‑output call in
[`planner.py`](../core/app/engine/planner/planner.py). That part is correct.

The failure is a **second regex layer wrapped around the LLM** to catch it lying,
re‑derive what the user meant, and rewrite its parameters. The tell is in the code itself:
there are patches for a single dropped STT letter (`"18th" → "18t"`), for the adverb
`"successfully"` landing between `"has been"` and `"booked"`, and for `"registered"` vs
`"booked"`. This is a system fighting its own model.

Every one of those patches exists to compensate for a gap below. Fix the gaps and the
patches become dead code.

---

## 2. P0 — Close the grounding bypass *(highest ROI change in the codebase)*

### The bug

The PRD (step 11) requires the customer‑facing reply to be **regenerated after execution,
from verified results only**, forbidden to state any fact not present. That grounded
responder exists — [`responder.py`](../core/app/engine/planner/responder.py) — and even
self‑reports `asserts_completed` for cross‑checking. **But it is skipped on the very turns
that need it most.** [`turn_runner.py:3047`](../core/app/engine/planner/turn_runner.py):

```python
needs_grounding = (bool(outcome) or bool(proposal.knowledge_used)
                   or astate in ("staged", "blocked"))
if needs_grounding:
    text = await responder.generate(...)     # safe, grounded path
else:
    text = proposal.response_draft            # RAW model draft → customer
```

When a turn runs no tool and retrieves no knowledge, the model's **draft — written during
planning, before anything executed** — goes straight to the caller. That is the source of
every *"I've registered your appointment" / "all set for the 19th" / "I'll schedule a
callback"* with nothing behind it. Every **"Responder claimed a completed book with no
book_appointment execution"** note in `Bugs.md` traces to this line.

This single gap is *why* `CLAIM_PATTERNS` and `PROMISE_PATTERNS` had to be built: a second
net to catch the lies the draft leaks.

### The fix

1. **The grounded responder becomes the only path to the customer.** `response_draft` is
   demoted to an internal hint; it is never emitted raw.
2. **Inject the deterministic action state into the responder prompt.** By the time the
   reply is generated we already *know* the ground truth — `book_appointment` either
   committed to the DB or it did not. We do not need the model to self‑report; we
   **constrain** it to the known state (`done` / `staged` / `blocked` / `failed` / `none`).
3. **Delete the post‑hoc claim/promise regex.** With no ungrounded draft reaching the
   caller and the reply constrained to verified state, `CLAIM_PATTERNS`,
   `PROMISE_PATTERNS`, `NO_AVAILABILITY_CLAIM`, `_false_claim`, `_promises_action`, and
   their retry machinery are redundant. Remove them incrementally, each behind a passing
   golden test (P5).

### Why this is also what makes streaming safe

Because execution is deterministic, the outcome is known **before** leg B runs. That is the
same property that lets us stream leg B token‑by‑token (§5.4) without risking a mid‑stream
false claim — the constraint is in the prompt and the state, not in a post‑hoc scan.

---

## 3. P1 — Restore the two‑tier understanding step

### Why the regex exists

Retrieval gating and "what did the user mean" run **before** the model has produced
anything, so today they *cannot* use a model label — hence they are regex
([`rag/__init__.py`](../core/app/engine/rag/__init__.py):`needs_knowledge`, `SCHEDULING`;
[`turn_runner.py`](../core/app/engine/planner/turn_runner.py):`_is_scheduling_aside`,
`_asks_booking_status`). The PRD specifies a **fast Haiku‑class classify/extract call as
step 2**; it was collapsed into `single_call` mode to save latency, and that collapse is
what forced these decisions to be pattern‑matched.

### The fix

Reintroduce a cheap understanding call (fast model — `gpt-4o-mini` or **Groq
`llama-3.1-8b-instant`**, ~200–400 ms) that returns `intent`, `entities`, and the routing
booleans (`needs_knowledge`, `is_scheduling`, `is_status_question`). Retrieval gating and
scheduling/status detection then become **semantic**, and the regexes are deleted.

This is also what makes the *"generator = power backup"* / *"daycare = creche"* class of
misunderstanding tractable: the understanding step (plus §6 synonym expansion) bridges
vocabulary that a keyword regex never will.

> **Latency note:** this adds one call, so it is only viable **after P4**. See §5.6 for how
> the extra call is absorbed (fast model + speculative retrieval overlap).

---

## 4. P2 — Consolidate conversation state *(detail in companion docs)*

The memory‑inconsistency bugs (Feedback #1 "2 BHK → quoted 3 BHK budget"; #8 "is my visit
scheduled"; the booking‑continuity family) are one root cause: **critical state is emergent
and scattered, and the LLM is expected to be the memory.** Full treatment already exists:

- **What the call is *about*** → [entity-slot-memory.md](./entity-slot-memory.md)
  (additively‑merged `active_entities` as a retrieval constraint).
- **What the call is *trying to do*** → [task-frame-continuity.md](./task-frame-continuity.md)
  and [booking-task-architecture.md](./booking-task-architecture.md)
  (explicit orchestrator‑owned booking state machine).
- **The umbrella** → [production-grade-fixes.md](./production-grade-fixes.md).

Two additions this plan makes explicit:

1. **Identify inbound callers by phone at conversation start.** Long‑term memory (facts +
   summaries) loads in
   [`turn_context.py:77`](../core/app/engine/planner/turn_context.py) **only when
   `customer` is set.** Most test calls are "Unknown caller" → `customer_id` is null → zero
   episodic/semantic memory. Resolve the customer from the caller ID before the first turn
   so history attaches.
2. **Answer appointment‑status questions from Postgres, not session state.** "Is my visit
   scheduled?" must be a DB read at context assembly — the durable truth — not a lookup in
   the Redis `booking_task`, which only covers in‑flight bookings.

---

## 5. P4 — Latency architecture (target: 3–5 s/turn, < 2.6 s to first token)

### 5.1 The current problem is per‑call cost, not call count

Transcripts show **5–15 s turns on one or two LLM calls.** A 12 s `knowledge_lookup`
(Bugs.md Tab 11) is not a call‑count problem — a two‑call model would make it 24 s. Four
multipliers are stacked:

1. **`gpt-4o` over the public API** — ~800–1500 ms fixed before the first token
   (matches the `~1300ms fixed overhead` comment in
   [`planner.py`](../core/app/engine/planner/planner.py)).
2. **No prompt / KV caching** — the large static prefix (persona + tool schemas + rules)
   is re‑processed every call.
3. **Buffered JSON** (`complete_json`) — the whole object is awaited before anything moves;
   no overlap with TTS, no early exit.
4. **Cross‑region** India → OpenAI RTT on every call.

**Profile before optimizing call count** (see P4‑task‑0 in §7).

### 5.2 Prompt caching + KV / prefix cache *(biggest single win)*

At the API layer, "KV cache reuse" is exposed as **prompt caching**: the provider keeps the
attention KV cache for a stable prompt **prefix** and skips recomputing it.

| Provider | Mechanism | Requirement |
|---|---|---|
| **OpenAI** (`gpt-4o`) | Automatic prefix caching for prompts > 1024 tokens; ~50% input cost + latency cut | Static content **first**, byte‑stable prefix |
| **Anthropic** (Claude, per PRD) | Explicit `cache_control` breakpoints; up to ~90% input cost cut, large latency cut; 5 min / 1 h TTL | Mark the static prefix breakpoint |
| **Self‑host (vLLM / Groq)** | Automatic prefix **KV** caching | Same prefix stability rule |

**Required refactor to actually benefit — prompt layout must become prefix‑stable:**

```
┌─ STATIC PREFIX (cacheable) ──────────────────┐
│  persona · tool JSON schemas · rule           │   ← identical every turn for a tenant
│  obligations · format contract                │   ← place FIRST, keep byte‑for‑byte stable
├─ ⟵ cache breakpoint ⟶ ────────────────────────┤
│  DYNAMIC SUFFIX (not cached)                  │
│  conversation history · current utterance ·   │   ← changes every turn, goes LAST
│  retrieved knowledge · action state           │
└───────────────────────────────────────────────┘
```

Today [`prompts.build_system_prompt`](../core/app/engine/planner/prompts.py) interpolates
dynamic turn context into the system prompt, which **defeats prefix caching**. Split it:
static system prefix (cached) + dynamic content in the user message.

### 5.3 Model tiering *(turn 1.3 s calls into 300–500 ms calls)*

Reserve the strong model for the one reasoning/proposal call; put understanding (P1) and
the grounded reply (P0) on a fast model. **Groq is the latency ace** here —
`llama-3.3-70b-versatile` is structured‑output capable and typically **3–5× faster** than
`gpt-4o`; `llama-3.1-8b-instant` covers understanding. Config already supports the switch
(`groq_model_main`, `groq_model_fast` in [`config.py`](../core/app/config.py)).

| Leg | Model | Rationale |
|---|---|---|
| Understanding (P1) | `llama-3.1-8b-instant` / `gpt-4o-mini` | classification, ~300 ms |
| Reason + propose | `gpt-4o` (cached) / `llama-3.3-70b` | the one call that needs the strong model |
| Grounded reply (P0) | fast model, **streamed** | constrained to verified state, so speed‑safe |

### 5.4 Stream the final leg (SSE now, WebSocket for voice)

**SSE reduces *perceived* latency, not total compute.** It only helps leg B (the generated
reply), letting TTS/UI start at the first token (~0.4 s) instead of waiting for the whole
reply. It does nothing for understand → plan → execute, which must finish first.

- **Chatbot today: SSE** is the right, simple choice — FastAPI `StreamingResponse` /
  `sse-starlette`, one long‑lived request.
- **Voice later: WebSocket**, not SSE — voice needs *bidirectional* streaming for barge‑in
  (caller interrupts mid‑reply) and partial‑STT upstream, which SSE cannot do. The PRD
  already scopes the Voice Session Service as a separate WebSocket service.

**Required change:** leg B currently returns **JSON** (`{response, asserts_completed}`),
which streams badly. Switch it to **plain‑text streaming** and move the claim‑safety check
to deterministic code (we already know the outcome — §2). Build one `astream` interface on
the provider so both SSE and WebSocket consume it.

### 5.5 Speculative retrieval + skip‑calls routing

- **Speculative retrieval:** fire the embedding + pgvector/BM25 search on the **raw
  utterance in parallel** with the understanding call; refine only if entities change.
  Hides ~300 ms behind a call that is happening anyway.
- **Skip calls:** chitchat, confirmations, and slot‑picks need 0–1 calls, not 2–3. Let the
  understanding step short‑circuit them to a canned or single‑call path.
- **Region colocation:** run the core near the model endpoint to shave RTT on every call.

### 5.6 The resulting budget — two LLM calls, under 3.5 s

Two customer‑facing LLM calls: **(A) reason/propose** + **(B) grounded reply**. Retrieval
and execution are deterministic and overlap. (P1's understanding call is the fast, mostly
hidden third leg.)

| Step | Model / work | Time | Note |
|---|---|---|---|
| Understanding (P1) | fast model | ~0.3–0.5 s | retrieval overlaps this |
| Retrieval | embed + pgvector | ~0.3 s | **speculative, overlapped** |
| Reason + propose (A) | `gpt-4o` **cached** | ~1.0–1.5 s | caching + `max_tokens=400` |
| Validate + execute | DB / row lock | ~0.2 s | deterministic |
| Grounded reply (B) | fast, **streamed** | first token ~0.4 s, full ~1.0 s | TTS starts at 0.4 s |
| **Total wall‑clock** | | **~2.8–3.5 s** | inside target |
| **Time to first token** | | **~2.2–2.6 s** | what the caller feels |

Push leg A onto Groq and total approaches ~2 s.

---

## 6. P3 — Retrieval quality & honest fallback

Feedback #2–#7 (park‑facing, creche/daycare, dog parks, generator/backup, bricks vs RCC,
vague location) are **not** keyword‑vs‑semantic — retrieval is already hybrid vector + BM25.
They are three separate quality gaps:

1. **Ingestion / chunking.** Facts in tables, floor plans, amenity grids, and site maps are
   destroyed by naive text chunking, so "dog parks" / "generator" are in **no retrievable
   chunk**. Fix: structured ingestion that extracts fielded facts (amenity lists,
   specifications, location) rather than prose. Consider a small curated fact table /
   mini‑KG for always‑answerable structured facts (location, RCC structure, amenity count).
2. **No synonym / alias bridge.** `generator ↔ DG ↔ power backup`, `creche ↔ daycare ↔
   childcare`. Fix: an alias layer or LLM query‑expansion in the understanding step (P1).
3. **Eager "callback" reflex.** When retrieval is weak the agent jumps to *"I'll arrange a
   callback, what's your number?"* — the worst fallback for a sales agent, and it appears
   in four scenarios. Fix: answer from knowledge → else an honest *"I don't have that
   specific detail, but…"* Make `schedule_callback` harder to reach in the sales stage
   (stage/tool‑availability + rule).

Also tune the persona so the agent stops steering every turn toward a site visit
(Feedback #2) — a prompt/state‑goal change, not a code bug.

---

## 7. Work breakdown, sequencing, and effort

| Task | Prio | Effort | Impact |
|---|---|---|---|
| **P4‑0** Turn profiler — attribute the current 5–15 s per stage/call | first | S | unblocks all latency work |
| **P4‑1** Prompt/KV caching — split static prefix from dynamic suffix | P4 | S | largest single latency win |
| **P4‑2** Model tiering — Groq/`mini` for understanding + reply legs | P4 | S–M | 1.3 s calls → 0.3–0.5 s |
| **P0‑1** Grounded responder is the only reply path; inject action state | **P0** | M | fixes most of `Bugs.md` |
| **P0‑2** Delete claim/promise regex behind golden tests | P0 | M | removes ~hundreds of lines |
| **P4‑3** Plain‑text streaming leg B + SSE endpoint | P4 | M | perceived latency ↓ ~1 s |
| **P1‑1** Fast understanding call; semantic routing booleans | P1 | M | deletes routing regex; robustness |
| **P1‑2** Delete `needs_knowledge`/scheduling/status regex | P1 | S | removes more regex |
| **P4‑4** Speculative retrieval + skip‑calls routing | P4 | M | absorbs P1's extra call |
| **P2‑1** Identify caller by phone at call start | P2 | S | long‑term memory attaches |
| **P2‑2** Appointment status from Postgres at context assembly | P2 | S | fixes "is my visit booked?" |
| **P2‑3** Entity‑slot + booking state machine | P2 | L | see companion docs |
| **P3‑1** Structured ingestion + curated fact table | P3 | L | fixes amenity/spec/map misses |
| **P3‑2** Synonym/expansion + honest fallback + de‑emphasise callback | P3 | M | fixes semantic‑miss class |
| **P5‑1** Golden‑conversation suite; CI gate | P5 | M | stops regex regrowth |

**Recommended order:** P4‑0 → P4‑1 → P4‑2 → P0‑1 → P0‑2 → P4‑3 → P1 → P4‑4 → P2 → P3 → P5.

---

## 8. What we are explicitly NOT changing

- Modular monolith (no microservices).
- Postgres + pgvector as the single relational + vector store.
- The 12‑step decision cycle and the LLM‑proposes / code‑validates‑and‑executes split.
- Tool contracts, rule engine, journey state machine, provider abstraction.

These are correct and production‑appropriate at target scale. The work is **three seams and
a latency pass**, not a redesign.

---

## 9. How we know it worked (exit criteria)

- **Accuracy:** the golden‑conversation suite (P5) passes, including every reproduced
  transcript in `Bugs.md` and `Voice Agent Feedback.md`; no reply asserts an action the
  tools did not perform (measured from the decision trace, not regex).
- **Latency:** p50 turn ≤ 3.5 s, p95 ≤ 5 s, time‑to‑first‑token ≤ 2.6 s.
- **Maintainability:** `turn_runner.py` regex guard count trends to near‑zero; new bugs land
  as failing golden tests, never as new patterns.
- **Memory:** cross‑turn and cross‑call recall verified for identified callers; appointment
  status answered from the database of record.
```
