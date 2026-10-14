# Tool-Turn Latency — Root Cause & Fix

**Status:** implemented on branch `dev`, behind a default-off flag · **Date:** 2026-09-19
**Audience:** engineers working on the voice-agent planner (`core/app/engine/planner/`)

---

## 1. TL;DR

A golden-dataset evaluation showed some turns taking 2–3× a normal turn. Profiling proved the
cost is **not** tool execution (~11 ms) but the number of **sequential LLM round-trips** a turn
makes. Each turn ran three model calls in series:

```
understand  →  propose  →  respond
```

The first call (`understand`, a fast classifier) was *supposed* to be free because it ran
"concurrently" with document retrieval. **Live measurement proved it was not free:** retrieval
is ~15 ms, but `understand` is a full ~1.3 s LLM round-trip, so overlapping them hid nothing —
`understand` was effectively a serial ~1.3 s cost on **every non-trivial turn**.

**The fix (P4-1):** run `understand` concurrently with `propose` (the *long* call, ~1.5 s)
instead of with retrieval. `understand` now finishes inside the time `propose` already takes, so
its ~1.3 s disappears from the critical path. Measured result: **~1–2 s off nearly every turn**,
with **no change to any answer** and prefix-cache hit rate unchanged (~91 %).

Shipped behind `understanding_concurrent_with_propose` (default **off**). Alongside it we fixed a
telemetry bug that was hiding re-plan cost, added prompt-cache instrumentation, and removed a dead
config switch.

---

## 2. Background: the shape of a turn

`process_turn` (`core/app/engine/planner/turn_runner.py`) drives every caller utterance. The
planner (`core/app/engine/planner/planner.py`, `plan()`) runs steps 1–7:

| Step | Name | Kind | Cost |
|---|---|---|---|
| 1 | assemble context | deterministic | small |
| 2 | **understanding** | **LLM (fast model)** | **~1.3 s** |
| 3 | knowledge retrieval | embedding + pgvector | ~15 ms |
| 4 | candidate actions | deterministic | small |
| 5 | rule gate | deterministic | small |
| 6 | **propose** | **LLM (main model)** | ~1.4–2.5 s |
| 7 | confidence banding | deterministic | small |

After the planner, the **responder** (`responder.py`, `generate()`) makes one more main-model
call to write the spoken sentence. So a normal turn = **3 serial LLM calls**: understand →
propose → respond.

Tool/scheduling turns can make *more* propose calls: after a read-only lookup (e.g.
`my_appointments`, `search_slots`) the engine may re-plan to decide the next step
(`turn_runner.py`, the chain loop bounded by `MAX_TOOL_STEPS`/`CHAIN_BUDGET_MS`), and a
mis-planned confirm triggers one corrective retry. Those are the `x > 1` turns in the profiler.

**Governing equation:** `turn wall time ≈ (number of serial LLM calls) × (per-call floor)`.

---

## 3. Root cause: `understand` was serial, not hidden

### 3.1 The original design assumption

`understand` decides *semantically* whether a document could answer the turn (replacing brittle
regexes). To keep it "off the critical path," it was run **concurrently with a speculative
retrieval** via `asyncio.gather` in `_understand_and_retrieve` (`planner.py`). The design comment
assumed retrieval was ~300 ms and would hide the ~200–400 ms classifier behind it.

### 3.2 What measurement actually showed

Using the replay harness (`core/scripts/replay_outliers.py`, which reports per-stage timings the
xlsx export drops), the concurrent leg splits like this on every turn:

```
understand_retrieve = 1237ms   (u=1237  r=18)
understand_retrieve = 1409ms   (u=1409  r=19)
understand_retrieve = 1305ms   (u=1305  r=14)
```

- `r` (retrieval) is **~15–30 ms** — embeddings/pgvector are far faster than assumed.
- `u` (understand) is **~1.1–1.7 s** — a full LLM round-trip, matching the bare
  `gpt-4o-mini` floor (nearly all time-to-first-token, not generation).

Because `gather` only hides the *shorter* leg, and retrieval is ~80× shorter than the classifier,
**`understand` was almost entirely exposed** — a serial ~1.3 s call on every non-trivial turn.
The engine was spending ~1.3 s to decide whether to keep a ~15 ms result.

> This is why the original investigation's "Finding 6" (a wasteful per-turn understanding call)
> was directionally right, even though its proposed mechanism — a dead `planner_mode="single_call"`
> config switch — never controlled anything.

---

## 4. The fix (P4-1): overlap `understand` with `propose`

### 4.1 Idea

Retrieval is trivially cheap, so there is no point overlapping the classifier with it. Overlap
the classifier with the **longest** call instead — `propose`. On a normal turn `propose` (~1.5 s)
is longer than `understand` (~1.3 s), so running them together hides `understand` entirely.

**Analogy:** don't boil the pasta and *then* make the sauce; make the sauce *while* the pasta
boils. Same meal, ready sooner.

### 4.2 Control flow

```
BEFORE (serial):
  context → [understand ∥ retrieve] → rules → build prompt → propose → respond
             └── ~1.3s (understand dominates) ──┘         └ ~1.5s ┘   └ ~1.3s ┘
  ≈ 1.3 + 1.5 + 1.3  = ~4.1s

AFTER (understanding_concurrent_with_propose = ON, non-ambiguous non-retry turn):
  context → retrieve(~15ms) → rules → build prompt → [understand ∥ propose] → respond
                                                      └── max(1.3, 1.5) ≈ 1.5s ──┘  └ 1.3s ┘
  ≈ 0.015 + 1.5 + 1.3  = ~2.8s
```

### 4.3 Implementation

Two seams in `planner.py`:

1. **`_understand_and_retrieve(..., skip_understanding=True)`** — new parameter. In skip mode it
   runs retrieval *only*, returns `understanding=None`, and **keeps the knowledge regardless of
   routing** (the routing decision isn't known yet). The classifier is not called here.

2. **`plan()`** — when the flag is on and the turn qualifies:
   ```python
   _concurrent_understand = (get_settings().understanding_concurrent_with_propose
                             and not skip_retrieval and not tc.project_ambiguous)
   ```
   - Retrieval runs alone (`skip_understanding=_concurrent_understand`).
   - At the propose call, `understand()` and `propose()` are gathered:
     ```python
     understanding, result = await asyncio.gather(
         understand(user_input, history=tc.history,
                    model=get_settings().llm_model_fast, candidates=None),
         _propose)
     ```
   - After the gather, `understand`'s `do_retrieve` is applied **to the responder's material
     only** (`proposal.knowledge_used`), so a turn routed away from documents still never hands
     the responder off-topic chunks.

### 4.4 The one behaviour change (and why it's safe)

Because we can't know `do_retrieve` before `propose` runs, **`propose` now always sees the
retrieved knowledge** — including on a scheduling/status turn that the serial path would have
discarded it for. This is the *only* behavioural difference, and it is bounded by three existing
guardrails:

- The `propose` prompt (`prompts.py`, `BASE_SYSTEM`) has strong rules that scheduling turns must
  call scheduling tools and "answer the question they asked."
- The **responder still only receives knowledge when `do_retrieve` is true** — its input and its
  grounding gate are byte-for-byte unchanged.
- The responder is the *only* path to the caller and speaks only from verified results
  (`turn_runner.py`: "the grounded responder is the ONLY path"), so it cannot fabricate from a
  stray chunk `propose` happened to see.

**Turns that keep the exact serial path (never touched by this flag):**
- **Project-ambiguous turns** — the classifier's `project_choice` must resolve the project
  *before* the prompt/retrieval scope is built.
- **Retry / chain legs** (`skip_retrieval=True`) — they already retrieved once this turn.

### 4.5 Rollout

Controlled by `understanding_concurrent_with_propose` in `core/app/config.py` (default `False`).
Enabling is a `.env` change (`UNDERSTANDING_CONCURRENT_WITH_PROPOSE=true`); reverting restores the
byte-for-byte serial behaviour with no code change.

---

## 5. Results (live replay, OpenAI `gpt-4o-mini`, 3 runs/turn)

**Mechanism proof** — the concurrent leg collapses to retrieval-only, and `u` now runs during
propose:

```
OFF:  understand_retrieve = 1305ms  (u=1305  r=14)
ON:   understand_retrieve =   19ms  (u=1306  r=17)   ← u now overlaps propose
```

**Wall-time deltas (flag OFF → ON):**

| Turn | Wall OFF | Wall ON | Δ |
|---|---:|---:|---:|
| dnd_registered_hi | 5164 | 2847 | **−2317** |
| whos_meeting | 4493 | 2788 | −1705 |
| have_details | 4264 | 2925 | −1339 |
| cancel_visit | 4305 | 3034 | −1271 |
| banks_approved | 4306 | 3079 | −1227 |
| registration_cost | 4182 | 3346 | −836 |
| dnd_registered | 3970 | 3291 | −679 |

~1–2 s off nearly every turn. Prefix-cache hit rate unchanged (~91 %).

**Accuracy gate:** the `say:` line of every scheduling/status turn was inspected. None drifted
into reciting property/brochure facts, and answers matched the serial run turn-for-turn in
substance (e.g. `budget_crore` still returns "3 BHK is 1.54 crore, above budget"; `cancel_visit`
still returns "Shall I cancel your visit scheduled for Friday…"). On this dataset the flag is safe.

> **Note on `x > 1` variance:** whether a turn re-plans is non-deterministic (the model's choice),
> so an individual turn can be `x=1` in one run and `x=2` in the next. This flag lowers the cost of
> each leg, not the number of legs, so a turn that happens to re-plan is still slow. The re-plan /
> corrective-retry tail is the next lever (see §8).

---

## 6. Supporting changes shipped alongside

### 6.1 Telemetry fix — honest `planner_calls` / `planner_ms` (`turn_runner.py`)

`timing["planner_ms"]` was overwritten and `timing["planner_calls"]` hard-reset to `1` *after* the
corrective-retry block had already incremented them — so a mis-planned tool turn (which makes 2
planner calls) reported `x=1`, erasing the retry from the very count the replay harness reads.
Now recorded on the opening leg and **accumulated** by the retry and chain re-plans. Without this,
the fix's own before/after numbers would be untrustworthy.

### 6.2 Prompt-cache instrumentation (`providers/llm/openai_provider.py`)

Captured `cached_tokens` (from `prompt_tokens_details`) into the per-call token dict. This is the
number that *proves* the deliberate prompt-prefix ordering (`prompts.build_system_prompt`) is
paying off. Live runs show **~91 % of prompt tokens served from cache** on OpenAI. (Groq/Cerebras
subclass the OpenAI adapter; they report `0` if the provider doesn't expose it — see §7.)

### 6.3 Removed dead config + fixed stale docstrings

- Removed `planner_mode` from `config.py` — it was read nowhere and its premise was obsolete.
- Fixed `planner.py` docstring ("folded into single_call mode") and `responder.py` docstring
  ("call is skipped for clarify/ack turns" — it is deliberately *not*; the grounded responder is
  the only path to the caller). These stale comments would send the next investigation chasing
  ghosts.

---

## 7. Production note: OpenAI vs Cerebras

Production uses **Cerebras**; the numbers above are from OpenAI and are inflated by its ~1.5–2.2 s
per-call floor. Two consequences:

1. **Absolute savings will be smaller on Cerebras** (faster inference), but the fix still helps:
   `understand` runs on `gpt-oss-120b` (main == fast) with `cerebras_min_max_tokens=1500`, so it
   still generates real reasoning tokens per turn.
2. **The prefix-cache win does not transfer.** Cerebras is unlikely to report/serve
   `cached_tokens`, so `cache%` will read `-`. That makes **cutting the *number* of calls (this
   fix, plus the re-plan tail) matter more on Cerebras, not less** — there's no cache to absorb the
   cost of re-reading the ~8k-token prompt.

---

## 8. How to validate / measure

The replay harness self-labels every run with the config it measured and prints per-stage timings.

```bash
# Baseline (serial), 3 runs per outlier, plus the bare-model floor:
cd core && python scripts/replay_outliers.py --baseline --runs 3

# With the fix on:
UNDERSTANDING_CONCURRENT_WITH_PROPOSE=true \
  docker compose exec core python scripts/replay_outliers.py --runs 3
```

**Reading the output:**
- `plan` / `x` — total planner time and how many planner calls (`x>1` = re-planned or corrective retry).
- `cache%` — share of prompt tokens served from prefix cache (OpenAI only).
- detail line `(u=.. r=..)` — the understand vs retrieval split inside `understand_retrieve`.
  With the fix on, `understand_retrieve` collapses to ~retrieval-only while `u` stays ~1.3 s
  (it now runs *during* propose).

**Acceptance criteria for enabling in production:**
1. `plan`/`wall` drop by ~the understand time on non-re-plan turns.
2. No scheduling/status turn's `say:` line answers from a document (read them).
3. Run against the **full** golden dataset (not just the 20 outliers) and diff responses —
   no regression vs the serial run.

---

## 9. Next lever (not yet done)

With `understand` off the critical path, the remaining tail is the `x > 1` turns. The clearest
safe win is the **corrective-retry over-fire**: a no-tool `respond` turn (e.g. "my budget is under
a crore") pays a full second `propose` call because `_hinted_tool` fires on "budget/crore" even
though the model already answered correctly from retrieved knowledge. Scoping: skip the corrective
retry when the model produced a grounded `respond` **and** the hinted tool is a read-only lookup
already covered by the retrieved knowledge — without touching the confirm/act safety net the retry
exists for. This is provider-independent (it hurts on Cerebras too, with no cache to mask it).

---

## 10. Changed files

| File | Change |
|---|---|
| `core/app/config.py` | Added `understanding_concurrent_with_propose` flag; removed dead `planner_mode` |
| `core/app/engine/planner/planner.py` | `_understand_and_retrieve(skip_understanding=…)`; concurrent understand+propose in `plan()`; docstring fix |
| `core/app/engine/planner/turn_runner.py` | Telemetry fix (`planner_ms`/`planner_calls` accumulate, not overwrite) |
| `core/app/providers/llm/openai_provider.py` | Capture `cached_tokens` |
| `core/app/engine/planner/responder.py` | Docstring corrected (responder is never skipped) |
| `core/scripts/replay_outliers.py` | `cache%` column, config/flag banner, `(u= r=)` split, "testing each lever" guide |
| `core/tests/test_concurrent_understanding.py` | New: pins the `skip_understanding` seam and the serial path |

**Tests:** `cd core && python -m pytest tests/ -q` → 31 passed.
