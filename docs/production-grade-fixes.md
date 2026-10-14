# Production‑Grade Fixes — Conversation State in the Voice Agent

**Status:** living document · capstone for [entity-slot-memory.md](./entity-slot-memory.md) and
[task-frame-continuity.md](./task-frame-continuity.md)
**Scope:** every multi‑turn "the agent forgot / answered the wrong thing" bug observed in
testing, its root cause in the code, and the fix.

---

## 0. The one root cause behind all of these

Every bug in this document is the same disease wearing a different mask:

> **Critical conversation state is emergent and scattered, and the LLM is expected to be
> the memory.**

State lives in a transcript that gets truncated and evicted, or in a dozen ad‑hoc session
keys with fragile write‑ordering, or nowhere at all. The model is then asked to *recall* or
*reconstruct* it — and it guesses, because a language model fills gaps with plausible text.

The engine already follows the right philosophy everywhere else: deterministic code
validates every LLM proposal (tools, rules, the state machine). These fixes extend that
same principle to **conversation state** — the one thing still left to the model to
remember.

**The division of labour every mature system uses** (Microsoft Bot Framework, Rasa,
Dialogflow CX, Sierra, Parloa): *the LLM does understanding and phrasing; the orchestrator
owns state, retrieval, tools, and resumption.*

What this is **not** (three tempting non‑solutions, each ruled out during investigation):

- **Not a materialised slot table.** Availability is correctly computed on demand from
  working hours − holidays − bookings ([services/scheduling.py](../core/app/services/scheduling.py));
  a slot table is a cache‑invalidation liability and would not touch these bugs.
- **Not prompt optimisation.** The base prompt is already maximal and rule‑dense, and the
  code repeatedly documents that "prompting did not hold." You cannot prompt in state that
  is not in the context window.
- **Not "history already covers it."** History is a lossy, size‑capped, evicted transcript
  that even carries the model's own mistakes forward. It is *what was said*, not *what is
  true*.

---

## 1. Status at a glance

| # | Bug | Fix | Status |
|---|---|---|---|
| 1 | Follow‑up retrieved the wrong variant (2.5 BHK → 2/3 BHK cost sheet) | Entity focus slots + focus‑term ranking | ✅ **implemented** |
| 2 | Slot date hallucinated / lost ("May 17th", "I don't have the exact date") | Surface `offered_slots` into the prompt | ⛔ pending (Phase A) |
| 3 | Document retrieval fires on scheduling / decline turns ("8 chunks" on "16th", "No thanks") | Fix the `needs_knowledge` gate | ⛔ pending (Phase A) |
| 4 | False "no slots on the 16th" without searching | Force `search_slots` on a named, uncovered day | ⛔ pending (Phase A) |
| 5 | Task forgotten across an interruption (reschedule after identity; booking dropped after registration) | Explicit Task frame + stack | ⛔ pending (Phase B) |

---

## 2. Bug 1 — Wrong variant retrieved on a follow‑up ✅ DONE

**Symptom.** "I want a 2.5 BHK" → "can you give me both cost breakdowns" returned the 2/3
BHK cost sheet.

**Why.** Retrieval searched the *raw current utterance*, which held no "2.5 BHK". The long,
number‑dense cost sheets always out‑rank the one document that lists every configuration,
and the prompt tells the model retrieved chunks are the only quotable material.

**Solution (shipped).** The planner already extracts entities every turn under its
`understood` contract; they are now persisted to working memory and reused:

- [focus.py](../core/app/engine/planner/focus.py) — `merge_entities`, `contextual_query`,
  `focus_terms`. Uses a **blocklist** (`NON_SUBJECT_SLOTS`) not an allowlist, so
  `flat_type` / `bhk` / `unit_type` are all captured whatever the model names them.
- Retrieval prepends the focus to the query **and** applies a `FOCUS_BOOST` substring
  ranking on chunk title+content ([retrieval.py:242](../core/app/engine/rag/retrieval.py)) —
  this is what settles a 1.000‑vs‑0.992 race between the wrong and right chunk.
- Surfaced to the model as a `CURRENT FOCUS` block ([prompts.py:393](../core/app/engine/planner/prompts.py)).
- Merged from the **final** proposal before save ([turn_runner.py:2076](../core/app/engine/planner/turn_runner.py)).

Full design: [entity-slot-memory.md](./entity-slot-memory.md). **Zero added latency** — no
new model call.

---

## 3. Bug 2 — Slot date hallucinated or lost ⛔ PENDING

**Symptom.**
- "what about 17th?" → *"**May** 17th available at 10 AM"* (the month is invented; today is
  August).
- Later, "what is the exact date?" → *"I don't have that information."*

**Why.** The slot engine returns the correct date. It never reaches the model cleanly:

1. `search_slots` formats each slot with the full date — `"when": "Sunday 17 August at
   4:00 PM"` ([scheduling_tools.py:25](../core/app/engine/tools/builtin/scheduling_tools.py)) —
   and stores it in `session["offered_slots"]`. **That session state is never rendered into
   the prompt** (confirmed: no `offered_slots` block in
   [prompts.py](../core/app/engine/planner/prompts.py) or
   [turn_context.py](../core/app/engine/planner/turn_context.py)).
2. The only copy that enters history is the tool note, **truncated to 600 characters**
   ([turn_runner.py](../core/app/engine/planner/turn_runner.py), `_tool_note`). Eight slots
   is ~720+ chars, so the JSON is chopped mid‑list.
3. History is only the last **12 messages** (`RECENT_TURNS`, ~3 turns), so a slot search a
   few turns back is fully evicted.
4. Booking needs the offset‑aware ISO timestamp (`2026-08-17T16:00:00+05:30`), which is
   **never spoken** and lives only in `offered_slots`.

So the model reconstructs the date from the "17th" it half‑remembers and confabulates the
month; and once it has spoken "May 17th", that wrong sentence is written into history and
fed back next turn.

**Solution.** Surface `offered_slots` as a structured prompt field, exactly like
`active_entities` already is. `search_slots` already writes `session["offered_slots"]` as
`[{"starts_at": ..., "when": "Sunday 16 August at 4:00 PM"}, …]` (`_remember_ids`), so the
`when` string with the full date already exists — it just never reaches the prompt.

**Implementation.**

1. Carry it on the `TurnContext` — [turn_context.py](../core/app/engine/planner/turn_context.py):
   ```python
   @dataclass
   class TurnContext:
       ...
       active_entities: dict = field(default_factory=dict)
       offered_slots: list = field(default_factory=list)      # NEW — times just read out

   async def build_turn_context(db, *, ..., active_entities: dict | None = None,
                                offered_slots: list | None = None) -> TurnContext:   # NEW
       ...
       return TurnContext(
           ..., active_entities=dict(active_entities or {}),
           offered_slots=list(offered_slots or []))           # NEW
   ```

2. Thread it through the planner — [planner.py:77](../core/app/engine/planner/planner.py):
   ```python
   async def plan(db, *, ..., active_entities: dict | None = None,
                  offered_slots: list | None = None) -> Proposal:                    # NEW
       tc = await build_turn_context(
           db, ..., active_entities=active_entities,
           offered_slots=offered_slots)                                              # NEW
   ```

3. Pass it from the session at every `plan()` call site in
   [turn_runner.py](../core/app/engine/planner/turn_runner.py) (the main call plus the
   corrective / chain / triage re‑plans — the same four sites that already pass
   `active_entities`):
   ```python
   proposal = await plan(
       db, ...,
       active_entities=session.get("active_entities") or {},
       offered_slots=session.get("offered_slots") or [],                             # NEW
       ...)
   ```

4. Render it in the prompt, right after the `CURRENT FOCUS` block —
   [prompts.py:399](../core/app/engine/planner/prompts.py):
   ```python
   if getattr(tc, "offered_slots", None):
       _slots = [f"- {s.get('when')}" for s in tc.offered_slots if s.get("when")]
       if _slots:
           parts.append(_block(
               "SLOTS I JUST OFFERED (quote a line from here exactly — copy the date and "
               "time as written; never reconstruct a month or day yourself)",
               "\n".join(_slots)))
   ```

5. Reinforce with one base‑prompt rule in `BASE_SYSTEM`:
   ```
   WHEN STATING A SLOT, COPY IT VERBATIM.
   Every time you name a visit time, copy a line from SLOTS I JUST OFFERED exactly —
   the day, the date and the month as written. Never assemble a date from memory; the
   month you "remember" is a guess and it will be wrong.
   ```

Grounds every read‑back and every "what's the exact date?" follow‑up. **Zero added
latency** — the block is ≤8 short lines and replaces a truncated, evicted transcript.

---

## 4. Bug 3 — Retrieval fires on the wrong turns ⛔ PENDING

**Symptom.** `8 chunks retrieved` appears on turns that need no document: "no i want
something on 16th", "does the slot is booked", "No thanks you". Worse, brochure chunks then
become the source the model answers from.

**Why.** `needs_knowledge()` ([rag/__init__.py:59](../core/app/engine/rag/__init__.py))
routes by turn shape, which is correct — but the `SCHEDULING` guard recognises weekday words
and "morning/evening" and **not a bare ordinal date**. "something on the 16th" matches no
scheduling pattern, so it falls through to `return True` and retrieves documents. The same
gap lets slot‑detail follow‑ups ("what date", "which day") retrieve brochures.

**Solution.** Two changes, both shape‑based and domain‑neutral.

**Implementation.**

1. Recognise bare ordinal dates as scheduling in
   [rag/__init__.py](../core/app/engine/rag/__init__.py):
   ```python
   # "something on the 16th", "the 16th", "anything on the 17th". The old SCHEDULING
   # pattern knew weekday words and morning/evening but not a bare date number, so a
   # date fell through to document retrieval.
   SCHEDULING_DATE = re.compile(r"\b(?:on |for )?(?:the )?\d{1,2}(?:st|nd|rd|th)\b", re.I)

   def needs_knowledge(text: str) -> bool:
       ...
       if SCHEDULING.search(low) or SCHEDULING_DATE.search(low):   # was: SCHEDULING only
           return False
       ...
   ```

2. Answer slot‑detail follow‑ups from state, not documents. Do this in the **session‑aware**
   `_is_scheduling_aside` ([turn_runner.py:638](../core/app/engine/planner/turn_runner.py)),
   *not* in the pure `needs_knowledge` — it has `session` and can require that slots were
   actually offered, which keeps the pattern from over‑suppressing a genuine question like
   "what are the payment dates?":
   ```python
   # Asking about a slot ALREADY offered — the answer is the offered-slots block, never a
   # document. Guarded on there being offered slots, so it only fires mid-booking.
   _SLOT_DETAIL = re.compile(
       r"\b(?:exact\s+)?(?:date|day|month|time)\b|\bwhich day\b|\bwhat time\b", re.I)

   def _is_scheduling_aside(session: dict, text: str) -> bool:
       low = (text or "").strip()
       if not low or len(low.split()) > 9:
           return False
       if _NOT_A_TIME.search(low):
           return False
       # NEW: a short "what's the exact date / which day" while slots are on the table
       if session.get("offered_slots") and _SLOT_DETAIL.search(low):
           return True
       return _scheduling_in_flight(session) and bool(_TIMEISH.search(low))
   ```
   `_is_scheduling_aside` already drives `skip_retrieval` at the main `plan()` call
   ([turn_runner.py:1101](../core/app/engine/planner/turn_runner.py)), so this immediately
   stops "what is the exact date?" from pulling brochure chunks.

Removes the stray retrievals and stops "something on 16th" being answered from the brochure.

---

## 5. Bug 4 — False "no availability" without looking ⛔ PENDING

**Symptom.** "no i want something on 16th" → *"I'm afraid I don't have any slots on the
sixteenth."* The 16th in fact had 10 AM, 11 AM, 4 PM and 5 PM slots — shown later in the
same call.

**Why.** The turn did **not** call `search_slots` for the named day. It answered "no
availability" from a stale `offered_slots` set (the first search's spread) — and, per Bug 3,
even fired document retrieval instead. The base prompt already says *"SEARCH THE DAY THEY
NAMED"* and *"NEVER OFFER A TIME YOU HAVE NOT LOOKED UP"* — and the model broke both. Prompt
rules are guidance the model drops under pressure; this needs an engine backstop.

**Solution.** A `TOOL_HINTS`‑style backstop in `process_turn`. Both helpers already exist —
`_wanted_day` ([turn_runner.py:957](../core/app/engine/planner/turn_runner.py)) parses the
day the customer named, and `_slot_date` converts an offered slot's ISO start to a date. The
backstop forces a real search when the named day is not already covered.

**Implementation.** Place this just **before** the `if action == "invoke_tool"` execution
gate, so the forced tool runs through the normal path. `search_slots` is in
`CHAINABLE_TOOLS`, so once it runs the chain re‑plans with the results in hand and offers
real times in the same turn:

```python
# Force a real search when they name a day we have not looked up.
#
# The model answers "no availability on the 16th" from a stale offered set — or from
# nothing — breaking its own "SEARCH THE DAY THEY NAMED" rule. Prompt rules do not hold
# under pressure; this makes the search actually happen. It may not claim availability
# for a day it never looked at.
_named_day = _wanted_day(user_input)
_covered = any(_slot_date(s.get("starts_at")) == _named_day
               for s in (session.get("offered_slots") or []))
_searched = "search_slots" in {o.tool_name for o in executed}
if (_named_day is not None and not _covered and not _searched
        and proposal.tool != "search_slots"):
    proposal.action = "invoke_tool"
    proposal.tool = "search_slots"
    proposal.params = {"date_from": _named_day.isoformat(),
                       "date_to": _named_day.isoformat()}
    if session.get("centre_id"):
        proposal.params["centre_id"] = session["centre_id"]
    _note = (f"Customer named {_named_day}; forced search_slots rather than let the "
             f"model claim availability it never looked up.")
    proposal.error = f"{proposal.error} | {_note}" if proposal.error else _note
```

Fixes the false "no slots on the 16th": the engine now looks before it answers.

---

## 6. Bug 5 — Task forgotten across an interruption ⛔ PENDING (structural)

**Symptom.**
- "reschedule to the 12th at 4 PM" → asks to verify identity → then *"I don't have the
  details"* / escalates.
- Mid‑booking, an unknown caller is asked "can I take your name?"; after giving it, the
  booking is **dropped** — "ok" ended the call instead of booking.

**Why.** There is no first‑class object for "the task in progress." A booking/reschedule is
reconstructed every turn from ~10 independent session keys (`pending_action`, `wanted_tool`,
`registration`, `offered_slots`, `centre_id`, `appointment_id`, `identity_verified`,
`strikes`, …), each written and cleared in a different block. An interruption (identity,
registration) mutates that shared state, and resumption depends on every key still lining up
— which it often does not. (`grep "class Task"` → nothing; this is not started.)

**Solution.** An explicit `Task` frame + a **task stack** on working memory, where an
interruption **pushes a child and pops back to the parent**, and the invariant is: *a task
is removed only when it is done or explicitly abandoned — never as a side effect of an
interruption.*

Full design, lifecycle chart, key‑mapping table, and an incremental
wrap‑then‑collapse migration: [task-frame-continuity.md](./task-frame-continuity.md).

---

## 7. Roadmap (impact‑first)

**Phase A — surface slot state + fix the gates (small, no added latency).**
Bugs 2, 3, 4. Self‑contained; touches
[turn_context.py](../core/app/engine/planner/turn_context.py),
[prompts.py](../core/app/engine/planner/prompts.py),
[rag/__init__.py](../core/app/engine/rag/__init__.py),
[turn_runner.py](../core/app/engine/planner/turn_runner.py). Clears almost everything in the
slot/booking transcripts.

**Phase B — the Task frame (structural).**
Bug 5. Migrate incrementally: (1) add `Task`/stack, (2) ship dark as a read‑only projection
logged to the decision trace, (3) flip *resume* to the stack and stop clearing
`pending_action` on interruptions — this step alone fixes the dropped booking, (4) collapse
the scattered keys one at a time.

**Phase C — hardening (later, only if needed).**
LLM query rewrite for messy coreference ("the cheaper one"); a reranker for finer variant
precision; a **holds/reservations** table *only if* concurrent double‑booking is real (this
is about reservation under concurrency, not about storing availability).

---

## 8. What "done" looks like

Re‑running the transcripts that surfaced these bugs:

- "give me both cost breakdowns" after "2.5 BHK" → the 2.5 BHK sheet. *(already true)*
- "what about the 17th?" → the correct month, quoted verbatim from the offered slots.
- "what is the exact date?" → answered from state, no document retrieval.
- "something on the 16th" → a real `search_slots` for the 16th, never a fabricated "no".
- "reschedule to the 12th at 4 PM" → survives the identity check and completes in the same
  flow; registration no longer drops a booking.

The common thread of every fix: **make state explicit and machine‑owned; keep the LLM for
understanding and phrasing.**
