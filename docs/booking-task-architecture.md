# Booking Task — Production Architecture for Multi‑Turn Scheduling

**Status:** proposed · implementation‑ready
**Supersedes the booking half of** [task-frame-continuity.md](./task-frame-continuity.md) with a
concrete, single‑workflow design. Companion to
[production-grade-fixes.md](./production-grade-fixes.md).
**Fixes:** the entire family of booking/scheduling continuity bugs (six reproduced
transcripts, all one root cause).

---

## 0. TL;DR for reviewers

A site‑visit booking is a **multi‑turn transaction**, but the engine currently treats each
turn as independent and *reconstructs* the booking from ~10 scattered session keys. Whether
a call succeeds depends on the **order** the turns happen to arrive in — the definition of an
emergent‑state bug.

The fix is to make the booking an **explicit state machine owned by the orchestrator**. The
LLM extracts entities and phrases sentences; it never decides what state the booking is in.
Six deterministic rules drive the machine, and each one removes one observed failure mode.

This is a **new orchestration layer over existing, correct tools** — `search_slots`,
`create_customer`, `book_appointment`, and the `slot_was_offered` guard are all fine. What is
missing is the machine that *sequences* them.

---

## 1. The evidence: six transcripts, one root cause

All six failing calls are the same flow — *greeting → flat → configuration → "see the flat"
→ offered Saturday 15 Aug 10 AM → customer wants a different day → break*. Four failure modes,
one root:

| # | Failure mode | Seen in |
|---|---|---|
| **A** | Registration interrupts and **drops** the day/time the customer just asked for | Tabs 1, 3, 4, 5, 6 |
| **B** | Offered slots are **stale/sparse** — a later day is never re‑searched, so `slot_was_offered` refuses the real choice | Tabs 1, 3 |
| **C** | **book↔reschedule confusion** — nothing is booked, yet the model relabels "change the time" as reschedule → `slot_start` deadlock | Tab 1 |
| **D** | Booking **never resumes** after registration — customer must ask "is it booked?" to re‑drive it | Tabs 4, 5, 6 |

**Tab 2 is the control case.** It is the only clean run, and the only thing different is
*ordering*: the customer's day request triggered a real day‑scoped `search_slots` **before**
registration interrupted. Same code, same intent, different sequence → different outcome.
That is the proof the bug is emergent state, not logic.

Root cause, stated once:

> **The booking is a transaction with state, but the orchestration reconstructs that state
> per turn from scattered keys, so an interruption or an out‑of‑order turn corrupts it.**

A, C, D are the **orchestration glue**; B is the **conversation‑state model**. Both are named
in this document's fix.

---

## 2. Scope — what this does and does not fix

**Fixes (the whole class):** every multi‑turn booking / reschedule / cancel / slot‑selection
/ registration / "is it booked?" variant, for any day, any configuration, any turn ordering.
This is 100% of the current bug stream, including orderings not yet tested.

**Does not fix (different subsystems, not the current stream):** knowledge‑retrieval ranking,
rules‑engine misconfiguration, STT errors, LLM phrasing quality. Out of scope here.

---

## 3. The design: a BookingTask owned by the orchestrator

### 3.1 The task object

```python
# core/app/engine/planner/booking_task.py
from dataclasses import dataclass, field

BOOKING_STATUSES = ("gathering", "searching", "offering", "picking",
                    "registering", "confirming", "booking", "done", "abandoned")

@dataclass
class BookingTask:
    id: str                                    # uuid4 — follows the task across turns in the trace
    kind: str = "book"                         # "book" | "reschedule"  (decided by FACT, see Rule 3)
    status: str = "gathering"
    slots: dict = field(default_factory=dict)  # configuration, requested_day, time_pref,
                                               # centre_id, verified_slot_start, appointment_id
    prereqs: dict = field(default_factory=lambda: {"customer_exists": False})
    confirmed: bool = False                    # customer said yes to the read-back
    created_turn: int = 0
    updated_turn: int = 0

    def ready_to_book(self) -> bool:
        return (bool(self.slots.get("verified_slot_start"))
                and all(self.prereqs.values())
                and self.confirmed)
```

Stored on working memory (Redis), one per conversation, alongside the existing session:

```jsonc
// session
{
  "booking_task": {
    "id": "…", "kind": "book", "status": "offering",
    "slots": {"configuration": "2.5 BHK", "requested_day": "2026-08-19",
              "time_pref": "afternoon", "centre_id": "…",
              "verified_slot_start": null},
    "prereqs": {"customer_exists": false},
    "confirmed": false
  }
}
```

The existing keys (`pending_action`, `wanted_tool`, `offered_slots`, `centre_id`,
`appointment_id`, `registration`, `identity_verified`) become **projections of, or inputs to,
this one object** — see §6 migration.

### 3.2 The state machine

```
                 customer wants a visit / names a day
                              │
                              ▼
                        ┌───────────┐
       fill slots ─────▶│ gathering │  configuration, requested_day, time_pref
      (LLM entities)    └───────────┘
                              │ requested_day not covered by offered_slots
                              ▼
                        ┌───────────┐   run search_slots(day, day)   [Rule 1]
                        │ searching │──────────────────────────────┐
                        └───────────┘                              ▼
                              ▲                             ┌───────────┐
                 named time not in offered set  [Rule 5]    │ offering  │  read the real slots back
                              └──────────────────────────── └───────────┘
                                                                   │ customer picks a time IN the set
                                                                   ▼
                                                             ┌───────────┐
                                          customer unknown   │  picking  │  verified_slot_start set
                                         ┌───────────────────└───────────┘
                                         ▼                         │ customer already known
                                   ┌─────────────┐                 ▼
                                   │ registering │──create_customer──┐   [Rule 2: child task,
                                   └─────────────┘  prereq satisfied │    booking slots kept]
                                         │ resume, same slots        ▼
                                         └──────────────────▶ ┌────────────┐
                                                              │ confirming │  "shall I book 2 PM?"
                                                              └────────────┘
                                                                    │ customer says yes
                                                                    ▼   ready_to_book()  [Rule 4]
                                                              ┌─────────┐        ┌──────┐
                                                              │ booking │──run──▶│ done │
                                                              └─────────┘        └──────┘
```

### 3.3 The six deterministic rules (each kills one failure mode)

| Rule | Statement | Kills |
|---|---|---|
| **1. Day → day‑scoped search, always** | When `requested_day` is not covered by `offered_slots`, run `search_slots(date_from=date_to=requested_day)` **before** offering anything. Never offer stale slots. | **B** |
| **2. Registration is a child task, not a detour** | Needing a customer pushes a `create_customer` sub‑task; the BookingTask keeps its slots. On completion the booking **resumes automatically** at `confirming`. | **A, D** |
| **3. book vs reschedule by fact** | `kind = "reschedule"` **iff** the customer has an active appointment (`my_appointments` → live/upcoming). Otherwise `book`. Never routed by the model's `intent` label. | **C** |
| **4. Execute only on the invariant** | `book`/`reschedule` fires only when `ready_to_book()` — verified slot ∈ offered set **and** `customer_exists` **and** `confirmed`. Nothing else can. | fabricated "booked" claims (removes the need for the false‑claim guard) |
| **5. Never strip `slot_start` into a schema error** | If a chosen time is not verified, transition to `searching` and re‑search that day — do **not** drop `slot_start` (which then fails the required‑field schema and deadlocks). | **Tab 1 loop** |
| **6. Surface the task + offered slots** | Render the BookingTask and `offered_slots` into the prompt as structured fields, so "is it booked?" and every read‑back are answered from state, not guessed. | date hallucinations, "is it booked?" confusion |

---

## 4. Where each rule lives in the code

The tools are correct; this is orchestration. All changes are in
[turn_runner.py](../core/app/engine/planner/turn_runner.py) `process_turn`, plus a new
`booking_task.py` and small edits to [prompts.py](../core/app/engine/planner/prompts.py) and
[turn_context.py](../core/app/engine/planner/turn_context.py).

### Rule 1 — day‑scoped force search
Helpers already exist: `_wanted_day` ([turn_runner.py:957](../core/app/engine/planner/turn_runner.py))
and `_slot_date`. Before the `invoke_tool` gate:
```python
task = load_booking_task(session)
requested_day = _wanted_day(user_input) or _slot_date(_proposed_slot_start(proposal))
covered = any(_slot_date(s.get("starts_at")) == requested_day
              for s in (session.get("offered_slots") or []))
searched = "search_slots" in {o.tool_name for o in executed}
if requested_day and not covered and not searched:
    proposal.action = "invoke_tool"
    proposal.tool = "search_slots"
    proposal.params = {"date_from": requested_day.isoformat(),
                       "date_to": requested_day.isoformat()}
    if session.get("centre_id"):
        proposal.params["centre_id"] = session["centre_id"]
    task["status"] = "searching"
```
`search_slots` is in `CHAINABLE_TOOLS`, so it runs and the chain re‑plans with real slots in
the same turn.

### Rule 2 — registration as a child task
Today (`turn_runner.py` ~1060–1175) registration mutates `wanted_tool` / `registration` and
hopes the booking is picked up later. Replace with: when `book_appointment` needs a customer,
set `task.status = "registering"` and push `create_customer` (already in `CHAINABLE_TOOLS`).
On its success set `task.prereqs["customer_exists"] = True`, `task.status = "confirming"`, and
**do not clear the booking slots**. The booking resumes on the same turn via the chain.

### Rule 3 — book vs reschedule by fact
Decide `kind` from data, not the label:
```python
def decide_kind(session, executed) -> str:
    if session.get("appointment_id"):
        return "reschedule"
    for o in executed:
        if o.tool_name == "my_appointments" and o.status == "success":
            if (o.output or {}).get("has_upcoming") or (o.output or {}).get("live"):
                return "reschedule"
    return "book"
```
Then force the tool to match `kind`, overriding a model that said "reschedule" for a booking
that was never made. (Tab 1: no appointment exists ⇒ `book`, never `reschedule`.)

### Rule 4 — the booking invariant
Gate execution on `task.ready_to_book()` rather than the scattered `pending_action` +
`verbally_confirmed` + `customer_already_agreed` conditions. This is one predicate in one
place, replacing several interacting ones.

### Rule 5 — no slot_start stripping
Remove the guard that drops an unverified `slot_start` (the one whose note reads *"was not
among the times actually offered … dropped"*). Its job — refuse an unverified slot — is
already done by the `slot_was_offered` **precondition** in
[contracts.py:185](../core/app/engine/tools/contracts.py). On that block, transition the task
to `searching` (Rule 1) instead of stripping the field. This ends the
`'slot_start' is a required property` deadlock ([definitions.py:159](../core/app/engine/tools/definitions.py)
keeps `slot_start` required — correctly).

### Rule 6 — surface state
- Carry `offered_slots` and a compact `booking_task` summary on the `TurnContext` (mirror how
  `active_entities` is already threaded through
  [planner.py:82](../core/app/engine/planner/planner.py) →
  [turn_context.py:43](../core/app/engine/planner/turn_context.py)).
- Render in `build_user_prompt`:
  ```
  ## SLOTS I JUST OFFERED (quote a line exactly — never reconstruct a date)
  - Wednesday 19 August at 1:00 PM
  - Wednesday 19 August at 2:00 PM

  ## BOOKING IN PROGRESS
  configuration: 2.5 BHK · day: Wed 19 Aug · time: afternoon · customer: not yet registered
  ```

---

## 5. Worked example — Tab 1, before and after

**Before (deadlock):** "19th" → registration drops it → books stale **10 AM** → customer says
"afternoon" → model relabels **reschedule** → `slot_start` (19th PM) not in offered set →
guard strips it → schema `'slot_start' required` fails → loops forever.

**After:**
1. "19th … afternoon" → `gathering`: `requested_day=19th`, `time_pref=afternoon`.
2. 19th not in `offered_slots` → **Rule 1** forces `search_slots(19th)` → `offering` with real
   PM slots (12/1/2/3/4).
3. "yes, 2 PM" → `picking`: `verified_slot_start = 2 PM` (it is in the offered set).
4. Customer unknown → **Rule 2** pushes `create_customer`; booking slots kept.
5. Registered → `prereqs.customer_exists = True` → `confirming` → read back "2 PM, shall I
   book?".
6. "yes" → **Rule 4** `ready_to_book()` true → `book_appointment` at 2 PM → `done`.

No reschedule, no stripping, no deadlock — and it matches Tab 2's clean path by construction,
not by luck.

---

## 6. Migration — incremental, low‑risk

Do **not** rewrite `process_turn` in one pass; it holds hard‑won guards.

1. **Introduce `booking_task.py` + the object.** Add `booking_task` to the session default in
   [working.py](../core/app/engine/memory/working.py). Build it each turn as a **read‑only
   projection** of today's keys (`wanted_tool` + `pending_action` + `offered_slots` +
   `registration` + `appointment_id`). Log it to the decision trace. Ship dark.
2. **Verify parity** on the six transcripts: assert the projected task matches what the keys
   imply, so the projection is trustworthy before it drives anything.
3. **Turn on Rule 1 + Rule 5 + Rule 3** (the smallest, highest‑impact set). This alone stops
   Tab 1's deadlock, the stale‑slot offers, and the book/reschedule mislabel — with the task
   still a projection.
4. **Turn on Rule 2 + Rule 4** (registration as child task; the single booking invariant).
   This fixes A and D and lets you delete the registration key‑juggling block.
5. **Turn on Rule 6** (surface state) and delete the false‑claim guard, which is now dead code
   — nothing can fabricate a booking once Rule 4 is the only path.
6. **Collapse** `wanted_tool` / `registration` / `strikes` into the task; delete each guard as
   it is subsumed.

Gate behind a flag so it can be toggled without redeploy:
```python
# core/app/config.py
booking_task_machine: bool = False
```

---

## 7. Testing — turn the six tabs into regressions

Replay each transcript through `process_turn` and assert the task reaches `done` with a real
booking on the day/time the customer actually asked for:

```python
async def test_tab1_afternoon_does_not_deadlock():
    # 19th + afternoon -> day search -> pick 2 PM -> register -> book. No reschedule.
    ...
    assert not any(o.tool_name == "reschedule_appointment" for o in all_tools)
    assert booked_slot.hour in (12, 13, 14, 15, 16)   # an afternoon slot, verified

async def test_registration_does_not_drop_the_day():
    # naming a day then giving a name must still book THAT day
    ...

async def test_is_it_booked_answered_from_state():
    # "is it booked?" before booking -> "not yet", never a document retrieval
    ...
```
Unit‑test `booking_task.py` (`ready_to_book`, transitions, `decide_kind`) in isolation — it is
pure.

**Acceptance:** all six tabs reach a correct booking with **no** deadlock, **no** stale‑slot
offer, **no** book/reschedule mislabel, and **no** "is it booked?" dead end — regardless of
the order the customer registers vs. picks a slot.

---

## 8. Why this ends the whack‑a‑mole

Every bug you have hit is success‑depends‑on‑ordering in the booking flow. A state machine
makes ordering irrelevant: the task holds what has been gathered, the six rules force the
missing step, and execution is gated on one invariant. New orderings can't produce new bugs,
because there is no per‑turn reconstruction left to get wrong.

The principle, one last time: **the LLM understands and phrases; the orchestrator owns and
sequences the transaction.** The tools were never the problem — the missing state machine was.
