# Booking Slot System — Architectural Change (Implementation Record)

**Status:** implemented · flag-gated · live-validated (partial)
**Flag:** `BOOKING_TASK_MACHINE` (env) → `settings.booking_task_machine` (default `False`)
**Implements:** [booking-task-architecture.md](./booking-task-architecture.md) — Rules 1–6, done as
validated increments. Rules 2 & 4 are the "complete the architecture" step.
**Design principle:** *the LLM understands and phrases; the orchestrator owns and sequences
the booking transaction.*

---

## 0. TL;DR

A site-visit booking is a **multi-turn transaction**. The engine used to reconstruct that
transaction every turn from ~10 scattered session keys, so whether a call succeeded depended
on the **order** the turns arrived in (an emergent-state bug). This change introduces an
explicit **`BookingTask`** the orchestrator owns, plus deterministic rules that force the
missing step regardless of turn order.

Everything new is gated behind `BOOKING_TASK_MACHINE`. With the flag **off**, behaviour is the
legacy path unchanged; the task is still built and logged (dark) for parity. With the flag
**on**, the rules sequence the booking. **No legacy guards were deleted** — they remain as
backstops, so flipping the flag off cleanly reverts.

---

## 1. The object — `BookingTask`

New module: [`core/app/engine/planner/booking_task.py`](../core/app/engine/planner/booking_task.py).
Pure, dependency-free (no DB/Redis/tool types), unit-tested via a built-in self-test:

```bash
docker compose exec core python -m app.engine.planner.booking_task   # → "all passed"
```

```python
@dataclass
class BookingTask:
    id: str                 # uuid4, follows the task across turns
    kind: str = "book"      # "book" | "reschedule"  (decided by FACT — Rule 3)
    status: str             # gathering|searching|offering|picking|registering|
                            # confirming|booking|done|abandoned
    slots: dict             # configuration, requested_day, time_pref, centre_id,
                            # verified_slot_start, appointment_id
    prereqs: dict           # {"customer_exists": bool}
    confirmed: bool         # customer said yes to the read-back

    def ready_to_book(self) -> bool:                     # the ONE execution invariant
        return (bool(self.slots.get("verified_slot_start"))
                and all(self.prereqs.values())
                and self.confirmed)
```

Stored on working memory (Redis) as `session["booking_task"]`, **rebuilt every turn** by a
**read-only projection** of the current session + this turn's tool outcomes
(`booking_task.project(...)`). The projection never mutates the session; persistence is the
caller's job. It is always built and attached to the decision trace
(`debug["booking_task"]`), whether or not the flag drives behaviour — so parity can be
observed before/while the machine is on.

Key helpers in the module:
- `decide_kind(session, executed, current_state)` — Rule 3.
- `project(...)` — the projection, incl. sticky-`done` and cancel handling.
- `carry_config(entities)` — key-agnostic capture of the flat configuration.
- `_format_slot(iso)` — one canonical spoken rendering of a slot time.
- `ready_to_book()` — Rule 4's invariant.

---

## 2. The state machine

```
        customer wants a visit / names a day/time
                        │
                        ▼
                 ┌───────────┐  fill slots (config, requested_day, time_pref)
                 │ gathering │
                 └───────────┘
                        │ requested_day/window not covered by offered_slots
                        ▼
                 ┌───────────┐  Rule 1: search_slots(day, day, part_of_day)
                 │ searching │  (locate_centre first if no centre — Rule 1a)
                 └───────────┘
                        ▼
                 ┌───────────┐  read the REAL slots back
                 │ offering  │
                 └───────────┘
                        │ customer picks a time IN the offered set + window
                        ▼
                 ┌───────────┐  verified_slot_start set (Rule 5: if unverified, re-search)
                 │  picking  │
                 └───────────┘
             customer unknown │            │ customer already known
                 ▼            │            ▼
          ┌─────────────┐     │      ┌────────────┐
          │ registering │─────┘      │ confirming │  read back the canonical time
          └─────────────┘  Rule 2    └────────────┘
           create_customer, slots kept       │ customer says yes
           → resume to confirming             ▼  ready_to_book()  → Rule 4
                                        ┌─────────┐      ┌──────┐
                                        │ booking │─────▶│ done │  (terminal, sticky)
                                        └─────────┘      └──────┘
```

---

## 3. The rules, as built, and where they live

All orchestration lives in
[`turn_runner.py`](../core/app/engine/planner/turn_runner.py) `process_turn`, gated by a
local `_machine = get_settings().booking_task_machine`. Tools were not changed.

| Rule | What it does | Key code |
|---|---|---|
| **1. Day/window → search** | When the requested day (or its time-of-day window) is not covered by `offered_slots`, force a **windowed** `search_slots(date_from=date_to=day, part_of_day=…)` before offering/booking. `search_slots` is chainable, so the chain re-plans with real slots the same turn. | machine block before the confirmation gate; `_day_covered_in_window`, `_wanted_part_of_day` |
| **1a. Centre first** | If a search is needed but no centre is established, force `locate_centre` first (once per call) — a centre-less search fails schema validation on multi-centre tenants. | same block |
| **2. Registration is a child task** | Registration engages whenever a booking is **in flight** (offered_slots present / booking is the standing goal), not only when a booking tool is proposed. Name/phone are captured from free text in any order. On `create_customer` success, the booking **resumes to a read-back** with the slot already chosen. | registration block; "Rule 2: registration complete" resume block |
| **3. book vs reschedule by FACT** | `kind = "reschedule"` iff an appointment exists — by `session.appointment_id`, a `my_appointments` result this turn (`has_upcoming`/`live`), **or the customer's `current_state == "site_visit_scheduled"`**. Never by the model's intent label. A model that mislabels "change the time" as a reschedule when nothing is booked is overridden to `book`. | `decide_kind`; Rule 3 remap in the machine block |
| **3a. Reschedule prerequisite** | For a reschedule with no known appointment id, force `my_appointments` first (once per call) to load the id + centre — the reschedule analog of registration. | `_needs_appt` in the machine block |
| **4. Execute only on the invariant** | A booking/reschedule fires only when `ready_to_book()` (verified slot + customer + consent). This **cross-checks the legacy consent path** (`already_confirmed`) so it never blocks a booking the legacy gate rightly allows; it holds and reads back otherwise. Gated behind the flag → revertible. | Rule 4 block at the execute site |
| **5. No `slot_start` stripping deadlock** | An unverified/out-of-window chosen time triggers a re-search (Rule 1) rather than being dropped into a required-field schema error. | `_needs_search`'s `(_wants_write and not _slot_verified)` clause |
| **6. Surface state** | The `BookingTask` is rendered into the prompt as **BOOKING IN PROGRESS — the single source of truth**; the `RETRIEVED KNOWLEDGE` block is scoped to product/price/policy so it no longer out-ranks booking state. Booking-status questions ("is it booked?", any phrasing) **skip retrieval** and are answered from state. | `prompts.build_user_prompt`; `_asks_booking_status`; `skip_retrieval` in `process_turn` |

### Supporting reliability fixes (also flag-related / general)

- **Weekday parsing** in `_wanted_day` ("tuesday", "next friday") — the requested day is no
  longer lost when named by weekday.
- **Time-of-day window persistence** (`session["time_pref"]`) — a window named before a
  registration detour survives it; cleared on booking success.
- **Canonical time rendering** — `verified_slot_start` is shown as one fixed string
  (`Tuesday 18 August at 2:00 PM`) so every turn quotes the same time instead of re-deriving
  it from a raw ISO timestamp and disagreeing with itself.
- **Post-booking guard checks the time, not just the date** — if the spoken hour ≠ the booked
  slot's hour, the reply is replaced with the real booked slot (deterministic, checks agent
  output against tool output).
- **Sticky `done`** — a completed booking keeps reporting as `done` (carrying its slot,
  consent, kind) on later turns, until a fresh booking supersedes it or the visit is
  cancelled. Fixes "is it booked?" being answered wrong one turn after a real booking.
- **Null-param strip** in `_fit_params` — the model's stray `{"city": null}` no longer fails
  schema validation ("city: None is not of type string"); a null is treated as "not
  provided" for every tool.
- **`locate_centre` city fallback** — a city that matches nothing (e.g. "Bangalore" vs the
  stored "Bengaluru") returns all active centres instead of dead-ending the booking.

---

## 4. Files changed

| File | Change |
|---|---|
| `core/app/engine/planner/booking_task.py` | **New.** The task object, projection, `decide_kind`, canonical formatting, self-test. |
| `core/app/config.py` | Added `booking_task_machine: bool = False`. |
| `core/app/engine/memory/working.py` | Added `booking_task: None` to the session default. |
| `core/app/engine/planner/turn_context.py` | Carries `booking_summary` on `TurnContext`. |
| `core/app/engine/planner/planner.py` | Threads `booking_summary` into `build_turn_context`. |
| `core/app/engine/planner/prompts.py` | Renders `BOOKING IN PROGRESS` (single source of truth); scopes `RETRIEVED KNOWLEDGE`. |
| `core/app/engine/planner/turn_runner.py` | The machine block (Rules 1/1a/3/3a/5), Rule 2 registration + resume, Rule 4 execute gate, weekday parsing, window persistence, canonical time guard, sticky-done wiring, `_fit_params` null-strip, booking-status retrieval skip. |
| `core/app/engine/tools/builtin/scheduling_tools.py` | `locate_centre` city fallback. |

---

## 5. Revertibility

- `BOOKING_TASK_MACHINE=false` (or unset) → every rule above is bypassed; the projection is
  still computed and logged but drives nothing; behaviour is the legacy scattered-key path.
- **No legacy guards were removed.** The false-claim guard, `_pick_offered_slot`, the
  hint-park staging, the strike counter, and the registration key-juggling all remain. Under
  the machine they act as **backstops** (cooperating), not competitors.
- `.env` changes require **recreating** the container (`docker compose up -d`), not
  `restart` — env is read once at process start.

---

## 6. Validation status

Validated live against the running Docker app with DB verification (before an LLM-credit
outage forced a provider switch to OpenRouter):

- unknown caller → register mid-call → book — ✅
- "19th afternoon" re-searched to real afternoon slots (not a stale/spread slot) — ✅
- "saturday/tuesday **evening**" books an **evening** slot (17:00), not afternoon — ✅ (DB)
- requested time honoured when free (picked 11 AM → booked 11 AM) — ✅ (DB)
- registration → booking resumes to a read-back → books — ✅ (DB)
- "is it booked?" answered from state across phrasings ("is it booked", "does the appointment
  is booked", "wait did that actually work") — ✅
- booked time == stated time on every turn (canonical rendering + time guard) — ✅
- Rule 4 is a no-op on the happy path (`rule4_held=False`, booking executes) — ✅
- reschedule-by-state chosen for a `site_visit_scheduled` lead; graceful when no live
  appointment exists — ✅
- zero 5xx / tracebacks throughout — ✅

**Pending:** a full replay of the 8 original transcript tabs end-to-end (blocked first by
OpenAI credit exhaustion, then by free-model speed; to be run on `openai/gpt-4o-mini` via
OpenRouter).

---

## 7. Deferred (not done, by choice)

Per the decision to keep the change **revertible**, the doc's final cleanup was **not** done:

- **Deleting** the legacy guards subsumed by Rule 4 (false-claim guard, `_pick_offered_slot`,
  hint-park, strike counter, registration key-juggling).
- Collapsing `wanted_tool` / `registration` / `strikes` into the task.

These remain as backstops so `BOOKING_TASK_MACHINE=false` fully restores prior behaviour.
Doing them would make the `BookingTask` the *sole* path (the doc's Phase 6) but removes the
revert switch.

---

## 8. LLM provider note (operational)

The planner uses **Structured Outputs** (`response_format: json_schema, strict: true`), so
the active model must support it. OpenRouter is wired through the OpenAI-compatible settings
(there is no separate `openrouter` provider in code):

```env
LLM_PROVIDER=openai
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=sk-or-v1-...
LLM_MODEL_MAIN=openai/gpt-4o-mini      # supports structured outputs; fast
EMBEDDING_PROVIDER=mock                 # OpenRouter has no embeddings API
```

Free models are often too slow / rate-limited and may not honour strict structured outputs;
`openai/gpt-4o-mini` (or `gpt-4o`) via OpenRouter is the reliable choice. Embeddings must run
on a real OpenAI-compatible embeddings endpoint or `mock` (retrieval degrades under `mock`,
but the booking flow is unaffected).
