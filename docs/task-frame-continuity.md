# Task‑Frame Continuity (Explicit Task Stack for Multi‑Turn Workflows)

**Status:** proposed
**Owner:** engine / planner
**Fixes:** multi‑step tasks (book / reschedule / cancel / callback / register) that are
*forgotten mid‑flow* — the agent collects a time, asks the customer to verify identity,
and then loses the booking task itself, ending in "I don't have the details" or a
needless escalation.

> Companion to [entity-slot-memory.md](./entity-slot-memory.md). Both documents are the
> same theme: **make conversational state explicit instead of emergent.** Entity slots fix
> *what the call is about*; this fixes *what the call is trying to do*.

---

## 1. The problem

A workflow like *reschedule my visit to the 12th at 4 PM* spans several turns:

```
collect the time  ->  read it back  ->  (interruption: verify identity)  ->  resume  ->  execute
```

In production we observe the task being dropped at the interruption. The customer verifies
successfully and is then told the agent no longer has the booking details, or is
transferred to a human for a task the agent was one step from completing. From the
customer's side the agent "forgot what we were doing."

This is **not** the language model losing context. The transcript still contains every
turn. It is the **orchestrator** losing the task, because there is no single object that
represents it.

---

## 2. The reason: task state is fragmented across ~10 session keys

There is no first‑class "task in progress." A multi‑step task is **reconstructed every
turn** from a scatter of independent Redis keys, each written and cleared in a different
block of the ~2,000‑line [`process_turn`](../core/app/engine/planner/turn_runner.py):

| Session key | Part of the task it holds | Touched around |
|---|---|---|
| `pending_action` | staged tool + params (+ `strikes`, `blocked_by`) | 1326, 1456, 1537, 1619 |
| `wanted_tool` | what the customer rang about | 1154, 1584 |
| `registration` | goal + name + phone (unregistered callers) | 1066, 1143, 1167 |
| `offered_slots` / `offered_centre_id` | the times the customer may pick from | 596 |
| `centre_id` / `appointment_id` | ids the model reliably loses | 574, 587 |
| `identity_verified` | a prerequisite of write tools | 590, 1175 |
| `strikes` | blocked‑attempt counter | 1535 |
| `last_registration_prompt`, `human_requests`, `escalated`, `triaged_human_request` | interruption bookkeeping | scattered |

The task is "remembered" only if **all of these line up correctly across every
interruption**. There is nowhere that records, as one fact:

> *"The customer is rescheduling appointment X to slot Y, pending identity verification."*

That statement is an **emergent property** of ten keys and their write‑ordering. Every
ordering between them is a chance to drop it — which is exactly why the file carries dozens
of `THE BUG THIS FIXES:` comments that are all, underneath, **the same bug**: an
interruption dropped part of the task state.

### 2.1 The exact failure, traced through the code

*"reschedule to the 12th at 4pm"* → asks for identity → forgets:

1. Customer states the change → `reschedule_appointment` staged in `pending_action`, read
   back to the customer. **The date and time now live only inside `pending_action.params`.**
2. Customer says *"yes"* → the rule gate **blocks pending identity**
   ([~1198](../core/app/engine/planner/turn_runner.py)); the agent asks for the last four
   digits.
3. Customer gives the digits → `verify_identity` is forced
   ([~1224](../core/app/engine/planner/turn_runner.py)), runs, and the chain re‑plans to
   resume.

Resuming now depends on **three independent things all still being intact**, each a
documented single point of failure:

- **`pending_action` still holds the reschedule** — but the `strikes` logic *drops it
  after two blocks* ([~1536](../core/app/engine/planner/turn_runner.py)). Identity block +
  one busy‑slot block = two strikes, and the consented date/time are discarded → "I don't
  have the details" → escalate. The incident is narrated verbatim at
  [~1515](../core/app/engine/planner/turn_runner.py).
- **`slot_start` is reconstructable** — it exists only inside `offered_slots`, and
  `_pick_offered_slot` requires the *day to still match*
  ([~879](../core/app/engine/planner/turn_runner.py)). If any re‑search replaced
  `offered_slots`, the 4 PM is unrecoverable → staging reports `missing` → nothing is
  staged ([~1618](../core/app/engine/planner/turn_runner.py)) → task forgotten.
- **`appointment_id` survived** on the session — if `_remember_ids` did not capture it,
  `_fit_params` cannot refill it and the release fails on a schema error.

Any single gap and the task evaporates. The engine is not losing *memory*; it is losing
*the task frame*, because one does not exist.

### 2.2 Why the current design produces this

The engine is right not to trust the LLM to hold task state. But the fallback is not a
structured state machine — it is **imperative patches over shared mutable keys**. Two
abstractions are missing:

1. **No first‑class Task object** with an explicit lifecycle. "In progress" is *inferred
   from* the keys above, never *stored*.
2. **No interruption / resume model.** Identity verification and registration are
   *interruptions* of the real task, but they are implemented by mutating the same flat
   state the task depends on. So "resume" means "hope the keys still line up," not "pop the
   sub‑task and continue the parent."

Every new interruption ordering is therefore a new bug, fixed with another guard — which is
the maintenance signature this document exists to end.

---

## 3. The solution: an explicit Task Frame + Task Stack

Make the in‑progress task a real object that the **orchestrator owns**. The LLM proposes
*intent* and *slot values*; the engine owns *what is in progress and what to resume*. This
is the standard dialog‑state / form‑filling‑with‑digression pattern (Section 6).

### 3.1 The Task frame

```python
# core/app/engine/planner/tasks.py
from dataclasses import dataclass, field

@dataclass
class Task:
    id: str                      # uuid4, so trace/logs can follow one task across turns
    type: str                    # book_appointment | reschedule | cancel | callback | ticket | register
    status: str = "collecting"   # collecting | awaiting_confirmation | blocked | ready | done | abandoned
    slots: dict = field(default_factory=dict)      # {appointment_id, slot_start, centre_id, date, time, ...}
    required: list[str] = field(default_factory=list)   # slots that MUST be filled before it can run
    prerequisites: list[str] = field(default_factory=list)  # ["identity_verified", "customer_exists"]
    confirmed: bool = False       # customer has said yes to the read-back
    attempts: int = 0             # blocked-execution counter (replaces `strikes`)
    created_turn: int = 0
    updated_turn: int = 0

    def missing_slots(self) -> list[str]:
        return [s for s in self.required if not self.slots.get(s)]

    def is_ready(self, satisfied_prereqs: set[str]) -> bool:
        return (not self.missing_slots()
                and self.confirmed
                and all(p in satisfied_prereqs for p in self.prerequisites))
```

### 3.2 The Task stack

Stored on working memory, so it survives the turn exactly like `pending_action` does today:

```jsonc
// session (Redis, one per conversation)
{
  "tasks": [                                  // <-- NEW: a stack, bottom = oldest goal
    { "id": "…", "type": "reschedule",
      "status": "awaiting_confirmation",
      "slots": { "appointment_id": "…", "slot_start": "2025-08-12T16:00:00+05:30" },
      "required": ["appointment_id", "slot_start"],
      "prerequisites": ["identity_verified"],
      "confirmed": true, "attempts": 1 }
  ],
  "satisfied_prereqs": ["customer_exists"]    // identity_verified added when it clears
}
```

Interruptions **push a child and pop back to the parent**:

```
[reschedule: slots={appt, 12th@4pm}, confirmed=true, prereq=identity]   <- parent, NEVER dropped
   push ->  [verify_identity]        (interruption / prerequisite)
   verify_identity succeeds -> satisfied_prereqs += "identity_verified"; pop
   pop  ->  parent resumes: is_ready() -> True -> execute reschedule
```

**The invariant that fixes the bug:** *a task is removed only when it reaches `done` or the
customer explicitly abandons it — never as a side effect of an interruption.* Identity
verification is modeled as a **prerequisite of** the reschedule (a child on the stack), not
as a mutation that competes with it. When it completes, the parent is still there with its
slots intact.

### 3.3 Lifecycle

```
                 customer states a goal
                         │
                         ▼
                   ┌───────────┐   all required slots filled
   fill slots ───▶ │ collecting │ ─────────────────────────┐
   (planner         └───────────┘                           ▼
    entities,             ▲                          ┌──────────────────────┐
    tool results,         │ customer changes a slot  │ awaiting_confirmation │
    session ids)          └──────────────────────────└──────────────────────┘
                                                             │ customer says "yes"
                          prerequisite unmet                 ▼
                    ┌──────────┐   push child           ┌────────┐  is_ready()  ┌──────┐
                    │ blocked  │◀────────────────────── │ ready  │ ───────────▶ │ done │
                    └──────────┘   pop child, resume    └────────┘   execute    └──────┘
                         │ attempts >= 2 with same params
                         ▼
                    re-elicit the specific missing slot / offer alternative
```

One place owns each transition, replacing the scattered guards:

- **`collecting`** — fill slots from `proposal.understood["entities"]`, tool results
  (`_remember_ids`), and session ids. This subsumes the `_fit_params` / `offered_slots` /
  `wanted_tool` juggling.
- **all `required` filled** → `awaiting_confirmation`, read back once.
- **customer confirms** + prerequisites met → `ready` → execute.
- **blocked** → `task.attempts += 1` (replaces `strikes`); after 2 identical failures,
  re‑elicit the *specific* missing/invalid slot instead of dropping the whole task.

### 3.4 The keys map cleanly onto it (so this is wrap‑then‑collapse, not a rewrite)

| Today | Becomes |
|---|---|
| `pending_action` | `task.status == awaiting_confirmation` + `task.slots` |
| `wanted_tool` | the task at the top of the stack |
| `registration` | a `register` task (child), popped on `create_customer` success |
| `offered_slots` / `offered_centre_id` | task‑scoped `candidate_slots` |
| `centre_id` / `appointment_id` | `task.slots["centre_id"]` / `["appointment_id"]` |
| `identity_verified` | a `prerequisite`, satisfied by a child `verify_identity` task |
| `strikes` | `task.attempts` |
| `_pick_offered_slot`, `_fit_params` slot‑filling | the `collecting` transition |

---

## 4. How it fixes the traced failure

Same transcript as §2.1, under the new model:

1. *"reschedule to the 12th at 4pm"* → push `Task(type=reschedule,
   required=[appointment_id, slot_start], prerequisites=[identity_verified])`. Slots filled
   from the offered list; `status=awaiting_confirmation`; read back.
2. *"yes"* → `task.confirmed = True`. `is_ready()` is False (identity prerequisite unmet) →
   push child `verify_identity`, ask for the digits. **The parent, with its slots and
   `confirmed=True`, stays on the stack untouched.**
3. Digits → `verify_identity` runs → `satisfied_prereqs += "identity_verified"`, pop the
   child → parent resumes → `is_ready()` True → execute the reschedule. Done in the same
   turn.

No combination of key‑ordering can drop the task, because the date, time, appointment and
consent are all fields of one object whose only exits are `done` and `abandoned`. The
`strikes`/`offered_slots`/`pending_action` race conditions cannot occur — there is a single
source of truth.

---

## 5. Integration points

| File | Change |
|---|---|
| `core/app/engine/planner/tasks.py` | **new** — `Task`, the stack helpers (`push`, `top`, `pop`, `resume`), `is_ready`, slot‑fill |
| `core/app/engine/memory/working.py` | `_empty()` gains `"tasks": []`, `"satisfied_prereqs": []` |
| `core/app/engine/planner/turn_runner.py` | replace the scattered key logic block by block with stack operations (see migration below) |
| `core/app/engine/planner/prompts.py` | render the top task as an `ACTIVE TASK` block so the model elicits the *missing slot*, not a fresh question |
| `core/app/engine/planner/turn_context.py` | expose `top_task` on the context for the prompt |

The interruption seam already exists: `CHAINABLE_TOOLS` includes `verify_identity` and
`create_customer` ([~335](../core/app/engine/planner/turn_runner.py)), which is precisely
"run the prerequisite, then resume the parent in the same turn." The stack formalizes what
the chain loop approximates today.

---

## 6. How the industry does this

Every mature dialog platform solved exactly this with an explicit task/stack model — none
rely on the model to remember the in‑flight task:

- **Microsoft Bot Framework** — a first‑class **dialog stack** (waterfall / component
  dialogs) with built‑in interruption handling. The canonical version of this design.
- **Rasa** — **Forms** with `required_slots` and slot mappings, plus documented
  "unhappy‑path" / digression handling that returns to the form.
- **Google Dialogflow CX** — Flows / Pages with **parameters (form filling)** and native
  **digression** so an interruption resumes the page it left.
- **Amazon Lex** — intents with slot elicitation and interruption/resume.
- **RavenClaw / TrIPS** (the academic lineage) — a **dialog task tree + agenda stack**,
  i.e. push sub‑task, pop, resume — the direct ancestor of this proposal.
- **Modern LLM agents (Sierra, Parloa, Cognigy)** — LLM for NLU + phrasing, a
  **deterministic workflow / state machine** for what is in progress. The division of
  labour is the point: *the LLM understands and speaks; the orchestrator owns and resumes
  state.*

Your engine already holds this philosophy — deterministic code validates every LLM
proposal. The missing piece is giving *the task in progress* the same first‑class,
deterministic representation you already gave tools, rules, and the customer state machine.

---

## 7. Migration (incremental, low‑risk)

Do **not** rewrite `process_turn` in one pass. Wrap, then collapse:

1. **Introduce `Task`/stack alongside** the existing keys. On each turn, build a `Task`
   from `wanted_tool` + `pending_action` + `offered_slots` (read‑only projection). Ship it
   dark — logged to the decision trace, not yet driving behaviour.
2. **Verify parity** on real transcripts: assert the derived task matches what the current
   keys imply, so the projection is trustworthy before it is authoritative.
3. **Flip resume to the stack.** Make the resume decision (`is_ready()` → execute) read the
   `Task`, and stop clearing `pending_action` on the identity/registration interruptions.
   This alone fixes the reported bug.
4. **Collapse the keys** one at a time into `task.slots` / `task.attempts` /
   prerequisites, deleting the corresponding guards as each is subsumed. The `THE BUG THIS
   FIXES` comments become tests (Section 8).

Gate behind a flag so it can be toggled without redeploy:

```python
# core/app/config.py  (Settings)
task_stack_enabled: bool = False   # explicit Task stack for multi-turn workflow continuity
```

---

## 8. Testing

Turn the historical incidents into regression tests. Replay each transcript through
`process_turn` and assert the task survives:

```python
async def test_reschedule_survives_identity_interruption():
    # "reschedule to the 12th at 4pm" -> "yes" -> blocked on identity ->
    # "1234" -> verify_identity succeeds -> reschedule executes in the same turn
    ...
    assert session["tasks"] == []                      # popped: done
    assert any(o.tool_name == "reschedule_appointment" and o.status == "success"
               for o in executed)

async def test_task_not_dropped_after_two_blocks():
    # identity block + one busy-slot block must NOT discard the collected date/time
    task = top_task(session)
    assert task.type == "reschedule"
    assert task.slots["slot_start"]                    # still held after 2 attempts

async def test_registration_is_a_child_task_that_pops():
    # unknown caller asks to book -> register (name, phone) -> book resumes automatically
    ...
```

Unit‑test `tasks.py` in isolation (`is_ready`, `missing_slots`, push/pop/resume) since it
is pure and deterministic.

---

## 9. What this does *not* change

- **The confirmation gate stays.** `awaiting_confirmation` → `ready` still requires the
  customer's yes; write tools never execute without a read‑back. The stack makes the gate
  *more* reliable, not looser.
- **Rules still govern.** Prerequisites are satisfied by the same rule/tool machinery
  (`verify_identity`, escalation policy); the stack only decides *what to resume*, never
  *what is allowed*.
- **The LLM's role is unchanged** — it proposes intent and slot values. It simply stops
  being the (unreliable) place task continuity is expected to live.
