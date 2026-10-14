# Entity‑Slot Memory (Active‑Focus Carry‑Forward)

**Status:** proposed
**Owner:** engine / planner
**Latency cost:** ~0 ms (no extra LLM call)
**Fixes:** follow‑up turns retrieving the wrong variant — e.g. asking about *2.5 BHK*, then "give me both cost breaks" returning the *2 BHK* cost sheet.

---

## 1. The problem this solves

Retrieval in the planner searches on the **raw current utterance only**:

```python
# core/app/engine/planner/planner.py  (~line 102)
rag = await retrieve(db, tenant_id=tenant_id, query=user_input, on_date=as_of_date)
```

When the customer says *"can you provide both cost break"*, that string contains no
`2.5`, no `BHK`, no `north/south`. The retriever therefore cannot prefer the 2.5 BHK
document, and — as the comments in [`retrieval.py`](../core/app/engine/rag/retrieval.py)
already note — the long, number‑dense 2 BHK / 3 BHK cost sheets win the ranking. The
prompt then tells the model those chunks are *"the only factual material you may quote"*
([`prompts.py`](../core/app/engine/planner/prompts.py)), so it answers with 2 BHK.

The conversation *history* still contains "2.5 BHK", and the model can see it — but the
retriever is **stateless** and overrides it with the wrong documents. The subject the
call is currently about (`configuration = 2.5 BHK`) is never carried forward as a
**constraint on retrieval**.

### Why this approach and not query rewriting

An LLM query‑rewrite step ("condense the follow‑up into a standalone query") also fixes
this, but it adds a serial model call (~150–400 ms even on a fast model) on the
retrieval path. Entity‑slot carry‑forward adds **zero** LLM calls because the planner
**already extracts the entities every turn** — we are currently throwing them away. This
is the cheapest, lowest‑risk fix and should ship first. Query rewriting can be layered on
later for messy coreference; the two are complementary.

---

## 2. The idea in one paragraph

Keep a small dictionary of the **entities currently in focus** on the per‑call working
memory (Redis). Update it each turn from what the planner already understood
(`proposal.understood["entities"]`). On the next turn, **prepend those entity values to
the retrieval query** and **surface them to the model** as a "CURRENT FOCUS" block. The
subject of the call now survives across turns and steers retrieval, with no extra latency.

```
Turn 1  "I want a 2.5 BHK flat"
        planner understands entities={configuration: "2.5 BHK"}
        -> session.active_entities = {configuration: "2.5 BHK"}

Turn 2  "can you provide both cost break"
        retrieval query = "2.5 BHK  can you provide both cost break"   <-- carried forward
        -> retrieves the 2.5 BHK cost sheet.  Correct answer.
```

---

## 3. Data model

`active_entities` is a flat `dict[str, str]` stored on the working‑memory session hash.
Keys are short, stable slot names; values are the last stated value.

```jsonc
// session (Redis, one per conversation)
{
  "pending_action": null,
  "identity_verified": false,
  "turn_index": 3,
  "recent_actions": ["action:respond"],
  "escalated": false,
  "active_entities": {                 // <-- NEW
    "configuration": "2.5 BHK",
    "region": "north|south",
    "budget": "1.2 cr"
  }
}
```

**Slot vocabulary is not hard‑coded.** We accept whatever entity names the planner emits
(`configuration`, `region`, `budget`, `product`, …), because the engine is domain‑neutral
by design — the same way `needs_knowledge()` avoids a domain keyword list. A small
**allow‑merge / decay policy** (Section 5) keeps the dict from accumulating stale junk.

---

## 4. Where it plugs in

Three files, all touch points already exist:

| File | Function | Change |
|---|---|---|
| `core/app/engine/memory/working.py` | `_empty()` | add `"active_entities": {}` to the default session |
| `core/app/engine/planner/turn_context.py` | `TurnContext`, `build_turn_context()` | accept `active_entities`, expose on the context, build the contextual search query |
| `core/app/engine/planner/planner.py` | `plan()` | accept `active_entities`, use the contextual query for `retrieve()` |
| `core/app/engine/planner/prompts.py` | `build_user_prompt()` | render a `CURRENT FOCUS` block |
| `core/app/engine/planner/turn_runner.py` | `process_turn()` | pass slots into `plan()`, merge the planner's entities back before `working.save()` |

Data flow:

```
process_turn (turn_runner.py)
  session = working.load()                         # line ~942  — active_entities read here
  plan(..., active_entities=session["active_entities"])
      build_turn_context(..., active_entities)     # exposes tc.active_entities
      search_query = contextual_query(active_entities, user_input)
      retrieve(query=search_query)                 # <-- the fix
      build_user_prompt(tc)                         # renders CURRENT FOCUS
  ... proposal.understood["entities"] produced ...
  session["active_entities"] = merge_entities(session["active_entities"],
                                              proposal.understood.get("entities"),
                                              user_input)
  working.save(session)                            # line ~1849
```

---

## 5. Implementation

### 5.1 `working.py` — add the slot to the default session

```python
# core/app/engine/memory/working.py
def _empty() -> dict[str, Any]:
    return {"pending_action": None, "identity_verified": False,
            "turn_index": 0, "recent_actions": [], "escalated": False,
            "active_entities": {}}                 # <-- NEW
```

Old sessions in Redis won't have the key; every read uses `session.get("active_entities")
or {}`, so this is backward compatible with in‑flight calls.

### 5.2 New helpers — build the query and merge entities

Create `core/app/engine/planner/focus.py`:

```python
"""Active-focus entity slots: what the call is currently about.

Carried on working memory so a follow-up turn retrieves the RIGHT variant. No LLM call —
the planner already extracts these entities; we persist and reuse them.
"""
import re

# Slots worth carrying into the retrieval query. Kept deliberately generic: the engine
# sells anything, so we do not enumerate a domain's nouns. Anything not listed is still
# stored (for the CURRENT FOCUS block) but not injected into the search string, so a
# stray "sentiment"-like entity can't pollute retrieval.
RETRIEVAL_SLOTS = ("configuration", "product", "variant", "model", "region",
                   "location", "plan", "package", "tier", "size")

# A turn that explicitly changes subject should not keep dragging the old entity along.
_RESET = re.compile(r"\b(?:actually|instead|different|other|change|forget that|"
                    r"never mind|not that)\b", re.I)

_MAX_SLOTS = 8
_MAX_VALUE = 60


def merge_entities(prior: dict, incoming: dict | None, user_input: str) -> dict:
    """Carry prior focus forward, overwriting only where the customer stated a new value.

    - a new non-empty value for a slot overwrites the old one (they changed their mind)
    - a slot the customer did not mention this turn is KEPT (still in focus)
    - an explicit subject-change utterance clears carried slots first, so "actually show
      me the 3 BHK instead" does not retain 2.5 BHK
    """
    merged = {} if _RESET.search(user_input or "") else dict(prior or {})
    for name, value in (incoming or {}).items():
        name = str(name).strip().lower()[:40]
        value = _clean(value)
        if name and value:
            merged[name] = value
    # bound the size; keep the most recently written (dicts preserve insertion order)
    if len(merged) > _MAX_SLOTS:
        merged = dict(list(merged.items())[-_MAX_SLOTS:])
    return merged


def contextual_query(active_entities: dict, user_input: str) -> str:
    """Prepend carried entity values to the raw utterance. Pure string work — no model."""
    low = (user_input or "").lower()
    carried = []
    for slot in RETRIEVAL_SLOTS:
        val = _clean((active_entities or {}).get(slot))
        # don't duplicate a value the customer just restated themselves
        if val and val.lower() not in low:
            carried.append(val)
    prefix = " ".join(carried)
    return f"{prefix} {user_input}".strip() if prefix else (user_input or "")


def _clean(v) -> str:
    if not isinstance(v, str):
        v = "" if v is None else str(v)
    return re.sub(r"\s+", " ", v).strip()[:_MAX_VALUE]
```

### 5.3 `turn_context.py` — carry the slots onto the context

```python
@dataclass
class TurnContext:
    ...
    active_entities: dict = field(default_factory=dict)   # <-- NEW

async def build_turn_context(db, *, tenant_id, conversation, user_input,
                             stt_confidence=None, identity_verified=False,
                             pending_action=None,
                             active_entities=None):          # <-- NEW
    ...
    return TurnContext(
        ...,
        pending_action=pending_action, stt_confidence=stt_confidence,
        recent_actions=recent_actions,
        active_entities=active_entities or {})              # <-- NEW
```

### 5.4 `planner.py` — search with the contextual query

```python
from app.engine.planner.focus import contextual_query

async def plan(db, *, tenant_id, conversation, user_input, stt_confidence=None,
               identity_verified=False, pending_action=None, as_of_date=None,
               model=None, skip_retrieval=False,
               active_entities=None):                        # <-- NEW
    ...
    tc = await build_turn_context(
        db, tenant_id=tenant_id, conversation=conversation, user_input=user_input,
        stt_confidence=stt_confidence, identity_verified=identity_verified,
        pending_action=pending_action,
        active_entities=active_entities)                     # <-- NEW

    if not skip_retrieval and needs_knowledge(user_input):
        search_query = contextual_query(active_entities, user_input)   # <-- the fix
        try:
            async with db.begin_nested():
                rag = await retrieve(db, tenant_id=tenant_id, query=search_query,
                                     on_date=as_of_date)
            tc.knowledge = rag.as_context()
            retrieval_meta = {"attempted": True, "query": search_query,   # log the real query
                              "raw_input": user_input, ...}
```

> **Note** `needs_knowledge()` still keys off the **raw** `user_input`, not the expanded
> query — the gate ("is this the kind of turn that needs a document?") is about the
> customer's actual words, and expanding it first could make a bare "yes" look like a
> question. Only the *search string* is expanded.

### 5.5 `prompts.py` — tell the model what's in focus

In `build_user_prompt()`, just before the `CONVERSATION SO FAR` block:

```python
if tc.active_entities:
    lines = [f"- {k}: {v}" for k, v in tc.active_entities.items() if v]
    if lines:
        parts.append(_block(
            "CURRENT FOCUS (what this call is about — keep answering about THIS "
            "unless the customer changes it)", "\n".join(lines)))
```

This is belt‑and‑braces: even if retrieval still returns a mixed set, the model is told
the answer must be about 2.5 BHK, which reinforces the existing "ANSWER THE QUESTION THEY
ASKED" rules already in `BASE_SYSTEM`.

### 5.6 `turn_runner.py` — pass in, merge back

Pass the slots into every `plan()` call (the main call ~line 950, the corrective retry
~line 992, and the triage re‑plans ~1429 / ~1679):

```python
proposal = await plan(
    db, tenant_id=tenant_id, conversation=conversation, user_input=user_input,
    stt_confidence=stt_confidence,
    identity_verified=bool(session.get("identity_verified")),
    pending_action=session.get("pending_action"),
    as_of_date=as_of_date,
    active_entities=session.get("active_entities") or {})   # <-- NEW
```

Then merge the planner's understanding back, right before the existing
`working.save()` (~line 1849, next to the `turn_index` / `recent_actions` updates):

```python
from app.engine.planner.focus import merge_entities

session["active_entities"] = merge_entities(
    session.get("active_entities") or {},
    (proposal.understood or {}).get("entities"),
    user_input)

session["turn_index"] = int(session.get("turn_index", 0)) + 1
session["recent_actions"] = (session.get("recent_actions", []) + [f"action:{action}"])[-5:]
await working.save(conversation.id, session)
```

Use `proposal.understood` from the **final** proposal (after any corrective retry), so a
re‑planned turn stores the entities it actually acted on.

---

## 6. Behaviour walk‑through

| Turn | Customer says | `understood.entities` | `active_entities` after | Retrieval query |
|---|---|---|---|---|
| 1 | "I want a 2.5 BHK flat" | `{configuration: "2.5 BHK"}` | `{configuration: "2.5 BHK"}` | `2.5 BHK I want a 2.5 BHK flat` |
| 2 | "can you provide both cost break" | `{}` | `{configuration: "2.5 BHK"}` (kept) | `2.5 BHK can you provide both cost break` ✅ |
| 3 | "actually show me the 3 BHK instead" | `{configuration: "3 BHK"}` | `{configuration: "3 BHK"}` (reset + overwrite) | `3 BHK actually show me the 3 BHK instead` |

Turn 2 is the bug today. With carry‑forward it retrieves the 2.5 BHK sheet.
Turn 3 shows the reset guard preventing stale carry‑over on an explicit subject change.

---

## 7. Edge cases & how they're handled

- **Customer restates the entity** ("what about the 2.5 BHK price") — `contextual_query`
  skips any carried value already present in the utterance, so the query isn't duplicated.
- **Subject change** ("actually, the 3 BHK") — `_RESET` clears carried slots before
  merging the new one in.
- **Model emits noise as an entity** (e.g. a sentiment word) — only slots in
  `RETRIEVAL_SLOTS` reach the search string; everything else is display‑only in CURRENT
  FOCUS and bounded by `_MAX_SLOTS`.
- **Stale value across a long call** — slots are per‑call (Redis TTL, working memory),
  never promoted to the durable customer record, so nothing leaks between calls.
- **Redis miss / cold session** — `active_entities` defaults to `{}`; behaviour degrades
  to today's (raw‑utterance) retrieval, never errors.
- **`needs_knowledge()` false** (chit‑chat, slot‑picking) — retrieval doesn't run, so the
  query is never built; slots are still merged/kept for the next knowledge turn.

---

## 8. Latency

Zero added model calls. The work per turn is:

- one `dict` merge (`merge_entities`) — microseconds
- one string concatenation (`contextual_query`) — microseconds
- a slightly longer retrieval query string — the extra tokens are negligible against the
  8‑chunk prompt already sent (config note in `config.py`: ~25 ms for 4× the tokens)

No change to the number of LLM round trips. The planner call count is unchanged.

---

## 9. Rollout

Gate behind a settings flag so it can be toggled without a redeploy:

```python
# core/app/config.py  (Settings)
entity_slot_memory: bool = True   # carry active entities across turns into retrieval
```

In `planner.py`, fall back to the raw utterance when disabled:

```python
search_query = (contextual_query(active_entities, user_input)
                if get_settings().entity_slot_memory else user_input)
```

**Suggested order:** ship with the flag `True` in staging, watch `retrieval_meta.query`
vs `raw_input` in the decision trace, confirm the carried entities look right, then enable
in production.

---

## 10. Testing

Unit (`focus.py` is pure and trivially testable):

```python
def test_carries_configuration_into_followup():
    slots = merge_entities({}, {"configuration": "2.5 BHK"}, "I want a 2.5 BHK flat")
    assert slots == {"configuration": "2.5 BHK"}
    q = contextual_query(slots, "can you provide both cost break")
    assert "2.5 bhk" in q.lower()

def test_reset_on_subject_change():
    slots = merge_entities({"configuration": "2.5 BHK"},
                           {"configuration": "3 BHK"},
                           "actually show me the 3 BHK instead")
    assert slots == {"configuration": "3 BHK"}

def test_no_duplication_when_restated():
    q = contextual_query({"configuration": "2.5 BHK"}, "what is the 2.5 BHK price")
    assert q.lower().count("2.5 bhk") == 1
```

Integration: replay the two‑turn transcript through `process_turn` against a seeded
knowledge base containing a 2 BHK and a 2.5 BHK cost sheet, and assert the turn‑2
`retrieval_meta.query` contains "2.5 BHK" and the retrieved `chunk_id`s belong to the
2.5 BHK document.

---

## 11. What this does *not* do (future work)

- **Coreference beyond named entities** — "the cheaper one", "that second option". An LLM
  query‑rewrite step handles these; layer it on top of this, gated to short anaphoric
  turns only, using the fast model (`llm_model_summary`).
- **Metadata‑filtered retrieval** — once `configuration` is a known slot, the SQL in
  `retrieval.py` already has a `doc_type` filter hook; a `configuration` tag on chunks at
  ingestion would let retrieval *filter* rather than merely *bias*.
- **Reranking** — a cross‑encoder over the fused candidates for finer variant precision.
