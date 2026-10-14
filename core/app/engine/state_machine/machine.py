"""Config-driven customer journey state machine.

Responsibilities:
- load the active definition for a tenant (cached per request scope)
- answer "is this transition allowed?" and "what transition does trigger X cause?"
- apply transitions: update customer, manage the interrupting-state stack,
  record a StateTransition row
The machine itself never decides *when* to fire — tools/signals do (later stages).
"""
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Customer, DncEntry, StateMachineDef, StateTransition

MAX_STACK_DEPTH = 2


class StateMachineError(Exception):
    pass


@dataclass
class TransitionResult:
    from_state: str
    to_state: str
    transition_id: uuid.UUID


async def load_definition(db: AsyncSession, tenant_id: uuid.UUID) -> dict:
    row = (await db.execute(
        select(StateMachineDef)
        .where(StateMachineDef.tenant_id == tenant_id, StateMachineDef.is_active == True)  # noqa: E712
        .order_by(StateMachineDef.version.desc())
    )).scalars().first()
    if row is not None:
        return row.definition
    # No StateMachineDef row yet — fall back to the tenant's assigned industry
    # pack's own default journey (Vocal Backend Tasksheet P0, Generic Sprint 1).
    # This sits on the live-call turn path (turn_context.py) with no caller-side
    # try/except, so a pack-lookup failure (DB hiccup, migration not yet applied)
    # must degrade to a safe default here rather than raise — the old behaviour
    # (a hardcoded constant, guaranteed to succeed) must not regress.
    from app.packs import PackRegistry
    try:
        pack = await PackRegistry.for_tenant(db, tenant_id)
    except Exception:
        pack = PackRegistry.get_default()
    return pack.state_machine


def validate_definition(defn: dict) -> list[str]:
    """Returns a list of problems; empty list = valid."""
    problems = []
    states = defn.get("states", {})
    if defn.get("initial") not in states:
        problems.append("initial state missing from states")
    for t in defn.get("transitions", []):
        frm = t.get("from")
        if frm != "*" and not isinstance(frm, list):
            problems.append(f"transition 'from' must be list or '*': {t}")
        if frm != "*":
            for s in frm:
                if s not in states:
                    problems.append(f"unknown from-state {s}")
        to = t.get("to")
        if to != "_return" and to not in states:
            problems.append(f"unknown to-state {to}")
        if not t.get("on"):
            problems.append(f"transition missing 'on': {t}")
    return problems


def find_transition(defn: dict, current_state: str, trigger: str) -> dict | None:
    """First matching transition wins (list order). Wildcards match last only if
    no specific-state transition matched, so specific rules take priority."""
    wildcard_match = None
    for t in defn["transitions"]:
        if trigger not in t["on"]:
            continue
        if t["from"] == "*":
            wildcard_match = wildcard_match or t
        elif current_state in t["from"]:
            return t
    return wildcard_match


def allowed_states_for_tool(defn: dict, tool_name: str) -> list[str]:
    """Which states allow a given tool trigger (used by the planner's candidate filter)."""
    trigger = f"tool:{tool_name}"
    out: set[str] = set()
    for t in defn["transitions"]:
        if trigger in t["on"]:
            if t["from"] == "*":
                return list(defn["states"].keys())
            out.update(t["from"])
    return sorted(out)


async def apply_trigger(
    db: AsyncSession,
    customer: Customer,
    trigger: str,
    *,
    actor: str = "system",
    tool_execution_id: uuid.UUID | None = None,
    confidence: float | None = None,
    reason: str | None = None,
) -> TransitionResult | None:
    """Fire a trigger against a customer. Returns the applied transition, or None
    if no transition matches (not an error — most triggers are no-ops in most states)."""
    defn = await load_definition(db, customer.tenant_id)
    current = customer.current_state

    if defn["states"].get(current, {}).get("terminal") and trigger != "signal:admin_override":
        return None  # terminal states never move (except explicit admin override)

    t = find_transition(defn, current, trigger)
    if t is None:
        return None

    to_state = t["to"]
    stack = list(customer.state_stack or [])

    if to_state == "_return":
        to_state = stack.pop() if stack else defn["initial"]
    else:
        if defn["states"].get(to_state, {}).get("interrupting"):
            if current not in stack and not defn["states"].get(current, {}).get("interrupting"):
                stack.append(current)
                stack = stack[-MAX_STACK_DEPTH:]
        else:
            stack = []  # normal forward movement clears the return stack

    if to_state == current:
        return None

    row = StateTransition(
        tenant_id=customer.tenant_id, customer_id=customer.id,
        from_state=current, to_state=to_state, trigger=trigger,
        tool_execution_id=tool_execution_id, actor=actor,
        confidence=confidence, reason=reason,
    )
    db.add(row)
    customer.current_state = to_state
    customer.state_stack = stack
    if to_state == "do_not_contact":
        await _ensure_dnc_entry(db, customer, trigger)
    await db.flush()
    return TransitionResult(from_state=current, to_state=to_state, transition_id=row.id)


async def _ensure_dnc_entry(db: AsyncSession, customer: Customer, trigger: str) -> None:
    """A customer entering do_not_contact is added to the DNC list automatically —
    one spoken request must protect them everywhere, not just in the journey state."""
    from datetime import datetime, timezone
    existing = (await db.execute(select(DncEntry).where(
        DncEntry.tenant_id == customer.tenant_id,
        DncEntry.phone_normalized == customer.phone_normalized,
        DncEntry.scope == "all"))).scalar_one_or_none()
    if existing is None:
        db.add(DncEntry(tenant_id=customer.tenant_id,
                        phone_normalized=customer.phone_normalized,
                        customer_id=customer.id, scope="all",
                        source="customer_request" if trigger.startswith("signal:") else "system",
                        added_at=datetime.now(timezone.utc)))


async def manual_override(
    db: AsyncSession, customer: Customer, to_state: str, *, actor: str, reason: str
) -> TransitionResult:
    """Supervisor/admin forced state change — bypasses transition table, always recorded."""
    defn = await load_definition(db, customer.tenant_id)
    if to_state not in defn["states"]:
        raise StateMachineError(f"Unknown state: {to_state}")
    current = customer.current_state
    row = StateTransition(
        tenant_id=customer.tenant_id, customer_id=customer.id,
        from_state=current, to_state=to_state,
        trigger="manual:override", actor=actor, reason=reason,
    )
    db.add(row)
    customer.current_state = to_state
    customer.state_stack = []
    if to_state == "do_not_contact":
        await _ensure_dnc_entry(db, customer, "manual:override")
    await db.flush()
    return TransitionResult(from_state=current, to_state=to_state, transition_id=row.id)
