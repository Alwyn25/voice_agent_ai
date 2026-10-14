"""The check library: named, deterministic pre/postcondition checks.

Pack authors compose these by name in tool contracts; they cannot invent new ones
(new checks are engine code, reviewed). Every check returns a CheckResult.
Argument values support "$"-references resolved before the check runs:
  $param.<name>     -> value from the tool call params
  $customer.id      -> current customer's id
  $result.<name>    -> value from the execution result (postconditions only)
"""
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Awaitable, Callable

from sqlalchemy import func, select

from app.db.models import (Appointment, CallbackRequest, Customer, CustomerFact,
                           DncEntry, Location, ShareableDocument, StaffMember, Ticket, Unit)
from app.engine.tools.types import CheckResult, ToolContext
from app.services import scheduling as sched

CheckFn = Callable[[ToolContext, dict], Awaitable[CheckResult]]
_REGISTRY: dict[str, CheckFn] = {}

# Checks that touch NO database — they read only ctx.customer / ctx.identity_verified /
# ctx.extra, already in memory. The executor uses this to decide which preconditions are
# worth dispatching onto their own connection when parallelising (task P2.05): an in-memory
# check runs inline in microseconds, so opening a side session for it would be pure waste.
#
# This is a PERFORMANCE hint only, never a correctness one. If a genuinely in-memory check is
# missing here it merely gets a side session it never uses (a wasted connection); if a
# DB-backed check were wrongly listed it would run inline on the shared session (correct, just
# not parallelised). So keeping it in sync matters for speed, not safety -- but do add new
# no-DB checks here when you write them.
IN_MEMORY_CHECKS: frozenset[str] = frozenset({
    "customer_identified", "identity_verified", "slot_was_offered",
})


def is_in_memory_check(name: str | None) -> bool:
    return name in IN_MEMORY_CHECKS



def _name_tokens(value) -> set[str]:
    return {t for t in "".join(c.lower() if c.isalnum() else " " for c in str(value or "")).split() if t}


def _match_centres(rows, given):
    """Centres whose name contains every word the caller used.

    A SQL LIKE fails the moment punctuation differs — "DSR Highland Greenz — Site Office"
    never matches "DSR Highland Greenz site office" because of one em-dash. The model
    writes names the way a person says them, so matching has to survive that.
    """
    want = _name_tokens(given)
    if not want:
        return []
    exact = [r for r in rows if _name_tokens(r.name) == want]
    return exact or [r for r in rows if want.issubset(_name_tokens(r.name))]



async def _centre_id_from(ctx, value):
    """A centre_id as a UUID, whether the model sent an id or a name.

    Every check that touches a centre needs this. Doing the parse independently in each one
    is how "site office" passed centre_exists and then failed slot_available with the
    scheduler's own error, which the agent read out to the customer.
    """
    if not value:
        return _uuid((getattr(ctx, "extra", None) or {}).get("known_centre_id"))
    parsed = _uuid(value)
    if parsed is not None:
        return parsed
    rows = (await ctx.db.execute(select(Location).where(
        Location.tenant_id == ctx.tenant_id, Location.is_active.is_(True)))).scalars().all()
    hits = _match_centres(rows, value)
    if len(hits) == 1:
        return hits[0].id
    # Fall back to the centre already established on this call. Checks run before the tool,
    # so without this the booking is refused before the tool's own fallback can help.
    return _uuid((getattr(ctx, "extra", None) or {}).get("known_centre_id"))


def check(name: str):
    def deco(fn: CheckFn):
        _REGISTRY[name] = fn
        return fn
    return deco


def available_checks() -> list[str]:
    return sorted(_REGISTRY)


def resolve_refs(spec: Any, params: dict, ctx: ToolContext, result: dict | None = None) -> Any:
    """Recursively resolve $param./$customer./$result. references in check args."""
    if isinstance(spec, dict):
        return {k: resolve_refs(v, params, ctx, result) for k, v in spec.items()}
    if isinstance(spec, list):
        return [resolve_refs(v, params, ctx, result) for v in spec]
    if isinstance(spec, str) and spec.startswith("$"):
        path = spec[1:].split(".")
        root = path[0]
        rest = path[1:]
        if root == "param":
            cur: Any = params
            for k in rest:
                cur = (cur or {}).get(k)
            return cur
        if root == "customer" and ctx.customer is not None:
            return getattr(ctx.customer, rest[0], None) if rest else ctx.customer.id
        if root == "result":
            cur = result or {}
            for k in rest:
                cur = (cur or {}).get(k)
            return cur
        if root == "conversation":
            return ctx.conversation_id
        return None
    return spec


async def run_check(name: str, ctx: ToolContext, args: dict) -> CheckResult:
    fn = _REGISTRY.get(name)
    if fn is None:
        return CheckResult(False, name, f"Unknown check '{name}'")
    try:
        return await fn(ctx, args)
    except Exception as e:  # a check must never crash the executor
        return CheckResult(False, name, f"Check error: {e.__class__.__name__}: {e}")


# ---------------- checks ----------------

@check("customer_identified")
async def _customer_identified(ctx: ToolContext, args: dict) -> CheckResult:
    ok = ctx.customer is not None
    return CheckResult(ok, "customer_identified",
                       "" if ok else "No customer is identified for this conversation")


@check("identity_verified")
async def _identity_verified(ctx: ToolContext, args: dict) -> CheckResult:
    return CheckResult(ctx.identity_verified, "identity_verified",
                       "" if ctx.identity_verified else "Caller identity has not been verified")


@check("centre_exists")
async def _centre_exists(ctx: ToolContext, args: dict) -> CheckResult:
    cid = args.get("centre_id")
    if not cid:
        return CheckResult(False, "centre_exists", "centre_id missing")
    cid_uuid = _uuid(cid)
    if cid_uuid is None:
        # The model frequently passes the centre's NAME here — "Vapi" — because that is
        # what it has been discussing with the customer, despite the tool contract saying
        # UUID. Rejecting outright produced an agent that asked the CALLER for a database
        # id, which no caller has. A centre name is unambiguous within a tenant, so accept
        # it when exactly one active centre matches; the tool implementation resolves the
        # same way before executing.
        if await _centre_id_from(ctx, cid) is not None:
            return CheckResult(True, "centre_exists", "")
        active = (await ctx.db.execute(select(Location).where(
            Location.tenant_id == ctx.tenant_id, Location.is_active.is_(True)))).scalars().all()
        matches = _match_centres(active, cid)
        # A single match is the answer. The line here used to be `if False:` — dead from
        # the day it was written, so a name that matched exactly one centre fell through
        # to the "matches several centres" branch below and the caller was asked for a
        # database id.
        if len(matches) == 1:
            return CheckResult(True, "centre_exists", "")
        if not matches:
            return CheckResult(False, "centre_exists",
                               f"No active centre matches '{cid}'. Call locate_centre first "
                               f"and pass the centre_id it returns.")
        names = ", ".join(m.name for m in matches[:5])
        return CheckResult(False, "centre_exists",
                           f"'{cid}' matches several centres ({names}). Call locate_centre "
                           f"and pass the exact centre_id.")
    row = (await ctx.db.execute(select(Location).where(
        Location.tenant_id == ctx.tenant_id, Location.id == cid_uuid,
        Location.is_active == True))).scalar_one_or_none()  # noqa: E712
    return CheckResult(row is not None, "centre_exists",
                       "" if row else "Centre not found or inactive")


@check("executive_exists")
async def _executive_exists(ctx: ToolContext, args: dict) -> CheckResult:
    eid = args.get("executive_id")
    if not eid:
        return CheckResult(True, "executive_exists", "")  # optional
    eid_uuid = _uuid(eid)
    if eid_uuid is None:
        return CheckResult(False, "executive_exists", f"'{eid}' is not a valid executive id")
    row = (await ctx.db.execute(select(StaffMember).where(
        StaffMember.tenant_id == ctx.tenant_id, StaffMember.id == eid_uuid,
        StaffMember.is_active == True))).scalar_one_or_none()  # noqa: E712
    return CheckResult(row is not None, "executive_exists",
                       "" if row else "Executive not found or inactive")


@check("slot_was_offered")
async def _slot_was_offered(ctx: ToolContext, args: dict) -> CheckResult:
    """The time being booked must be one a slot search actually returned on this call.

    This is the check that makes an invented booking impossible rather than caught.

    The planner is a language model choosing parameters freely, and it will happily
    resolve "tomorrow at four" into a timestamp of its own — one that may be outside
    working hours, already taken, or on the wrong day entirely. Until now the booking was
    attempted, refused by the scheduler, and the agent then re-offered times the customer
    had already picked from. Several days of patching went into detecting that after the
    fact: a search-first redirect, a slot corrector, day matching, a strike counter on the
    parked action. All of them exist because the tool would accept bad input.

    Refusing it here removes the whole class. The engine knows exactly which times it
    offered — search_slots put them on the session — so this is a comparison against fact,
    not a guess about intent. And the failure is useful: the message carries the real free
    times, so the agent has something true to say next.
    """
    offered = (ctx.extra or {}).get("offered_slots") or []
    slot_start = args.get("slot_start")

    if not offered:
        return CheckResult(
            False, "slot_was_offered",
            "No slot search has run on this call, so no time has been confirmed as free. "
            "Call search_slots for the day the customer wants, then offer what it returns.")

    if not slot_start:
        return CheckResult(False, "slot_was_offered", "slot_start is required")

    wanted = _dt(slot_start)
    if wanted is None:
        return CheckResult(False, "slot_was_offered", f"'{slot_start}' is not a valid time")

    for o in offered:
        got = _dt(o.get("starts_at"))
        # Compared as instants, so an offset written differently still matches.
        if got is not None and got == wanted:
            return CheckResult(True, "slot_was_offered", "")

    # Say WHICH DAY the offered times belong to, and notice when it is a different one.
    #
    # The stored slots are whatever the last search returned. A customer who asked for the
    # 16th was refused and then offered five times on the 15th — the day of the previous
    # search — as though those were alternatives to what she wanted. They were not; nobody
    # had looked at the 16th at all. Re-offering yesterday's search as though it answered
    # today's question is the same mistake the check exists to prevent, made by the check
    # itself.
    offered_day = None
    for o in offered:
        d = _dt(o.get("starts_at"))
        if d is not None:
            offered_day = d.date()
            break

    if offered_day is not None and wanted.date() != offered_day:
        return CheckResult(
            False, "slot_was_offered",
            f"Nothing has been checked for {wanted.date():%A %d %B}. The times on hand are "
            f"for {offered_day:%A %d %B}, which is a different day. Run search_slots for "
            f"{wanted.date():%Y-%m-%d} before offering anything.")

    spoken = ", ".join(o.get("when") for o in offered[:5] if o.get("when"))
    day_text = f" on {offered_day:%A %d %B}" if offered_day else ""
    return CheckResult(
        False, "slot_was_offered",
        f"That time is not free{day_text}. Free then: {spoken or 'nothing'}. "
        f"Offer one of these.")


@check("slot_available")
async def _slot_available(ctx: ToolContext, args: dict) -> CheckResult:
    cid, start = args.get("centre_id"), args.get("slot_start")
    if not cid or not start:
        return CheckResult(False, "slot_available", "centre_id and slot_start are required")
    start_dt = _dt(start)
    if start_dt is None:
        return CheckResult(False, "slot_available",
                           f"'{start}' is not a valid date-time. Use the exact value returned "
                           f"by search_slots.")
    centre_id = await _centre_id_from(ctx, cid)
    if centre_id is None:
        return CheckResult(False, "slot_available",
                           f"'{cid}' does not identify one active centre. Call locate_centre "
                           f"and pass the centre_id it returns.")
    slots = await sched.compute_slots(ctx.db, ctx.tenant_id, centre_id,
                                      start_dt.date(), start_dt.date(),
                                      executive_id=_uuid(args.get("executive_id")))
    match = [s for s in slots if s.starts_at == start_dt]
    if match:
        return CheckResult(True, "slot_available", "", {"ends_at": match[0].ends_at.isoformat(),
                                                        "capacity_left": match[0].capacity_left})
    alts = [s.starts_at.isoformat() for s in slots[:3]]
    return CheckResult(False, "slot_available",
                       "Requested slot is not available (taken, outside hours, or too soon)",
                       {"alternatives": alts})


@check("no_conflicting_appointment")
async def _no_conflict(ctx: ToolContext, args: dict) -> CheckResult:
    if ctx.customer is None:
        return CheckResult(False, "no_conflicting_appointment", "No customer in context")
    start = _dt(args.get("slot_start"))
    if start is None:
        return CheckResult(False, "no_conflicting_appointment", "slot_start is not a valid date-time")
    n = (await ctx.db.execute(select(func.count()).select_from(Appointment).where(
        Appointment.tenant_id == ctx.tenant_id,
        Appointment.customer_id == ctx.customer.id,
        Appointment.status.in_(sched.ACTIVE_APPT_STATUSES),
        Appointment.starts_at < start.replace(hour=23, minute=59),
        Appointment.ends_at > start.replace(hour=0, minute=0),
    ))).scalar_one()
    return CheckResult(n == 0, "no_conflicting_appointment",
                       "" if n == 0 else "Customer already has an appointment that day")


@check("appointment_owned_by_customer")
async def _appt_owned(ctx: ToolContext, args: dict) -> CheckResult:
    if ctx.customer is None:
        return CheckResult(False, "appointment_owned_by_customer", "No customer in context")
    appt = await _get_appt(ctx, args.get("appointment_id"))
    ok = appt is not None and appt.customer_id == ctx.customer.id
    return CheckResult(ok, "appointment_owned_by_customer",
                       "" if ok else "Appointment does not belong to this customer")


@check("appointment_active")
async def _appt_active(ctx: ToolContext, args: dict) -> CheckResult:
    appt = await _get_appt(ctx, args.get("appointment_id"))
    ok = appt is not None and appt.status in sched.ACTIVE_APPT_STATUSES
    return CheckResult(ok, "appointment_active",
                       "" if ok else f"Appointment is {appt.status if appt else 'missing'}")


@check("appointment_matches_request")
async def _appointment_matches_request(ctx: ToolContext, args: dict) -> CheckResult:
    """Postcondition: the row actually written has the slot and centre that were validated
    before the tool ran, not just that a row with the right status exists.

    row_exists alone confirms the write happened; it does not confirm the write landed on
    the SAME slot slot_was_offered/slot_available checked, and the tool implementation
    resolves the slot's timezone and the centre id along a separate code path from the one
    the preconditions use (scheduling_tools._parse_slot_start looks up the centre's own
    timezone; this file's _dt always assumes Asia/Kolkata). A resolution bug on either side
    could write a technically-valid appointment at a technically-valid time that is quietly
    not the one the customer agreed to, and row_exists would still call that a success.
    """
    appt = await _get_appt(ctx, args.get("appointment_id"))
    if appt is None:
        return CheckResult(False, "appointment_matches_request", "Appointment not found")

    wanted_start = _dt(args.get("slot_start"))
    if wanted_start is not None and appt.starts_at != wanted_start:
        return CheckResult(
            False, "appointment_matches_request",
            f"Booked row starts at {appt.starts_at.isoformat()}, not the requested "
            f"{wanted_start.isoformat()}")

    if args.get("centre_id"):
        wanted_centre = await _centre_id_from(ctx, args.get("centre_id"))
        if wanted_centre is not None and appt.centre_id != wanted_centre:
            return CheckResult(
                False, "appointment_matches_request",
                f"Booked row is at centre {appt.centre_id}, not the requested {wanted_centre}")

    return CheckResult(True, "appointment_matches_request", "")


@check("customer_profile_updates_persisted")
async def _customer_profile_updates_persisted(ctx: ToolContext, args: dict) -> CheckResult:
    """Postcondition for update_customer_profile: the fields the tool reported as updated
    actually hold those values on the customer row.

    update_customer_profile can touch any subset of seven optional fields per call, so
    unlike row_exists's fixed where-clause, this reads the tool's OWN reported
    updated_fields to know what to check -- row_exists cannot express "whichever fields
    happened to change this call."
    """
    customer_id = args.get("customer_id")
    updated = args.get("updated_fields") or {}
    if not customer_id or not updated:
        return CheckResult(True, "customer_profile_updates_persisted", "")

    row = (await ctx.db.execute(select(Customer).where(
        Customer.tenant_id == ctx.tenant_id, Customer.id == _uuid(customer_id)
    ))).scalar_one_or_none()
    if row is None:
        return CheckResult(False, "customer_profile_updates_persisted", "Customer not found")

    # Only the plain text fields are checked here. The time fields come back formatted
    # ("18:00") for the reply and would need re-parsing to compare against the stored
    # `time` column, and "clamped_to_calling_hours" is a note ABOUT the fields, not a
    # column itself -- both are skipped rather than guessed at.
    text_fields = {"full_name", "email", "location", "region_code", "preferred_language"}
    for key, value in updated.items():
        if key not in text_fields:
            continue
        actual = getattr(row, key, None)
        if actual != value:
            return CheckResult(
                False, "customer_profile_updates_persisted",
                f"{key} is {actual!r} on the row, not the reported {value!r}")
    return CheckResult(True, "customer_profile_updates_persisted", "")


@check("appointments_belong_to_customer")
async def _appointments_belong_to_customer(ctx: ToolContext, args: dict) -> CheckResult:
    """Postcondition for my_appointments: every appointment reported back actually belongs
    to the calling customer.

    The query itself already filters by customer_id -- this re-verifies the OUTPUT
    independently, so a future refactor that loosens or breaks that filter is caught here
    rather than reading one caller's booking to another.
    """
    customer_id = args.get("customer_id")
    # my_appointments splits results across "live" and "past" (there is no single
    # "appointments" list in its result) -- both need checking, since include_past can
    # populate either independently.
    items = (args.get("live") or []) + (args.get("past") or [])
    ids = [a.get("appointment_id") for a in items if isinstance(a, dict) and a.get("appointment_id")]
    if not customer_id or not ids:
        return CheckResult(True, "appointments_belong_to_customer", "")
    rows = (await ctx.db.execute(select(Appointment.id).where(
        Appointment.tenant_id == ctx.tenant_id, Appointment.id.in_([_uuid(i) for i in ids]),
        Appointment.customer_id == _uuid(customer_id)))).scalars().all()
    found = {str(r) for r in rows}
    missing = [i for i in ids if i not in found]
    if missing:
        return CheckResult(False, "appointments_belong_to_customer",
                           f"{len(missing)} of {len(ids)} reported appointment(s) are not this customer's")
    return CheckResult(True, "appointments_belong_to_customer", "")


@check("tickets_belong_to_customer")
async def _tickets_belong_to_customer(ctx: ToolContext, args: dict) -> CheckResult:
    """Postcondition for my_tickets -- same reasoning as appointments_belong_to_customer."""
    customer_id = args.get("customer_id")
    items = args.get("tickets") or []
    ids = [t.get("ticket_id") for t in items if isinstance(t, dict) and t.get("ticket_id")]
    if not customer_id or not ids:
        return CheckResult(True, "tickets_belong_to_customer", "")
    rows = (await ctx.db.execute(select(Ticket.id).where(
        Ticket.tenant_id == ctx.tenant_id, Ticket.id.in_([_uuid(i) for i in ids]),
        Ticket.customer_id == _uuid(customer_id)))).scalars().all()
    found = {str(r) for r in rows}
    missing = [i for i in ids if i not in found]
    if missing:
        return CheckResult(False, "tickets_belong_to_customer",
                           f"{len(missing)} of {len(ids)} reported ticket(s) are not this customer's")
    return CheckResult(True, "tickets_belong_to_customer", "")


@check("slots_are_valid")
async def _slots_are_valid(ctx: ToolContext, args: dict) -> CheckResult:
    """Postcondition for search_slots: nothing offered is already in the past, and nothing
    offered is at zero capacity -- a slot the agent should never be able to read out as
    available in the first place.

    Deliberately does NOT check the slots fall inside the requested date/time window: the
    tool's own result is documented as minimal on purpose (no searched_from/searched_to
    exposed, to stop the model quoting a window boundary as an appointment date), so there
    is no independent value to compare against without re-deriving the tool's own date
    defaulting logic here too -- which would just be a second place for that logic to drift
    out of sync with the first.
    """
    slots = args.get("slots") or []
    now = datetime.now(timezone.utc)
    for s in slots:
        starts = _dt(s.get("starts_at")) if isinstance(s, dict) else None
        if starts is not None and starts < now:
            return CheckResult(False, "slots_are_valid", f"Offered a past slot: {s.get('starts_at')}")
        cap = s.get("capacity_left") if isinstance(s, dict) else None
        if cap is not None and cap <= 0:
            return CheckResult(False, "slots_are_valid",
                               f"Offered a full slot: {s.get('starts_at')} (capacity {cap})")
    return CheckResult(True, "slots_are_valid", "")


@check("centres_are_active")
async def _centres_are_active(ctx: ToolContext, args: dict) -> CheckResult:
    """Postcondition for locate_centre: every centre handed back is actually active and
    belongs to this tenant -- catches a stale id or a cross-tenant leak independently of
    the query that produced it."""
    items = args.get("centres") or []
    ids = [c.get("centre_id") for c in items if isinstance(c, dict) and c.get("centre_id")]
    if not ids:
        return CheckResult(True, "centres_are_active", "")
    rows = (await ctx.db.execute(select(Location.id).where(
        Location.tenant_id == ctx.tenant_id, Location.id.in_([_uuid(i) for i in ids]),
        Location.is_active == True))).scalars().all()  # noqa: E712
    found = {str(r) for r in rows}
    missing = [i for i in ids if i not in found]
    if missing:
        return CheckResult(False, "centres_are_active",
                           f"{len(missing)} of {len(ids)} returned centre(s) are inactive or not this tenant's")
    return CheckResult(True, "centres_are_active", "")


@check("inventory_units_match_query")
async def _inventory_units_match_query(ctx: ToolContext, args: dict) -> CheckResult:
    """Postcondition for inventory_query: every unit it handed back is a real row for this
    tenant (and its resolved project), the price it reported is the price ON that row, and —
    when a budget constraint was applied — the row genuinely satisfies it.

    This is the result-correctness net the numeric answer depends on. The whole point of the
    tool is to move the price comparison off the sentence-writer and onto real rows; this
    proves the comparison was actually done right, so a filter bug (an off-by-one on the band,
    a stale price, a unit from another project) is caught here instead of being read out to a
    caller as fact. Mirrors appointments_belong_to_customer / centres_are_active.
    """
    units = args.get("units") or []
    constraint = args.get("constraint") or {}
    lo = constraint.get("min")
    hi = constraint.get("max")
    for u in units:
        if not isinstance(u, dict):
            continue
        uid = _uuid(u.get("id"))
        if uid is None:
            return CheckResult(False, "inventory_units_match_query",
                               "A returned unit has no valid id")
        row = (await ctx.db.execute(select(Unit).where(
            Unit.tenant_id == ctx.tenant_id, Unit.id == uid))).scalar_one_or_none()
        if row is None:
            return CheckResult(False, "inventory_units_match_query",
                               f"Unit {uid} is not this tenant's")
        # Project isolation: a project-scoped call must not return another project's unit.
        if (ctx.project_id is not None and row.project_id is not None
                and row.project_id != ctx.project_id):
            return CheckResult(False, "inventory_units_match_query",
                               f"Unit {uid} belongs to a different project")
        reported = u.get("price_exact")
        if reported is not None and row.all_in_total != reported:
            return CheckResult(
                False, "inventory_units_match_query",
                f"Unit {uid} reported {reported} but the row holds {row.all_in_total}")
        price = row.all_in_total
        if price is None:
            return CheckResult(False, "inventory_units_match_query",
                               f"Unit {uid} was returned as a priced match but has no price")
        if lo is not None and price < lo:
            return CheckResult(False, "inventory_units_match_query",
                               f"Unit {uid} at {price} is below the applied minimum {lo}")
        if hi is not None and price > hi:
            return CheckResult(False, "inventory_units_match_query",
                               f"Unit {uid} at {price} is above the applied maximum {hi}")
    return CheckResult(True, "inventory_units_match_query", "")


@check("documents_belong_to_tenant")
async def _documents_belong_to_tenant(ctx: ToolContext, args: dict) -> CheckResult:
    """Postcondition for send_document: every document it reports having sent is a real, active
    row for THIS tenant and — when the call is project-scoped — this project.

    A send is an outward action carrying a file URL to a customer's phone. The one thing that
    must never happen is delivering another tenant's or another project's document (a resolver
    bug, a loosened scope filter). The tool's own query already scopes correctly; this re-checks
    the OUTPUT independently, so a future change that breaks the scoping is caught here rather
    than by a caller receiving the wrong project's floor plan. Mirrors
    inventory_units_match_query / appointments_belong_to_customer.
    """
    items = args.get("documents") or []
    ids = [d.get("id") for d in items if isinstance(d, dict) and d.get("id")]
    if not ids:
        return CheckResult(True, "documents_belong_to_tenant", "")
    for raw in ids:
        did = _uuid(raw)
        if did is None:
            return CheckResult(False, "documents_belong_to_tenant",
                               "A sent document has no valid id")
        row = (await ctx.db.execute(select(ShareableDocument).where(
            ShareableDocument.tenant_id == ctx.tenant_id,
            ShareableDocument.id == did))).scalar_one_or_none()
        if row is None:
            return CheckResult(False, "documents_belong_to_tenant",
                               f"Document {did} is not this tenant's")
        if not row.is_active:
            return CheckResult(False, "documents_belong_to_tenant",
                               f"Document {did} is retired and must not be sent")
        if (ctx.project_id is not None and row.project_id is not None
                and row.project_id != ctx.project_id):
            return CheckResult(False, "documents_belong_to_tenant",
                               f"Document {did} belongs to a different project")
    return CheckResult(True, "documents_belong_to_tenant", "")


@check("row_exists")
async def _row_exists(ctx: ToolContext, args: dict) -> CheckResult:
    """Generic postcondition. args: {table, where: {col: value}}"""
    tables = {"appointments": Appointment, "tickets": Ticket, "customers": Customer,
             "dnc_list": DncEntry, "customer_facts": CustomerFact,
             "callback_requests": CallbackRequest}
    model = tables.get(args.get("table"))
    if model is None:
        return CheckResult(False, "row_exists", f"Table not allowed: {args.get('table')}")
    q = select(func.count()).select_from(model).where(model.tenant_id == ctx.tenant_id)
    for col, val in (args.get("where") or {}).items():
        attr = getattr(model, col, None)
        if attr is None:
            return CheckResult(False, "row_exists", f"Unknown column {col}")
        if isinstance(val, str) and _looks_uuid(val):
            val = _uuid(val)
        q = q.where(attr == val)
    n = (await ctx.db.execute(q)).scalar_one()
    return CheckResult(n > 0, "row_exists",
                       "" if n > 0 else f"No matching row in {args.get('table')}")


# ---------------- helpers ----------------

def _uuid(v) -> uuid.UUID | None:
    """Never raises — a malformed id yields None so checks fail with a useful message."""
    if v is None:
        return None
    if isinstance(v, uuid.UUID):
        return v
    try:
        return uuid.UUID(str(v))
    except (ValueError, AttributeError, TypeError):
        return None


def _looks_uuid(v: str) -> bool:
    try:
        uuid.UUID(v)
        return True
    except (ValueError, AttributeError):
        return False


# Slot times are stored with an offset. A model writing "2026-08-16T16:00:00" produces a
# naive one, and the two never compare equal however identical the instant.
_LOCAL_TZ = ZoneInfo("Asia/Kolkata")


def _dt(v) -> datetime | None:
    """Parsed and made offset-aware, so two spellings of the same moment match.

    THE BUG THIS FIXES: a customer asked for Sunday at 4 PM, the search returned Sunday at
    4 PM, and the check refused it — then offered her the very slot it had just rejected.
    "That time is not free on Sunday 16 August. Free then: Sunday 16 August at 4:00 PM."
    Twice. The offered slot carried +05:30 and the requested one did not, so an equality
    test between them was False no matter what.

    Naive input is read as local time, which is what anyone typing a bare timestamp for an
    Indian site office means.
    """
    dt = v if isinstance(v, datetime) else None
    if dt is None:
        try:
            dt = datetime.fromisoformat(str(v))
        except (ValueError, TypeError):
            return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=_LOCAL_TZ)


async def _get_appt(ctx: ToolContext, appt_id) -> Appointment | None:
    # Accept a short reference or fall back to the appointment already under discussion,
    # matching what the tool itself does. Checks run first, so without this the booking is
    # refused before the tool's own resolution can help.
    raw = str(appt_id or "").strip()
    if _uuid(raw) is None:
        # Scoped to THIS customer. Matching a short reference across the whole tenant could
        # select someone else's booking, which the ownership check then rejects — the
        # customer is told their own appointment is not theirs, and one caller's reference
        # could reach another's record.
        rows = (await ctx.db.execute(select(Appointment).where(
            Appointment.tenant_id == ctx.tenant_id,
            Appointment.customer_id == (ctx.customer.id if ctx.customer else None)))).scalars().all()
        token = raw.replace("-", "").upper()
        hits = [a for a in rows
                if len(token) >= 6 and str(a.id).replace("-", "").upper().startswith(token[:8])]
        if len(hits) == 1:
            return hits[0]
        appt_id = (getattr(ctx, "extra", None) or {}).get("known_appointment_id") or appt_id

    if not appt_id:
        return None
    return (await ctx.db.execute(select(Appointment).where(
        Appointment.tenant_id == ctx.tenant_id,
        Appointment.id == _uuid(appt_id)))).scalar_one_or_none()