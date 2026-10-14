"""Built-in CRM / support tool implementations."""
import logging
import re
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import select

from app.db.models import CallbackRequest, Conversation, Customer, CustomerFact, Ticket
from app.engine.tools.types import ToolContext
from app.services.phone import normalize_phone

log = logging.getLogger(__name__)


async def create_customer(ctx: ToolContext, params: dict) -> dict:
    """Register a caller who is not in the CRM yet.

    THE GAP THIS FILLS: a rule blocks book_appointment when there is no customer record —
    correctly, since a booking has to belong to someone. But nothing could create one, so
    the agent was told to identify the caller and given no way to do it. It asked for a
    name, then for verification, then for the last four digits of a number it had no record
    of, looping until the caller gave up. Every one of those turns was the agent trying to
    satisfy a requirement its toolset could not meet.

    Taking a name and a number from someone who wants to visit is the ordinary thing a
    human would do. It is also the least the CRM needs to be useful afterwards.
    """
    name = (params.get("full_name") or "").strip()
    phone = (params.get("phone") or "").strip()

    # A phone number is digits. "<customer's phone number>" is not, and a record created
    # from it is worse than no record — it looks registered, cannot be called, and quietly
    # breaks every later lookup by number.
    if phone and len(re.sub(r"\D", "", phone)) < 8:
        log.warning("create_customer refused: %r is not a phone number (conv=%s)",
                    phone, ctx.conversation_id)
        return {"created": False, "needs": "phone",
                "note": (f"'{phone}' is not a usable phone number. Ask the caller for it "
                         f"— this is not a system fault and must not be described as one.")}

    if not name or not phone:
        # Returned, not raised. An exception here surfaces as an execution error, and the
        # agent tells the customer there was "a technical issue with our system" — for a
        # missing field it could simply have asked for. Saying which one is missing lets
        # the next turn ask the right question.
        missing = "a name" if not name else "a phone number"
        # Logged, because five rounds of debugging had nothing to read. The tool failed
        # silently and the only evidence was the agent apologising for a "system issue" —
        # which is not searchable, not specific, and not true.
        log.warning("create_customer refused: missing %s (name=%r phone=%r conv=%s)",
                    missing, name, phone, ctx.conversation_id)
        return {"created": False, "needs": "full_name" if not name else "phone",
                "note": f"Cannot register anyone without {missing}. Ask the caller for it "
                        f"— this is not a system fault and should not be described as one."}

    normalized = normalize_phone(phone)

    # Someone may ring from a new number and already be on file, or give a number that has
    # since been added. Creating a second record for the same person splits their history
    # in two, and neither half is right afterwards.
    existing = (await ctx.db.execute(select(Customer).where(
        Customer.tenant_id == ctx.tenant_id,
        Customer.phone_normalized == normalized))).scalar_one_or_none()

    if existing is not None:
        customer = existing
        created = False
    else:
        customer = Customer(tenant_id=ctx.tenant_id, full_name=name, phone=phone,
                            phone_normalized=normalized,
                            email=params.get("email") or None,
                            preferred_language=params.get("preferred_language") or None,
                            current_state="lead",
                            lead_source="inbound_call")
        ctx.db.add(customer)
        await ctx.db.flush()
        created = True

    # Attach them to THIS conversation, so everything after this turn — booking, facts,
    # the decision trace — belongs to a real person rather than to nobody.
    conv = (await ctx.db.execute(select(Conversation).where(
        Conversation.id == ctx.conversation_id))).scalar_one_or_none()
    if conv is not None and conv.customer_id is None:
        conv.customer_id = customer.id
        await ctx.db.flush()

    # The caller gave us their own number on their own call, so identity is not in
    # question for the rest of it — asking them to verify a record they just dictated
    # would be theatre.
    ctx.identity_verified = True

    return {"customer_id": str(customer.id), "full_name": customer.full_name,
            "phone": customer.phone_normalized, "created": created,
            "note": ("Registered and attached to this call."
                     if created else "Already on file; attached to this call.")}


async def create_ticket(ctx: ToolContext, params: dict) -> dict:
    t = Ticket(tenant_id=ctx.tenant_id, customer_id=ctx.customer.id,
               conversation_id=ctx.conversation_id, subject=params["subject"],
               description=params.get("description"), category=params.get("category"),
               priority=params.get("priority", "medium"))
    ctx.db.add(t)
    await ctx.db.flush()
    return {"ticket_id": str(t.id), "subject": t.subject, "status": t.status,
            "priority": t.priority}


# A stated preference narrows the permitted calling window; it can never widen it.
# Without this clamp, "mornings before 10" becomes 00:00-10:00 and calling_policy
# would treat a 3am call as inside the customer's preferred hours.
BUSINESS_START = time(9, 0)
BUSINESS_END = time(21, 0)
# Callbacks are a local-time promise. "Call me at six" means six where the customer is,
# and treating a naive time as UTC put every callback five and a half hours out.
LOCAL_TZ = ZoneInfo("Asia/Kolkata")

TIME_FIELDS = ("preferred_contact_start", "preferred_contact_end")
TEXT_FIELDS = ("full_name", "email", "location", "region_code", "preferred_language")


def _to_time(v) -> time | None:
    """Accept "18:00", "18:00:00" or a time object. Anything else is rejected."""
    if isinstance(v, time):
        return v
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::\d{2})?", str(v).strip())
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    return time(h, mi) if 0 <= h <= 23 and 0 <= mi <= 59 else None


async def update_customer_profile(ctx: ToolContext, params: dict) -> dict:
    updated: dict[str, str] = {}
    rejected: dict[str, str] = {}

    for k in TEXT_FIELDS:
        v = params.get(k)
        if v is None:
            continue
        text = str(v).strip()[:255]
        if text:
            setattr(ctx.customer, k, text)
            updated[k] = text

    for k in TIME_FIELDS:
        v = params.get(k)
        if v is None:
            continue
        parsed = _to_time(v)
        if parsed is None:
            rejected[k] = f"'{v}' is not a valid time — use 24-hour HH:MM"
            continue
        setattr(ctx.customer, k, parsed)
        updated[k] = parsed.strftime("%H:%M")

    # A half-open contact window is worse than none: the calling-hours rule reads it.
    start, end = ctx.customer.preferred_contact_start, ctx.customer.preferred_contact_end
    if start is not None and end is None:
        end = BUSINESS_END
    elif end is not None and start is None:
        start = BUSINESS_START

    if start is not None and end is not None:
        clamped_start = max(start, BUSINESS_START)
        clamped_end = min(end, BUSINESS_END)
        if clamped_start >= clamped_end:          # preference falls outside legal hours
            clamped_start, clamped_end = BUSINESS_START, BUSINESS_END
        if (clamped_start, clamped_end) != (start, end):
            updated["clamped_to_calling_hours"] = "09:00-21:00"
        ctx.customer.preferred_contact_start = clamped_start
        ctx.customer.preferred_contact_end = clamped_end
        updated["preferred_contact_start"] = clamped_start.strftime("%H:%M")
        updated["preferred_contact_end"] = clamped_end.strftime("%H:%M")

    if not updated:
        raise ValueError(f"No valid fields to update. {rejected or 'Nothing recognised.'}")

    await ctx.db.flush()
    return {"customer_id": str(ctx.customer.id), "updated_fields": updated,
            "rejected": rejected}


async def record_customer_fact(ctx: ToolContext, params: dict) -> dict:
    """Store something learned about the customer (live extraction path)."""
    f = CustomerFact(tenant_id=ctx.tenant_id, customer_id=ctx.customer.id,
                     category=params["category"], key=params["key"],
                     value={"text": params["value"]} if isinstance(params["value"], str)
                     else params["value"],
                     confidence=float(params.get("confidence", 0.8)),
                     source_conversation_id=ctx.conversation_id, status="active")
    ctx.db.add(f)
    await ctx.db.flush()
    return {"fact_id": str(f.id), "category": f.category, "key": f.key}


def _to_datetime(v) -> datetime | None:
    """Accept a full ISO datetime, a date, or a bare time.

    A bare time ("18:00") means the next occurrence of that time — the model naturally
    says "call me at 6" and should not have to compute a date to be understood.
    """
    if isinstance(v, datetime):
        return v
    text = str(v).strip()
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::\d{2})?", text)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        if not (0 <= h <= 23 and 0 <= mi <= 59):
            return None
        now = datetime.now(timezone.utc)
        candidate = now.replace(hour=h, minute=mi, second=0, microsecond=0)
        return candidate if candidate > now else candidate + timedelta(days=1)
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), 10, 0,
                        tzinfo=timezone.utc)
    return None


async def schedule_callback(ctx: ToolContext, params: dict) -> dict:
    when = _to_datetime(params.get("requested_for"))
    if when is None:
        # Do not refuse. The customer has asked to be called back, which is the part that
        # matters; a time we could not parse is our problem, not theirs. Booking it for the
        # start of the next calling window and saying so beats "that is not a valid time"
        # — and beats what actually happened, which was the model giving up on the tool and
        # promising a callback nobody ever booked.
        when = _next_calling_slot()
    if when.tzinfo is None:
        when = when.replace(tzinfo=LOCAL_TZ)

    # Clamp into legal calling hours. 6am is not a time we may ring anybody, and accepting
    # it silently would put a TCCCPR breach in the queue with the customer's blessing.
    original, when = when, _within_calling_hours(when)
    cb = CallbackRequest(tenant_id=ctx.tenant_id, customer_id=ctx.customer.id,
                         requested_for=when, source_conversation_id=ctx.conversation_id)
    ctx.db.add(cb)
    await ctx.db.flush()
    out = {"callback_id": str(cb.id), "requested_for": when.isoformat(),
           "when": when.astimezone(LOCAL_TZ).strftime("%A %-d %B at %-I:%M %p")
           if hasattr(when, "strftime") else str(when)}
    if when != original:
        out["moved_from"] = original.astimezone(LOCAL_TZ).strftime("%-I:%M %p")
        out["moved_because"] = ("we may only call between 9am and 9pm — tell the customer "
                                "the new time plainly")
    return out


def _next_calling_slot() -> datetime:
    now = datetime.now(LOCAL_TZ)
    return _within_calling_hours(now + timedelta(hours=1))


def _within_calling_hours(when: datetime) -> datetime:
    """Move a time into the 9am-9pm window, forwards, never backwards."""
    local = when.astimezone(LOCAL_TZ)
    if local.time() < BUSINESS_START:
        local = local.replace(hour=BUSINESS_START.hour, minute=BUSINESS_START.minute,
                              second=0, microsecond=0)
    elif local.time() > BUSINESS_END:
        local = (local + timedelta(days=1)).replace(
            hour=BUSINESS_START.hour, minute=BUSINESS_START.minute, second=0, microsecond=0)
    return local


async def record_do_not_contact(ctx: ToolContext, params: dict) -> dict:
    """Honour an opt-out request: move the customer to do_not_contact and add them
    to the DNC list. The state machine hook writes the DNC entry."""
    from app.engine.state_machine.machine import apply_trigger

    result = await apply_trigger(ctx.db, ctx.customer, "signal:dnc_request",
                                 actor=ctx.actor,
                                 reason=params.get("reason") or "customer requested opt-out")
    await ctx.db.flush()
    return {"customer_id": str(ctx.customer.id),
            "state": ctx.customer.current_state,
            "previous_state": result.from_state if result else None,
            "scope": params.get("scope", "all"),
            "confirmed": True}