"""Built-in scheduling tool implementations. Each returns a dict (the tool output)."""
import re
import uuid
from datetime import date, datetime, time, timedelta

from zoneinfo import ZoneInfo

from sqlalchemy import select

from app.db.models import Appointment, Location
from app.engine.tools.types import ToolContext
from app.services import scheduling as sched
from app.services.events import publish
from app.services.spoken import relative_day_prefix


# A single-day search that starts from today returns nothing every afternoon, because the
# minimum-notice window has already eaten the remaining slots. The agent then tells the
# customer there is no availability while the rest of the week is wide open. So an
# unspecified end date now means "a week", and an empty window is widened once more before
# giving up — with next_available reported so the agent can offer the real answer instead
# of a dead end.
IST = ZoneInfo("Asia/Kolkata")


def _spoken(value) -> str | None:
    """Local time, already formatted. An ISO string with an offset invites the model to
    normalise it to UTC and quote the wrong hour to someone standing in Bengaluru."""
    if not value:
        return None
    local = value.astimezone(IST) if value.tzinfo else value.replace(tzinfo=IST)
    return f"{relative_day_prefix(local)}{local:%A %d %B at %-I:%M %p}"


DEFAULT_SEARCH_DAYS = 7
FALLBACK_SEARCH_DAYS = 21
DEFAULT_LIMIT = 8

# Colloquial Indian-English boundaries, not clock-textbook ones: a caller asking for
# "evening" in Bengaluru means from about four o'clock, and the old assumption that evening
# began at 17:00 would have hidden the 16:00 slot this whole feature exists to surface.
PART_OF_DAY = {
    "morning":   (time(0, 0), time(12, 0)),
    "afternoon": (time(12, 0), time(16, 0)),
    "evening":   (time(16, 0), time(23, 59)),
}


def _as_date(value):
    if not value:
        return None
    return (datetime.fromisoformat(value).date() if "T" in value
            else datetime.strptime(value, "%Y-%m-%d").date())


def _as_time(value):
    """Accept "16:00", "16:00:00", "4pm", "4 PM", "16"."""
    if not value:
        return None
    text = str(value).strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::\d{2})?", text)
    if m:
        h, mi = int(m.group(1)), int(m.group(2))
        return time(h, mi) if 0 <= h <= 23 and 0 <= mi <= 59 else None
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(am|pm)", text)
    if m:
        h, mi = int(m.group(1)), int(m.group(2) or 0)
        if not (1 <= h <= 12):
            return None
        if m.group(3) == "pm" and h != 12:
            h += 12
        if m.group(3) == "am" and h == 12:
            h = 0
        return time(h, mi)
    m = re.fullmatch(r"(\d{1,2})", text)
    if m and 0 <= int(m.group(1)) <= 23:
        return time(int(m.group(1)), 0)
    return None


def _window_from(params) -> tuple[time | None, time | None, str | None]:
    """The time-of-day window the customer asked for, if any."""
    part = str(params.get("part_of_day") or "").strip().lower()
    if part in PART_OF_DAY:
        lo, hi = PART_OF_DAY[part]
        return lo, hi, part
    lo = _as_time(params.get("time_from"))
    hi = _as_time(params.get("time_to"))
    if lo and not hi:
        hi = time(23, 59)
    if hi and not lo:
        lo = time(0, 0)
    if lo and hi:
        # A caller-named exact time comes in as time_from == time_to; render it as one time
        # ("14:00") not a degenerate range ("14:00-14:00") so the not-available note reads
        # naturally to the model ("Nothing free at 14:00" → "2 PM isn't available").
        label = f"{lo:%H:%M}" if lo == hi else f"{lo:%H:%M}-{hi:%H:%M}"
        return lo, hi, label
    return None, None, None


def _in_window(slot, lo: time, hi: time) -> bool:
    local = slot.starts_at.astimezone(IST).time()
    return lo <= local <= hi


def _spread(slots: list, limit: int) -> list:
    """Evenly spaced across everything found, not the first N.

    THE BUG THIS FIXES: this used to be slots[:limit] with limit defaulting to 5. The site
    office is open 10:00-18:30 on 60-minute slots, so a single day holds eight of them —
    and the model was only ever shown 10:00 to 14:00. A caller asking for four o'clock was
    told, truthfully as far as the agent could see, that the next slot was 10:00 AM. She
    asked six times. Taking a spread means the model always sees the shape of the day,
    including its end.
    """
    if len(slots) <= limit:
        return slots
    step = (len(slots) - 1) / (limit - 1) if limit > 1 else 0
    picked = {round(i * step) for i in range(limit)}
    return [slots[i] for i in sorted(picked)]


def _shape(slots) -> list[dict]:
    return [{"when": _spoken(s.starts_at),
             "starts_at": s.starts_at.isoformat(),
             "capacity_left": s.capacity_left} for s in slots]


async def search_slots(ctx: ToolContext, params: dict) -> dict:
    centre_id = await _resolve_centre_id(ctx, params.get("centre_id"))
    executive_id = uuid.UUID(params["executive_id"]) if params.get("executive_id") else None

    d_from = _as_date(params.get("date_from")) or date.today()
    explicit_to = _as_date(params.get("date_to"))
    d_to = explicit_to or (d_from + timedelta(days=DEFAULT_SEARCH_DAYS))

    slots = await sched.compute_slots(ctx.db, ctx.tenant_id, centre_id, d_from, d_to,
                                      executive_id=executive_id)

    widened = False
    if not slots and not explicit_to:
        widened = True
        d_to = d_from + timedelta(days=FALLBACK_SEARCH_DAYS)
        slots = await sched.compute_slots(ctx.db, ctx.tenant_id, centre_id, d_from, d_to,
                                         executive_id=executive_id)

    limit = int(params.get("limit", DEFAULT_LIMIT))
    lo, hi, asked_for = _window_from(params)

    matching, note = slots, None
    if lo and hi:
        matching = [s for s in slots if _in_window(s, lo, hi)]
        if not matching:
            # Never a dead end. The customer asked for a window we cannot fill, so say so
            # AND hand over what does exist — otherwise the agent stalls on "nothing in the
            # evening" while the rest of the day is wide open.
            note = (f"Nothing free in the {asked_for} window. These are the times that ARE "
                    f"free — offer the closest one and say plainly that the requested "
                    f"window is full.")
            matching = slots

    shown = _spread(matching, limit)

    # Deliberately minimal. An earlier version also returned searched_from/searched_to,
    # and the model read the window's end date as the appointment date — telling the
    # customer "August 6" when the first slot was July 31. Every extra date in a tool
    # result is another thing that can be quoted back as fact, so only the slots
    # themselves and one explicit next_available are exposed.
    return {
        "slots": _shape(shown),
        "total_found": len(matching),
        "asked_for": asked_for,
        "note": note,
        # The earliest slot INSIDE what the customer asked for, so the agent stops
        # answering "I want the evening" with tomorrow morning.
        "next_available": _spoken(matching[0].starts_at) if matching else None,
        "next_available_iso": matching[0].starts_at.isoformat() if matching else None,
        "searched_days": (d_to - d_from).days,
        "widened_search": widened,
    }


# The tool contract already tells the model that centre_id must be the UUID returned by
# locate_centre. It does not always comply — it passes the centre's NAME ("Vapi"), because
# that is what it has been discussing. The old behaviour failed the centre_exists check and
# the agent then asked the CUSTOMER for a valid centre ID, which is nonsense: no caller has
# a UUID. Strictness bought nothing here, because a centre name is unambiguous within a
# tenant. So resolve it. Same for slot_start, which the model sometimes returns without the
# timezone offset it was given.

def _name_tokens(value) -> set[str]:
    return {t for t in "".join(c.lower() if c.isalnum() else " " for c in str(value or "")).split() if t}


def _match_centres(rows, given):
    """Centres whose name contains every word the caller used.

    A plain SQL LIKE fails the moment punctuation differs — "DSR Highland Greenz — Site
    Office" never matches "DSR Highland Greenz site office", because of one em-dash. The
    model writes names the way a person says them, so matching has to survive that. Words
    present, punctuation ignored.
    """
    want = _name_tokens(given)
    if not want:
        return []
    exact = [r for r in rows if _name_tokens(r.name) == want]
    if exact:
        return exact
    return [r for r in rows if want.issubset(_name_tokens(r.name))]


async def _resolve_centre_id(ctx: ToolContext, value) -> uuid.UUID:
    # Whatever the model sent, a centre already established on this call wins when the
    # value does not resolve. It has sent names, descriptions and text lifted from its own
    # instructions; none of that should cost the customer a booking.
    known = (ctx.extra or {}).get("known_centre_id")
    raw = str(value or "").strip()
    # A stringly-typed null is not a centre. The model emits the literal word — "null",
    # "none", "undefined" — as the value of a field it has nothing for; it is truthy, so
    # `str(value or "")` keeps it and the code below would fail to match any centre and raise
    # the misleading "No active centre matches 'null'". The planner strips these before a
    # tool runs, but this is the tool's own boundary and cannot assume it: a placeholder is
    # "no centre given", so fall through to the known-centre / call-locate_centre path.
    if raw.lower() in ("null", "none", "nil", "undefined", "n/a", "na"):
        raw = ""
    if not raw:
        if known:
            return uuid.UUID(str(known))
        raise ValueError("centre_id is required — call locate_centre first.")
    try:
        parsed = uuid.UUID(raw)
    except (ValueError, AttributeError):
        parsed = None
    if parsed is not None:
        # A well-formed UUID is NOT proof of a live centre. The model invents syntactically
        # valid ids, and a real one goes stale when demo data is reseeded — either way the old
        # code returned it unchecked and the scheduler then raised the misleading "Centre not
        # found or inactive", as though a centre had been named and rejected. Trust the id only
        # when it names an ACTIVE centre in this tenant; otherwise fall back to the centre
        # already established on the call. (The planner's _fit_params normally resolves this
        # before the tool runs — from the resolved project or the call's own centre — so this
        # is the tool's own last line of defence for any path that reaches it directly.)
        live = (await ctx.db.execute(select(Location).where(
            Location.tenant_id == ctx.tenant_id, Location.id == parsed,
            Location.is_active.is_(True)))).scalar_one_or_none()
        if live is not None:
            return parsed
        if known:
            return uuid.UUID(str(known))
        # Nothing better to offer — keep the id so the scheduler's own check reports it, exactly
        # as before this guard existed.
        return parsed

    active = (await ctx.db.execute(select(Location).where(
        Location.tenant_id == ctx.tenant_id, Location.is_active.is_(True)))).scalars().all()
    rows = _match_centres(active, raw)
    if len(rows) == 1:
        return rows[0].id
    # One project with a site office and a sales gallery is the normal shape, so "the site
    # office" should resolve even though both names start the same way.
    if len(rows) > 1:
        narrowed = [r for r in rows if _name_tokens(r.name) == _name_tokens(raw)]
        if len(narrowed) == 1:
            return narrowed[0].id
    if not rows:
        if known:
            return uuid.UUID(str(known))
        raise ValueError(f"No active centre matches {raw!r}. Call locate_centre and use the "
                         f"centre_id it returns.")
    names = ", ".join(r.name for r in rows[:5])
    raise ValueError(f"{raw!r} matches more than one centre ({names}). Call locate_centre and "
                     f"use the exact centre_id.")


async def _parse_slot_start(ctx: ToolContext, value, centre_id: uuid.UUID) -> datetime:
    """Accept the slot timestamp with or without its offset."""
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is not None:
        return parsed
    centre = (await ctx.db.execute(select(Location).where(Location.id == centre_id))).scalar_one_or_none()
    tz = ZoneInfo(getattr(centre, "timezone", None) or "Asia/Kolkata")
    return parsed.replace(tzinfo=tz)


async def book_appointment(ctx: ToolContext, params: dict) -> dict:
    centre_id = await _resolve_centre_id(ctx, params.get("centre_id"))
    # The unit the caller chose comes from the engine's tracked preference, NOT from a model
    # param — the model loses a configuration across a multi-turn booking as reliably as it
    # loses a centre id, and this is a decision we already recorded. Bound onto the visit so
    # "the flat I selected" is answerable from the appointment ever after.
    desired_unit = (ctx.extra or {}).get("desired_unit")
    unit_context = dict((ctx.extra or {}).get("desired_unit_context") or {})
    appt = await sched.book(
        ctx.db, tenant_id=ctx.tenant_id, customer=ctx.customer,
        centre_id=centre_id,
        starts_at=await _parse_slot_start(ctx, params["slot_start"], centre_id),
        executive_id=uuid.UUID(params["executive_id"]) if params.get("executive_id") else None,
        booked_via="agent", source_conversation_id=ctx.conversation_id,
        # Bind the resolved project to the visit (multi-project-per-tenant), so it can be read
        # back off the appointment on the caller's next call.
        project_id=ctx.project_id,
        unit_type=desired_unit or None, unit_context=unit_context)
    centre = (await ctx.db.execute(select(Location).where(Location.id == appt.centre_id))).scalar_one()
    publish(ctx.tenant_id, "appointment.booked", appointment_id=str(appt.id),
            customer_id=str(ctx.customer.id), starts_at=appt.starts_at.isoformat(),
            centre=centre.name, booked_by="agent")
    out = {"appointment_id": str(appt.id), "starts_at": appt.starts_at.isoformat(),
           "ends_at": appt.ends_at.isoformat(), "centre_name": centre.name,
           "centre_address": centre.address, "status": appt.status,
           "unit_type": appt.unit_type}
    # A location may carry a fixed reference to read back with every booking. Only GTM demo
    # tenants set one (their sample reference, e.g. VIS-1187, app/demo/agent.py); for every
    # other location the key is absent and the output is exactly as before.
    ref = (centre.location_type or {}).get("booking_reference")
    if ref:
        out["booking_reference"] = ref
    return out



async def _resolve_appointment(ctx: ToolContext, value):
    """An appointment, whether the model sent the id, the short reference, or nothing.

    It sends all three. my_appointments hands back a reference like ADED37CF for the
    customer to repeat, and the model then puts that where a UUID belongs; other times it
    loses the id entirely. The appointment under discussion is known to the conversation,
    so that is the fallback rather than failing in front of the caller.
    """
    raw = str(value or "").strip()
    known = (ctx.extra or {}).get("known_appointment_id")

    if raw:
        try:
            found = (await ctx.db.execute(select(Appointment).where(
                Appointment.tenant_id == ctx.tenant_id,
                Appointment.id == uuid.UUID(raw)))).scalar_one_or_none()
            if found is not None:
                return found
        except (ValueError, AttributeError):
            pass

        # A short reference — the first eight hex characters, as read out to the customer.
        token = raw.replace("-", "").upper()
        if len(token) >= 6:
            rows = (await ctx.db.execute(select(Appointment).where(
                Appointment.tenant_id == ctx.tenant_id,
                Appointment.customer_id == (ctx.customer.id if ctx.customer else None)))).scalars().all()
            hits = [a for a in rows if str(a.id).replace("-", "").upper().startswith(token[:8])]
            if len(hits) == 1:
                return hits[0]

    if known:
        found = (await ctx.db.execute(select(Appointment).where(
            Appointment.tenant_id == ctx.tenant_id,
            Appointment.customer_id == (ctx.customer.id if ctx.customer else None),
            Appointment.id == uuid.UUID(str(known))))).scalar_one_or_none()
        if found is not None:
            return found

    raise ValueError("Which appointment? Call my_appointments to get the customer's "
                     "booking, then pass the appointment_id it returns.")


async def reschedule_appointment(ctx: ToolContext, params: dict) -> dict:
    appt = await _resolve_appointment(ctx, params.get("appointment_id"))
    old = appt.starts_at

    # THE BUG THIS FIXES: this was a bare datetime.fromisoformat(), which produces a NAIVE
    # datetime whenever the model omits the offset — which it usually does. The scheduler's
    # slots are timezone-aware, and `s.starts_at == starts_at` between an aware and a naive
    # datetime is simply False (no exception raised), so a slot that was genuinely free came
    # back as SLOT_NOT_AVAILABLE. The customer was told a time was available and then, one
    # turn later, that it was not, and finally that there had been a system error.
    # book_appointment already went through _parse_slot_start; reschedule did not.
    new_starts_at = await _parse_slot_start(ctx, params["slot_start"], appt.centre_id)

    # Check before attempting, so a genuine clash comes back with real alternatives rather
    # than surfacing as an opaque failure. The agent can only offer another time if it is
    # handed one.
    day = await sched.compute_slots(ctx.db, ctx.tenant_id, appt.centre_id,
                                    new_starts_at.date(), new_starts_at.date(),
                                    executive_id=appt.executive_id)
    if not any(s.starts_at == new_starts_at for s in day):
        alternatives = [_spoken(s.starts_at) for s in day[:6]]
        raise ValueError(
            f"{_spoken(new_starts_at)} is not free. "
            + (f"Free that day: {', '.join(alternatives)}. Offer one of these to the customer."
               if alternatives
               else "Nothing is free that day at all — offer a different day."))

    appt = await sched.reschedule(ctx.db, tenant_id=ctx.tenant_id, appt=appt,
                                  new_starts_at=new_starts_at)
    return {"appointment_id": str(appt.id), "reference": str(appt.id).replace("-", "")[:8].upper(),
            "previous_start": _spoken(old),
            "when": _spoken(appt.starts_at),
            "starts_at": appt.starts_at.isoformat(), "status": appt.status}


async def cancel_appointment(ctx: ToolContext, params: dict) -> dict:
    appt = await _resolve_appointment(ctx, params.get("appointment_id"))
    appt.status = "cancelled"
    appt.cancel_reason = params.get("reason")
    await ctx.db.flush()
    return {"appointment_id": str(appt.id), "status": appt.status}


async def locate_centre(ctx: ToolContext, params: dict) -> dict:
    limit = int(params.get("limit", 3))
    q = select(Location).where(Location.tenant_id == ctx.tenant_id, Location.is_active == True)  # noqa: E712
    if params.get("city"):
        q = q.where(Location.city.ilike(f"%{params['city']}%"))
    if params.get("region_code"):
        q = q.where(Location.region_code == params["region_code"])
    rows = (await ctx.db.execute(q.limit(limit))).scalars().all()

    # A city that matches nothing must not dead-end the whole booking. The caller says
    # "Bangalore" but the record reads "Bengaluru"; the substring filter finds nothing, and
    # without this the agent tells them there are no centres in their city and the visit
    # stalls. When a city/region filter yields nothing, fall back to every active centre so
    # the agent can offer what actually exists (and, with more than one, ask which they
    # mean) instead of denying they have any.
    if not rows and (params.get("city") or params.get("region_code")):
        rows = (await ctx.db.execute(select(Location).where(
            Location.tenant_id == ctx.tenant_id, Location.is_active == True)  # noqa: E712
            .limit(limit))).scalars().all()
    # Address and phone are deliberately NOT returned. The model kept reading them out as
    # an answer — "I've found the Vapi centre, you can reach them at +91..." — which turns
    # a booking agent into a directory and sends the caller away to do the job themselves.
    # Prompting against it did not hold. It cannot quote a number it was never given.
    # Callers who genuinely want an address get it from the Centres and Working Hours
    # document through knowledge_lookup, which is the right source anyway.
    return {"next_step": "Call search_slots with one of these centre_id values. "
                         "Finding the centre is not an answer on its own.",
            "centres": [{"centre_id": str(c.id), "name": c.name, "city": c.city} for c in rows]}