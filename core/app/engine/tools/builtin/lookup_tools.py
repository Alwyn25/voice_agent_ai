"""Tools that read the caller's own records back to them, and one that checks who they are.

The agent could create appointments and tickets but had no way to answer "when is my site
visit?" or "what happened to my complaint?" — so it either guessed or dead-ended. These
close that gap.

Identity verification exists because some of what these return is sensitive. On an inbound
call the number already identifies the caller, so asking them to read it back proves
nothing — anyone holding the phone can do that. Asking for the last four digits of the
number ON RECORD is a real check when the call arrives from anywhere else, and it is the
weakest useful thing that still tests knowledge rather than possession.
"""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import select

from app.db.models import Appointment, Location, Ticket
from app.engine.tools.types import ToolContext
from app.services import scheduling as sched
from app.services.spoken import relative_day_prefix


IST = ZoneInfo("Asia/Kolkata")

# The single definition of a "finished" ticket — anything else is still open and owed a
# reply. Kept here so my_tickets filters on it in SQL rather than re-listing the strings.
_CLOSED_TICKET_STATUSES = ("closed", "resolved")


def _is_current_visit(starts_at, today) -> bool:
    """True when an appointment still counts as the customer's current visit.

    An active booking stays "current" until the END of its calendar day, judged in IST — a
    3 pm slot is not history at 3:05, and a caller ringing about it must not be told they
    have nothing booked. A visit whose local day has fully passed is history. A row with no
    start time is never current. This is the one date rule my_appointments uses to split the
    (already status-filtered) active bookings into current vs elapsed, pulled out here so it
    is unit-tested without a database (see the self-check at the bottom).
    """
    return bool(starts_at and starts_at.astimezone(IST).date() >= today)


def _spoken_time(value) -> str | None:
    """A time the agent can say out loud, already in local time.

    Handing the model an ISO string with an offset invites it to normalise: it read
    13:00+05:30 and told the customer 7:30 AM, which is the same instant in UTC and the
    wrong answer on a phone call in Bengaluru. Formatting here removes the choice.
    """
    if not value:
        return None
    local = value.astimezone(IST) if value.tzinfo else value.replace(tzinfo=IST)
    return f"{relative_day_prefix(local)}{local:%A %d %B %Y at %-I:%M %p}"


def _reference(value) -> str:
    """A booking reference a person can repeat. Nobody reads a UUID down a phone."""
    return str(value).replace("-", "")[:8].upper()


# The database status is engine vocabulary, and the responder reads whatever it is given
# straight out loud. A customer who asked what time her visit was got "Your appointment is
# rescheduled for Monday, 10 August at 1:00 PM" — accurate about the row, alarming to her,
# and an answer to a question she had not asked. A booking that stands is confirmed,
# whether or not it was moved to get there.
CALLER_STATUS = {
    "booked": "confirmed",
    "rescheduled": "confirmed",
    "completed": "completed",
    "cancelled": "cancelled",
    "no_show": "missed",
}


def _digits(value) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit())


async def verify_identity(ctx: ToolContext, params: dict) -> dict:
    """Check the last four digits of the caller's registered number.

    Deliberately does not reveal the correct answer on failure, and does not say how many
    attempts remain — both would turn a weak check into a guessing game.
    """
    customer = ctx.customer
    if customer is None:
        return {"verified": False, "reason": "No customer record on this call."}

    on_record = _digits(getattr(customer, "phone_normalized", None) or getattr(customer, "phone", ""))
    given = _digits(params.get("last_four"))

    if len(given) != 4:
        return {"verified": False,
                "reason": "Ask the caller for the last four digits of their registered "
                          "mobile number, digits only."}

    verified = bool(on_record) and on_record[-4:] == given
    return {
        "verified": verified,
        "customer_name": customer.full_name if verified else None,
        "reason": None if verified
                  else "Those digits do not match our records. Offer to call them back on "
                       "the registered number instead of continuing.",
    }


async def my_appointments(ctx: ToolContext, params: dict) -> dict:
    """The caller's own site visits — upcoming by default, past on request."""
    if ctx.customer is None:
        return {"appointments": [], "note": "No customer record on this call."}

    include_past = bool(params.get("include_past"))
    now = datetime.now(timezone.utc)
    # An active booking counts as the customer's current visit until the END of its calendar
    # day — a 3 pm slot is not history at 3:05 — so "today or later" is judged on the local
    # date in IST, not the exact instant.
    today = now.astimezone(IST).date()

    centres = {c.id: c.name for c in (await ctx.db.execute(
        select(Location).where(Location.tenant_id == ctx.tenant_id))).scalars().all()}

    def shape(a):
        return {
            # Say the reference and the spoken time; the raw id and timestamp are here for
            # other tools to use, not for reading aloud.
            "reference": _reference(a.id),
            "when": _spoken_time(a.starts_at),
            # An appointment earlier today is still live — it can be moved or cancelled —
            # but it is NOT "upcoming", and saying so to someone whose 3pm slot passed two
            # hours ago is plainly wrong. The agent needs to know which it is.
            "already_passed": bool(a.starts_at and a.starts_at < now),
            "status": CALLER_STATUS.get(a.status, a.status),
            "location": centres.get(a.centre_id, "our office"),
            # Carried so a reschedule started from "I have a booking" has a centre to work
            # with. Without it the model has to name one from memory, and it cannot.
            "centre_id": str(a.centre_id) if a.centre_id else None,
            "appointment_id": str(a.id),
            "starts_at_iso": a.starts_at.isoformat() if a.starts_at else None,
        }

    # LIVE bookings are filtered by status IN THE QUERY, not fetched blindly and split in
    # Python. A customer who rebooked or rescheduled many times accumulates a pile of
    # `cancelled` rows — often at the very same timestamp as their one standing visit — and a
    # plain `ORDER BY starts_at DESC LIMIT n` let those cancelled duplicates fill the whole
    # window and crowd the live booking out of it. The tool then reported "nothing booked" to
    # a caller who had a confirmed visit the next morning. Scoping the query to the active
    # statuses removes the crowding at the source: a cancelled/completed row can never
    # displace a live one, no matter how many of them exist. ACTIVE_APPT_STATUSES is the
    # engine's single definition of "still a live visit" (booked OR rescheduled) — the same
    # one context assembly reads status on — so the two paths can no longer disagree. Active
    # visits are few by nature; the cap is only a runaway guard, and soonest-first ordering
    # keeps the nearest visit first should it ever bite.
    _ACTIVE_CAP = 100
    active_rows = (await ctx.db.execute(
        select(Appointment).where(
            Appointment.tenant_id == ctx.tenant_id,
            Appointment.customer_id == ctx.customer.id,
            Appointment.status.in_(sched.ACTIVE_APPT_STATUSES),
        ).order_by(Appointment.starts_at.asc()).limit(_ACTIVE_CAP))).scalars().all()

    current, elapsed_active = [], []
    for a in active_rows:
        if _is_current_visit(a.starts_at, today):
            current.append(a)
        else:
            # An active booking whose day has fully passed with no cancel/complete update —
            # rare, but it is history, not an upcoming visit.
            elapsed_active.append(a)

    live = [shape(a) for a in current]
    upcoming = [a for a in live if not a["already_passed"]]
    earlier_today = [a for a in live if a["already_passed"]]

    # HISTORY is a bounded, most-recent sample of the caller's non-active visits (cancelled,
    # completed, no-show) plus any elapsed active ones. Always queried — it is cheap and
    # capped — so the closing note can tell "you have earlier visits on record" apart from
    # "nothing booked at all"; it is handed back to the agent only when the caller asked for
    # past visits.
    _HISTORY_SAMPLE = 10
    history_rows = (await ctx.db.execute(
        select(Appointment).where(
            Appointment.tenant_id == ctx.tenant_id,
            Appointment.customer_id == ctx.customer.id,
            Appointment.status.not_in(sched.ACTIVE_APPT_STATUSES),
        ).order_by(Appointment.starts_at.desc()).limit(_HISTORY_SAMPLE))).scalars().all()
    history = [shape(a) for a in elapsed_active] + [shape(a) for a in history_rows]
    has_history = bool(elapsed_active or history_rows)

    return {
        # "live" rather than "upcoming": these are bookings that still stand, some of
        # which may have been earlier today.
        "live": live,
        "upcoming": upcoming,
        "earlier_today": earlier_today,
        "past": history if include_past else [],
        "has_upcoming": bool(upcoming),
        "phrasing": ("Their booking was earlier today and that time has passed. Say so "
                     "plainly; do not call it upcoming. It can still be moved or cancelled."
                     if earlier_today and not upcoming else None),
        # Never say "no appointments" while any exist — say what there is.
        "note": None if live
                else ("They have no live booking. There are earlier visits on record; check "
                      "those before offering a new one." if has_history
                      else "Nothing booked at all. Offer to arrange a visit."),
    }


async def my_tickets(ctx: ToolContext, params: dict) -> dict:
    """The caller's own tickets and their current status."""
    if ctx.customer is None:
        return {"tickets": [], "note": "No customer record on this call."}

    def shape(t):
        return {
            "reference": _reference(t.id),
            "ticket_id": str(t.id),
            "subject": t.subject,
            "status": t.status,
            "priority": t.priority,
            "raised_on": t.created_at.date().isoformat() if t.created_at else None,
            # The internal resolution note is not read out — it is written for colleagues, not
            # for the customer, and often names people or systems they should not hear about.
            "has_resolution": bool(getattr(t, "resolution", None)),
        }

    # OPEN tickets are filtered in the query, not fetched blindly and filtered in Python.
    # This is the my_appointments crowding bug in ticket form: `ORDER BY created_at DESC
    # LIMIT n` followed by a Python "drop the closed ones" let a stack of recently-closed
    # tickets fill the whole window and hide an older but still-open one, so the agent told a
    # caller nothing was open while their complaint sat unresolved. Scoping to the open
    # statuses in SQL removes that: closed/resolved rows can never displace an open one. Open
    # tickets are few by nature; the cap is only a runaway guard.
    _OPEN_CAP = 100
    open_rows = (await ctx.db.execute(
        select(Ticket).where(
            Ticket.tenant_id == ctx.tenant_id,
            Ticket.customer_id == ctx.customer.id,
            Ticket.status.not_in(_CLOSED_TICKET_STATUSES),
        ).order_by(Ticket.created_at.desc()).limit(_OPEN_CAP))).scalars().all()

    closed_rows = []
    if params.get("include_closed"):
        # A bounded, most-recent sample of resolved history, only when the caller asked for it.
        _CLOSED_SAMPLE = 10
        closed_rows = (await ctx.db.execute(
            select(Ticket).where(
                Ticket.tenant_id == ctx.tenant_id,
                Ticket.customer_id == ctx.customer.id,
                Ticket.status.in_(_CLOSED_TICKET_STATUSES),
            ).order_by(Ticket.created_at.desc()).limit(_CLOSED_SAMPLE))).scalars().all()

    tickets = [shape(t) for t in open_rows] + [shape(t) for t in closed_rows]
    return {
        "tickets": tickets,
        "open_count": len(open_rows),
        "note": None if open_rows else "Nothing open. If they are describing a new problem, raise one.",
    }


# --- self-check -------------------------------------------------------------------------
# Both my_appointments and my_tickets now filter status IN THE QUERY so a pile of
# cancelled/closed rows can never crowd a live one out of a bounded row window — that fix
# lives in SQL and is exercised against a real database, not here. What is unit-testable
# without a database is the date rule that then splits the (active-only) rows into current
# vs elapsed, and the timezone handling around it. That is what this self-check pins.
if __name__ == "__main__":
    def _dt(y, m, d, hh, mm=0, tz=timezone.utc):
        return datetime(y, m, d, hh, mm, tzinfo=tz)

    # "Today" is judged in IST. Pick a UTC instant during 26 Aug IST and derive today from it,
    # exactly as the tool does.
    _now = _dt(2026, 8, 25, 12)                     # 25 Aug 17:30 IST
    _today = _now.astimezone(IST).date()            # 2026-08-25

    failures = 0

    def check(label, cond):
        global failures
        print(("ok  " if cond else "FAIL") + "  " + label)
        if not cond:
            failures += 1

    # A visit tomorrow morning — the exact shape of the bug we fixed (booked, next day) — is
    # current. This is stored as a UTC instant whose IST date is 26 Aug.
    check("tomorrow-morning visit is current",
          _is_current_visit(_dt(2026, 8, 26, 4, 30), _today))       # 26 Aug 10:00 IST
    # Later today, even if the hour is already past, is still current (movable/cancellable).
    check("earlier-today visit is still current",
          _is_current_visit(_dt(2026, 8, 25, 3, 0), _today))        # 25 Aug 08:30 IST
    # A visit whose local day has fully passed is history, not current.
    check("yesterday visit is not current",
          not _is_current_visit(_dt(2026, 8, 24, 4, 30), _today))
    # A row with no start time is never current.
    check("missing start time is not current", not _is_current_visit(None, _today))
    # Timezone boundary: 25 Aug 20:00 UTC is already 26 Aug 01:30 IST — tomorrow, not today.
    check("UTC evening that is next-day in IST is current",
          _is_current_visit(_dt(2026, 8, 25, 20, 0), _today))

    print("\nLOOKUP SELFTEST:", "PASS" if not failures else f"{failures} FAILURE(S)")
    import sys
    sys.exit(1 if failures else 0)