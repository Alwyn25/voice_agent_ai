"""One shared rule for the relative day of a spoken date ("today"/"tomorrow").

Every surface that reads a date back to the caller — offered slots, a booking confirmation,
"when is my visit", the visit-on-file block — formats its own absolute string ("Wednesday 19
August at 1:00 PM"). What was missing is the day the caller actually thinks in: whether that
is TODAY or TOMORROW. The model is unreliable at deriving it (date arithmetic against a
training-cutoff sense of "now" is exactly what the planner's TODAY block exists to prevent),
so it is computed here, deterministically, and folded into the spoken string. The wording the
model finally speaks stays its own; only this fact is made reliable.

Kept in ONE place so the three formatters (scheduling_tools._spoken, lookup_tools._spoken_time,
turn_context._spoken_local) cannot drift apart on what "tomorrow" means.
"""
from datetime import date, datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def relative_day_prefix(value, *, today: date | None = None) -> str:
    """``"today, "`` / ``"tomorrow, "`` / ``""`` for a datetime or date, judged on the local
    (IST) calendar day. Prepend it to an already-formatted spoken date so the caller hears the
    relative day: "tomorrow, Wednesday 20 August at 1:00 PM". ``today`` overrides "now" for the
    test harness; without it, real wall-clock IST is used."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        d = (value.astimezone(IST) if value.tzinfo else value.replace(tzinfo=IST)).date()
    elif isinstance(value, date):
        d = value
    else:
        return ""
    ref = today or datetime.now(IST).date()
    delta = (d - ref).days
    if delta == 0:
        return "today, "
    if delta == 1:
        return "tomorrow, "
    return ""
