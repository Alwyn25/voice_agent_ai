"""Guard: a spoken "Weekday, D Month" must name the day that date actually falls on.

THE BUG THIS FIXES: a real call had the agent ask "Would 5 PM on Monday, 10 October work
for your evening visit?" when 10 October (that year) is a Saturday — confirmed self-
consistent elsewhere in the SAME call, which called it "Saturday, 10 October" both before
and after this one turn. No tool ran on the turn that got it wrong, so there was no tool
output to check the sentence against (the existing guard in turn_runner.py that compares a
spoken date/time to a just-booked slot's real starts_at only fires after book_appointment/
reschedule_appointment succeeds). The model had a list of earlier-offered slots in context
("Monday 12 October at 5:00 PM", "Saturday 10 October at 10:00 AM" among them) and, asked
to confirm "the 10th, evening", spliced the date number from the caller's words onto the
weekday+time of a DIFFERENT cached slot instead of re-checking — same failure family as
booking_claim.py (a confident sentence proves nothing about what actually happened this
turn), just for a free-form proposal instead of a claimed booking.

Unlike that family, this one is NOT ground-truth-dependent — a weekday is a pure function
of its date, so whether the stated day is right or wrong is always computable here, with no
caller-side context needed and no risk of a false positive. Applied to every reply,
unconditionally: a correct "Saturday, 10 October" is left untouched; a wrong "Monday, 10
October" has only its weekday word corrected, in place, nothing else about the sentence
rewritten.

Deliberately matches full month names and the common 3-letter abbreviations used in
natural speech ("10 Oct"), but not ambiguous numeric dates (10/10) or relative phrases
("next Monday") — those carry no self-contained weekday+date pair to check.

Pure and self-tested (``python -m app.engine.planner.date_claim``).
"""
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"]

_WEEKDAY_RE = "|".join(_WEEKDAYS)
_MONTH_RE = "|".join(_MONTHS) + "|" + "|".join(m[:3] for m in _MONTHS)

# "Monday, 10 October" / "Saturday 10 October 2026" / "Mon 10 Oct" -- weekday first (how the
# agent always speaks it, per app/services/spoken.py's `%A, %d %B`), then day, then month,
# with an optional ordinal suffix and an optional explicit year.
_CLAIM_RE = re.compile(
    rf"\b({_WEEKDAY_RE})\b,?\s+(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\b"
    rf"(?:,?\s+(\d{{4}}))?",
    re.I)


def _month_num(token: str) -> int | None:
    low = token.lower()
    for i, name in enumerate(_MONTHS, start=1):
        if low == name.lower() or low == name[:3].lower():
            return i
    return None


def _resolve_year(month: int, day: int, stated_year: int | None, *, today: date) -> int | None:
    """Same next-occurrence rule turn_runner._spoken_day already uses for a bare date with
    no year stated: assume this year unless that date has already passed, then next year."""
    if stated_year:
        return stated_year
    year = today.year
    try:
        candidate = date(year, month, day)
    except ValueError:
        return None
    return year + 1 if candidate < today else year


def correct_weekday_date_claims(text: str, *, today: date | None = None) -> tuple[str, list[str]]:
    """Fix any "Weekday, D Month" claim in `text` whose weekday is wrong for that date.

    Returns ``(corrected_text, notes)`` -- ``notes`` is empty when nothing needed fixing, and
    otherwise one human-readable line per correction, for the caller to log. Never raises: an
    unparsable month token or an impossible day-of-month (e.g. "31 February") is left alone
    rather than guessed at.
    """
    if not text:
        return text, []
    ref = today or datetime.now(IST).date()
    notes: list[str] = []

    def _fix(m: re.Match) -> str:
        stated_weekday, day_txt, month_txt, year_txt = m.groups()
        month = _month_num(month_txt)
        day = int(day_txt)
        if month is None:
            return m.group(0)
        year = _resolve_year(month, day, int(year_txt) if year_txt else None, today=ref)
        if year is None:
            return m.group(0)
        try:
            real_date = date(year, month, day)
        except ValueError:
            return m.group(0)
        real_weekday = _WEEKDAYS[real_date.weekday()]
        if real_weekday.lower() == stated_weekday.lower():
            return m.group(0)
        notes.append(f"said {stated_weekday}, {day} {month_txt} -- that date is actually "
                     f"a {real_weekday}")
        return real_weekday + m.group(0)[len(stated_weekday):]

    return _CLAIM_RE.sub(_fix, text), notes


def _selftest() -> None:
    # The real bug, pinned to the call's actual date so the test doesn't rot: 2026-10-10 is a
    # real Saturday, not a Monday.
    _oct8_2026 = date(2026, 10, 8)
    fixed, notes = correct_weekday_date_claims(
        "Would 5 PM on Monday, 10 October work for your evening visit?", today=_oct8_2026)
    assert fixed == "Would 5 PM on Saturday, 10 October work for your evening visit?", fixed
    assert notes and "Monday" in notes[0] and "Saturday" in notes[0]

    # Already correct -- untouched, no note.
    ok = "Would you prefer 4 PM or 5 PM on Saturday, 10 October?"
    fixed, notes = correct_weekday_date_claims(ok, today=_oct8_2026)
    assert fixed == ok
    assert notes == []

    # No weekday+date pair at all -- left alone.
    fixed, notes = correct_weekday_date_claims(
        "Shall I book your visit for tomorrow at 10:00 AM?", today=_oct8_2026)
    assert fixed == "Shall I book your visit for tomorrow at 10:00 AM?"
    assert notes == []

    # 3-letter month abbreviation, explicit year, no comma -- still caught and fixed.
    fixed, notes = correct_weekday_date_claims(
        "Your visit is confirmed for Tuesday 10 Oct 2026 at 5 PM.", today=_oct8_2026)
    assert fixed == "Your visit is confirmed for Saturday 10 Oct 2026 at 5 PM.", fixed
    assert notes

    # No year stated and the date has already passed this year -- rolls to next year, and is
    # checked against THAT year's weekday, not this year's.
    fixed, notes = correct_weekday_date_claims(
        "Would Wednesday, 10 October work?", today=date(2027, 11, 1))
    # 10 October 2028 (the next occurrence after 2027-11-01) is a Tuesday.
    assert fixed == "Would Tuesday, 10 October work?", fixed

    # Empty/None text -- no crash, nothing to fix.
    assert correct_weekday_date_claims("") == ("", [])
    assert correct_weekday_date_claims(None) == (None, [])

    print("date_claim selftest: all passed")


if __name__ == "__main__":
    _selftest()
