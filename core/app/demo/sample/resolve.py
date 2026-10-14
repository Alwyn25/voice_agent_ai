"""Turn a validated dummy dataset into what one call actually uses (docs/gtm-demo-links.md §6.4).

Called once at call start; the result is frozen into demo_sessions.resolved_context so the
transcript can always be checked against exactly what the agent was given.

  * text_template  → placeholders filled from the record's facts
  * relative_slot  → a real date and time in the record's timezone, plus the spoken form
                     ("tomorrow, Thursday 1 October at 11 am"). A slot that has already
                     passed (day_offset 0, time gone) is dropped; a slot landing on a day the
                     industry skips (e.g. Sunday OPD) moves to the next allowed day.
  * relative_day   → a real date plus its spoken form ("yesterday, Monday 28 September")

Pure function: `now` is passed in, so tests (and the admin preview at any date) are exact.
"""
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.demo.sample.validate import PLACEHOLDER_RE

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


@dataclass(frozen=True)
class ResolvedContext:
    data: dict
    resolved_at: str        # ISO timestamp of "now" in the record's timezone
    timezone: str

    def as_json(self) -> dict:
        return {"data": self.data, "resolved_at": self.resolved_at, "timezone": self.timezone}


def spoken_time(t: time) -> str:
    """11:00 → "11 am", 16:30 → "4:30 pm", 12:00 → "12 noon", 00:00 → "12 midnight"."""
    if t.minute == 0 and t.hour == 12:
        return "12 noon"
    if t.minute == 0 and t.hour == 0:
        return "12 midnight"
    hour = t.hour % 12 or 12
    suffix = "am" if t.hour < 12 else "pm"
    return f"{hour} {suffix}" if t.minute == 0 else f"{hour}:{t.minute:02d} {suffix}"


def relative_word(d: date, today: date) -> str:
    return {-1: "yesterday", 0: "today", 1: "tomorrow"}.get((d - today).days, "")


def spoken_date(d: date, today: date) -> str:
    """"tomorrow, Thursday 1 October"; "Saturday 3 October"; adds the year only when it
    differs from today's, so a call in late December still says the right year."""
    text = f"{d:%A} {d.day} {d:%B}"
    if d.year != today.year:
        text += f" {d.year}"
    rel = relative_word(d, today)
    return f"{rel}, {text}" if rel else text


def _fill(text: str, facts: dict[str, str | None]) -> str:
    return PLACEHOLDER_RE.sub(lambda m: (facts.get(m.group(1)) or "").strip(), text)


def _resolve_slot(slot: dict, now: datetime, skip: set[int]) -> dict | None:
    day = now.date() + timedelta(days=slot["day_offset"])
    for _ in range(7):
        if day.weekday() not in skip:
            break
        day += timedelta(days=1)
    hh, mm = (int(x) for x in slot["time"].split(":"))
    at = datetime.combine(day, time(hh, mm), tzinfo=now.tzinfo)
    if at <= now:
        return None
    out = {**slot, "date": day.isoformat(), "at": at.isoformat(),
           "spoken": f"{spoken_date(day, now.date())} at {spoken_time(at.timetz())}"}
    return out


def _resolve(value: Any, schema: dict, facts: dict, now: datetime, skip: set[int]) -> Any:
    kind = schema.get("x-demo-type")
    if kind == "relative_slot" and isinstance(value, dict):
        return _resolve_slot(value, now, skip)
    if kind == "relative_day" and isinstance(value, int):
        d = now.date() + timedelta(days=value)
        return {"offset": value, "date": d.isoformat(), "spoken": spoken_date(d, now.date())}
    if kind == "text_template" and isinstance(value, str):
        return _fill(value, facts)
    if isinstance(value, dict):
        props = schema.get("properties", {})
        return {k: _resolve(v, props.get(k, {}), facts, now, skip) for k, v in value.items()}
    if isinstance(value, list):
        items = schema.get("items", {})
        resolved = [_resolve(v, items, facts, now, skip) for v in value]
        if items.get("x-demo-type") == "relative_slot":
            # Past slots are dropped; slots shifted onto the same moment are de-duplicated.
            seen, kept = set(), []
            for s in resolved:
                if s is not None and s["at"] not in seen:
                    seen.add(s["at"])
                    kept.append(s)
            return sorted(kept, key=lambda s: s["at"])
        return resolved
    return value


def resolve_dataset(data: dict, schema: dict, facts: dict[str, str | None], *, now: datetime,
                    tz: str, skip_weekdays: list[str] | tuple[str, ...] = ()) -> ResolvedContext:
    """Resolve a dataset that has ALREADY passed validate_dataset() for these facts."""
    zone = ZoneInfo(tz)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    local_now = now.astimezone(zone)
    skip = {WEEKDAYS.index(d.lower()) for d in skip_weekdays}
    if len(skip) >= 7:
        raise ValueError("skip_weekdays cannot skip every day")
    resolved = _resolve(data, schema, facts, local_now, skip)
    return ResolvedContext(data=resolved, resolved_at=local_now.isoformat(), timezone=tz)
