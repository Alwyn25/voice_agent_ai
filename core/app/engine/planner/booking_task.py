"""The BookingTask — an explicit state machine for the multi-turn site-visit booking.

A site-visit booking is a transaction with state, but the rest of the engine reconstructs
that state per turn from ~10 scattered session keys (``pending_action``, ``wanted_tool``,
``offered_slots``, ``registration``, ``appointment_id`` …). Whether a call succeeds then
depends on the ORDER the turns happen to arrive in — the definition of an emergent-state
bug, and the single root cause behind the whole family of booking/scheduling continuity
failures documented in ``docs/booking-task-architecture.md``.

This module makes the booking one object the orchestrator owns. The LLM extracts entities
and phrases sentences; it never decides what state the booking is in — six deterministic
rules do (see ``turn_runner.process_turn``). This file holds the object, the invariant it is
gated on (:meth:`BookingTask.ready_to_book`), the book-vs-reschedule decision
(:func:`decide_kind`), and the projection of the session keys onto the object
(:func:`project`) that those rules read each turn.

Deliberately dependency-free: no SQLAlchemy, no Redis, no tool types. Everything it touches
is a plain dict or a duck-typed outcome (``.tool_name`` / ``.status`` / ``.output``), so it
is pure and unit-testable in isolation — run ``python -m app.engine.planner.booking_task``
for the built-in self-test.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Any, Iterable

# The lifecycle a booking moves through, in order. Named so the trace and the tests can
# assert on a status rather than on the presence/absence of a handful of session keys.
BOOKING_STATUSES = (
    "gathering",     # collecting configuration / requested_day / time_pref
    "searching",     # a day-scoped search_slots is being run for the requested day
    "offering",      # real slots for the requested day have been read back
    "picking",       # the customer named a time that IS in the offered set
    "registering",   # a create_customer child task is in flight; booking slots kept
    "confirming",    # "shall I book 2 PM?" — waiting for the customer's yes
    "booking",       # ready_to_book() is true; the write tool is firing
    "done",          # book/reschedule succeeded
    "abandoned",     # the customer walked away from the booking
)

# The write tools this machine sequences, and the statuses the executor uses for "it
# worked" (``skipped_duplicate`` carries the same real output as ``success``, deduped).
_BOOK_TOOLS = ("book_appointment", "reschedule_appointment")
_SUCCEEDED = ("success", "skipped_duplicate")


@dataclass
class BookingTask:
    """One booking, owned by the orchestrator and carried across turns.

    ``slots`` holds what has been gathered (configuration, requested_day, time_pref,
    centre_id, verified_slot_start, appointment_id). ``prereqs`` holds what must be true
    before the write tool may fire. ``confirmed`` is the customer's yes to the read-back.
    Execution is gated on exactly one predicate — :meth:`ready_to_book` — so no scattered
    combination of keys can ever fabricate a "booked" claim.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    kind: str = "book"                                  # "book" | "reschedule" (by FACT)
    status: str = "gathering"
    slots: dict = field(default_factory=dict)
    prereqs: dict = field(default_factory=lambda: {"customer_exists": False})
    confirmed: bool = False                             # customer said yes to the read-back
    created_turn: int = 0
    updated_turn: int = 0

    def ready_to_book(self) -> bool:
        """The one invariant every write is gated on.

        Verified slot ∈ offered set, every prerequisite satisfied, and the customer has
        agreed. Nothing else can put a booking on the wire — so a reply can only truthfully
        claim a booking when this held and the write actually ran.
        """
        return (bool(self.slots.get("verified_slot_start"))
                and bool(self.prereqs)
                and all(self.prereqs.values())
                and self.confirmed)

    def ready_to_reschedule(self) -> bool:
        """A reschedule may fire only for a genuinely DIFFERENT time (RC2).

        ready_to_book() already demands a verified slot, the customer, and consent. A
        reschedule adds one thing: the new slot must not be the one they already have.
        Without this, a reschedule staged and read back with the CURRENT time — "shall I
        reschedule it?" before any new time was chosen — passed every check and "moved" the
        visit to the hour it was already at. When the current slot is unknown (never looked
        up) the extra guard cannot apply, so it falls back to ready_to_book().
        """
        if not self.ready_to_book():
            return False
        cur = self.slots.get("current_slot_start")
        new = self.slots.get("verified_slot_start")
        if not cur:
            return True
        return not _same_instant(new, cur)

    # -- serialisation: the task lives on working memory (Redis) as a plain dict --------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "BookingTask | None":
        if not data:
            return None
        known = {f for f in cls.__dataclass_fields__}          # noqa: SLF001
        return cls(**{k: v for k, v in data.items() if k in known})

    def summary_line(self) -> str:
        """One compact line for the prompt's BOOKING IN PROGRESS block (Rule 6).

        Answered from state, so "is it booked?" and every read-back are grounded rather
        than guessed. Deliberately terse — every word is latency the caller hears.
        """
        s = self.slots
        bits = [f"kind: {self.kind}", f"status: {self.status}"]
        if s.get("configuration"):
            bits.append(f"configuration: {s['configuration']}")
        if s.get("requested_day"):
            bits.append(f"day: {s['requested_day']}")
        if s.get("time_pref"):
            bits.append(f"time: {s['time_pref']}")
        if s.get("verified_slot_start"):
            # ONE canonical rendering of the time, formatted from the timestamp itself, so
            # every turn quotes the same words. Handing the model a raw ISO string let it
            # re-derive the time turn to turn and disagree with itself ("2 PM" one turn,
            # "3 PM" the next) — the slot never changed; only the model's rendering did.
            bits.append(f"chosen slot: {_format_slot(s['verified_slot_start'])}")
        bits.append("customer: "
                    + ("registered" if self.prereqs.get("customer_exists") else "not yet registered"))
        bits.append("confirmed: " + ("yes" if self.confirmed else "no"))
        return " · ".join(bits)


# ---------------------------------------------------------------------------------------
# Rule 3 — book vs reschedule BY FACT, never by the model's intent label.
# ---------------------------------------------------------------------------------------
# Journey states that mean the customer ALREADY HAS a visit on the books. book_appointment
# is deliberately not allowed in site_visit_scheduled (reschedule is the right tool there),
# so classifying such a customer as "book" and forcing that tool gets it blocked at
# execution — "Tool not allowed in state 'site_visit_scheduled'". The state is the fact:
# an existing visit means the change they want is a RESCHEDULE, even before my_appointments
# has run to load the id. Kept narrow — offer_shared / negotiation / booking_token /
# documentation all permit a genuinely NEW booking too, so they are NOT listed here.
RESCHEDULE_STATES = ("site_visit_scheduled",)


def decide_kind(session: dict, executed: Iterable[Any] | None = None,
                current_state: str | None = None) -> str:
    """``"reschedule"`` iff the customer has an active appointment; otherwise ``"book"``.

    This is the fix for the book↔reschedule confusion (Tab 1): nothing was booked, yet the
    model relabelled "change the time" as a reschedule, deadlocking on a required
    ``slot_start``. The decision is data, not the ``intent`` string — an appointment exists
    or it does not. "Exists" is read from four independent facts: the appointment id on the
    session, the id of a visit already on file from an earlier call
    (``on_file_appointment_id``), a ``my_appointments`` result this turn, and the customer's
    journey STATE (a lead sitting in ``site_visit_scheduled`` has a visit even if nothing
    looked it up yet). The on-file id sets kind only — it never seeds the slots that render a
    "BOOKING IN PROGRESS" line — so a passive turn from a returning caller stays silent.
    """
    if session.get("appointment_id"):
        return "reschedule"
    if session.get("on_file_appointment_id"):
        return "reschedule"
    if current_state in RESCHEDULE_STATES:
        return "reschedule"
    for o in (executed or []):
        if getattr(o, "tool_name", None) == "my_appointments" \
                and getattr(o, "status", None) in _SUCCEEDED:
            out = getattr(o, "output", None) or {}
            if out.get("has_upcoming") or out.get("live"):
                return "reschedule"
    return "book"


# ---------------------------------------------------------------------------------------
# Small date helpers — kept local so the module has no dependency on turn_runner.
# ---------------------------------------------------------------------------------------
def _slot_date(iso: str | None) -> date | None:
    try:
        return datetime.fromisoformat(iso).date() if iso else None
    except (ValueError, TypeError):
        return None


def _same_instant(a: str | None, b: str | None) -> bool:
    """True when two ISO timestamps name the same moment, even if written differently.

    Offsets can be spelled two ways for the same instant, so the comparison is on parsed
    datetimes, not raw strings. A parse failure falls back to string equality.
    """
    if not a or not b:
        return False
    try:
        return datetime.fromisoformat(a) == datetime.fromisoformat(b)
    except (ValueError, TypeError):
        return a == b


def _format_slot(iso: str | None) -> str:
    """One canonical spoken rendering of a slot time, e.g. "Tuesday 18 August at 2:00 PM".

    Derived from the timestamp so it is identical on every turn — the read-back, the
    "is it booked?" answer, and the confirmation all quote the same words, instead of the
    model reconstructing the time from a raw ISO string and disagreeing with itself.
    """
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return iso
    hour12 = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{dt:%A %d %B} at {hour12}:{dt.minute:02d} {ampm}"


def offered_starts(session: dict) -> set[str]:
    return {s.get("starts_at") for s in (session.get("offered_slots") or [])
            if s.get("starts_at")}


def day_is_covered(session: dict, day: date | None) -> bool:
    """True when a slot for ``day`` is already in ``offered_slots``. Backs Rule 1."""
    if day is None:
        return False
    return any(_slot_date(s.get("starts_at")) == day
               for s in (session.get("offered_slots") or []))


# The planner names its entities freely (the `understood` contract just says
# {"<name>": "<value>"}), so the configuration arrives under whatever key it picked —
# bhk_type, configuration, flat_type, unit_type, property_type, size … Enumerating keys is
# the mistake focus.py's blocklist was written to avoid, so this is key-agnostic: a couple
# of obvious keys, then a value-shape fallback (anything that reads like "2.5 BHK").
_CONFIG_KEYS = ("bhk_type", "configuration", "flat_type", "unit_type", "property_type",
                "bhk", "flat", "unit", "size", "config")
_BHK_VALUE = re.compile(r"\b\d(?:\.\d)?\s*bhk\b", re.I)


def carry_config(entities: dict | None) -> dict:
    """Pull the flat configuration out of the extracted entities, whatever it was named."""
    entities = entities or {}
    for key in _CONFIG_KEYS:
        val = entities.get(key)
        if isinstance(val, str) and val.strip():
            return {"configuration": val.strip()}
    for val in entities.values():
        if isinstance(val, str) and _BHK_VALUE.search(val):
            return {"configuration": val.strip()}
    return {}


def verified_slot_from_session(session: dict) -> str | None:
    """The chosen slot, but ONLY if it is a time a search actually offered this call.

    Mirrors the ``slot_was_offered`` precondition: a ``slot_start`` sitting in a parked
    action is trustworthy only when it is in the offered set, so a stale slot the customer
    never heard cannot be projected as "verified".
    """
    offered = offered_starts(session)
    if not offered:
        return None
    pending = (session.get("pending_action") or {}).get("params") or {}
    start = pending.get("slot_start")
    return start if start in offered else None


# ---------------------------------------------------------------------------------------
# The projection: build the task each turn as a projection of the session keys, log it to
# the trace, and let the deterministic rules in process_turn sequence the booking from it.
# ---------------------------------------------------------------------------------------
def project(session: dict, *, customer_exists: bool, executed: Iterable[Any] | None = None,
            turn_index: int = 0, active_entities: dict | None = None,
            current_state: str | None = None) -> BookingTask:
    """Derive a :class:`BookingTask` from the current session + this turn's tool outcomes.

    Pure and side-effect-free: it reads ``session`` and ``executed`` and returns a fresh
    object. It never mutates the session — persistence is the caller's job — so it is safe
    to call on every turn regardless of the feature flag.
    """
    executed = list(executed or [])
    entities = active_entities or session.get("active_entities") or {}

    # -- carry a prior task's identity/created_turn forward, so the id follows the task --
    prior = BookingTask.from_dict(session.get("booking_task"))

    task = BookingTask(
        id=prior.id if prior else str(uuid.uuid4()),
        created_turn=prior.created_turn if prior else turn_index,
        updated_turn=turn_index,
    )
    task.kind = decide_kind(session, executed, current_state=current_state)
    task.prereqs = {"customer_exists": bool(customer_exists)}

    # -- slots ---------------------------------------------------------------------------
    slots: dict = {}
    slots.update(carry_config(entities))
    if session.get("wanted_day"):
        slots["requested_day"] = session["wanted_day"]
    time_pref = entities.get("time_pref") or entities.get("part_of_day")
    if time_pref:
        slots["time_pref"] = time_pref
    if session.get("centre_id"):
        slots["centre_id"] = session["centre_id"]
    if session.get("appointment_id"):
        slots["appointment_id"] = session["appointment_id"]
    # The time the current appointment sits at, so ready_to_reschedule() can require a
    # DIFFERENT slot rather than letting a reschedule fire against the existing one (RC2).
    if session.get("appointment_current_start"):
        slots["current_slot_start"] = session["appointment_current_start"]

    verified = verified_slot_from_session(session)

    # A booking that actually went through this turn is the strongest signal of all — take
    # its real slot_start straight from the tool output.
    booked_now = next((o for o in executed
                       if getattr(o, "tool_name", None) in _BOOK_TOOLS
                       and getattr(o, "status", None) in _SUCCEEDED), None)
    if booked_now is not None:
        real = (getattr(booked_now, "output", None) or {}).get("starts_at")
        if real:
            verified = real
    if verified:
        slots["verified_slot_start"] = verified
    task.slots = slots

    # -- confirmed -----------------------------------------------------------------------
    # A read-back the customer agreed to is what sets this. The projection cannot see the
    # customer's words directly, so it infers consent from the state the engine already
    # committed: a booking that succeeded was, by construction, confirmed.
    task.confirmed = bool(prior and prior.confirmed) or booked_now is not None

    task.status = _derive_status(session, task, executed, booked_now is not None)

    # "done" is TERMINAL and STICKY. A booking that succeeded on an earlier turn must keep
    # reporting as done — otherwise the very next turn recomputes status from scratch, and
    # with the appointment now on file but no pending slot it regresses to "offering",
    # telling the model the visit is NOT booked. That is exactly what made "is it booked?"
    # get answered with "I'm not seeing a booking" one turn after a real, successful book.
    # Carry the completed booking forward (its slot, its consent, its kind) until a FRESH
    # booking/reschedule succeeds this turn and supersedes it — or the visit is cancelled,
    # after which it must stop reporting as booked.
    cancelled_now = any(getattr(o, "tool_name", None) == "cancel_appointment"
                        and getattr(o, "status", None) in _SUCCEEDED for o in executed)

    # A reschedule of the COMPLETED booking is in flight — the customer has a done visit and
    # is now moving it. The done booking must stop masquerading as the live task, or its slot
    # and its consent leak into the new reschedule.
    #
    # THE BUG THIS FIXES: caller books the 18th at 11:00 (task → done, verified_slot_start =
    # the 18th). Next they say "reschedule … the 19th at 2pm". Their new pick has not been
    # searched/offered yet, so it is dropped as unverified — and the sticky-done block below
    # then backfilled verified_slot_start with the 18th (the CURRENT appointment time) and
    # forced confirmed=True. Rule 4 (RC2) compared the new time against that stale "verified"
    # slot, found they matched the current one, blocked every different time as "the same
    # time you already have", and looped "what day and time would you like?" forever. Leaving
    # the freshly-projected task (no verified slot, not yet confirmed) lets the search force
    # run for the new day and the reschedule proceed on a real, newly-chosen slot.
    pending_tool = (session.get("pending_action") or {}).get("tool")
    ran = {getattr(o, "tool_name", None) for o in executed}
    _reschedule_in_flight = (
        pending_tool == "reschedule_appointment"
        or "reschedule_appointment" in ran
        or session.get("wanted_tool") == "reschedule_appointment")

    if (prior is not None and prior.status == "done" and booked_now is None
            and not cancelled_now and not _reschedule_in_flight):
        task.status = "done"
        task.confirmed = True
        task.kind = prior.kind
        if not task.slots.get("verified_slot_start") and prior.slots.get("verified_slot_start"):
            task.slots["verified_slot_start"] = prior.slots["verified_slot_start"]
    elif _reschedule_in_flight and booked_now is None:
        # Rescheduling: the OLD booking's consent does not carry to the NEW time. Require a
        # fresh yes to the new slot (Rule 4 also re-derives consent per turn, but the prompt's
        # BOOKING IN PROGRESS block reads this, and "confirmed: yes" here would tell the model
        # the move is already agreed). The stale verified slot is likewise not carried — it is
        # only ever the current appointment time, never the customer's new choice.
        task.confirmed = booked_now is not None
    return task


def _derive_status(session: dict, task: BookingTask, executed: list,
                   booked_now: bool) -> str:
    pending = session.get("pending_action") or {}
    reg = session.get("registration") or {}
    wanted = session.get("wanted_tool")
    ran = {getattr(o, "tool_name", None) for o in executed}

    if booked_now or task.kind == "reschedule" and any(
            getattr(o, "tool_name", None) == "reschedule_appointment"
            and getattr(o, "status", None) in _SUCCEEDED for o in executed):
        return "done"

    # A registration is the only thing standing between a chosen slot and the booking.
    booking_goal = (reg.get("goal") in _BOOK_TOOLS) or wanted in _BOOK_TOOLS
    if reg.get("goal") in _BOOK_TOOLS and not task.prereqs.get("customer_exists"):
        return "registering"
    if "create_customer" in ran and not task.prereqs.get("customer_exists"):
        return "registering"

    if task.ready_to_book():
        return "booking"

    if task.slots.get("verified_slot_start"):
        # A chosen slot with a customer on file and no consent yet is a read-back in
        # flight; without a customer it is the registration step.
        if task.prereqs.get("customer_exists"):
            return "confirming"
        return "picking"

    if "search_slots" in ran or wanted == "search_slots" \
            or pending.get("tool") == "search_slots":
        # A search ran but produced no verified pick yet — either we are offering what it
        # found, or (no slots at all) still searching.
        return "offering" if session.get("offered_slots") else "searching"

    if session.get("offered_slots"):
        return "offering"

    if booking_goal or task.slots.get("requested_day") or task.slots.get("configuration"):
        return "gathering"

    return "gathering"


def store(session: dict, task: BookingTask) -> None:
    """Persist the projected task onto the session (one per conversation)."""
    session["booking_task"] = task.to_dict()


# ---------------------------------------------------------------------------------------
# Built-in self-test — runs without pytest (which is not installed in every environment).
#   python -m app.engine.planner.booking_task
# ---------------------------------------------------------------------------------------
class _FakeOutcome:
    def __init__(self, tool_name, status, output=None):
        self.tool_name, self.status, self.output = tool_name, status, output or {}


def _selftest() -> int:
    failures: list[str] = []

    def check(name, cond):
        if not cond:
            failures.append(name)

    # ready_to_book: all three legs required
    t = BookingTask(slots={"verified_slot_start": "2026-08-19T14:00:00+05:30"},
                    prereqs={"customer_exists": True}, confirmed=True)
    check("ready_all_true", t.ready_to_book())

    # ready_to_reschedule (RC2): a NEW time is required, not the current one.
    _r_ok = BookingTask(kind="reschedule", confirmed=True, prereqs={"customer_exists": True},
                        slots={"verified_slot_start": "2026-08-19T11:00:00+05:30",
                               "current_slot_start": "2026-08-18T17:00:00+05:30"})
    check("resched_new_time_ok", _r_ok.ready_to_reschedule())
    _r_same = BookingTask(kind="reschedule", confirmed=True, prereqs={"customer_exists": True},
                          slots={"verified_slot_start": "2026-08-18T17:00:00+05:30",
                                 "current_slot_start": "2026-08-18T17:00:00+05:30"})
    check("resched_same_time_blocked", not _r_same.ready_to_reschedule())
    # Same instant written with a different offset spelling still counts as the same time.
    _r_same_offset = BookingTask(kind="reschedule", confirmed=True,
                                 prereqs={"customer_exists": True},
                                 slots={"verified_slot_start": "2026-08-18T11:30:00+00:00",
                                        "current_slot_start": "2026-08-18T17:00:00+05:30"})
    check("resched_same_instant_offset_blocked", not _r_same_offset.ready_to_reschedule())
    # Current slot unknown ⇒ fall back to ready_to_book (cannot enforce difference).
    _r_unknown = BookingTask(kind="reschedule", confirmed=True,
                             prereqs={"customer_exists": True},
                             slots={"verified_slot_start": "2026-08-19T11:00:00+05:30"})
    check("resched_unknown_current_falls_back", _r_unknown.ready_to_reschedule())
    check("resched_needs_consent", not BookingTask(
        kind="reschedule", confirmed=False, prereqs={"customer_exists": True},
        slots={"verified_slot_start": "2026-08-19T11:00:00+05:30",
               "current_slot_start": "2026-08-18T17:00:00+05:30"}).ready_to_reschedule())
    check("ready_needs_confirm", not BookingTask(
        slots={"verified_slot_start": "x"}, prereqs={"customer_exists": True},
        confirmed=False).ready_to_book())
    check("ready_needs_slot", not BookingTask(
        prereqs={"customer_exists": True}, confirmed=True).ready_to_book())
    check("ready_needs_customer", not BookingTask(
        slots={"verified_slot_start": "x"}, prereqs={"customer_exists": False},
        confirmed=True).ready_to_book())

    # decide_kind: fact, not label
    check("kind_book_default", decide_kind({}, []) == "book")
    check("kind_reschedule_by_appt",
          decide_kind({"appointment_id": "a"}, []) == "reschedule")
    check("kind_reschedule_by_lookup", decide_kind({}, [
        _FakeOutcome("my_appointments", "success", {"has_upcoming": True})]) == "reschedule")
    check("kind_book_when_no_upcoming", decide_kind({}, [
        _FakeOutcome("my_appointments", "success", {"has_upcoming": False, "live": []})]) == "book")
    check("kind_reschedule_by_state",
          decide_kind({}, [], current_state="site_visit_scheduled") == "reschedule")
    check("kind_book_other_state", decide_kind({}, [], current_state="lead") == "book")
    # A visit already on file (booked on an earlier call) is a reschedule signal too...
    check("kind_reschedule_by_on_file",
          decide_kind({"on_file_appointment_id": "a"}, []) == "reschedule")
    # ...but it must NOT, on its own, fabricate a "BOOKING IN PROGRESS" line: with no slot,
    # no offered times and nothing in flight, the projection stays at gathering with empty
    # slots, so _booking_summary()'s "slots or status != gathering" gate keeps it silent.
    _on_file_only = project({"on_file_appointment_id": "a"}, customer_exists=True,
                            current_state="lead")
    check("on_file_kind_reschedule", _on_file_only.kind == "reschedule")
    check("on_file_no_slots", not _on_file_only.slots)
    check("on_file_status_gathering", _on_file_only.status == "gathering")
    check("project_state_reschedule",
          project({}, customer_exists=True, current_state="site_visit_scheduled").kind
          == "reschedule")

    # Tab 1: nothing booked, model says reschedule -> projection must stay "book"
    tab1 = {"wanted_day": "2026-08-19", "offered_slots": [], "wanted_tool": "search_slots"}
    pt = project(tab1, customer_exists=False)
    check("tab1_kind_book", pt.kind == "book")
    check("tab1_not_ready", not pt.ready_to_book())

    # configuration is captured whatever key the planner used, and by value shape
    check("cfg_bhk_type", carry_config({"bhk_type": "2.5 BHK"}) == {"configuration": "2.5 BHK"})
    check("cfg_flat_type", carry_config({"flat_type": "2.5 BHK"}) == {"configuration": "2.5 BHK"})
    check("cfg_by_value", carry_config({"interest": "a 2.5 BHK unit"})
          == {"configuration": "a 2.5 BHK unit"})
    check("cfg_none", carry_config({"sentiment": "positive"}) == {})
    check("cfg_in_project", project({"active_entities": {"unit_type": "2.5 BHK"}},
                                    customer_exists=False).slots.get("configuration") == "2.5 BHK")

    # day coverage / verified slot gating
    sess = {"offered_slots": [{"starts_at": "2026-08-19T14:00:00+05:30", "when": "Wed"}]}
    check("day_covered", day_is_covered(sess, date(2026, 8, 19)))
    check("day_uncovered", not day_is_covered(sess, date(2026, 8, 20)))
    sess["pending_action"] = {"params": {"slot_start": "2026-08-19T14:00:00+05:30"}}
    check("verified_in_set", verified_slot_from_session(sess) == "2026-08-19T14:00:00+05:30")
    sess["pending_action"] = {"params": {"slot_start": "2026-08-15T10:00:00+05:30"}}
    check("verified_rejects_stale", verified_slot_from_session(sess) is None)

    # status: registering (chosen slot, no customer, reg goal is a booking)
    reg_sess = {
        "offered_slots": [{"starts_at": "2026-08-19T14:00:00+05:30", "when": "Wed"}],
        "pending_action": {"params": {"slot_start": "2026-08-19T14:00:00+05:30"}},
        "registration": {"goal": "book_appointment"}, "wanted_day": "2026-08-19"}
    rt = project(reg_sess, customer_exists=False)
    check("status_registering", rt.status == "registering")

    # status: confirming (chosen slot, customer on file, not yet confirmed)
    ct = project({**reg_sess, "registration": {}}, customer_exists=True)
    check("status_confirming", ct.status == "confirming")

    # status: done after a successful booking, with the real slot verified
    done = project({"offered_slots": [{"starts_at": "2026-08-19T14:00:00+05:30"}]},
                   customer_exists=True,
                   executed=[_FakeOutcome("book_appointment", "success",
                                          {"starts_at": "2026-08-19T14:00:00+05:30",
                                           "appointment_id": "a"})])
    check("status_done", done.status == "done")
    check("done_confirmed", done.confirmed)
    check("done_verified", done.slots.get("verified_slot_start") == "2026-08-19T14:00:00+05:30")

    # "is it booked?" is answered from state: after search + register but BEFORE the write,
    # the task is not done and not confirmed (Tabs 4, 5, 6 — the customer had to ask).
    pre_book = project(
        {"offered_slots": [{"starts_at": "2026-08-19T17:00:00+05:30", "when": "Wed"}],
         "pending_action": {"params": {"slot_start": "2026-08-19T17:00:00+05:30"}},
         "wanted_day": "2026-08-19"},
        customer_exists=True)
    check("is_it_booked_not_done", pre_book.status != "done")
    check("is_it_booked_not_confirmed", not pre_book.confirmed)
    check("is_it_booked_slot_verified", pre_book.slots.get("verified_slot_start"))

    # Registration keeps the requested day (Tab regression A: registration must not drop
    # the day the customer just asked for).
    keep_day = project(
        {"offered_slots": [{"starts_at": "2026-08-19T14:00:00+05:30"}],
         "pending_action": {"params": {"slot_start": "2026-08-19T14:00:00+05:30"}},
         "registration": {"goal": "book_appointment"}, "wanted_day": "2026-08-19"},
        customer_exists=False)
    check("registration_keeps_day", keep_day.slots.get("requested_day") == "2026-08-19")
    check("registration_keeps_slot",
          keep_day.slots.get("verified_slot_start") == "2026-08-19T14:00:00+05:30")

    # "done" survives the NEXT turn ("is it booked?" one turn after a real booking). After a
    # successful book, appointment_id is on the session and nothing runs on the follow-up
    # turn — the projection must still report done, not regress to offering.
    booked_session = {
        "offered_slots": [{"starts_at": "2026-08-19T14:00:00+05:30"}],
        "appointment_id": "11111111-1111-1111-1111-111111111111"}
    store(booked_session, project(
        booked_session, customer_exists=True,
        executed=[_FakeOutcome("book_appointment", "success",
                               {"starts_at": "2026-08-19T14:00:00+05:30",
                                "appointment_id": "11111111-1111-1111-1111-111111111111"})]))
    check("done_stored", booked_session["booking_task"]["status"] == "done")
    follow_up = project(booked_session, customer_exists=True)   # "is it booked?" — nothing ran
    check("done_sticky_next_turn", follow_up.status == "done")
    check("done_sticky_confirmed", follow_up.confirmed)
    check("done_sticky_slot",
          follow_up.slots.get("verified_slot_start") == "2026-08-19T14:00:00+05:30")

    # A reschedule of the done booking must NOT inherit the old slot or its consent. After
    # booking the 18th, the customer moves the visit; a pending reschedule is in flight and no
    # new slot has been offered yet. The projection must drop the sticky 18th slot and reset
    # confirmed, so RC2 does not compare a new time against the current one and loop forever.
    resched_session = {
        "offered_slots": [{"starts_at": "2026-08-18T11:00:00+05:30"}],
        "appointment_id": "22222222-2222-2222-2222-222222222222",
        "appointment_current_start": "2026-08-18T11:00:00+05:30",
        "pending_action": {"tool": "reschedule_appointment", "params": {}}}
    store(resched_session, project(
        resched_session, customer_exists=True,
        executed=[_FakeOutcome("book_appointment", "success",
                               {"starts_at": "2026-08-18T11:00:00+05:30",
                                "appointment_id": "22222222-2222-2222-2222-222222222222"})]))
    rs = project(resched_session, customer_exists=True)   # "move it to the 19th at 2pm"
    check("reschedule_drops_stale_slot", not rs.slots.get("verified_slot_start"))
    check("reschedule_resets_confirmed", not rs.confirmed)
    check("reschedule_not_ready", not rs.ready_to_reschedule())

    # id follows the task across turns
    s2 = {}
    store(s2, project({}, customer_exists=False, turn_index=1))
    first_id = s2["booking_task"]["id"]
    store(s2, project(s2, customer_exists=False, turn_index=2))
    check("id_stable", s2["booking_task"]["id"] == first_id)
    check("created_turn_stable", s2["booking_task"]["created_turn"] == 1)
    check("updated_turn_moves", s2["booking_task"]["updated_turn"] == 2)

    if failures:
        print(f"booking_task self-test: {len(failures)} FAILED -> {failures}")
        return 1
    print("booking_task self-test: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
