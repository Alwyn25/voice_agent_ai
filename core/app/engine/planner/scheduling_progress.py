"""Deterministic guards for the ONE thing a scheduling turn must never do: narrate.

WHY THIS EXISTS. A phone agent has no way to say a second thing without the caller speaking
first. So a reply that PROMISES a lookup — "let me check the available appointment times for
you" — with no tool behind it does not defer the work by a moment; it defers it by a whole
customer utterance, and the caller has to say "ok" to push the agent through a step it was
already able to take. Repeated, that is the stall that makes an agent feel broken:

    Caller: what are the available options
    Agent:  Could you specify what options you're interested in?
    Caller: appointment time
    Agent:  Let me check the available appointment times for DSR Highland Greenz or
            DSR Techno Cube in Bengaluru for you.          <- nothing ran. Turn over.

THE ROOT CAUSE this backs up. ``turn_runner._next_scheduling_lookup`` already knows the next
deterministic step of an in-flight booking (locate_centre -> search_slots), but it was only
ever reachable from INSIDE the chain loop — which runs only after some tool has already
executed this turn. A turn where the planner narrates instead of naming any tool never
entered the loop, so nothing advanced the flow. The structural fix hoists that driver to the
top level; this module supplies the two pure judgements it needs, plus the honest fallback
for the residual case where no lookup can run (a multi-site tenant with no site chosen yet).

Two DIFFERENT kinds of text are matched here, and the distinction matters:

  * ``is_scheduling_intent`` reads an intent LABEL — produced by our own classifier, from a
    closed-ish vocabulary. Not the caller's raw words; the semantic read of them.
  * ``reads_like_lookup_promise`` reads the AGENT'S OWN drafted sentence. This is not an
    attempt to interpret a customer (the failure mode the P1 routing-regex deletion was
    about) — it is detecting an ENGINE fault in text the engine itself just wrote, where a
    fixed set of phrasings is exactly what a model produces.

Pure and dependency-free (stdlib ``re`` only), so it unit-tests without a database or any
engine types — run ``python -m app.engine.planner.scheduling_progress`` for the self-test.
"""
from __future__ import annotations

import re

# Intent labels that mean "this turn is about slots, times, or the act of booking".
#
# The classifier writes these in its own words (understanding.py deliberately leaves `intent`
# a free string), so this matches on STEMS rather than an enum: "book_appointment", "booking",
# "reschedule", "schedule_visit", "appointment_time", "availability_query", "slot_selection"
# all have to land. `availab` covers "availability"/"available_options" — the exact label the
# stalled turn above carried.
_SCHEDULING_INTENT_RE = re.compile(
    r"book|reschedul|schedul|appointment|slot|availab|site[_ ]?visit|viewing|time[_ ]?pref",
    re.I)

# Subjects a lookup promise has to be ABOUT before it counts as a scheduling stall. Without
# this, "let me check the price for you" — a knowledge turn, handled by an entirely different
# path — would be dragged into the booking machine.
_LOOKUP_SUBJECT = (r"(?:slot|slots|time|times|timing|timings|availab\w*|appointment|"
                   r"appointments|visit|visits|booking|bookings|schedule|opening\w*)")

# The agent narrating a lookup it did not perform. Every alternative requires BOTH a promise
# verb and a scheduling subject, so a reply that merely mentions times ("we have 10 AM and
# 4 PM free") never matches — only one that says it is going to go and find them.
_LOOKUP_PROMISE_ALTERNATIVES = (
    # "let me check the available times", "I'll look up the slots", "let me find a time"
    r"\b(?:let me|i'?ll|i will|i'?m going to|i am going to|allow me to|give me a moment to|"
    r"one moment while i|hold on while i)\b[^.?!]{0,40}"
    r"\b(?:check|look\w*|find|fetch|pull\w*|see|search|retriev\w*)\b[^.?!]{0,40}",
    # "checking availability for you now", "just checking the slots"
    r"\b(?:just )?(?:checking|looking (?:up|into)|fetching|pulling up|searching)\b"
    r"[^.?!]{0,40}",
    # "I'll get back to you with the times", "let me come back to you on availability"
    r"\b(?:get back to you|come back to you|revert to you)\b[^.?!]{0,40}",
)
_LOOKUP_PROMISE_RE = re.compile(
    "|".join(alt + _LOOKUP_SUBJECT for alt in _LOOKUP_PROMISE_ALTERNATIVES), re.I)

# Said when the flow genuinely CANNOT advance itself and the reply was about to narrate.
#
# Neither line promises anything the engine is not doing this instant, and both end in a
# question the caller can answer — which is what turns a dead turn back into a live one. The
# site question comes first because a multi-site tenant cannot search anything until it is
# answered; that is the one gap the caller must fill, and it is genuinely theirs to fill.
STALLED_ASK_CENTRE = ("Which of our sites would you like to visit? Once you tell me, I can "
                      "read out the exact times that are free.")
STALLED_ASK_TIME = ("Which day suits you, and roughly what time — morning or evening? "
                    "I'll read out the exact slots that are free for it.")


def is_scheduling_intent(label: str | None) -> bool:
    """True when an intent label means the turn is about slots, times or booking.

    Reads OUR classifier's label, never the caller's raw words — the semantic routing has
    already happened by the time this is asked.
    """
    return bool(_SCHEDULING_INTENT_RE.search(str(label or "")))


def reads_like_lookup_promise(text: str | None) -> bool:
    """True when the AGENT's own sentence promises a scheduling lookup it has not performed.

    Used as a backstop only: a turn where this fires and no tool ran is, by definition, a
    turn the caller has to spend an utterance on for nothing.
    """
    return bool(_LOOKUP_PROMISE_RE.search(text or ""))


def stalled_reply(*, centre_known: bool) -> str:
    """The honest question to ask in place of a narrated lookup that cannot be performed."""
    return STALLED_ASK_TIME if centre_known else STALLED_ASK_CENTRE


# ---------------------------------------------------------------------------------------
# Built-in self-test — runs without pytest.
#   python -m app.engine.planner.scheduling_progress
# ---------------------------------------------------------------------------------------
def _selftest() -> int:
    failures: list[str] = []

    def check(name, cond):
        if not cond:
            failures.append(name)

    # --- intent labels that ARE scheduling ---
    for label in ("book_appointment", "booking", "book", "reschedule",
                  "reschedule_appointment", "schedule_visit", "scheduling",
                  "appointment_time", "availability_query", "available_options",
                  "slot_selection", "site_visit", "viewing_request", "cancel_appointment"):
        check(f"sched::{label}", is_scheduling_intent(label))

    # --- intent labels that are NOT scheduling ---
    for label in ("product_question", "greeting", "affirm", "unclear", "complaint",
                  "pricing_query", "smalltalk", "request_human", None, ""):
        check(f"not_sched::{label}", not is_scheduling_intent(label))

    # --- the agent narrating a lookup it did not run ---
    for draft in (
            # the exact reported sentence
            "Let me check the available appointment times for DSR Highland Greenz or "
            "DSR Techno Cube in Bengaluru for you.",
            "Let me check the available slots for the evening of August 12th.",
            "I'll look up the available times for you.",
            "I will find an available slot in the evening for August 12th.",
            "Let me pull up the appointment availability for tomorrow.",
            "One moment while I check the visit timings.",
            "Just checking the slots for Friday.",
            "Let me see what appointment times we have free.",
            "I'm going to search for available slots now.",
            "I'll get back to you with the available times."):
        check(f"promise::{draft[:32]}", reads_like_lookup_promise(draft))

    # --- replies that ACTUALLY deliver, or ask, must NOT match ---
    for draft in (
            "We have 10:00 AM and 4:00 PM free on Tuesday — which suits you?",
            "Your visit is confirmed for Tuesday, 18 August at 5:00 PM.",
            "Which day works for you?",
            "The site is open from ten to six-thirty every day.",
            "Shall I book your visit for Tuesday at 5 PM?",
            "I don't see a visit on record for you. Could I have your name?",
            # a lookup promise about a NON-scheduling subject belongs to another path
            "Let me check the price of the 3 BHK for you.",
            "I'll look up the floor plan details.",
            "",
            None):
        check(f"no_promise::{str(draft)[:32]}", not reads_like_lookup_promise(draft))

    # --- the stalled fallback asks, and never promises a lookup of its own ---
    check("stall_centre", stalled_reply(centre_known=False) == STALLED_ASK_CENTRE)
    check("stall_time", stalled_reply(centre_known=True) == STALLED_ASK_TIME)
    for line in (STALLED_ASK_CENTRE, STALLED_ASK_TIME):
        check(f"stall_asks::{line[:20]}", "?" in line)
        check(f"stall_not_promise::{line[:20]}", not reads_like_lookup_promise(line))

    if failures:
        print(f"scheduling_progress self-test: {len(failures)} FAILED -> {failures}")
        return 1
    print("scheduling_progress self-test: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
