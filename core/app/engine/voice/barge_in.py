"""Keeping the caller's words when a barge-in cancels the turn that was planning them.

THE BUG THIS FIXES. A voice turn runs inline on TurnBridge's pipecat frame task, and that
task is cancelled the moment pipecat decides a new user turn has started
(UserTurnProcessor -> broadcast_interruption -> FrameProcessor._start_interruption). While
the bot is silent -- which is exactly the 2-20s it spends planning a turn -- the start
strategy (voice_ws.py's MinWordsUserTurnStartStrategy) needs ONE interim word to fire. So
a caller who says "I want to book a site visit", pauses, then adds "tomorrow morning" (or
just "hello?") cancels the first turn mid-plan. bridge.py then rolls the DB back, which is
correct for consistency but also erases the customer Message: the request is in neither
the DB history nor working memory. The next turn plans "tomorrow morning" alone -- no
booking intent anywhere in its context -- and the planner reaches for whatever that
fragment looks like on its own (a greeting, a status check, an unrelated lookup), or for
no tool at all. A chat turn can never lose a message this way, which is why the same
conversation behaves correctly typed and wrongly spoken.

The fix: a cancelled, rolled-back turn's text is carried into the next transcript, so the
planner sees "I want to book a site visit tomorrow morning" -- what the caller actually
said, as one request. Pure functions here; bridge.py owns the state and the wiring.
"""
import re
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class CarryOver:
    """The words of a turn that was cancelled before it was persisted."""
    text: str
    stt_confidence: float | None
    # When the FIRST carried utterance was cancelled. A merged turn that is itself cancelled
    # keeps this original stamp, so the max-age bound measures how old the oldest carried
    # words are, not just the latest fragment.
    since: float


def _norm(text: str) -> str:
    return re.sub(r"[^\w]+", " ", (text or "").lower()).strip()


def carry(previous: CarryOver | None, text: str, stt_confidence: float | None,
          now: float | None = None) -> CarryOver:
    """The carry-over left behind by a cancelled turn whose (possibly already merged) text
    was ``text``. ``previous`` is the carry-over that turn consumed, if any -- its stamp is
    kept so a chain of cancellations still ages out from the first one."""
    now = time.monotonic() if now is None else now
    return CarryOver(text=text.strip(), stt_confidence=stt_confidence,
                     since=previous.since if previous is not None else now)


def merge(carried: CarryOver | None, text: str, stt_confidence: float | None, *,
          max_age_s: float, now: float | None = None
          ) -> tuple[str, float | None, str]:
    """Fold a carried utterance into the transcript about to run.

    Returns ``(text, stt_confidence, outcome)`` where outcome is one of:
      none     -- nothing carried; inputs returned unchanged
      expired  -- carried words older than max_age_s; dropped, new text runs alone
      repeated -- the caller already said the carried words again; not duplicated
      merged   -- carried words prepended to the new text

    The merged turn's STT confidence is the LOWER of the two, so the low-confidence
    guardrail (confidence.py) still sees it if either half was poorly heard.
    """
    if carried is None or not carried.text:
        return text, stt_confidence, "none"
    now = time.monotonic() if now is None else now
    if now - carried.since > max_age_s:
        return text, stt_confidence, "expired"
    # Whole-word containment (padded with spaces), so a carried "no" is not "repeated" by
    # "i know" -- only a genuine re-say of the same words is.
    _c = _norm(carried.text)
    if _c and f" {_c} " in f" {_norm(text)} ":
        return text, stt_confidence, "repeated"
    known = [c for c in (carried.stt_confidence, stt_confidence) if c is not None]
    return (f"{carried.text} {text.strip()}".strip(),
            min(known) if known else None, "merged")
