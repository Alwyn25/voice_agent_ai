"""Scope matching for advisory instructions.

An advisory row carries a ``scope`` = ``{direction, state, intent, valid_from,
valid_until}`` where every key is optional. A row applies to a turn when EVERY key it
DOES set matches the turn; an unset key is a wildcard. This is deliberately deterministic
and enumerable (spec §2.2): every applicable row is applied every turn, never selected by
similarity.

Kept pure (no DB, no I/O) so it is trivially unit-testable and cheap to run per turn.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

# normalize_intent maps the planner's free-form label onto a canonical one, so an
# intent-scoped advisory ("...when they ask about loans") matches the same way a routing
# rule would. Advisory intent-scoping is best-effort by design (a missed advisory nudge is
# not an incident); routing gets the stricter closed-vocabulary treatment in Phase 2.
from app.engine.planner.confidence import normalize_intent


@dataclass(frozen=True)
class TurnScope:
    """The turn's own facts an advisory scope is matched against."""
    direction: str | None = None      # inbound | outbound
    state: str | None = None          # customer.current_state
    intent: str | None = None         # planner-resolved intent (free-form)
    today: date | None = None


def _parse_date(value) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        # A malformed scope date must never crash a live turn; treat it as "unbounded"
        # on that edge so the row still behaves, and surface the problem at author time.
        return None


def matches(scope: dict | None, turn: TurnScope) -> bool:
    """True when ``scope`` (a stored jsonb dict) applies to ``turn``."""
    scope = scope or {}

    want_dir = scope.get("direction")
    if want_dir and turn.direction and str(want_dir).lower() != str(turn.direction).lower():
        return False

    want_state = scope.get("state")
    if want_state and turn.state and str(want_state) != str(turn.state):
        return False

    want_intent = scope.get("intent")
    if want_intent:
        turn_intent = normalize_intent(turn.intent) or (turn.intent or "")
        if normalize_intent(str(want_intent)) not in (turn_intent, str(want_intent).lower()):
            return False

    today = turn.today or date.today()
    vf = _parse_date(scope.get("valid_from"))
    if vf and today < vf:
        return False
    vu = _parse_date(scope.get("valid_until"))
    if vu and today > vu:
        return False

    return True


# ---------------------------------------------------------------------------------------
# Built-in self-test — runs without pytest:  python -m app.custom_instructions.scope
# ---------------------------------------------------------------------------------------
def _selftest() -> int:
    failures: list[str] = []

    def check(name, cond):
        if not cond:
            failures.append(name)

    t = TurnScope(direction="inbound", state="qualified",
                  intent="loan_enquiry", today=date(2026, 9, 4))

    check("empty_is_wildcard", matches({}, t) is True)
    check("dir_match", matches({"direction": "inbound"}, t) is True)
    check("dir_mismatch", matches({"direction": "outbound"}, t) is False)
    check("dir_case_insensitive", matches({"direction": "INBOUND"}, t) is True)
    check("state_match", matches({"state": "qualified"}, t) is True)
    check("state_mismatch", matches({"state": "new"}, t) is False)
    # normalize_intent bridges the planner's free-form label to the scoped canonical one.
    check("intent_normalized", matches({"intent": "loan"}, t) is True)
    check("intent_exact", matches({"intent": "loan_enquiry"}, t) is True)
    check("intent_mismatch", matches({"intent": "booking_status"}, t) is False)
    # Date window.
    check("in_window", matches({"valid_from": "2026-01-01",
                                "valid_until": "2026-12-31"}, t) is True)
    check("before_window", matches({"valid_from": "2026-10-01"}, t) is False)
    check("after_window", matches({"valid_until": "2026-08-31"}, t) is False)
    check("open_ended_from", matches({"valid_from": "2026-09-04"}, t) is True)  # today == from
    # A malformed date must never crash a live turn — treated as unbounded on that edge.
    check("malformed_date_unbounded", matches({"valid_until": "not-a-date"}, t) is True)
    # A turn with no signal for a set key still matches (only a CONFLICT excludes).
    blank = TurnScope(today=date(2026, 9, 4))
    check("no_turn_signal_matches", matches({"direction": "inbound"}, blank) is True)

    if failures:
        print(f"scope self-test: {len(failures)} FAILED -> {failures}")
        return 1
    print("scope self-test: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
