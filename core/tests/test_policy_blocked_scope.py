"""planner.is_no_further_contact_block: only a block verdict that means "stop
contacting/selling to this caller entirely" (DNC, marketing opt-out) may switch the
responder into no-pitch "policy_blocked" mode. A tool-scoped business rule (a tenant's
"no site visits on Sundays", the built-in weekend-booking restriction) must not -- the
caller is still a hot lead and "Would Saturday work?" is the right answer, not something
to be suppressed. Pure, no DB.

    cd core && python -m pytest tests/test_policy_blocked_scope.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.planner.planner import is_no_further_contact_block  # noqa: E402


def _verdict(scope):
    return SimpleNamespace(body={"scope": scope, "reason": "because"})


def test_dnc_and_marketing_scopes_are_no_further_contact():
    assert is_no_further_contact_block(_verdict("all"))
    assert is_no_further_contact_block(_verdict("marketing"))


def test_tool_scoped_business_rules_are_not():
    # The exact shape of a tenant custom instruction ("don't book site visits on Sundays")
    # and the built-in no_weekend_booking_for_restricted_regions rule.
    assert not is_no_further_contact_block(_verdict("tool:book_appointment"))
    assert not is_no_further_contact_block(_verdict("tool:reschedule_appointment"))
    assert not is_no_further_contact_block(_verdict("outbound_call"))


def test_missing_scope_is_not_no_further_contact():
    # A tenant-compiled custom instruction's block verdict carries no scope key at all
    # (app/custom_instructions/compiler.py only emits {"block": {"reason": "..."}}).
    assert not is_no_further_contact_block(SimpleNamespace(body={"reason": "because"}))
