"""The booking-claim guard (app/engine/planner/booking_claim.py): a reply may not tell the
caller a visit exists unless one really does. Pure function, no DB or LLM.

    cd core && python -m pytest tests/test_booking_claim.py -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.planner import booking_claim as bc  # noqa: E402

CLAIMS = [
    # adjacent forms (already caught before this fix)
    "Your visit is confirmed for Saturday at 2 PM.",
    "You're already booked for a site visit.",
    "You have an appointment scheduled for tomorrow.",
    # a qualifier between noun and verb — the live false confirmation that slipped through
    "Your visit for the 2 BHK at Sumadhura Edition is confirmed for tomorrow, Tuesday 06 "
    "October at 11:00 AM. Looking forward to seeing you then!",
    "Your site visit on Tuesday at 11 AM has been booked.",
    "Your appointment with our sales team is now confirmed.",
    "Great, your visit for Saturday at 2 PM is confirmed.",
    "Once again, your visit for Saturday is confirmed.",
]

NOT_CLAIMS = [
    "Would you like me to book a visit for you?",
    "Would you like your visit for Saturday to be confirmed?",
    "Your visit for Saturday is not confirmed yet — shall I go ahead?",
    "Once you say yes, your visit for tomorrow at 11 AM is booked.",
    "If your visit for Saturday is confirmed, we'll send the location.",
    "Your visit to the site is for about 30 minutes.",
    "Your visit for the 2 BHK can be tomorrow at 11 AM or 4 PM.",
    "I don't have a visit on record for you on this call.",
]


@pytest.mark.parametrize("text", CLAIMS)
def test_ungrounded_claims_are_caught(text):
    assert bc.is_ungrounded_visit_claim(text, has_appointment_on_file=False)


@pytest.mark.parametrize("text", CLAIMS)
def test_claims_stand_when_an_appointment_exists(text):
    # A booking made this turn, or one on file, makes the same sentence true.
    assert not bc.is_ungrounded_visit_claim(text, has_appointment_on_file=True)


@pytest.mark.parametrize("text", NOT_CLAIMS)
def test_offers_questions_negations_conditions_and_descriptions_pass(text):
    assert not bc.is_ungrounded_visit_claim(text, has_appointment_on_file=False)


def test_gap_is_bounded_to_one_sentence():
    # "your visit" in one sentence and "is confirmed" in the next is not one claim.
    text = "Thanks for asking about your visit. The price list is confirmed by the team."
    assert not bc.asserts_existing_visit(text)


def test_selftest_still_passes():
    bc._selftest()


CANCEL_CLAIMS = [
    # the exact real call: planner said action=respond, no tool ran, reasoning was "customer
    # declined location pin offer" -- nothing to do with cancelling at all
    "Your visit scheduled for tomorrow, Saturday 10 October at 11:00 AM, has been cancelled. "
    "Let me know if you'd like to reschedule.",
    "Your appointment is cancelled.",
    "I've cancelled your visit.",
    "I've cancelled your 3 BHK visit scheduled for tomorrow, Saturday 10 October at "
    "10 AM. Let me know if you'd like to arrange another time.",
    "We've removed your visit from the calendar.",
    "Your booking is no longer scheduled.",
]

CANCEL_NOT_CLAIMS = [
    "Shall I cancel your visit for Saturday?",
    "Would you like me to cancel it?",
    "I haven't cancelled anything yet.",
    "I can't cancel that without your confirmation first.",
    "Your visit is still booked for Saturday at 2 PM.",
]


@pytest.mark.parametrize("text", CANCEL_CLAIMS)
def test_ungrounded_cancellation_claims_are_caught(text):
    assert bc.is_ungrounded_cancellation_claim(text, cancellation_grounded=False)


@pytest.mark.parametrize("text", CANCEL_CLAIMS)
def test_cancellation_claims_stand_when_actually_cancelled(text):
    assert not bc.is_ungrounded_cancellation_claim(text, cancellation_grounded=True)


@pytest.mark.parametrize("text", CANCEL_NOT_CLAIMS)
def test_cancel_offers_questions_and_negations_pass(text):
    assert not bc.is_ungrounded_cancellation_claim(text, cancellation_grounded=False)
