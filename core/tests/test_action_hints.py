"""Keyword action hints (turn_runner.TOOL_HINTS) must not turn a question ABOUT an action into
a request to perform it — and must defer to the models when both read the turn as informational.

THE BUG: "How long does a site visit take and what are the maintenance charges?" matched the
bare "site visit" booking pattern; the caller got "Before I can book your visit, can I take
your name?" with neither question answered, after two wasted re-plans.

    cd core && python -m pytest tests/test_action_hints.py -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.planner import turn_runner as tr  # noqa: E402


def test_questions_about_an_action_are_not_requests():
    for q in ("How long does a site visit take and what are the maintenance charges?",
              "How long does a site visit take?",
              "What happens if I cancel my visit?",
              "Is there a fee to reschedule my appointment?",
              "Is the site visit free?",
              "Do I need to book a site visit in advance?",
              "Can I bring my parents to the site visit?",
              "What should I bring for the site visit?"):
        assert tr._hinted_tool(q) is None, (q, tr._hinted_tool(q))


def test_real_requests_still_hint():
    cases = {
        "I want to see the flat": "book_appointment",
        "Can you book a site visit for Saturday?": "book_appointment",
        "Schedule a site visit please": "book_appointment",
        "How long does a site visit take? Book me one for Saturday.": "book_appointment",
        "how long is the visit, and can you book me in tomorrow at 11": "book_appointment",
        "Cancel my visit": "cancel_appointment",
        "Please move my appointment to another day": "reschedule_appointment",
        "Call me back tomorrow": "schedule_callback",
        # never deferred, even inside a question
        "Can you stop calling me?": "record_do_not_contact",
        "How do I raise a complaint about the seepage? Please log it.": "create_ticket",
    }
    for q, want in cases.items():
        assert tr._hinted_tool(q) == want, (q, tr._hinted_tool(q))


def _proposal(intent, *, is_scheduling, source="model"):
    return SimpleNamespace(understood={"intent": intent},
                           routing={"source": source, "is_scheduling": is_scheduling})


def test_hint_defers_only_when_both_models_agree_it_is_informational():
    q = "I'd love to see the property sometime"          # keyword says book_appointment
    assert tr._turn_hint(q, _proposal("product_question", is_scheduling=False)) is None
    # Either model disagreeing keeps the backstop — a mislabelled visit request still books.
    assert tr._turn_hint(q, _proposal("product_question", is_scheduling=True)) == \
        "book_appointment"
    assert tr._turn_hint(q, _proposal("book_appointment", is_scheduling=False)) == \
        "book_appointment"
    # No real classification (trivial-skip / failure fallback) = no deference.
    assert tr._turn_hint(q, _proposal("product_question", is_scheduling=False,
                                      source="fallback")) == "book_appointment"
    assert tr._turn_hint(q, None) == "book_appointment"


def test_opt_out_is_never_deferred():
    p = _proposal("product_question", is_scheduling=False)
    assert tr._turn_hint("please stop calling me", p) == "record_do_not_contact"


def test_booking_signal_uses_the_deferring_hint():
    p = _proposal("product_question", is_scheduling=False)
    assert not tr._booking_signalled({}, "I'd love to see the property sometime",
                                     "product_question", p)
    assert tr._booking_signalled({}, "I'd love to see the property sometime", None, None)
