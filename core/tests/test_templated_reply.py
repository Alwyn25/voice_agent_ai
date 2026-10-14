"""Unit tests for P3.04 — the deterministic responder template cache.

The decision to skip the responder LLM is driven by a PURE function,
``templated_reply.render``, precisely so it can be tested without a database, the LLM, or
pipecat — none of which these tests touch. Every eligibility gate and every language
template is exercised here; the fail-safe cases assert the function returns None (→ the
caller falls back to the LLM) rather than ever rendering a wrong or half-filled line.

Run: cd core && python -m pytest tests/test_templated_reply.py -q
"""
import sys
from pathlib import Path

# core/ on the path so `import app...` resolves the same way the app does at runtime.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.voice import templated_reply as tr


class _Outcome:
    """A duck-typed stand-in for tools.types.ToolOutcome (render only reads these three)."""
    def __init__(self, tool_name, status, output):
        self.tool_name = tool_name
        self.status = status
        self.output = output


# A canonical booked slot reused across cases: Sunday 15 March 2026, 2:00 PM IST.
_BOOK = {"starts_at": "2026-03-15T14:00:00+05:30", "centre_name": "Andheri", "status": "booked"}


def _book_outcome(status="success", output=None):
    return _Outcome("book_appointment", status, _BOOK if output is None else output)


# --- happy path: renders in every language and names the real slot -----------------------

def test_book_renders_english():
    r = tr.render(_book_outcome(), action_state="done", language="english", reframing=False)
    assert r is not None
    assert "2:00 PM" in r.text and "Sunday, 15 March" in r.text and "Andheri" in r.text
    assert r.template_id == "book_appointment.english"


def test_book_renders_hindi_and_hinglish():
    hi = tr.render(_book_outcome(), action_state="done", language="hindi", reframing=False)
    hg = tr.render(_book_outcome(), action_state="done", language="hinglish", reframing=False)
    # The date/time token stays in English form in every language (TTS number rule), but the
    # surrounding words are translated.
    assert hi is not None and "2:00 PM" in hi.text and "कन्फर्म" in hi.text
    assert hg is not None and "2:00 PM" in hg.text and "confirm ho gayi" in hg.text


def test_meta_matches_responder_grounded_shape():
    r = tr.render(_book_outcome(), action_state="done", language="english", reframing=False)
    m = r.meta
    # Shaped like responder.generate's healthy return so every downstream guard passes.
    assert m["source"] == "template"
    assert m["verifier_ok"] is True
    assert m["grounded"] is True
    assert m["feature_answer"] == "none"
    assert m["scope_claim"] == "none"
    assert m["answer_grounded"] is True
    assert m["asserts_completed"] == "book"
    assert m["action_state"] == "done"
    assert m["latency_ms"] == 0


def test_book_without_centre_uses_neutral_place():
    r = tr.render(_book_outcome(output={"starts_at": _BOOK["starts_at"], "status": "booked"}),
                  action_state="done", language="english", reframing=False)
    assert r is not None and "at the site office" in r.text


# --- every allow-listed tool renders on a valid success ----------------------------------

def test_all_write_tools_render():
    cases = {
        "cancel_appointment": ({"status": "cancelled"}, "cancelled"),
        "reschedule_appointment": ({"starts_at": "2026-03-20T10:00:00+05:30", "status": "booked"}, "moved"),
        "schedule_callback": ({"requested_for": "2026-03-15T15:00:00+05:30"}, "callback"),
        "create_ticket": ({"subject": "cleanliness", "status": "open"}, "logged"),
        "record_do_not_contact": ({"confirmed": True}, "contact list"),
        "send_document": ({"sent": True, "sent_titles": ["3BHK Brochure"], "to_masked": "98****55"}, "3BHK Brochure"),
    }
    for tool, (output, needle) in cases.items():
        r = tr.render(_Outcome(tool, "success", output),
                      action_state="done", language="english", reframing=False)
        assert r is not None, f"{tool} should render"
        assert needle in r.text, f"{tool} text missing {needle!r}: {r.text!r}"


def test_ticket_appends_subject_when_present():
    with_subj = tr.render(_Outcome("create_ticket", "success", {"subject": "parking"}),
                          action_state="done", language="english", reframing=False)
    no_subj = tr.render(_Outcome("create_ticket", "success", {}),
                        action_state="done", language="english", reframing=False)
    assert with_subj is not None and "about parking" in with_subj.text
    assert no_subj is not None and "about" not in no_subj.text


def test_idempotency_replay_is_deterministic():
    # skipped_duplicate carries the same real output as a fresh success → still templated.
    r = tr.render(_book_outcome(status="skipped_duplicate"),
                  action_state="done", language="english", reframing=False)
    assert r is not None and "2:00 PM" in r.text


# --- fail-safe gates: each must return None so the caller uses the LLM -------------------

def test_skip_when_not_done():
    for state in ("staged", "blocked", "failed", "none", None):
        assert tr.render(_book_outcome(), action_state=state, language="english",
                         reframing=False) is None


def test_skip_when_status_not_success():
    for status in ("failed", "blocked", "error"):
        assert tr.render(_book_outcome(status=status), action_state="done",
                         language="english", reframing=False) is None


def test_skip_when_reframing_signal_active():
    assert tr.render(_book_outcome(), action_state="done", language="english",
                     reframing=True) is None


def test_skip_unknown_tool():
    assert tr.render(_Outcome("search_slots", "success", {"slots": []}),
                     action_state="done", language="english", reframing=False) is None


def test_skip_no_outcome():
    assert tr.render(None, action_state="done", language="english", reframing=False) is None


def test_skip_unknown_language():
    assert tr.render(_book_outcome(), action_state="done", language="tamil",
                     reframing=False) is None


def test_skip_missing_or_bad_slot():
    # No slot time to state, or an unparseable one → never guess, defer to the LLM.
    assert tr.render(_book_outcome(output={"centre_name": "Andheri"}),
                     action_state="done", language="english", reframing=False) is None
    assert tr.render(_book_outcome(output={"starts_at": "not-a-date"}),
                     action_state="done", language="english", reframing=False) is None


def test_skip_cancel_not_actually_cancelled():
    assert tr.render(_Outcome("cancel_appointment", "success", {"status": "booked"}),
                     action_state="done", language="english", reframing=False) is None


def test_skip_send_document_not_sent_or_no_title():
    assert tr.render(_Outcome("send_document", "success", {"sent": False}),
                     action_state="done", language="english", reframing=False) is None
    assert tr.render(_Outcome("send_document", "success", {"sent": True, "sent_titles": []}),
                     action_state="done", language="english", reframing=False) is None


def test_skip_dnc_not_confirmed():
    assert tr.render(_Outcome("record_do_not_contact", "success", {"confirmed": False}),
                     action_state="done", language="english", reframing=False) is None


def test_renderer_exception_is_swallowed(monkeypatch):
    # A renderer must never raise into process_turn — a bad output shape falls back to LLM.
    def _boom(_out):
        raise ValueError("unexpected shape")
    monkeypatch.setitem(tr._RENDERERS, "book_appointment", _boom)
    assert tr.render(_book_outcome(), action_state="done", language="english",
                     reframing=False) is None


# --- booking reference (GTM demo tenants' locations carry one) ---------------------------

def test_book_without_reference_is_unchanged():
    r = tr.render(_book_outcome(), action_state="done", language="english", reframing=False)
    assert r.text == "Your visit is confirmed for Sunday, 15 March at 2:00 PM, at Andheri."


def test_book_reads_back_reference_in_one_sentence_every_language():
    out = {**_BOOK, "booking_reference": "VIS-1187"}
    en = tr.render(_book_outcome(output=out), action_state="done", language="english", reframing=False)
    hi = tr.render(_book_outcome(output=out), action_state="done", language="hindi", reframing=False)
    hg = tr.render(_book_outcome(output=out), action_state="done", language="hinglish", reframing=False)
    assert en.text == ("Your visit is confirmed for Sunday, 15 March at 2:00 PM, at Andheri, "
                       "booking reference VIS-1187.")
    assert "VIS-1187" in hi.text and "VIS-1187" in hg.text
    assert all(t.text.count(".") + t.text.count("।") == 1 for t in (en, hg, hi) if t)


def test_blank_reference_is_ignored():
    out = {**_BOOK, "booking_reference": "  "}
    r = tr.render(_book_outcome(output=out), action_state="done", language="english", reframing=False)
    assert "reference" not in r.text
