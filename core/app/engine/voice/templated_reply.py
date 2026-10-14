"""P3.04 — Deterministic response templates (the responder-LLM cache).

WHAT THIS SOLVES
----------------
On every tool turn, ``turn_runner.process_turn`` calls the responder LLM
(``responder.generate``) to phrase the customer-facing reply. That call is the RSP
stage — roughly 150 ms of TTFA on *every* tool turn, plus a per-call token cost.

For a whole class of outcomes the reply is *deterministic*. When ``book_appointment``
succeeds, the honest reply is always "Your visit is confirmed for <day> at <time>…" —
and every fact in it (the slot, the centre, the status) is already sitting in
``outcome.output``. Paying an LLM round-trip to re-phrase a fixed confirmation is pure
latency waste. Benchmark P3.04 estimates this covers ~40% of tool turns.

This module renders those confirmations directly from the tool output, in the caller's
language, with NO model call — saving the full RSP latency on qualifying turns.

WHY IT IS SAFE
--------------
1. It is *fail-safe by construction*. ``render()`` returns ``None`` the moment anything
   is even slightly off (unknown tool, non-success status, action not actually done, a
   required field missing, a "reframing" signal in play). The caller then falls back to
   the LLM exactly as before, so an unrenderable turn is never degraded — it is simply
   not cached.
2. The template still runs the SAME downstream guards. turn_runner substitutes this
   text for the LLM's and then runs every post-responder check unchanged — the
   wrong-slot date/time guard, grounding_check, the scope / visit-claim guards, price
   rounding, the repeated-closer trimming. The template goes through the identical
   validation net an LLM reply would, so this cannot regress correctness.
3. It emits a "clean grounded" meta (see ``_meta``) so those guards see exactly the
   shape ``responder.generate`` produces on a healthy grounded turn.

DEBUGGABILITY
-------------
* Gated behind a single env flag — ``RESPONDER_TEMPLATE_CACHE_ENABLED`` (off by
  default). Flip it in ``.env`` to turn the whole feature on; unset it to roll back
  instantly with zero code change.
* Every decision is logged on the ``app.engine.voice.templated_reply`` logger: a match
  logs the tool + language + template id; a skip logs the exact reason. Watch that
  logger to see precisely when the cache fires and why it declines.
* The returned meta carries ``source="template"`` and the ``template_id`` so the turn
  trace / TTFA telemetry can measure the hit-rate.

PURITY
------
No DB, no I/O, no pipecat — everything is computed from the tool output and a language
string. It therefore unit-tests without a database, both via pytest
(``tests/test_templated_reply.py``) and standalone::

    python -m app.engine.voice.templated_reply
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# A tool "succeeded" whenever its status is either of these — mirrors turn_runner's
# TOOL_SUCCEEDED. "skipped_duplicate" carries the SAME real output as a fresh success
# (idempotency replay), so its confirmation is just as deterministic.
_SUCCEEDED = ("success", "skipped_duplicate")

# The three reply languages, matching responder.detect_language().
_LANGS = ("english", "hindi", "hinglish")

# Which "I did it" claim each tool maps to, for the responder-meta contract. Tools whose
# completion the responder does not model as a claim (callback, document) map to None.
_CLAIM = {
    "book_appointment": "book",
    "reschedule_appointment": "reschedule",
    "cancel_appointment": "cancel",
    "create_ticket": "ticket",
    "record_do_not_contact": "dnc",
}


@dataclass(frozen=True)
class TemplatedReply:
    """A deterministically rendered reply plus the meta turn_runner expects from the
    responder. ``template_id`` (e.g. ``"book_appointment.hinglish"``) is carried purely
    for logging / telemetry so a cached turn is identifiable in the trace."""
    text: str
    meta: dict = field(default_factory=dict)
    template_id: str = ""


# ---------------------------------------------------------------------------------------
# Time formatting — Windows-safe.
#
# strftime("%-I") / "%-d" (no zero-pad) are NOT portable: they raise on Windows, where a
# chunk of this project is developed. The existing wrong-slot guard in turn_runner
# already formats the 12-hour clock by hand for exactly this reason; we do the same, so
# the spoken time is identical to what that guard reconstructs and the two never disagree.
#
# The date/time token stays in ENGLISH form in all three languages. That is deliberate and
# matches the codebase's TTS number rule (responder._LANGUAGE_INSTRUCTION): the
# multilingual voice pronounces a number by the words around it, so a plain English clock
# reading inside a Hindi/Hinglish sentence is what actually speaks cleanly on a real call.
# ---------------------------------------------------------------------------------------
def _spoken_dt(iso: str | None) -> str | None:
    """"Thursday, 15 March at 2:00 PM" from an ISO timestamp, or None if it can't parse.

    Returning None (rather than a half-formatted string) is a hard gate: a confirmation
    whose time we cannot state is not one we will speak deterministically — the caller
    falls back to the LLM instead.
    """
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(str(iso))
    except (TypeError, ValueError):
        return None
    hour12 = dt.hour % 12 or 12
    ampm = "AM" if dt.hour < 12 else "PM"
    return f"{dt.strftime('%A')}, {dt.day} {dt.strftime('%B')} at {hour12}:{dt.minute:02d} {ampm}"


def _clean(value) -> str | None:
    """A non-empty stripped string, or None. Guards every field a template interpolates —
    an empty centre name or title must never be rendered into the spoken line."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


# ---------------------------------------------------------------------------------------
# Per-tool renderers.
#
# Each takes the tool's ``output`` dict and returns ``{language: text}`` for all three
# languages, or ``None`` if a required field is missing/invalid (→ LLM fallback). Every
# template is a SINGLE sentence on purpose: turn_runner's repeated-"anything else?" and
# repeated-visit-reminder trims only drop a sentence when the reply has more than one, so
# a one-sentence confirmation is guaranteed to survive those passes intact.
# ---------------------------------------------------------------------------------------
def _book_appointment(out: dict) -> dict[str, str] | None:
    when = _spoken_dt(out.get("starts_at"))
    if when is None:
        return None  # no verifiable slot time → not deterministic, defer to the LLM
    centre = _clean(out.get("centre_name"))
    # "at <centre>" when we know it, a neutral "at the site office" when we don't — never
    # an invented place.
    at_en = f", at {centre}" if centre else ", at the site office"
    at_hi = f", {centre} पर" if centre else ", हमारे साइट ऑफिस पर"
    at_hg = f", {centre} par" if centre else ", hamare site office par"
    # A booking reference is read back only when the tool returned one (GTM demo tenants;
    # see book_appointment). Still ONE sentence, so the trims above can't drop it. Without
    # the key the three lines are exactly as before.
    ref = _clean(out.get("booking_reference"))
    ref_en = f", booking reference {ref}" if ref else ""
    ref_hi = f", बुकिंग रेफ़रेंस {ref}" if ref else ""
    ref_hg = f", booking reference {ref}" if ref else ""
    return {
        "english": f"Your visit is confirmed for {when}{at_en}{ref_en}.",
        "hindi": f"आपकी विज़िट {when} के लिए कन्फर्म हो गई है{at_hi}{ref_hi}।",
        "hinglish": f"Aapki visit {when} ke liye confirm ho gayi hai{at_hg}{ref_hg}.",
    }


def _reschedule_appointment(out: dict) -> dict[str, str] | None:
    # Prefer the ISO the tool returns (formatted here, Windows-safe) over its own spoken
    # "when" string, so the reply's time is byte-identical to what the wrong-slot guard
    # rebuilds from the same ISO.
    when = _spoken_dt(out.get("starts_at"))
    if when is None:
        return None
    return {
        "english": f"Done — your visit is now moved to {when}.",
        "hindi": f"हो गया — आपकी विज़िट अब {when} के लिए मूव कर दी गई है।",
        "hinglish": f"Done — aapki visit ab {when} ke liye move kar di gayi hai.",
    }


def _cancel_appointment(out: dict) -> dict[str, str] | None:
    # A cancel confirmation carries no time or place — the deterministic fact is simply
    # that the appointment is gone. Guarded on the tool having actually set the row to
    # cancelled, so we never announce a cancellation the DB did not make.
    if _clean(out.get("status")) != "cancelled":
        return None
    return {
        "english": "Your appointment has been cancelled.",
        "hindi": "आपकी अपॉइंटमेंट कैंसिल कर दी गई है।",
        "hinglish": "Aapki appointment cancel kar di gayi hai.",
    }


def _schedule_callback(out: dict) -> dict[str, str] | None:
    when = _spoken_dt(out.get("requested_for"))
    if when is None:
        return None
    return {
        "english": f"Sure — I've scheduled a callback for {when}.",
        "hindi": f"ठीक है — मैंने {when} के लिए कॉलबैक शेड्यूल कर दिया है।",
        "hinglish": f"Theek hai — maine {when} ke liye callback schedule kar diya hai.",
    }


def _create_ticket(out: dict) -> dict[str, str] | None:
    # create_ticket backs both complaints and general requests; the neutral "request"
    # wording is honest for either. The subject is appended only when it is a real value,
    # never a placeholder empty string.
    subject = _clean(out.get("subject"))
    about_en = f" about {subject}" if subject else ""
    about_hi = f" ({subject} के बारे में)" if subject else ""
    about_hg = f" ({subject} ke baare mein)" if subject else ""
    return {
        "english": f"I've logged your request{about_en}, and our team will follow up.",
        "hindi": f"मैंने आपका अनुरोध{about_hi} दर्ज कर लिया है, हमारी टीम फ़ॉलो-अप करेगी।",
        "hinglish": f"Maine aapki request{about_hg} register kar li hai, hamari team follow-up karegi.",
    }


def _record_do_not_contact(out: dict) -> dict[str, str] | None:
    # Only confirm the opt-out the tool actually recorded.
    if not out.get("confirmed"):
        return None
    return {
        "english": "Done — I've removed you from our contact list; you won't receive further calls from us.",
        "hindi": "हो गया — मैंने आपको हमारी कॉन्टैक्ट लिस्ट से हटा दिया है; अब आपको हमारी ओर से कॉल नहीं आएँगी।",
        "hinglish": "Done — maine aapko hamari contact list se hata diya hai; ab aapko humari taraf se call nahi aayengi.",
    }


def _send_document(out: dict) -> dict[str, str] | None:
    # Deterministic only on a genuine send with a delivered title to name.
    if not out.get("sent"):
        return None
    titles = [t for t in (out.get("sent_titles") or []) if _clean(t)]
    if not titles:
        return None
    named = ", ".join(titles)
    to = _clean(out.get("to_masked"))
    on_en = f" on {to}" if to else ""
    on_hi = f" ({to} पर)" if to else ""
    on_hg = f" ({to} par)" if to else ""
    return {
        "english": f"I've sent {named} to your WhatsApp{on_en}.",
        "hindi": f"मैंने {named} आपके WhatsApp{on_hi} पर भेज दिया है।",
        "hinglish": f"Maine {named} aapke WhatsApp{on_hg} par bhej diya hai.",
    }


# The allow-list. A tool absent here is never templated — the single source of truth for
# "which outcomes have a fixed confirmation". Adding a tool is a one-line change plus its
# renderer above.
_RENDERERS = {
    "book_appointment": _book_appointment,
    "reschedule_appointment": _reschedule_appointment,
    "cancel_appointment": _cancel_appointment,
    "schedule_callback": _schedule_callback,
    "create_ticket": _create_ticket,
    "record_do_not_contact": _record_do_not_contact,
    "send_document": _send_document,
}


def _meta(tool_name: str, action_state: str | None, template_id: str) -> dict:
    """The responder-meta a templated reply carries, shaped exactly like
    ``responder.generate``'s healthy-grounded return so every downstream guard behaves
    identically. The reply is grounded BY CONSTRUCTION (it states only fields the tool
    returned), so: no feature/scope claim to verify, ``verifier_ok`` true so
    grounding_check runs in normal mode and passes, and ``asserts_completed`` set truthfully
    (the action genuinely completed — action_state is "done" whenever we get here)."""
    return {
        "model": "template",           # names the RSP "provider" in the trace as the cache
        "tokens": {},
        "latency_ms": 0,
        "grounded": True,
        "asserts_completed": _CLAIM.get(tool_name),
        "action_state": action_state,
        "feature_answer": "none",
        "answer_grounded": True,
        "support_quote": None,
        "scope_claim": "none",
        "scope_grounded": False,
        "verifier_ok": True,
        "source": "template",          # telemetry: RSP served from template vs "llm"
        "template_id": template_id,
    }


def render(
    outcome,
    *,
    action_state: str | None,
    language: str,
    reframing: bool = False,
) -> TemplatedReply | None:
    """Render a deterministic confirmation for ``outcome``, or None to defer to the LLM.

    A template is returned ONLY when every one of these holds; any miss logs its reason
    and returns None so the caller runs ``responder.generate`` exactly as before:

    * ``outcome`` exists and its status is a success (``success`` / ``skipped_duplicate``).
    * ``action_state == "done"`` — the action genuinely completed. A ``staged`` / ``blocked``
      / ``failed`` turn must ask or explain, which is the LLM's job, not a confirmation.
    * ``outcome.tool_name`` is in the allow-list (``_RENDERERS``).
    * ``reframing`` is False. When the turn also carries a presence-check / backchannel
      continuation, a pitch-ladder rung, a forced project-disambiguation, or an outbound
      lead-forward line, the reply is NOT a plain tool confirmation — those need the LLM,
      so we step aside.
    * ``language`` is one we template, and the renderer produced text with all its
      required fields present.

    Args:
        outcome: the ToolOutcome that just executed (duck-typed: ``.tool_name`` /
            ``.status`` / ``.output``), or None on a no-tool turn.
        action_state: turn_runner's resolved ground-truth state
            (``done`` / ``staged`` / ``blocked`` / ``failed`` / ``none``).
        language: ``responder.detect_language(user_input)`` — the reply language.
        reframing: True when a continuation / pitch / forced-clarify / outbound-lead
            signal is active on this turn (see above).
    """
    if outcome is None:
        return None  # no tool ran — nothing deterministic to confirm

    tool_name = getattr(outcome, "tool_name", None)
    status = getattr(outcome, "status", None)

    # --- eligibility gates, each logged so a decline is never silent ---
    if reframing:
        logger.debug("template skip: reframing signal active (tool=%s)", tool_name)
        return None
    if action_state != "done":
        logger.debug("template skip: action_state=%r not 'done' (tool=%s)", action_state, tool_name)
        return None
    if status not in _SUCCEEDED:
        logger.debug("template skip: status=%r not a success (tool=%s)", status, tool_name)
        return None
    renderer = _RENDERERS.get(tool_name)
    if renderer is None:
        logger.debug("template skip: tool=%r not in allow-list", tool_name)
        return None
    if language not in _LANGS:
        logger.debug("template skip: language=%r not templated (tool=%s)", language, tool_name)
        return None

    output = getattr(outcome, "output", None) or {}
    try:
        rendered = renderer(output)
    except Exception:
        # A renderer must never take down a live turn: any unexpected shape in the tool
        # output falls back to the LLM rather than raising into process_turn.
        logger.exception("template skip: renderer raised (tool=%s)", tool_name)
        return None

    if not rendered:
        logger.debug("template skip: required field missing (tool=%s)", tool_name)
        return None

    text = _clean(rendered.get(language))
    if text is None:
        logger.debug("template skip: empty rendered text (tool=%s, lang=%s)", tool_name, language)
        return None

    template_id = f"{tool_name}.{language}"
    logger.info("template hit: %s served deterministically (RSP LLM skipped)", template_id)
    return TemplatedReply(text=text, meta=_meta(tool_name, action_state, template_id),
                          template_id=template_id)


# ---------------------------------------------------------------------------------------
# Built-in self-test — runs without pytest or a database:
#   python -m app.engine.voice.templated_reply
# The pytest suite (tests/test_templated_reply.py) covers the same ground in more detail.
# ---------------------------------------------------------------------------------------
def _selftest() -> int:
    failures: list[str] = []

    class _O:
        def __init__(self, tool_name, status, output):
            self.tool_name, self.status, self.output = tool_name, status, output

    def ok(name, cond):
        if not cond:
            failures.append(name)

    book = _O("book_appointment", "success",
              {"starts_at": "2026-03-15T14:00:00+05:30", "centre_name": "Andheri", "status": "booked"})

    # --- happy path, all three languages render and name the slot ---
    for lang in _LANGS:
        r = render(book, action_state="done", language=lang, reframing=False)
        ok(f"book_{lang}_renders", r is not None and "2:00 PM" in r.text)
        ok(f"book_{lang}_meta", r is not None and r.meta.get("source") == "template"
           and r.meta.get("verifier_ok") is True and r.meta.get("asserts_completed") == "book")

    # --- every allow-listed tool renders on a valid success ---
    ok("cancel_renders", render(_O("cancel_appointment", "success", {"status": "cancelled"}),
                                action_state="done", language="english", reframing=False) is not None)
    ok("reschedule_renders", render(_O("reschedule_appointment", "success",
                                       {"starts_at": "2026-03-20T10:00:00+05:30", "status": "booked"}),
                                    action_state="done", language="english", reframing=False) is not None)
    ok("callback_renders", render(_O("schedule_callback", "success",
                                     {"requested_for": "2026-03-15T15:00:00+05:30"}),
                                  action_state="done", language="hinglish", reframing=False) is not None)
    ok("ticket_renders", render(_O("create_ticket", "success", {"subject": "cleanliness", "status": "open"}),
                                action_state="done", language="english", reframing=False) is not None)
    ok("dnc_renders", render(_O("record_do_not_contact", "success", {"confirmed": True}),
                             action_state="done", language="hindi", reframing=False) is not None)
    ok("send_doc_renders", render(_O("send_document", "success",
                                     {"sent": True, "sent_titles": ["3BHK Brochure"], "to_masked": "98****55"}),
                                  action_state="done", language="english", reframing=False) is not None)

    # --- fail-safe: every gate must return None ---
    ok("skip_staged", render(book, action_state="staged", language="english", reframing=False) is None)
    ok("skip_failed_status", render(_O("book_appointment", "failed", {}),
                                    action_state="done", language="english", reframing=False) is None)
    ok("skip_reframing", render(book, action_state="done", language="english", reframing=True) is None)
    ok("skip_unknown_tool", render(_O("search_slots", "success", {"slots": []}),
                                   action_state="done", language="english", reframing=False) is None)
    ok("skip_no_outcome", render(None, action_state="done", language="english", reframing=False) is None)
    ok("skip_missing_slot", render(_O("book_appointment", "success", {"centre_name": "Andheri"}),
                                   action_state="done", language="english", reframing=False) is None)
    ok("skip_bad_iso", render(_O("book_appointment", "success", {"starts_at": "not-a-date"}),
                              action_state="done", language="english", reframing=False) is None)
    ok("skip_cancel_not_cancelled", render(_O("cancel_appointment", "success", {"status": "booked"}),
                                           action_state="done", language="english", reframing=False) is None)
    ok("skip_send_not_sent", render(_O("send_document", "success", {"sent": False}),
                                    action_state="done", language="english", reframing=False) is None)
    ok("skip_unknown_lang", render(book, action_state="done", language="tamil", reframing=False) is None)

    # --- idempotency replay is still deterministic ---
    ok("skipped_duplicate_renders", render(_O("book_appointment", "skipped_duplicate",
                                              {"starts_at": "2026-03-15T14:00:00+05:30", "status": "booked"}),
                                           action_state="done", language="english", reframing=False) is not None)

    if failures:
        print(f"templated_reply self-test: {len(failures)} FAILED -> {failures}")
        return 1
    print("templated_reply self-test: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
