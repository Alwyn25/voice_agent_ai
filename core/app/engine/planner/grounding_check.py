"""grounding_check — the verdict on whether an affirmed feature may stand.

The RESPONDER call is the semantic verifier: in one pass it reads the caller's question and the
retrieved material FOR MEANING (responder.py Rule 13), writes the answer, and self-assesses
(``feature_answer`` / ``answer_grounded`` / ``support_quote``). This module turns that
self-assessment into an accept/replace decision — with NO model call of its own. Meaning is
judged by the model that already has the query and the chunks in front of it, because no string
rule can bridge "generator" ⇄ "power backup" the way understanding does. (A first attempt did
exactly that string-matching and wrongly suppressed a real "yes" about the generator; this is
the fix — the deterministic layer no longer overrules the model's semantics.)

NORMAL MODE (``verifier_ok`` — the self-assessment is present). A reply that affirms a feature
stands iff the model judged it grounded AND pointed at a supporting passage. An affirm the model
itself marks ungrounded (or cannot point to anything for) is replaced with an honest "I can't
confirm that" — the model caught its own guess. The support quote is NOT re-matched as a string
here: that is the exact brittleness this design removed.

SCOPE/POLICY claims ("we don't rent", "we only sell") are NOT decided here. The model is trusted
to DETECT that its sentence makes such a claim (responder Rule 14, ``scope_claim``), but never to
judge whether it is grounded — it marks a fabricated "we only sell" as grounded because a sales
project "obviously" only sells. That grounding decision is made deterministically, against the
actual material, in ``scope_claim.py`` + turn_runner — see that module.

DEGRADED MODE (``verifier_ok`` false — the verifier call failed). No semantic verdict exists to
trust, so we fall back to the one thing a string CAN prove: an affirm whose support was quoted
from the material word-for-word is demonstrably grounded and stands; an unquoted affirm cannot
be vouched for and is replaced. Never fires on a healthy turn — the safety net for an LLM outage.

Pure and dependency-light (only the sibling pure helpers), so it unit-tests without a database —
``python -m app.engine.planner.grounding_check``.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.engine.planner.feature_grounding import cited_verbatim


@dataclass(frozen=True)
class Verdict:
    ok: bool            # may the reply stand as written?
    reason: str = ""    # audit note, recorded on the trace when the reply is replaced


def check(resp_meta: dict | None, grounding: str) -> Verdict:
    """Decide whether the responder's reply may stand, from its own semantic self-assessment.

    ``resp_meta`` is the meta dict from :func:`responder.generate`; ``grounding`` is
    :func:`feature_grounding.grounding_text` over the same passages + tool output the reply saw
    (used only in degraded mode).
    """
    meta = resp_meta or {}
    affirmed = meta.get("feature_answer") == "affirm"
    quote = (meta.get("support_quote") or "").strip()

    if meta.get("verifier_ok"):
        if not affirmed:
            return Verdict(True)                       # nothing affirmed → nothing to verify
        if meta.get("answer_grounded") and quote:
            return Verdict(True)                       # model grounded it and pointed at support
        return Verdict(False, f"model self-marked its affirm ungrounded or unsupported "
                              f"(answer_grounded={meta.get('answer_grounded')!r}, "
                              f"quote={quote!r})")

    # Degraded: the semantic verifier did not run this turn.
    if affirmed and not cited_verbatim(quote, grounding):
        return Verdict(False, "verifier unavailable and affirm not quoted verbatim")
    return Verdict(True)


# ---------------------------------------------------------------------------------------
# Built-in self-test — runs without pytest.
#   python -m app.engine.planner.grounding_check
# ---------------------------------------------------------------------------------------
def _selftest() -> int:
    failures: list[str] = []

    def check_(name, cond):
        if not cond:
            failures.append(name)

    G = "unit 1 west facing. 100 percent power backup. amenities: trampoline park, a gymnasium."

    # --- normal mode: trust the model's SEMANTIC verdict ---
    # generator affirmed and grounded (via meaning, quoting the doc's own words) → stands.
    check_("generator_grounded_stands", check({
        "verifier_ok": True, "feature_answer": "affirm", "answer_grounded": True,
        "support_quote": "100 percent power backup"}, G).ok)
    # the model quoting the caller's OWN word is fine in normal mode — no string re-match,
    # so the "power backup for generator" case is NOT punished for wording.
    check_("generator_grounded_caller_wording_stands", check({
        "verifier_ok": True, "feature_answer": "affirm", "answer_grounded": True,
        "support_quote": "electricity generator"}, G).ok)
    # park-facing: the model, reading for meaning, marks its own affirm ungrounded → replaced.
    check_("park_facing_ungrounded_replaced", not check({
        "verifier_ok": True, "feature_answer": "affirm", "answer_grounded": False,
        "support_quote": "west facing"}, G).ok)
    # grounded=true but nothing to point at → treated as unsupported, replaced.
    check_("grounded_but_no_quote_replaced", not check({
        "verifier_ok": True, "feature_answer": "affirm", "answer_grounded": True,
        "support_quote": None}, G).ok)
    # a deny or a non-feature turn is never touched, whatever the other fields say.
    check_("deny_stands", check({
        "verifier_ok": True, "feature_answer": "deny", "answer_grounded": False,
        "support_quote": None}, G).ok)
    check_("none_stands", check({
        "verifier_ok": True, "feature_answer": "none", "answer_grounded": False,
        "support_quote": None}, G).ok)

    # --- degraded mode: verifier failed, fall back to verbatim proof ---
    check_("degraded_affirm_verbatim_stands", check({
        "verifier_ok": False, "feature_answer": "affirm",
        "support_quote": "power backup"}, G).ok)
    check_("degraded_affirm_unquoted_replaced", not check({
        "verifier_ok": False, "feature_answer": "affirm",
        "support_quote": "solar generator on every floor"}, G).ok)
    # the real error path reports feature_answer "none" (draft can't be classified) → stands.
    check_("degraded_none_stands", check({
        "verifier_ok": False, "feature_answer": "none", "support_quote": None}, G).ok)

    # --- robustness: empty / missing meta must not throw and must not block ---
    check_("empty_meta_stands", check({}, G).ok)
    check_("none_meta_stands", check(None, G).ok)

    if failures:
        print(f"grounding_check self-test: {len(failures)} FAILED -> {failures}")
        return 1
    print("grounding_check self-test: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
