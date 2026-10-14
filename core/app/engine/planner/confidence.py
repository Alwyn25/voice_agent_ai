"""Step 7: turn the model's self-assessment into a trustworthy band.

The model's own score is adjusted by objective signals it cannot judge:
poor audio, answering factually with nothing retrieved, admitted missing info,
and going round in circles asking the same clarification.
"""
import re
from dataclasses import dataclass, field

from app.config import get_settings

# Models label intents freely ("inquire_price", "pricing_question", "ask_cost"),
# so match on keywords rather than exact strings.
FACTUAL_KEYWORDS = ("pric", "cost", "charge", "fee", "product", "policy", "coverage",
                    "eligib", "offer", "discount", "package", "plan", "feature")

# Canonical intents the rule engine keys on. The model's free-form label is mapped
# onto one of these so rules stay stable regardless of phrasing.
#
# Needles are matched on WORD BOUNDARIES, not raw substrings — see normalize_intent. The
# old substring match silently misread words that merely CONTAINED a needle: "come in
# person" → request_human (the word "person"), "issue" → legal_complaint (the word "sue"),
# "inhuman" → request_human. The "person" collision is the one that bit in production: a
# caller who said "I'll come by 2 PM today" was labelled with an in-person-visit intent,
# matched "person", was normalised to request_human, and the escalation_policy rule
# `explicit_human_request` forced a handover mid-reschedule.
#
# Because "person" and "agent" are ordinary words a caller uses without wanting a handover
# (come in person; the sales agent), they are NOT bare needles — a genuine human request is
# matched by "human", "a person", "a human", "human agent", "live agent", or an explicit
# manager/supervisor/escalate ask. Boundary matching also tolerates the stems the model
# actually emits ("cancellation", "booking", "escalated") — see _needle_matches.
#
# NOTE: the HANDOVER decision does NOT depend on this mapping at all any more. The planner
# emits an explicit `human_handoff_requested` boolean and the escalation rule
# (rules/defaults.py `explicit_human_request`) keys on that alone. This mapping's
# "request_human" canonical label used to ALSO be OR'd into that same rule as a secondary
# signal — removed after a live case disproved the assumption that it was harmless: the
# model can correctly set human_handoff_requested=False (genuinely the right call) while
# still free-text-labelling the turn's intent "request_human" as a topic tag (the call is
# ABOUT getting human-channel help — e.g. the sales team's email — not a literal "transfer
# me now"), and the word-boundary match still resolves that label to this canonical value,
# overriding the model's own correct decision. The label is a descriptive category, not a
# commitment to transfer, so it must never again drive the handover decision by itself.
# "request_human" still exists in this mapping for the other, non-escalation consumers that
# key on it (e.g. scheduling_progress.py's intent checks) — only the escalation OR-branch
# was removed.
INTENT_ALIASES = {
    "request_human": ("human", "manager", "supervisor", "escalate", "a person",
                      "a human", "human agent", "live agent", "real agent"),
    "dnc_request": ("dnc", "do_not_contact", "opt_out", "unsubscribe", "stop_call",
                    "remove_number", "stop_contact"),
    "legal_complaint": ("legal", "lawyer", "consumer_court", "sue"),
    "payment_dispute": ("payment_dispute", "refund_dispute", "billing_dispute", "chargeback"),
    "complaint": ("complaint", "grievance"),
    "book_appointment": ("book", "appointment", "schedule_visit"),
    "reschedule": ("reschedule", "postpone", "change_appointment"),
    "cancel": ("cancel",),
    "call_later": ("call_later", "callback", "busy_now"),
}


def _needle_matches(intent: str, needle: str) -> bool:
    """True when ``needle`` occurs in ``intent`` as whole word(s), stem-tolerant.

    Both sides are tokenised on any non-alphanumeric run (so ``change_appointment`` and
    ``change appointment`` are the same), then the needle's words must appear as a
    CONSECUTIVE run in the intent — with the final needle word allowed a trailing suffix so
    ``cancel`` still catches ``cancellation`` and ``book`` still catches ``booking``, while
    ``sue`` no longer catches ``issue`` and ``a person`` never catches ``in person``.
    """
    n_words = re.findall(r"[a-z0-9]+", needle.lower())
    if not n_words:
        return False
    # Prefix on the last word only; earlier words match whole so "a person" needs the "a".
    body = r"[^a-z0-9]+".join(re.escape(w) for w in n_words[:-1])
    last = re.escape(n_words[-1]) + r"[a-z0-9]*"
    pattern = (body + r"[^a-z0-9]+" if body else "") + last
    return re.search(r"(?<![a-z0-9])" + pattern + r"(?![a-z0-9])", intent.lower()) is not None


def normalize_intent(raw: str | None,
                     extra_aliases: dict[str, "list[str] | tuple[str, ...]"] | None = None
                     ) -> str | None:
    """Map a free-form model intent label onto a canonical one (or return it unchanged).

    ``extra_aliases`` — a tenant's intent-vocabulary additions ({canonical: [needles]}),
    checked AFTER the built-in aliases so a tenant entry can never shadow a core safety
    intent (request_human, dnc_request, …). This is how a Phase-2 routing rule keyed on a
    tenant intent like ``loan_enquiry`` fires reliably regardless of the planner's phrasing.
    Passing nothing preserves the original behaviour exactly.
    """
    if not raw:
        return None
    low = str(raw).lower()
    for canonical, needles in INTENT_ALIASES.items():
        if any(_needle_matches(low, n) for n in needles):
            return canonical
    for canonical, needles in (extra_aliases or {}).items():
        if any(_needle_matches(low, n) for n in needles):
            return canonical
    return low


def is_factual_intent(raw: str | None) -> bool:
    low = str(raw or "").lower()
    return any(k in low for k in FACTUAL_KEYWORDS)


@dataclass
class ConfidenceResult:
    score: float
    band: str                      # high | medium | low
    self_confidence: float
    adjustments: list[dict] = field(default_factory=list)


def score_confidence(*, self_confidence: float, intent: str | None,
                     stt_confidence: float | None, knowledge_count: int,
                     missing_info: list[str] | None, recent_actions: list[str]) -> ConfidenceResult:
    s = get_settings()
    score = max(0.0, min(1.0, float(self_confidence or 0)))
    adjustments: list[dict] = []

    if stt_confidence is not None and stt_confidence < s.conf_stt_threshold:
        score *= s.conf_stt_penalty
        adjustments.append({"reason": "low_stt_confidence", "factor": s.conf_stt_penalty})

    if is_factual_intent(intent) and knowledge_count == 0:
        score *= s.conf_ungrounded_factor
        adjustments.append({"reason": "factual_answer_without_retrieval",
                            "factor": s.conf_ungrounded_factor})

    if missing_info:
        score *= s.conf_ambiguity_factor
        adjustments.append({"reason": "model_reported_missing_info",
                            "factor": s.conf_ambiguity_factor})

    clarifies = [a for a in recent_actions if a.endswith("clarify")]
    if len(clarifies) >= 2:
        score -= s.conf_repeat_penalty
        adjustments.append({"reason": "repeated_clarification",
                            "penalty": s.conf_repeat_penalty})

    score = max(0.0, min(1.0, score))
    band = "high" if score >= s.confidence_high else ("low" if score < s.confidence_low else "medium")
    return ConfidenceResult(score=round(score, 3), band=band,
                            self_confidence=float(self_confidence or 0), adjustments=adjustments)


# ---------------------------------------------------------------------------------------
# Built-in self-test for normalize_intent — runs without pytest.
#   python -m app.engine.planner.confidence
# ---------------------------------------------------------------------------------------
def _selftest() -> int:
    failures: list[str] = []

    def check(name, cond):
        if not cond:
            failures.append(name)

    # THE PRODUCTION BUG: an in-person-visit label must NOT be read as a human request.
    for label in ("visit_in_person", "walk_in_person", "come_in_person",
                  "confirm_in_person_visit", "in person"):
        check(f"in_person_not_human[{label}]", normalize_intent(label) != "request_human")
    # ...and the reschedule these turns actually were still normalises correctly. (The
    # model emits "reschedule"/"postpone" here; a raw "change_appointment" would hit the
    # pre-existing alias precedence and land on book_appointment via its "appointment"
    # needle — a separate concern from this fix and deliberately not touched.)
    check("reschedule_kept", normalize_intent("reschedule_the_visit") == "reschedule")
    check("postpone_kept", normalize_intent("postpone_visit") == "reschedule")

    # Genuine human requests still map through.
    for label in ("talk_to_a_human", "speak to a person", "get me a manager",
                  "escalate", "escalated", "human agent", "live agent"):
        check(f"human_ok[{label}]", normalize_intent(label) == "request_human")

    # Mid-word collisions the old substring match got wrong are now clean.
    check("issue_not_legal", normalize_intent("billing_issue") != "legal_complaint")  # 'sue'
    check("inhuman_guard", normalize_intent("inhumane_wait") != "request_human")
    check("management_agent_guard",   # 'agent' inside a non-handover label
          normalize_intent("assigned_management") != "request_human")

    # Stem tolerance the old match relied on is preserved.
    check("cancellation_stem", normalize_intent("cancellation_request") == "cancel")
    check("booking_stem", normalize_intent("booking_request") == "book_appointment")

    # Unmatched labels pass through lowercased; empty stays None.
    check("passthrough", normalize_intent("Product_Question") == "product_question")
    check("none_stays_none", normalize_intent(None) is None)

    if failures:
        print(f"confidence self-test: {len(failures)} FAILED -> {failures}")
        return 1
    print("confidence self-test: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
