"""Guard: the agent may not declare a rent/lease/resale scope boundary the documents never state.

THE BUG THIS FIXES. Asked "what's the approx rent of this?", the agent answered "we only sell
apartments and do not offer rentals" — a POLICY invented from the absence of any rent passage,
not read from one. The seeded project facts say nothing about sale-vs-rent at all, so there is
no source for the claim; the model asserted it because a sales project "obviously" only sells.

WHY A DETERMINISTIC NET (not just the responder's ``scope_grounded`` self-assessment). An LLM
is unreliable at telling "explicitly stated" from "obviously inferred": faced with a luxury
apartment project it will mark "we only sell" as grounded even when no passage says so, so the
self-report alone lets the fabrication through. This is the same lesson the feature-grounding
guard learned. So the primary check here is a fact about the MATERIAL, not the model's opinion:
a statement that we do not rent / lease / do resale, or that we sell exclusively, can only be
grounded if the source material actually mentions that scope. If the words that would carry it
("rent", "lease", "resale", "for sale", "ownership", "outright") appear NOWHERE in the facts,
knowledge or tool output the reply drew on, the claim is unfounded by definition — replace it.

The responder self-assessment (Rule 14) still runs and still helps the model phrase honestly in
the first place; this is the reliable backstop underneath it.

Pure and self-tested — ``python -m app.engine.planner.scope_claim``. The wiring in turn_runner
supplies the grounding string (the same one grounding_check uses).
"""
from __future__ import annotations

import re

# Present-tense assertions that BOUND what the project offers: we do not rent/lease/resell, or
# we sell exclusively. Offers and questions are excluded below. Kept to the rent/lease/resale/
# sale-only family — the high-frequency, high-damage claim — not every conceivable scope remark.
_CLAIM_PATTERNS = (
    # "we only sell", "we just sell", "we exclusively sell", "we sell ... only"
    r"\bwe (?:only|just|exclusively) sell\b",
    r"\bwe sell\b[^.?!]{0,40}\bonly\b",
    r"\bfor sale only\b", r"\bsale[-\s]only\b", r"\bsell(?:ing)? (?:them )?outright\b",
    r"\bonly (?:available )?(?:for|to) (?:buy|purchase|own)\b",
    r"\bonly for ownership\b", r"\bownership only\b",
    # "do not offer rentals", "don't do leasing", "no rentals", "not for rent", "we don't rent"
    r"\b(?:do not|don'?t|doesn'?t|does not) (?:offer|do|provide|have|handle) "
    r"[^.?!]{0,25}\b(?:rent|rental|rentals|renting|leas\w*|resale|resell\w*)\b",
    r"\bno (?:rentals?|renting|leasing|leases|resale)\b",
    r"\bnot (?:available )?for (?:rent|lease)\b",
    r"\bwe (?:do not|don'?t) (?:rent|lease|resell)\b",
    r"\b(?:rentals?|leasing|resale) (?:is|are) not (?:offered|available|something we)\b",
)
_CLAIM_RES = tuple(re.compile(p, re.I) for p in _CLAIM_PATTERNS)

# An offer or a question shares the scope nouns but asserts nothing — never a fabricated claim.
_OFFER_OR_QUESTION = re.compile(
    r"\?|\b(?:would you like|shall i|do you want|can i (?:help|set|book|arrange)|"
    r"are you (?:looking|interested)|if you(?:'d| would) like)\b", re.I)

# What GROUNDS a rent/lease/sale-scope claim is an EXPLICIT scope DECLARATION in the material —
# NOT the incidental appearance of "sale", "ownership" or "outright", which pepper any sales
# project's cost sheets and payment schedules ("outright purchase price", "ownership at handover")
# and say nothing about whether renting is offered. Grounding on those bare words is what made the
# guard FLAKY: a rent question retrieves the cost sheet, the cost sheet contains "outright", the
# gate wrongly reads the fabricated "we only sell" as grounded, and it slips through — while a turn
# that retrieved plainer chunks caught it. So the gate matches only DELIBERATE scope statements,
# in two shapes a tenant writes on purpose and a price list never contains by accident:
#   (a) EXCLUSIVITY — sale/purchase/ownership qualified by "only";
#   (b) EXCLUSION — an explicit negation of renting/leasing/resale.
# Word-boundaried so "current" can't satisfy "rent"; structure-bearing so "apartments for sale"
# (marketing, no "only") and "outright purchase price" (a figure, no "only") do NOT ground.
_SCOPE_DECLARED_RE = re.compile(
    # (a) exclusivity: "for sale only", "sale only", "only for purchase", "ownership only", ...
    r"\bfor sale only\b|\bsale only\b|\bonly for sale\b|\bsold only\b|"
    r"\bonly (?:available )?(?:for |to )?(?:buy|purchase|own|ownership|sale)\b|"
    r"\bavailable only (?:for |to )?(?:buy|purchase|sale|own)\b|"
    r"\bownership only\b|\bfor ownership only\b|\boutright (?:sale|purchase) only\b|"
    r"\bexclusively (?:for )?(?:sale|purchase|ownership)\b|"
    # (b) exclusion of rent / lease / resale
    r"\bnot (?:available )?for rent\b|\bnot (?:available )?for lease\b|\bnot for rental\b|"
    r"\bno rentals?\b|\bno leasing\b|\bno leases\b|\bno resale\b|"
    r"\brentals?(?: are| is)? not (?:offered|available|permitted|possible|allowed)\b|"
    r"\bleasing(?: is)? not (?:offered|available|permitted|allowed)\b|"
    r"\b(?:we )?do(?:es)? not (?:offer|do|provide|allow|permit|handle) "
    r"(?:rent|rental|rentals|renting|leasing|lease|resale)\b|"
    r"\b(?:we )?don'?t (?:offer|do|provide|allow|permit|handle) "
    r"(?:rent|rental|rentals|renting|leasing|lease|resale)\b|"
    r"\b(?:we )?do(?:es)? not (?:rent|lease)\b|\b(?:we )?don'?t (?:rent|lease)\b",
    re.I)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.?!])\s+|\n+", text or "") if s.strip()]


def asserts_closed_scope(text: str) -> bool:
    """True when SOME sentence states, as fact, a rent/lease/resale/sale-only scope boundary —
    and that sentence is not itself an offer or a question."""
    for sent in _sentences(text):
        if _OFFER_OR_QUESTION.search(sent):
            continue
        if any(r.search(sent) for r in _CLAIM_RES):
            return True
    return False


def scope_declared(grounding: str) -> bool:
    """True when the material EXPLICITLY declares the rent/sale scope — an exclusivity statement
    ("for sale only") or an explicit exclusion of renting/leasing ("not available for rent"). This
    is the deterministic grounding gate: the model is never trusted to judge grounding for a scope
    claim (it rationalises "we only sell" as obvious), so grounding is decided here, against the
    actual source words. Deliberately does NOT fire on incidental "sale"/"ownership"/"outright" in
    a price list — matching those made the guard flaky (see _SCOPE_DECLARED_RE)."""
    return bool(_SCOPE_DECLARED_RE.search(grounding or ""))


def is_ungrounded_scope_claim(text: str, grounding: str) -> bool:
    """The reply declares a rent/lease/sale-scope boundary the material never states.

    ``grounding`` is the flattened, normalised material the reply was allowed to draw on
    (facts + retrieved knowledge + this turn's tool output — feature_grounding.grounding_text).
    A scope claim is legitimate ONLY if that material EXPLICITLY declares the scope
    (scope_declared); absent such a declaration, the claim was invented and must be replaced.

    This uses the REGEX detector for the claim. turn_runner additionally treats the responder's
    own ``scope_claim=="closed"`` self-report as a detection trigger for the SAME grounding gate,
    so a paraphrase the regex misses is still caught — see turn_runner. Detection is a union of
    two signals; the grounding decision (scope_declared) is the single deterministic arbiter.
    """
    return asserts_closed_scope(text) and not scope_declared(grounding)


# ---------------------------------------------------------------------------------------
# Built-in self-test — runs without pytest.
#   python -m app.engine.planner.scope_claim
# ---------------------------------------------------------------------------------------
def _selftest() -> int:
    failures: list[str] = []

    def check(name, cond):
        if not cond:
            failures.append(name)

    # Material with NO sale/rent scope vocabulary (the real DSR facts) — any scope claim here is
    # fabricated. "current pricing" must NOT satisfy "rent"; "sales team" must NOT satisfy "sale".
    G_SILENT = ("dsr highland greenz, a gated luxury-apartment community. location off sarjapur "
                "road. rcc framed structure. 1, 1.5, 2, 2.5 and 3 bhk, 619-1422 sq ft. the sales "
                "team confirms current pricing.")
    # Material that DOES state the scope — a claim is then grounded and must stand.
    G_STATED = "these units are for sale only and are not available for rent or lease."
    # THE FLAKINESS CASE: a retrieved cost sheet / payment schedule that mentions "outright",
    # "ownership" and "purchase" INCIDENTALLY — it says nothing about renting, so it must NOT
    # ground a "we only sell" claim. This is exactly what made the guard intermittent: on turns
    # where retrieval returned a chunk like this, the old topical-vocab gate wrongly stood down.
    G_INCIDENTAL = ("payment schedule: 20% on booking, balance on ownership handover. the "
                    "outright purchase price includes one covered car park. sale deed at "
                    "possession. gst extra.")

    # --- fabricated scope claims that MUST be caught against silent material ---
    fabricated = [
        "We do not offer rentals or leases; we only sell apartments outright for ownership.",
        "We only sell apartments at DSR Highland Greenz and do not offer rentals.",
        "We don't do rentals here.",
        "These flats are for sale only.",
        "We do not offer leasing or resale.",
        "No rentals — the units are only available to buy.",
    ]
    for t in fabricated:
        check(f"fabricated::{t[:30]}", is_ungrounded_scope_claim(t, G_SILENT))
        # ...but the SAME claim stands when the material actually DECLARES the scope.
        check(f"grounded_stands::{t[:30]}", not is_ungrounded_scope_claim(t, G_STATED))
        # ...and it is STILL caught when the material only mentions sale/ownership incidentally
        # (the flakiness fix — incidental cost-sheet vocab must not ground a scope claim).
        check(f"incidental_caught::{t[:30]}", is_ungrounded_scope_claim(t, G_INCIDENTAL))

    # --- non-claims: offers, questions, honest deflections — never replaced ---
    safe = [
        "I don't have that detail in front of me — shall I have the sales team call you back?",
        "Are you looking to rent or to buy?",
        "Would you like me to arrange a callback about rental options?",
        "The 3 BHK ranges from 1,239 to 1,560 square feet.",
        "I can help you schedule a visit to see the 3 BHK units.",
        "I'm not able to confirm that from the details I have.",
    ]
    for t in safe:
        check(f"safe::{t[:30]}", not is_ungrounded_scope_claim(t, G_SILENT))

    # --- the grounding gate itself (scope_declared): only EXPLICIT declarations count ---
    check("declared_absent_silent", not scope_declared(G_SILENT))
    check("declared_absent_incidental", not scope_declared(G_INCIDENTAL))
    check("declared_for_sale_only", scope_declared("the homes are for sale only"))
    check("declared_not_for_rent", scope_declared("units are not available for rent"))
    check("declared_no_rentals", scope_declared("no rentals are offered at this project"))
    check("declared_we_dont_rent", scope_declared("we do not rent or lease these apartments"))
    check("declared_ownership_only", scope_declared("these are ownership only"))
    # incidental / marketing vocabulary must NOT count as a declaration
    check("not_declared_outright_price", not scope_declared("the outright purchase price is 1.2 cr"))
    check("not_declared_ownership_handover", not scope_declared("ownership transfers at handover"))
    check("not_declared_apartments_for_sale", not scope_declared("luxury apartments for sale in bengaluru"))
    check("not_declared_current", not scope_declared("current pricing on request"))
    check("not_declared_salesteam", not scope_declared("the sales team will assist you"))
    check("not_declared_sale_deed", not scope_declared("sale deed executed at possession"))

    # --- word-boundary / structure safety: prose that merely CONTAINS the words must not ground ---
    check("current_not_rent",
          is_ungrounded_scope_claim("We only sell here.", "current pricing from the team"))
    check("outright_price_not_declaration",
          is_ungrounded_scope_claim("We only sell here.", "the outright purchase price is 1.2 cr"))
    check("empty_grounding_ungrounded",
          is_ungrounded_scope_claim("We do not offer rentals.", ""))

    if failures:
        print(f"scope_claim self-test: {len(failures)} FAILED -> {failures}")
        return 1
    print("scope_claim self-test: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
