"""Indian-currency parsing and phrasing — deterministic, no model in the loop.

WHY THIS EXISTS. A caller who says "anything around 1.1 cr" is stating a numeric budget
band, and answering it means comparing that figure against real unit prices. Letting a
language model do that comparison over retrieved prose is how "around 1.1 cr" came back as
"nothing under one crore" — a scripted line about a DIFFERENT question (budget == 1 crore)
that happened to rank highest in the pricing document. The number 1.1 was simply dropped.

Crore/lakh parsing is the one part of this that MUST be exact, and it is pure arithmetic, so
it is done here in code rather than guessed by the model: "1.1 cr" is 11,000,000 rupees,
every time, and "under 90 lakh" is a ceiling of 9,000,000, not a point. The inventory tool
then filters actual rows against the parsed constraint. That is the whole design — take the
numeric reasoning away from the sentence-writer and give it to a comparison against fact.

Word-numbers ("one crore") are deliberately NOT parsed here: they are rare in a budget
phrase, ambiguous ("a couple of crore"), and the planner's entity extraction already
normalises them to digits before this sees them. This parser is the digit path, kept small
and certain, not a general natural-language number engine.
"""
import re
from dataclasses import dataclass

CRORE = 10_000_000        # 1 crore  = 1,00,00,000
LAKH = 100_000            # 1 lakh   = 1,00,000

# Words that turn a bare figure into a one-sided or two-sided constraint. Order matters only
# in that longer phrases are tried before the words they contain (see _BOUND_PATTERNS).
_MAX_WORDS = ("under", "below", "less than", "lesser than", "upto", "up to", "at most",
              "maximum", "max", "within", "no more than", "not more than", "budget of",
              "budget is", "cheaper than")
_MIN_WORDS = ("over", "above", "more than", "at least", "minimum", "min", "starting",
              "north of", "upwards of")
_NEAR_WORDS = ("around", "about", "approx", "approximately", "roughly", "close to",
               "near", "~", "in the range of", "somewhere around")

# A number optionally followed by a unit. Group 1 = the figure, group 2 = the unit (may be
# empty, in which case the figure is already in rupees, e.g. "1,10,00,000").
_UNIT = r"(cr(?:ore)?s?|l(?:a?kh?s?|acs?)?|k|m(?:n|illion)?)?"
_NUM = r"(\d[\d,]*\.?\d*)"
_AMOUNT_RE = re.compile(_NUM + r"\s*" + _UNIT, re.IGNORECASE)


@dataclass
class Budget:
    """A parsed budget constraint, all figures in whole rupees.

    Exactly one shape is populated per parse:
      * point  — ``target`` set (an "around X" / bare "X" figure); ``minimum``/``maximum``
                 hold the tolerance band the caller's "around" implies, for convenience.
      * max    — ``maximum`` set ("under X", "up to X").
      * min    — ``minimum`` set ("over X", "at least X").
      * range  — ``minimum`` and ``maximum`` set ("between X and Y", "X to Y").
    """
    kind: str                       # point | max | min | range
    target: int | None = None
    minimum: int | None = None
    maximum: int | None = None
    raw: str = ""

    def as_dict(self) -> dict:
        return {"kind": self.kind, "target": self.target, "minimum": self.minimum,
                "maximum": self.maximum, "raw": self.raw}


def _unit_multiplier(unit: str | None) -> int:
    u = (unit or "").lower()
    if u.startswith("cr"):
        return CRORE
    if u.startswith("l") or u.startswith("lac") or u.startswith("lakh"):
        return LAKH
    if u == "k":
        return 1_000
    if u.startswith("m"):
        return 1_000_000
    return 1


def _to_rupees(figure: str, unit: str | None) -> int | None:
    """One "<number><unit>" token to whole rupees, or None if it is not a real amount."""
    try:
        value = float(figure.replace(",", ""))
    except (ValueError, AttributeError):
        return None
    if value <= 0:
        return None
    rupees = value * _unit_multiplier(unit)
    # A bare figure under ~1000 with no unit ("around 90") is a lakh/crore figure the caller
    # abbreviated, not 90 rupees — but we cannot know which, so we do NOT guess. Only accept a
    # unit-less figure when it is already a plausible full price (>= 1 lakh). This keeps a
    # stray "3" (as in "3 BHK") from ever being read as a budget of three rupees.
    if not unit and rupees < LAKH:
        return None
    return int(round(rupees))


def _amounts(text: str) -> list[int]:
    """Every rupee amount named in the text, left to right."""
    out = []
    for m in _AMOUNT_RE.finditer(text):
        rupees = _to_rupees(m.group(1), m.group(2))
        if rupees is not None:
            out.append(rupees)
    return out


# "between X and Y", "X to Y", "X - Y" — a two-figure span. Kept ahead of the single-figure
# path so a range is never misread as a bare point on its first number.
_RANGE_RE = re.compile(
    r"(?:between\s+)?" + _NUM + r"\s*" + _UNIT +
    r"\s*(?:to|-|–|and|&)\s*" + _NUM + r"\s*" + _UNIT,
    re.IGNORECASE)


def _has_word(text: str, words) -> bool:
    return any(re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", text) for w in words)


def parse_budget(text: str | None, *, near_tolerance: float = 0.15) -> Budget | None:
    """Parse a budget expression into a :class:`Budget`, or None when there is no figure.

    ``near_tolerance`` is the fraction above and below an "around X" target that the implied
    band spans (0.15 → ±15 %). It only informs ``minimum``/``maximum`` on a ``point`` result;
    the inventory tool uses the band to decide which real units are "near" the target.
    """
    if not text:
        return None
    low = str(text).strip().lower()
    if not low:
        return None

    # A range wins outright — "1.2 to 1.4 cr", "between 90 lakh and 1.1 cr". A trailing unit
    # applies to the first figure too when the first carries none ("1.2 to 1.4 cr").
    rm = _RANGE_RE.search(low)
    if rm:
        lo_unit = rm.group(2) or rm.group(4)          # borrow the high figure's unit if needed
        hi_unit = rm.group(4) or rm.group(2)
        lo = _to_rupees(rm.group(1), lo_unit)
        hi = _to_rupees(rm.group(3), hi_unit)
        if lo is not None and hi is not None:
            return Budget(kind="range", minimum=min(lo, hi), maximum=max(lo, hi), raw=low)

    amounts = _amounts(low)
    if not amounts:
        return None
    figure = amounts[0]

    if _has_word(low, _MAX_WORDS):
        return Budget(kind="max", maximum=figure, raw=low)
    if _has_word(low, _MIN_WORDS):
        return Budget(kind="min", minimum=figure, raw=low)

    # Bare figure or an explicit "around" — a target with a tolerance band. Treating a bare
    # "1.1 cr" as a point (not a ceiling) is deliberate: a caller who states a number wants
    # what sits AT it, and the nearest-below/above the tool returns covers both directions.
    band = int(round(figure * near_tolerance))
    return Budget(kind="point", target=figure,
                  minimum=max(0, figure - band), maximum=figure + band, raw=low)


def crore_phrase(rupees: int | None) -> str | None:
    """A spoken-friendly figure: "1.04 crore", "72 lakh". None passes through as None.

    Rounded to two decimals at crore scale and one at lakh scale — enough to be recognisable
    without implying false precision. The exact integer is always carried ALONGSIDE this in a
    tool result, so this is the phrasing, never the sole source of the number.
    """
    if rupees is None:
        return None
    if rupees >= CRORE:
        value = rupees / CRORE
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        return f"{text} crore"
    if rupees >= LAKH:
        value = rupees / LAKH
        text = f"{value:.1f}".rstrip("0").rstrip(".")
        return f"{text} lakh"
    return f"{rupees:,} rupees"


def indian_format(rupees: int | None) -> str | None:
    """The exact figure grouped Indian-style: 10393750 -> "1,03,93,750"."""
    if rupees is None:
        return None
    s = str(int(rupees))
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
    return f"{head},{tail}"


# An Indian-grouped rupee figure: 1-2 leading digits, one or more ",XX" groups, then a
# final ",XXX" -- "1,54,32,011", "96,25,000". Western grouping ("15,432,011", one comma
# every 3 digits) and a plain sqft figure ("1,239") never match this shape, so this is safe
# to run unconditionally over any reply text rather than needing to know it is a price.
_INDIAN_GROUPED_RUPEES_RE = re.compile(r"\b\d{1,2}(?:,\d{2})+,\d{3}\b")


def round_spoken_prices(text: str) -> str:
    """Rewrite any raw Indian-grouped rupee figure still in the text as the rounded
    crore/lakh phrase -- e.g. "1,54,32,011" -> "1.54 crore".

    Shared by turn_runner.py (applied to the final validated reply, the very last thing
    done before a reply is spoken) and responder.py (applied to each sentence AS IT
    STREAMS, so early-pushed TTS audio never says a raw digit figure that the final pass
    would have rewritten). Both call sites must use the exact same function: applying it
    only at the final-text stage let a streamed sentence say "1,54,32,011 rupees" while
    the validated text said "1.54 crore rupees" for the very same fact — a real
    duplicate/wrong-then-corrected spoken price this fixes at the source instead of only
    catching after the fact.

    prompts.py already instructs the model to always speak price_phrase, never a raw
    digit string -- even when the exact figure comes from a retrieved knowledge document
    rather than a tool (a cost-sheet chunk literally contains "Total 1,54,32,011", written
    that way because that is how the source document is filed). On a real call the model
    followed that instruction inconsistently: sometimes right, sometimes reading the raw
    figure straight out of the source text regardless of what it was told. A prompt
    instruction is a request; this is the guarantee.
    """
    return _INDIAN_GROUPED_RUPEES_RE.sub(
        lambda m: crore_phrase(int(m.group(0).replace(",", ""))), text or "")


# A real call had the responder write "...going up to roughly ₹1.33 lakh carpet (total about
# ₹1.42 crore)." -- the NUMBER was right (the unit's own flat_cost is genuinely ~1.33 crore),
# only the unit word was wrong. round_spoken_prices above cannot catch this: it only rewrites
# a raw digit string still sitting in the text, and here the model had already chosen its own
# (wrong) unit word in prose -- there is no raw figure left to re-derive the correct one from.
#
# THE BUG THIS FIXES. Nothing validated that a spoken "X lakh" is actually lakh-scale. The
# one place this is checkable without the tool's original rupee figure: a base/carpet price
# and its own paired "total" figure, stated together in the same breath, must be close to
# each other -- a real flat's base cost is always a large majority of its own all-in total
# (GST/registration/extras add maybe 5-15%, never anywhere close to 100x). "1.33 lakh" next to
# "total 1.42 crore" is a ~100x mismatch; "96.25 lakh" next to "total 1.04 crore" (a correct,
# real example from the same call) is a normal ~92%. That ratio is the signal.
_BASE_TOTAL_PAIR_RE = re.compile(
    r"₹?\s*(\d[\d,]*\.?\d*)\s*lakh\b[^.?!]{0,40}?\btotal\b[^.?!]{0,20}?"
    r"₹?\s*(\d[\d,]*\.?\d*)\s*crore\b", re.IGNORECASE)

# Below this fraction of its own paired total, a "lakh" figure is not a plausible base price
# for that total -- it is almost certainly the same number meant as crore, not lakh. A real
# base-vs-all-in gap is a few percent; 50% is a generous floor that will never flag a genuine
# fee/markup difference, only an actual unit mislabeling.
_BASE_TOTAL_MIN_RATIO = 0.5


def fix_lakh_crore_mismatch(text: str) -> str:
    """Correct a base-price figure spoken in lakh when paired with its own "total ... crore"
    in the same breath implies it should have been crore -- see the module comment above.
    """
    def _fix(m: re.Match) -> str:
        base_lakh = float(m.group(1).replace(",", ""))
        total_crore = float(m.group(2).replace(",", ""))
        base_rupees = base_lakh * LAKH
        total_rupees = total_crore * CRORE
        if total_rupees > 0 and base_rupees < _BASE_TOTAL_MIN_RATIO * total_rupees:
            # The figure itself (e.g. 1.33) was right -- only "lakh" should have been "crore".
            # Replaced via its own sub (not a plain string op) so odd spacing in the original
            # match can never make the fix silently no-op.
            return re.sub(
                r"(\d[\d,]*\.?\d*)\s*lakh\b",
                lambda n: f"{n.group(1)} crore", m.group(0), count=1, flags=re.IGNORECASE)
        return m.group(0)

    return _BASE_TOTAL_PAIR_RE.sub(_fix, text or "")


# --- self-check -------------------------------------------------------------------------
if __name__ == "__main__":
    def check(label, cond):
        print(("ok  " if cond else "FAIL") + "  " + label)
        assert cond, label

    b = parse_budget("do you have anything around 1.1 cr")
    check("around 1.1 cr -> point 1.1cr", b.kind == "point" and b.target == 11_000_000)
    check("1.1 cr band ~15%", b.minimum == 9_350_000 and b.maximum == 12_650_000)

    b = parse_budget("under 1 crore")
    check("under 1 crore -> max 1cr", b.kind == "max" and b.maximum == 10_000_000)

    b = parse_budget("90 lakh")
    check("90 lakh -> point 90L", b.kind == "point" and b.target == 9_000_000)

    b = parse_budget("between 1.2 and 1.4 cr")
    check("range 1.2-1.4 cr", b.kind == "range" and b.minimum == 12_000_000
          and b.maximum == 14_000_000)

    b = parse_budget("1.2 to 1.4 crore")
    check("1.2 to 1.4 crore (unit on 2nd only)",
          b.kind == "range" and b.minimum == 12_000_000 and b.maximum == 14_000_000)

    b = parse_budget("at least 1.5 cr")
    check("at least 1.5 cr -> min", b.kind == "min" and b.minimum == 15_000_000)

    b = parse_budget("1,10,00,000")
    check("plain grouped rupees", b.kind == "point" and b.target == 11_000_000)

    check("no figure -> None", parse_budget("do you have a 3 bhk") is None)
    check("bare small int ignored", parse_budget("3 bhk") is None)

    check("crore phrase 1.04", crore_phrase(10_393_750) == "1.04 crore")
    check("crore phrase whole", crore_phrase(15_000_000) == "1.5 crore")
    check("lakh phrase", crore_phrase(9_000_000) == "90 lakh")
    check("indian format", indian_format(10_393_750) == "1,03,93,750")

    # fix_lakh_crore_mismatch -- the exact real bug: "lakh" should have been "crore".
    buggy = ("...going up to roughly ₹1.33 lakh carpet (total about ₹1.42 crore).")
    fixed = fix_lakh_crore_mismatch(buggy)
    check("mismatch corrected to crore", "1.33 crore" in fixed and "1.33 lakh" not in fixed)

    # The real CORRECT sentence from the same call must never be touched -- 96.25 lakh is a
    # genuine, plausible base price for a 1.04 crore total (~92%), not a mismatch.
    correct = "starting at about ₹96.25 lakh (total around ₹1.04 crore)"
    check("genuine lakh figure left alone", fix_lakh_crore_mismatch(correct) == correct)

    # A lakh figure with no paired total at all must never be touched -- nothing to compare.
    no_total = "priced at about ₹72 lakh for the smallest unit"
    check("no paired total -> untouched", fix_lakh_crore_mismatch(no_total) == no_total)

    print("\nall money.py self-checks passed")
