"""Guard: every measured figure the agent states must come from the material it was given.

THE BUG THIS FIXES. Asked "how far is the project from Wipro?" alongside a price question, the
agent said "approximately 5 kilometers from Wipro" on one call and "approximately seven
kilometres" on another. The knowledge base says 2.5 km. Retrieval had not surfaced the
location document (fixed separately — query_split.py), so the model had NO distance in front
of it and invented one, despite responder rule 1 ("if a number is not in the material, you
may not say it"). A prompt rule is a request; this is the guarantee.

WHAT IT CHECKS. A quantity is a number with a measurement unit: distance (kilometres, metres),
duration (minutes, hours), area (square feet / metres) and percentages. Each one in the reply
must match a quantity of the same kind in the turn's GROUNDING material — retrieved passages,
tool results, curated project facts, the caller's own words and the agent's previous line —
after unit normalisation (2.5 km == 2,500 metres, an hour == 60 minutes, "seven" == 7). A
hedged figure ("about 2 kilometres" for a passage's 2.2 km) may round; an unhedged one must
match to within rounding. Domain-agnostic: it knows units, not real estate.

Deliberately NOT checked: money (rupee figures are re-expressed in lakh/crore by
round_spoken_prices and checked by their own guards), clock times and dates (date_claim.py,
the booking guards), and bare counts with no unit ("3 clubhouses") — a unit is what makes a
figure a measurement a caller will act on.

WHAT IT DOES. Removes the clause carrying the unsupported figure (or the sentence, if the
figure is the whole of it) and says once, honestly, that the exact figure is not to hand. It
never substitutes a figure of its own. Pure and self-tested
(``python -m app.engine.planner.figure_claim``); applied per streamed sentence (responder.py)
and to the final text (turn_runner.py) against the SAME grounding text, so both reach the same
verdict.
"""
import json
import re
from dataclasses import dataclass

from app.engine.planner.text_normalize import normalize_quotes

_WORD_NUMS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "twenty-five": 25, "thirty": 30, "forty": 40,
    "forty-five": 45, "fifty": 50, "sixty": 60, "ninety": 90, "hundred": 100,
    "half a": 0.5, "half an": 0.5, "a half": 0.5, "one and a half": 1.5,
    "couple of": 2, "a couple of": 2,
}
_WORD_NUM_ALT = "|".join(sorted((re.escape(k) for k in _WORD_NUMS), key=len, reverse=True))
_NUM = rf"(?P<num>\d[\d,]*(?:\.\d+)?|{_WORD_NUM_ALT})"

# unit pattern -> (kind, factor to the kind's base unit)
_UNITS: list[tuple[str, str, float]] = [
    (r"kilomet(?:re|er)s?|kms?\b", "distance", 1000.0),
    (r"met(?:re|er)s?|mtrs?\b", "distance", 1.0),
    (r"minutes?|mins?\b", "duration", 1.0),
    (r"hours?|hrs?\b", "duration", 60.0),
    (r"square\s+f(?:ee|oo)t|sq\.?\s*f(?:ee)?t\.?|sqft|sft\b", "area", 1.0),
    (r"square\s+met(?:re|er)s?|sq\.?\s*m\b|sqm\b", "area", 10.7639),
    (r"percent|per\s+cent|%", "percent", 1.0),
]
_UNIT_ALT = "|".join(f"(?P<u{i}>{p})" for i, (p, _, _) in enumerate(_UNITS))
_QTY_RE = re.compile(rf"(?<![\w.]){_NUM}\s*-?\s*(?:{_UNIT_ALT})", re.IGNORECASE)
_HEDGE_RE = re.compile(r"\b(?:about|around|roughly|approximately|approx|nearly|almost|under|"
                       r"less than|within|over|more than|just|close to|some)\s*$", re.I)

# Ubiquitous phrasings that are not a claim about the tenant's facts.
_NOT_CLAIMS = re.compile(r"\b(?:24\s*(?:hours?|hrs?)|24\s*by\s*7|24/7|100\s*(?:percent|%)\s+sure)\b",
                         re.I)


@dataclass(frozen=True)
class Quantity:
    kind: str
    value: float          # in the kind's base unit (metres, minutes, square feet, percent)
    hedged: bool
    text: str
    # Content words of the sentence the figure sits in — what the figure is ABOUT. A rounded
    # ("about 5 km") match must share one with the material's sentence, or "Manipal Hospital
    # 5.5 km" would vouch for an invented "about 5 km from Wipro".
    subject: frozenset = frozenset()


# Words that say nothing about WHICH thing a figure measures.
_GENERIC = {
    "the", "a", "an", "is", "are", "was", "it", "its", "it's", "this", "that", "there", "here",
    "of", "to", "from", "in", "on", "at", "for", "by", "and", "or", "with", "about", "around",
    "roughly", "approximately", "approx", "nearly", "almost", "under", "over", "within", "just",
    "away", "far", "distance", "takes", "take", "taking", "long", "only", "so", "you", "your",
    "you're", "we", "our", "us", "i", "my", "me", "project", "site", "property", "place",
    "located", "location", "be", "will", "would", "can", "could", "has", "have", "do", "does",
    "less", "more", "than", "some", "close", "also", "which", "while", "but", "very", "quite",
}
_UNIT_WORDS = re.compile(rf"^(?:{'|'.join(p for p, _, _ in _UNITS)})$", re.I)


def _subject_words(sentence: str) -> frozenset:
    words = re.findall(r"[a-z][a-z'\-]+", sentence.lower())
    return frozenset(w for w in words if w not in _GENERIC and w not in _WORD_NUMS
                     and not _UNIT_WORDS.match(w) and len(w) > 2)


def _to_number(raw: str) -> float | None:
    raw = raw.strip().lower()
    if raw in _WORD_NUMS:
        return float(_WORD_NUMS[raw])
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def quantities(text: str) -> list[Quantity]:
    """Every measured quantity in `text`, normalised to base units, with the content words of
    its own sentence (split on sentence ends, newlines AND the ". " between list items such as
    "RGA Tech Park 2.2 km. Wipro Corporate Office 2.5 km.")."""
    out: list[Quantity] = []
    t = normalize_quotes(text or "")
    for sentence in re.split(r"(?<=[.?!।;])\s+|\n+", t):
        for m in _QTY_RE.finditer(sentence):
            n = _to_number(m.group("num"))
            if n is None:
                continue
            for i, (_, kind, factor) in enumerate(_UNITS):
                if m.group(f"u{i}"):
                    hedged = bool(_HEDGE_RE.search(sentence[:m.start()]))
                    out.append(Quantity(kind, n * factor, hedged, m.group(0),
                                        _subject_words(sentence)))
                    break
    return out


# Tool results are JSON: a figure's unit lives in its KEY ("sqft": 1560, "duration_min": 60),
# not next to the number. Rendered as "<value> <unit>" lines so the same matcher reads them —
# otherwise a correct "1,560 square feet" read straight off an inventory result would be cut.
_KEY_UNITS = [
    (re.compile(r"sq_?ft|sqft|area|carpet|built_?up", re.I), "square feet"),
    (re.compile(r"(?:^|_)km(?:$|_)|kilomet", re.I), "kilometres"),
    (re.compile(r"distance_?m$|met(?:re|er)s?$", re.I), "metres"),
    (re.compile(r"min(?:ute)?s?$|duration", re.I), "minutes"),
    (re.compile(r"hours?$|hrs?$", re.I), "hours"),
    (re.compile(r"percent|pct", re.I), "percent"),
]


def _keyed_quantities(obj, out: list[str], key: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _keyed_quantities(v, out, str(k))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _keyed_quantities(v, out, key)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool) and key:
        for rx, unit in _KEY_UNITS:
            if rx.search(key):
                out.append(f"{key.replace('_', ' ')} {obj} {unit}.")
                break


def grounding_text(*, knowledge: list[dict] | None = None, tool_outputs: list | None = None,
                   project_facts: list[dict] | None = None, extra: list[str] | None = None) -> str:
    """Flatten everything the reply may legitimately quote into one searchable string."""
    bits: list[str] = []
    for k in knowledge or []:
        if isinstance(k, dict):
            bits.append(str(k.get("content") or ""))
    for o in tool_outputs or []:
        try:
            bits.append(json.dumps(o, default=str, ensure_ascii=False))
        except (TypeError, ValueError):
            bits.append(str(o))
        keyed: list[str] = []
        _keyed_quantities(o, keyed)
        bits.extend(keyed)
    for f in project_facts or []:
        if isinstance(f, dict):
            bits.append(f"{f.get('label', '')}: {f.get('value', '')}")
    bits.extend(x for x in (extra or []) if x)
    return "\n".join(bits)


def _supported(q: Quantity, pool: list[Quantity]) -> bool:
    """Exact (to rounding) match of the same kind: supported. A hedged figure may round further
    (up to 25%), but only against a figure about the SAME thing — the two sentences must share a
    subject word, unless the reply names no subject at all ("about 12 km away")."""
    for g in pool:
        if g.kind != q.kind:
            continue
        rel = abs(g.value - q.value) / max(abs(g.value), 1e-9)
        if rel <= 0.02:
            return True
        if q.hedged and rel <= 0.25 and (not q.subject or (q.subject & g.subject)):
            return True
    return False


def unsupported(text: str, grounding: str) -> list[Quantity]:
    """The quantities in `text` that nothing in `grounding` supports."""
    pool = quantities(grounding)
    return [q for q in quantities(text)
            if not _NOT_CLAIMS.search(q.text) and not _supported(q, pool)]


_CLAUSE_SPLIT_RE = re.compile(r"\s*;\s*|\s+[—–]\s+|,\s+(?=(?:and|but|while|which|so)\b)")


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.?!।])\s+|\n+", text or "") if s.strip()]


_MISSING_LINE = {
    "english": {"distance": "I don't have the exact distance for that to hand.",
                "duration": "I don't have the exact timing for that to hand.",
                "area": "I don't have the exact area for that to hand.",
                "percent": "I don't have the exact figure for that to hand."},
    "hindi": {"distance": "उसकी सटीक दूरी अभी मेरे पास नहीं है।",
              "duration": "उसका सटीक समय अभी मेरे पास नहीं है।",
              "area": "उसका सटीक एरिया अभी मेरे पास नहीं है।",
              "percent": "उसका सटीक आंकड़ा अभी मेरे पास नहीं है।"},
    "hinglish": {"distance": "Uski exact distance abhi mere paas nahi hai.",
                 "duration": "Uska exact time abhi mere paas nahi hai.",
                 "area": "Uska exact area abhi mere paas nahi hai.",
                 "percent": "Uska exact figure abhi mere paas nahi hai."},
}


def missing_line(kind: str, *, language: str = "english") -> str:
    lines = _MISSING_LINE.get(language, _MISSING_LINE["english"])
    return lines.get(kind, lines["percent"])


def scrub(text: str, grounding: str, *, language: str = "english",
          said: set | None = None) -> str:
    """`text` with every clause/sentence that states an unsupported figure removed, and one
    honest "I don't have the exact <kind> to hand" line IN ITS PLACE — at most once per kind.

    `said` carries the kinds already given a missing-line across calls: the responder's stream
    passes one set for all the sentences of a reply, so streaming sentence by sentence produces
    exactly the text that one call on the whole reply does (bridge.py replays on any
    difference). Unchanged text is returned as the same string."""
    if not text:
        return text or ""
    pool = quantities(grounding)
    said = set() if said is None else said

    def bad(fragment: str) -> list[Quantity]:
        return [q for q in quantities(fragment)
                if not _NOT_CLAIMS.search(q.text) and not _supported(q, pool)]

    kept: list[str] = []
    changed = False
    for sentence in _sentences(text):
        hits = bad(sentence)
        if not hits:
            kept.append(sentence)
            continue
        changed = True
        clauses = [c for c in _CLAUSE_SPLIT_RE.split(sentence) if c.strip()]
        clean = [c for c in clauses if not bad(c)]
        if clean and len(clean) < len(clauses):
            s = " ".join(clean).strip().rstrip(",")
            s = s[0].upper() + s[1:]
            kept.append(s if s[-1] in ".?!।" else s + ".")
        for kind in dict.fromkeys(q.kind for q in hits):
            if kind not in said:
                said.add(kind)
                kept.append(missing_line(kind, language=language))
    if not changed:
        return text
    return " ".join(kept)


def _selftest() -> None:
    kb = ("Location\nChikkanayakanahalli, off Sarjapur Road. RGA Tech Park 2.2 km. Wipro "
          "Corporate Office 2.5 km. Manipal Hospital 5.5 km.\nA visit takes about an hour.\n"
          "3 BHK — 1,239 to 1,560 square feet. 100 percent DG backup. Floor rise from the 4th.")

    # the reported inventions
    for bad_reply in ("The project is approximately 5 kilometers from Wipro.",
                      "It's approximately seven kilometres from Wipro.",
                      "The site visit takes about 3 hours.",
                      "The largest 3 BHK is 1,800 square feet."):
        assert unsupported(bad_reply, kb), bad_reply

    for good in ("The project is 2.5 kilometres from Wipro Corporate Office.",
                 "It's about 2 kilometres from RGA Tech Park.",           # hedged rounding of 2.2
                 "It's 2,500 metres from Wipro.",                         # unit conversion
                 "A site visit takes about an hour.",
                 "A site visit takes around 60 minutes.",
                 "Sizes run from 1,239 to 1,560 square feet.",
                 "There is 100 percent power backup.",
                 "Security is available 24 hours a day.",
                 "The 3 BHK has three bedrooms and two balconies.",       # counts, no unit
                 "The rate is 11,499 rupees per square foot.",            # a rate, not an area
                 ""):
        assert not unsupported(good, kb), (good, unsupported(good, kb))

    out = scrub("The 3 BHK rate is 11,499 rupees per square foot. The project is approximately "
                "5 kilometers from Wipro.", kb)
    assert out == ("The 3 BHK rate is 11,499 rupees per square foot. I don't have the exact "
                   "distance for that to hand."), out

    out = scrub("It's 2.5 kilometres from Wipro, and about 9 kilometres from Electronic City.", kb)
    assert out == ("It's 2.5 kilometres from Wipro. I don't have the exact distance for that "
                   "to hand."), out

    same = "The project is 2.5 kilometres from Wipro Corporate Office."
    assert scrub(same, kb) is same

    # streaming sentence by sentence with a shared `said` == one call on the whole reply
    reply = ("It is about 6 kilometres from Wipro. The visit takes 3 hours. It is also 8 km "
             "from Bellandur. Shall I book a visit?")
    whole = scrub(reply, kb)
    said: set = set()
    streamed = " ".join(x for x in (scrub(s, kb, said=said) for s in _sentences(reply)) if x)
    assert whole == streamed, (whole, streamed)
    assert whole == ("I don't have the exact distance for that to hand. I don't have the exact "
                     "timing for that to hand. Shall I book a visit?"), whole

    # a tool result's unit lives in its key
    g = grounding_text(tool_outputs=[{"units": [{"bhk_type": "3 BHK", "sqft": 1560,
                                                 "carpet_sqft": 1032}]}])
    assert not unsupported("The 1,560 square foot unit has a carpet area of 1,032 square feet.", g)
    assert unsupported("The 1,800 square foot unit is available.", g)

    # the caller's own figure is grounding too
    g = grounding_text(knowledge=[{"content": kb}], extra=["I live 12 km from the site"])
    assert not unsupported("So you're about 12 kilometres away.", g)
    assert missing_line("distance", language="hinglish").startswith("Uski")
    print("figure_claim selftest: all passed")


if __name__ == "__main__":
    _selftest()
