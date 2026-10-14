"""R5.08 — chunk metadata enrichment: structured price / BHK / area facts per chunk.

WHAT THIS IS FOR. Retrieval could only guess what a chunk is about from its words and its
document's ``doc_type``. A price question lifted every chunk of a ``pricing`` document —
including ones that state no number — and gave nothing to a brochure paragraph that quotes
"1,03,93,750 rupees". A "2.5 BHK" question matched chunks by the substring "2.5 bhk" and
could not tell which configurations a cost-sheet row actually covers.

This module reads each chunk ONCE, at ingestion, and records what it states as data:

    {"v": 1,
     "has_price": true, "has_rate": true, "has_area": true,
     "bhk":    [2.0, 2.5],                 # configurations the chunk mentions
     "prices": [250000, 10393750, ...],    # absolute amounts in rupees (totals, charges)
     "rates":  [11000.0],                  # rupees per sq ft
     "areas":  [875, 880, 1099],           # square feet
     "price_min": ..., "price_max": ..., "rate_min": ..., "area_min": ..., ...}

Retrieval then asks the matching question of the QUERY (:func:`analyze_query`) and uses the
two together to lift the chunks that actually hold what the caller asked for (see
``retrieval.py`` — ``_apply_metadata_boosts``).

HOW NUMBERS ARE CLASSIFIED — the hard part. Real cost sheets rarely write "₹". They write
"Flat cost 96,25,000. Total 1,03,93,750." and "Clubhouse charges 3,25,000". The same
documents are full of numbers that are NOT prices: "Carpet 592", "Balcony and utility 99",
"sr3", "4th floor", "1 June 2026", "Tower D". So a number is only a price when there is
positive evidence for it, and the rules run most-specific first, each CLAIMING the span it
matched so a later, looser rule can never re-read the same digits as something else:

  1. BHK        "2.5 BHK", "3bhk", "1, 2 and 3 BHK"             -> bhk, never money
  2. rate       "11,000 rupees per square foot", "₹11,499/sq ft" -> rate (per sq ft)
  3. area       "875 sq ft", "619, 625 and 646 square feet"      -> area
  4. area label "Carpet 592", "super built-up 1,325"             -> area
  5. money      "₹ 60 L", "Rs. 3,25,000", "30,000 rupees", "1.54 crore", "5.5 lakh"
  6. money label"Total 1,03,93,750", "car parking is 3,00,000"   -> money, only if >= 1,000
  7. grouping   a bare Indian-grouped figure >= 1,00,000 ("3,25,000") -> money. Areas and
                unit counts never reach a lakh; phone numbers and years are never grouped.

Anything no rule claims is ignored. A missed price costs one boost; a false price would
mislead ranking — so the rules err towards missing.

DOMAIN NOTE. Unlike ``chunkers.py`` this module IS real-estate aware (BHK, sq ft, lakh,
crore) — that is the point of the task. It is still tenant-neutral: no project names, no
tower names, nothing tied to one customer's documents.

VERSIONING. ``FACTS_VERSION`` is stored on every record. Bump it whenever extraction output
changes; ``scripts/backfill_chunk_metadata.py`` re-tags exactly the chunks on an older
version, without re-embedding them.

Pure and stdlib-only (``re``) — unit-testable with no DB and no network:
``python -m app.engine.rag.enrichment`` runs a self-test; ``tests/test_chunk_enrichment.py``
has the full suite.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Bump when extraction rules change so stored facts can be found and re-tagged (backfill).
#   1 — initial rules.
#   2 — spelled-out configurations ("ek BHK", "two bhk") and bedroom lists ("2 and 3 bedroom").
FACTS_VERSION = 2

# Bounds on what one chunk stores. A cost-sheet chunk can list dozens of figures; the full
# list is only for debugging, the booleans and min/max are what ranking reads. Capping keeps
# a pathological document from bloating the JSONB column that every retrieval row returns.
_MAX_VALUES = 24

# Sanity windows. Outside these a "match" is far more likely a mis-parse than a real value.
_AREA_MIN, _AREA_MAX = 100, 50_000              # sq ft — a studio to a large villa
_RATE_MIN, _RATE_MAX = 0.5, 200_000             # ₹/sq ft — incl. "3.50 per sq ft per month"
_MONEY_MIN, _MONEY_MAX = 1, 100_000_000_000     # ₹ — up to 10,000 crore
_LABEL_MONEY_MIN = 1_000                        # rule 6 needs a real amount, not "charges for a 2"
_BHK_MIN, _BHK_MAX = 1.0, 10.0

# --- number grammar ------------------------------------------------------------------------
# One number: Indian ("1,03,93,750") or Western ("1,000,000") grouping, or plain ("875",
# "3.50"). The grouping alternative comes first so "1,239" is read whole, not as "1".
# The lookbehinds stop a match starting inside a token ("sr3", "A1") or in the middle of
# another number ("2.5" -> "5", "1,325" -> "325"). The lookahead forces the WHOLE number:
# without it the regex engine may backtrack "1,325" to "1,32" to satisfy a later negative
# lookahead, and read the fragment as a separate value.
_NUM = (r"(?<!\w)(?<!\d[.,])(?:\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
        r"(?!\d|,\d|\.\d)")
# A list of numbers sharing one unit: "619, 625, 643 and 646". List commas need a following
# space — that is what tells "1,099, 1,103" (two numbers) from "1,099" (one).
_LIST_SEP = r"(?:\s*,\s+(?:and\s+|or\s+)?|\s+(?:and|or|to)\s+|\s*[/–-]\s*)"
_NUM_LIST = rf"{_NUM}(?:{_LIST_SEP}{_NUM})*"

_SQFT = r"(?:sq\.?\s*ft\.?|sqft|sft|sq\.?\s*feet|square\s+f(?:ee|oo)t)\b"
_RUPEE_WORD = r"(?:\brupees?\b|\brupaye\b|\brupaiye\b|\brs\b\.?|\binr\b|/-)"
# Word boundaries matter: without them "hours 3" reads as "rs 3".
_CURRENCY_PREFIX = r"(?:₹|\brs\b\.?|\binr\b)"
# Scale words. A bare "L" ("60L") is handled separately (_MONEY_L_RE): as a spaced word it is
# too ambiguous ("5 L" of water) to trust without a currency sign.
_SCALE = r"(?:lakhs?|lacs?|crores?|cr\b\.?|karod|करोड़|लाख)"

# 1. BHK — a single configuration or a list ("1, 2 and 3 BHK", "2/3 BHK").
_BHK_RE = re.compile(rf"({_NUM_LIST})\s*-?\s*bhk\b", re.IGNORECASE)
# Spelled-out configurations, English and romanised Hindi ("ek BHK hai kya", "two bedroom").
# Hindi words are accepted ONLY directly before "bhk" — "do" alone is an English verb.
_NUMBER_WORDS = {"one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0,
                 "ek": 1.0, "do": 2.0, "teen": 3.0, "char": 4.0, "chaar": 4.0, "paanch": 5.0}
_WORD_BHK_RE = re.compile(r"\b(one|two|three|four|five|ek|do|teen|chaar|char|paanch)"
                          r"[\s-]*bhk\b", re.IGNORECASE)
_BEDROOM_RE = re.compile(
    rf"(?:\b(one|two|three|four|five)|({_NUM_LIST}))[\s-]*bed(?:room)?s?\b", re.IGNORECASE)

# 2. Rate — money per square foot. Currency is optional ("11,000 per square foot" in a title)
#    because "<number> per sq ft" is a rate in any real-estate document.
_RATE_RE = re.compile(
    rf"(?:{_CURRENCY_PREFIX}\s*)?({_NUM})\s*(?:{_RUPEE_WORD}\s*)?(?:/|per)\s*{_SQFT}",
    re.IGNORECASE)

# 3. Area with an explicit unit, possibly a list: "619, 625, 643 and 646 sq ft".
_AREA_RE = re.compile(rf"({_NUM_LIST})\s*-?\s*{_SQFT}", re.IGNORECASE)

# 4. Area by label, no unit: "Carpet 592", "super built-up area 1,325".
_AREA_LABEL_RE = re.compile(
    rf"\b(?:carpet|super\s+built[\s-]*up|built[\s-]*up|saleable|rera\s+carpet)"
    rf"(?:\s+area)?(?:\s+(?:of|is))?\s*[:\-–]?\s*({_NUM})\b(?!\s*(?:{_RUPEE_WORD}|₹))",
    re.IGNORECASE)

# 5. Money with an explicit currency or scale word.
_MONEY_PREFIX_RE = re.compile(
    rf"{_CURRENCY_PREFIX}\s*({_NUM})\s*((?:{_SCALE})(?![a-z])|l\b)?", re.IGNORECASE)
_MONEY_SCALE_RE = re.compile(rf"({_NUM})\s*({_SCALE})(?![a-z])", re.IGNORECASE)
# "60L", "1.2L" — a capital L glued to the number is the common written shorthand for lakh.
# Case-sensitive on purpose: a lowercase "l" is far more often litres or a typo.
_MONEY_L_RE = re.compile(rf"({_NUM})(L)\b")
_MONEY_SUFFIX_RE = re.compile(rf"({_NUM})\s*{_RUPEE_WORD}", re.IGNORECASE)

# 6. Money by label: a cost-sheet word, then (within a few non-digit characters) a number.
_MONEY_LABELS = (r"cost|total|price|priced|charges?|fees?|parking|infrastructure|amount|"
                 r"deposit|maintenance|corpus|starts?\s+at|all\s+in|budget|emi")
_MONEY_LABEL_RE = re.compile(
    rf"\b(?:{_MONEY_LABELS})\b[^\d\n]{{0,25}}?({_NUM})(?!\s*(?:bhk|{_SQFT}|%|st\b|nd\b|rd\b|th\b))",
    re.IGNORECASE)

# 7. A bare Indian-grouped figure of at least one lakh: "3,25,000", "1,54,32,011".
_INDIAN_LAKH_RE = re.compile(r"(?<![\w.,])(\d{1,2}(?:,\d{2})+,\d{3})(?!\d|,\d)")

_SCALE_VALUE = {"l": 1e5, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "लाख": 1e5,
                "cr": 1e7, "cr.": 1e7, "crore": 1e7, "crores": 1e7, "karod": 1e7,
                "करोड़": 1e7}


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _split_list(raw: str) -> list[float]:
    """Numbers out of a ``_NUM_LIST`` match, in order."""
    out = []
    for m in re.finditer(_NUM, raw):
        v = _to_float(m.group(0))
        if v is not None:
            out.append(v)
    return out


def _nice(v: float) -> float | int:
    """Whole numbers stored as int (readable JSON); fractional ones (2.5 BHK, ₹3.50) kept."""
    return int(v) if float(v).is_integer() else round(v, 2)


@dataclass
class ChunkFacts:
    """What one chunk states. ``to_meta()`` is the JSON stored under ``metadata.facts``."""
    bhk: set[float] = field(default_factory=set)
    prices: set[float] = field(default_factory=set)     # absolute rupee amounts
    rates: set[float] = field(default_factory=set)      # rupees per sq ft
    areas: set[float] = field(default_factory=set)      # sq ft

    @property
    def has_price(self) -> bool:
        # A rate IS price information — "what does it cost" is answered by "11,000 per sq ft".
        return bool(self.prices or self.rates)

    def to_meta(self) -> dict:
        def capped(vals: set[float]) -> list:
            return [_nice(v) for v in sorted(vals)[:_MAX_VALUES]]

        meta: dict = {"v": FACTS_VERSION,
                      "has_price": self.has_price,
                      "has_rate": bool(self.rates),
                      "has_area": bool(self.areas),
                      "bhk": capped(self.bhk),
                      "prices": capped(self.prices),
                      "rates": capped(self.rates),
                      "areas": capped(self.areas)}
        # min/max are computed over ALL values, not the capped list, so a range query in
        # future (budget fit) is never misled by truncation.
        for name, vals in (("price", self.prices), ("rate", self.rates), ("area", self.areas)):
            if vals:
                meta[f"{name}_min"] = _nice(min(vals))
                meta[f"{name}_max"] = _nice(max(vals))
        return meta


class _Spans:
    """Character ranges already claimed by a more specific rule."""

    def __init__(self) -> None:
        self._taken: list[tuple[int, int]] = []

    def free(self, start: int, end: int) -> bool:
        return all(end <= a or start >= b for a, b in self._taken)

    def claim(self, start: int, end: int) -> None:
        self._taken.append((start, end))


def extract_facts(text: str) -> ChunkFacts:
    """Extract BHK / price / rate / area facts from one chunk's text. Never raises on
    ordinary input; returns an empty :class:`ChunkFacts` when nothing is found."""
    facts = ChunkFacts()
    if not text:
        return facts
    spans = _Spans()

    # 1. BHK — claimed first so "2.5" in "2.5 BHK" can never be read as money.
    for m in _BHK_RE.finditer(text):
        spans.claim(*m.span())
        for v in _split_list(m.group(1)):
            if _BHK_MIN <= v <= _BHK_MAX and (v * 2).is_integer():   # 1, 1.5, 2, 2.5 ...
                facts.bhk.add(v)
    for m in _WORD_BHK_RE.finditer(text):
        if spans.free(*m.span()):
            facts.bhk.add(_NUMBER_WORDS[m.group(1).lower()])
            spans.claim(*m.span())
    for m in _BEDROOM_RE.finditer(text):
        if not spans.free(*m.span()):
            continue
        vals = ([_NUMBER_WORDS[m.group(1).lower()]] if m.group(1)
                else _split_list(m.group(2)))
        vals = [v for v in vals if _BHK_MIN <= v <= _BHK_MAX and (v * 2).is_integer()]
        if vals:
            facts.bhk.update(vals)
            spans.claim(*m.span())

    # 2. Rate per sq ft — before area, so "per square foot" is never read as an area.
    for m in _RATE_RE.finditer(text):
        if not spans.free(*m.span()):
            continue
        v = _to_float(m.group(1))
        if v is not None and _RATE_MIN <= v <= _RATE_MAX:
            facts.rates.add(v)
            spans.claim(*m.span())

    # 3./4. Area — with a unit (possibly a list), then by carpet/built-up label.
    for m in _AREA_RE.finditer(text):
        if not spans.free(*m.span()):
            continue
        vals = [v for v in _split_list(m.group(1)) if _AREA_MIN <= v <= _AREA_MAX]
        if vals:
            facts.areas.update(vals)
            spans.claim(*m.span())
    for m in _AREA_LABEL_RE.finditer(text):
        if not spans.free(*m.span(1)):
            continue
        v = _to_float(m.group(1))
        if v is not None and _AREA_MIN <= v <= _AREA_MAX:
            facts.areas.add(v)
            spans.claim(*m.span(1))

    # 5. Money with explicit currency / scale.
    def add_money(start: int, end: int, raw: str, scale: str | None) -> None:
        v = _to_float(raw)
        if v is None:
            return
        if scale:
            v *= _SCALE_VALUE.get(scale.lower().rstrip("."), 1)
        if _MONEY_MIN <= v <= _MONEY_MAX:
            facts.prices.add(round(v, 2))
            spans.claim(start, end)

    for m in _MONEY_PREFIX_RE.finditer(text):
        if spans.free(*m.span()):
            add_money(*m.span(), m.group(1), m.group(2))
    for m in _MONEY_SCALE_RE.finditer(text):
        if spans.free(*m.span()):
            add_money(*m.span(), m.group(1), m.group(2))
    for m in _MONEY_L_RE.finditer(text):
        if spans.free(*m.span()):
            add_money(*m.span(), m.group(1), m.group(2))
    for m in _MONEY_SUFFIX_RE.finditer(text):
        if spans.free(*m.span()):
            add_money(*m.span(), m.group(1), None)

    # 6. Money by cost-sheet label. Only the NUMBER's span must be free; the label may be
    #    shared ("Clubhouse charges are 2,50,000 for a 2 BHK and 3,25,000 ..." — the second
    #    figure is picked up by rule 7).
    for m in _MONEY_LABEL_RE.finditer(text):
        if not spans.free(*m.span(1)):
            continue
        v = _to_float(m.group(1))
        if v is not None and v >= _LABEL_MONEY_MIN and v <= _MONEY_MAX:
            facts.prices.add(v)
            spans.claim(*m.span(1))

    # 7. Bare Indian-grouped figure of a lakh or more.
    for m in _INDIAN_LAKH_RE.finditer(text):
        if not spans.free(*m.span(1)):
            continue
        v = _to_float(m.group(1))
        if v is not None and 100_000 <= v <= _MONEY_MAX:
            facts.prices.add(v)
            spans.claim(*m.span(1))

    return facts


def enrich_chunk_meta(text: str) -> dict:
    """The ``metadata.facts`` record for a chunk — the single entry point ingestion and the
    backfill script both use, so a chunk tagged either way gets identical facts."""
    return extract_facts(text).to_meta()


# ---------------------------------------------------------------- query side

# What makes a QUERY a price question. Broader than retrieval.PRICING_MARKERS (which gates
# the document-level PRICING_BOOST and is deliberately left unchanged): callers ask in
# Hinglish — "clubhouse ka charge kitna hai", "corner aur premium mein charge ka farak" — and
# none of those carry an English pricing word. Whole-word matches only, so "charging point"
# or "prices" inside a URL-like token do not trip it.
#
# NOT "total": measured on the eval set, "How many flats in total?" was read as a price
# question, the price lift filled the window with cost sheets and pushed out the chunk holding
# the unit count (EN027 regressed). A real "total" price question always carries another cue
# ("total cost", "total kitna padega").
_PRICE_Q = re.compile(
    r"\b(?:price[sd]?|pricing|cost(?:s|ing)?|rates?|psf|budget|emi|lakhs?|lacs?|crores?|cr|"
    r"rupees?|rs|inr|charge[sd]?|fees?|amount|expensive|cheap(?:er|est)?|afford(?:able)?|"
    r"ballpark|floor\s+rise|kharch[ae]?|"
    r"keemat|kimat|qeemat|daam|dam|paisa|paise|rupaye|rupaiye|mehenga|mehnga|sasta|"
    # NOT bare "registration": "RERA registration number" is a compliance question (A/B Q38/
    # Q39 lost their location chunks to price sheets). A registration FEE question always
    # carries cost/fee/charge/kharcha/stamp duty, which match on their own.
    r"maintenance|corpus|deposit|gst|stamp\s+duty)\b|₹|/sq|per\s+sq",
    re.IGNORECASE)
# "kitna / kitne / kitni" (how much / how many) is a price cue ONLY when nothing in the query
# points to size, distance, count or time instead: "kitna bada hai" is an area question,
# "kitne floors" is a count, "kitni door" is a distance. Unit words ("square feet") are NOT in
# this guard: "1390 square feet ka kitna banta hai" asks what that size COSTS — the size is the
# subject, not the ask (eval HI008). A size ask is carried by bada/area/size/carpet, which stay.
_KITNA = re.compile(r"\bkit(?:na|ne|ni)\b", re.IGNORECASE)
_NOT_PRICE_KITNA = re.compile(
    r"\b(?:bada|badi|bade|area|size|carpet|floors?|manzil|door|dur|km|"
    r"minute|minutes|min|time|din|saal|mahine|log|units?|flats?\s+hai|towers?|lifts?|"
    r"parking\s+slots?)\b", re.IGNORECASE)
_RATE_Q = re.compile(r"\b(?:rates?|psf|per\s+sq(?:uare)?|/sq|sq\s*ft\s+rate|sft\s+rate)\b|/sq",
                     re.IGNORECASE)
# A COUNT question ("how many flats", "kitne units hain") is not a price question even though
# "kitne" would otherwise read as "how much". Only an explicit money word overrides it.
_COUNT_Q = re.compile(
    r"\bhow\s+many\b|\bkit(?:ne|ni)\s+(?:flats?|units?|towers?|floors?|wings?|apartments?|"
    r"blocks?|lifts?|clubhouses?)\b|\bnumber\s+of\b", re.IGNORECASE)
_MONEY_WORD_Q = re.compile(
    r"\b(?:price[sd]?|pricing|cost(?:s|ing)?|rates?|budget|lakhs?|crores?|rupees?|rs|inr|"
    r"charges?|fees?|keemat|kimat|daam|emi)\b|₹", re.IGNORECASE)
# "per square foot" / "/sq ft" is the unit of a RATE, not a question about size — stripped
# before the area check so "per square feet kitna padta hai" is a price question only.
# "area" that names a THING, not a size: "common area maintenance" is a charge, "play area" /
# "parking area" are amenities. Stripped before the area check (A/B: the CAM question read as a
# size question, area chunks were lifted, precision fell 1.00 -> 0.50).
_NOT_SIZE_AREA_Q = re.compile(
    r"\b(?:common|play|parking|kids|children'?s|sitting|seating|green|open|service|utility|"
    r"lobby|landscaped?|recreation(?:al)?)\s+areas?\b", re.IGNORECASE)
_RATE_UNIT_Q = re.compile(
    r"(?:/|\bper)\s*(?:sq\.?\s*ft\.?|sqft|sft|sq\.?\s*feet|square\s+f(?:ee|oo)t|sq(?:uare)?)\b",
    re.IGNORECASE)
_AREA_Q = re.compile(
    r"\b(?:area|size|sizes|carpet|super\s+built|built[\s-]*up|sq\.?\s*ft|sqft|sft|square\s+"
    r"f(?:ee|oo)t|how\s+big|how\s+large|kitna\s+bada|kitni\s+badi|kitne\s+bade|dimensions?)\b",
    re.IGNORECASE)


@dataclass
class QueryIntent:
    """What the caller's question asks for, in the same vocabulary as :class:`ChunkFacts`."""
    wants_price: bool = False
    wants_rate: bool = False
    wants_area: bool = False
    bhk: set[float] = field(default_factory=set)
    # Figures the caller named ("under 60 lakh") — recorded for the trace and future budget-
    # fit ranking; not used for scoring today.
    budget: list[float] = field(default_factory=list)

    @property
    def any(self) -> bool:
        return self.wants_price or self.wants_area or bool(self.bhk)

    def describe(self) -> dict:
        return {"price": self.wants_price, "rate": self.wants_rate, "area": self.wants_area,
                "bhk": [_nice(b) for b in sorted(self.bhk)],
                "budget": [_nice(b) for b in self.budget]}


def analyze_query(query: str) -> QueryIntent:
    """Classify a caller's question into the facts it wants. Cheap (a few regexes) — it runs
    on every retrieval, so it must stay well under a millisecond."""
    intent = QueryIntent()
    q = query or ""
    if not q.strip():
        return intent
    facts = extract_facts(q)
    intent.bhk = set(facts.bhk)
    intent.budget = sorted(facts.prices)
    intent.wants_rate = bool(_RATE_Q.search(q)) or bool(facts.rates)
    intent.wants_price = (bool(_PRICE_Q.search(q)) or intent.wants_rate or facts.has_price
                          or (bool(_KITNA.search(q)) and not _NOT_PRICE_KITNA.search(q)))
    if intent.wants_price and _COUNT_Q.search(q) and not _MONEY_WORD_Q.search(q):
        intent.wants_price = intent.wants_rate = False
    size_q = _NOT_SIZE_AREA_Q.sub(" ", _RATE_UNIT_Q.sub(" ", q))
    intent.wants_area = bool(_AREA_Q.search(size_q)) or bool(facts.areas)
    return intent


def facts_of(row_meta) -> dict | None:
    """The ``facts`` record out of a chunk's stored metadata, or None if the chunk was never
    tagged (ingested before R5.08 and not yet backfilled). Tolerates the JSONB arriving as a
    string, which some drivers do without a JSON codec."""
    if isinstance(row_meta, str):
        import json
        try:
            row_meta = json.loads(row_meta)
        except ValueError:
            return None
    if not isinstance(row_meta, dict):
        return None
    facts = row_meta.get("facts")
    return facts if isinstance(facts, dict) else None


# ------------------------------------------------------------------ self-test (no DB, no net)

def _selftest() -> int:
    """Quick sanity pass. The full suite is tests/test_chunk_enrichment.py."""
    failures = 0

    def check(name: str, cond: bool) -> None:
        nonlocal failures
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        if not cond:
            failures += 1

    f = extract_facts("875 sq ft, East and West facing. Carpet 592. Flat cost 96,25,000. "
                      "Total 1,03,93,750.")
    check("cost-sheet row: areas", f.areas == {875, 592})
    check("cost-sheet row: prices", f.prices == {9625000, 10393750})
    f = extract_facts("Rate 11,000 rupees per square foot on super built-up area.")
    check("rate per sq ft", f.rates == {11000} and not f.prices and not f.areas)
    f = extract_facts("2.5 BHK, 1,325 sq ft, South facing.")
    check("bhk not money", f.bhk == {2.5} and not f.prices and f.areas == {1325})
    f = extract_facts("Tower A wing 1 sr3, 4th floor, from 1 June 2026.")
    check("no false facts", not (f.bhk or f.prices or f.rates or f.areas))
    check("hinglish price query", analyze_query("clubhouse ka charge kitna hai").wants_price)
    check("area kitna is not price", not analyze_query("flat kitna bada hai").wants_price)
    check("query bhk", analyze_query("price of 2.5bhk").bhk == {2.5})

    print("ENRICHMENT SELFTEST:", "PASS" if not failures else f"{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
