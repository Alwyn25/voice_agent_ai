"""Rewrite a reply into the words a person would SAY: no abbreviations, no symbols.

THE BUG THIS FIXES. Replies are written as text and then spoken by ElevenLabs. The model writes
the way its source material is written — "1,560 sq ft", "₹11,499/sq ft", "2.2 km", "approx.",
"80cm x 80cm" — because the cost sheets and brochure are filed that way. A TTS engine reads
those inconsistently: "sq ft" as "ess-cue eff-tee", "/" as "slash", "₹" sometimes not at all.
The caller hears a recording reading a spreadsheet, not a person.

The prompt asks the model to write spoken words (responder.py rule 17, prompts.py rule 6) — but,
as with round_spoken_prices (app/services/money.py), a prompt instruction is a request and this
is the guarantee. Shared by the same two call sites, for the same reason: responder.py's
streaming path (each sentence AS IT STREAMS, so early TTS audio is already clean) and
turn_runner.py's final pass (the validated text). Both must use this exact function, or a
streamed "sq ft" and a validated "square feet" differ and bridge.py replays the sentence.

DESIGN RULES — every rule below is:
  * DETERMINISTIC and table-driven: an expansion is a fixed rewrite, never a model call.
  * ANCHORED: units only expand next to a number ("1,560 sq ft", "2 km"), never as a bare word,
    so ordinary English ("the sun", "a min of fuss") is never touched.
  * IDEMPOTENT: running it twice gives the same text (both call sites may see one sentence).
  * SENTENCE-SAFE: an abbreviation's own dot is consumed, but a dot that also ENDS the sentence
    is kept — "1,560 sq. ft." must stay one sentence ending in "square feet.", or the bridge's
    sentence split (and the caller's pause) shifts.
Acronyms people genuinely say letter by letter — BHK, RERA, GST, EMI, AC, CCTV — are NOT
expanded; that is how they are spoken.
"""
import re

# The prompt half of this guarantee — one string for the planner (prompts.py rule 6) and the
# responder (responder.py rule 18), so the model is asked for exactly what this module enforces.
PROMPT_RULE = (
    "WRITE EVERY WORD THE WAY IT IS SAID ALOUD — this text is read by a speech engine. No "
    "abbreviations and no symbols: \"square feet\" (never sq ft, sq. ft., sqft, sft), \"per "
    "square foot\" (never /sq ft or psf), \"rupees\" after the amount (never ₹ or Rs), "
    "\"crore\" and \"lakh\" (never Cr or L), \"kilometres\", \"minutes\", \"hours\", "
    "\"percent\", \"approximately\", \"for example\", \"and\" (never &), and \"800 to 1,200\" "
    "for a range (never a dash). Your source material is written in abbreviations; say it in "
    "words. Acronyms people really do say letter by letter stay as they are: BHK, RERA, GST, "
    "EMI, AC, CCTV.")

_NUM = r"\d[\d,]*(?:\.\d+)?"

# --- currency -----------------------------------------------------------------------------
# "₹1.54 crore", "Rs. 30,000", "INR 5 lakh", "₹11,499" -> "1.54 crore rupees", "30,000 rupees".
# The amount moves first because that is the spoken order; "rupees" is not doubled when the
# model already wrote it after the figure ("₹30,000 rupees").
_SCALE_WORDS = {"cr": "crore", "crs": "crore", "crore": "crore", "crores": "crore",
                "l": "lakh", "lac": "lakh", "lacs": "lakh", "lakh": "lakh", "lakhs": "lakh",
                "k": "thousand", "thousand": "thousand"}
_CURRENCY_RE = re.compile(
    rf"(?:₹|\bRs\.?|\bINR\b)\s*({_NUM})"
    r"(?:\s*((?i:crores?|crs?|lakhs?|lacs?|thousand)|L|[kK])\b\.?)?"
    r"(\s+rupees\b)?")


def _currency(m: re.Match) -> str:
    amount, scale = m.group(1), m.group(2)
    words = amount + (f" {_SCALE_WORDS[scale.lower()]}" if scale else "")
    return f"{words} rupees"


# "5,000/-" — the Indian "and no paise" suffix; meaningless aloud.
_SLASH_DASH_RE = re.compile(r"(\d)\s*/-")

# Indian scale abbreviations after a figure: "1.54 Cr", "50L", "45 lacs", "20k".
# "L" is matched upper-case only — a lone lower-case "l" after a digit is too ambiguous.
_CRORE_RE = re.compile(r"(\d)\s*(?:Cr|cr|CR|Crs|crs)\b\.?")
_LAKH_RE = re.compile(r"(\d)\s*(?:L|lacs?|Lacs?|lakhs|Lakhs)\b\.?")
_THOUSAND_RE = re.compile(r"(\d)\s*[kK]\b")

# --- area ---------------------------------------------------------------------------------
# Every way a source document writes square feet: sq ft, sq. ft., sq.ft, sqft, sft, sq feet,
# square ft. A RATE ("/sq ft", "per sq ft", "psf") becomes "per square foot" — singular, as
# it is said; an AREA becomes "square feet".
_SQFT = r"(?:sq\.?\s*f(?:ee)?t\.?|sq\.?\s*foot|sqft|sft|square\s+ft\.?)"
_RATE_SQFT_RE = re.compile(rf"(?:\s*/\s*|\s+per\s+){_SQFT}(?![a-z])|\bpsf\b", re.I)
_AREA_SQFT_RE = re.compile(rf"(?<![a-z]){_SQFT}(?![a-z])", re.I)   # "1560sqft" too
_SQM_RE = re.compile(r"(\d)\s*(?:sq\.?\s*m(?:trs?|eters?|etres?)?\.?|sqm)(?![a-z])", re.I)

# --- length / time, only directly after a figure --------------------------------------------
# (pattern, singular, plural). "1 min" -> "1 minute", "5 mins" -> "5 minutes".
_UNIT_AFTER_NUMBER = [
    (r"kms?|KMs?|Kms?", "kilometre", "kilometres"),
    (r"mtrs?|mts", "metre", "metres"),
    (r"cm", "centimetre", "centimetres"),
    (r"mm", "millimetre", "millimetres"),
    (r"ft", "foot", "feet"),
    (r"mins?|Mins?", "minute", "minutes"),
    (r"hrs?|Hrs?", "hour", "hours"),
    (r"secs?", "second", "seconds"),
    (r"yrs?|Yrs?", "year", "years"),
    (r"mths?", "month", "months"),
]
_UNIT_RES = [(re.compile(rf"({_NUM})\s*(?:{p})\b\.?"), one, many)
             for p, one, many in _UNIT_AFTER_NUMBER]

# "80cm x 80cm", "300x300mm" -> "by".
_DIMENSION_X_RE = re.compile(r"(\d)\s*[xX×]\s*(\d)")
_UNIT_X_RE = re.compile(r"\b(centimetres?|millimetres?|metres?|feet|foot)\s*[xX×]\s*(?=\d)")

# Number ranges: "1,239–1,560" (en/em dash), "1,239 - 1,560" (spaced hyphen), and an unspaced
# hyphen only between comma-grouped figures ("1,239-1,560") — a bare "080-4912" phone
# fragment or "2026-10-13" is never a range read aloud.
_RANGE_DASH_RE = re.compile(rf"({_NUM})\s*[–—]\s*(?=\d)|({_NUM})\s+-\s+(?=\d)")
_RANGE_GROUPED_RE = re.compile(r"(\d{1,3}(?:,\d{2,3})+)-(?=\d{1,3}(?:,\d{2,3})+\b)")

# --- per-period -----------------------------------------------------------------------------
_PER_PERIOD_RE = re.compile(r"\s*/\s*(month|year|day|week|night|annum)\b", re.I)
_PER_ABBR_RE = re.compile(r"\s*/\s*(?:mo|mth|yr)\b\.?", re.I)
_PA_RE = re.compile(r"\bp\.a\.", re.I)

# --- words ----------------------------------------------------------------------------------
_APPROX_RE = re.compile(r"\bapprox\b\.?", re.I)
_EG_RE = re.compile(r"\be\.g\.", re.I)
_IE_RE = re.compile(r"\bi\.e\.", re.I)
_VS_RE = re.compile(r"\bvs\b\.?(?=\s)", re.I)
_PVT_LTD_RE = re.compile(r"\bPvt\.?\s*Ltd\b\.?", re.I)
_LTD_RE = re.compile(r"\bLtd\b\.?")
_GOVT_RE = re.compile(r"\bgovt\b\.?", re.I)
_DEPT_RE = re.compile(r"\bdept\b\.?", re.I)
_APPT_RE = re.compile(r"\bappt\b\.?", re.I)
# "Plot No. 221", "flat no 5", "#5" -> "number". Only after a noun that takes a number, so a
# sentence-final "no." ("I said no.") is never read as "number".
_NUMBER_NO_RE = re.compile(
    r"\b(plot|flat|unit|tower|wing|floor|house|door|room|shop|survey|block|phase|road|"
    r"site|sr|serial|registration|phone|mobile|contact)\s+no\b\.?\s*(?=\d)", re.I)
_HASH_NUMBER_RE = re.compile(r"#\s*(?=\d)")
_PERCENT_RE = re.compile(r"(\d)\s*%")
_TILDE_RE = re.compile(r"~\s*(?=\d)")
_PLUS_RE = re.compile(r"(\d)\+(?=\s|$|[.,;!?])")
_AMP_RE = re.compile(r"\s*&\s*")
_247_RE = re.compile(r"\b24\s*/\s*7\b")

# Month / weekday abbreviations next to a date — "10 Oct", "Oct 10", "Sat, 10 October".
# Capitalised only, and only beside a day number, so "Sun", "Mar", "Sat" as words never match.
_MONTHS = {"Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April",
           "Jun": "June", "Jul": "July", "Aug": "August", "Sep": "September",
           "Sept": "September", "Oct": "October", "Nov": "November", "Dec": "December"}
_DAYS = {"Mon": "Monday", "Tue": "Tuesday", "Tues": "Tuesday", "Wed": "Wednesday",
         "Thu": "Thursday", "Thur": "Thursday", "Thurs": "Thursday", "Fri": "Friday",
         "Sat": "Saturday", "Sun": "Sunday"}
_MON_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DAY_ALT = "|".join(sorted(_DAYS, key=len, reverse=True))
_DAY_MONTH_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MON_ALT})\b\.?")
_MONTH_DAY_RE = re.compile(rf"\b({_MON_ALT})\b\.?\s+(\d{{1,2}})\b")
_WEEKDAY_RE = re.compile(rf"\b({_DAY_ALT})\b\.?(?=,?\s+\d)")


def _keep_sentence_dot(m: re.Match, words: str) -> str:
    """`words` replaces an abbreviation whose match may have swallowed a trailing '.'. Give the
    dot back when it was ALSO the end of the sentence (end of text, or followed by whitespace
    and a capital/quote), so "…1,560 sq. ft. Shall I…" stays two sentences."""
    if not m.group(0).endswith("."):
        return words
    rest = m.string[m.end():]
    if not rest.strip() or re.match(r"\s+[A-Z\"'“‘(]", rest):
        return words + "."
    return words


def _sub_keep_dot(pattern: re.Pattern, words, text: str) -> str:
    return pattern.sub(lambda m: _keep_sentence_dot(
        m, words(m) if callable(words) else words), text)


def to_spoken_form(text: str) -> str:
    """Return `text` with abbreviations and symbols written out as spoken words.

    Pure, idempotent, safe to run over any reply in any language (every rule is anchored on
    ASCII abbreviations or digits, so Devanagari text passes through untouched)."""
    if not text:
        return text or ""
    t = text

    # Order matters: currency and rates before bare areas/units, ranges after units so the
    # figures they join are already settled.
    t = _SLASH_DASH_RE.sub(r"\1", t)
    t = _sub_keep_dot(_CURRENCY_RE, _currency, t)
    t = _sub_keep_dot(_CRORE_RE, lambda m: f"{m.group(1)} crore", t)
    t = _sub_keep_dot(_LAKH_RE, lambda m: f"{m.group(1)} lakh", t)
    t = _THOUSAND_RE.sub(r"\1 thousand", t)

    t = _sub_keep_dot(_RATE_SQFT_RE, " per square foot", t)
    t = _sub_keep_dot(_AREA_SQFT_RE, "square feet", t)
    t = _sub_keep_dot(_SQM_RE, lambda m: f"{m.group(1)} square metres", t)
    t = re.sub(r"(\d)(square|per square)", r"\1 \2", t)        # "1560sqft" -> "1560 square feet"
    t = re.sub(r"\s{2,}per square foot", " per square foot", t)

    t = _DIMENSION_X_RE.sub(r"\1 by \2", t)
    for rx, one, many in _UNIT_RES:
        t = _sub_keep_dot(
            rx, lambda m, one=one, many=many:
            f"{m.group(1)} {one if m.group(1) == '1' else many}", t)
    t = _UNIT_X_RE.sub(r"\1 by ", t)                      # "80 centimetres x 80 ..." -> "by"

    t = _PER_PERIOD_RE.sub(lambda m: f" per {m.group(1).lower()}", t)
    t = _sub_keep_dot(_PER_ABBR_RE, lambda m: " per " + (
        "year" if m.group(0).strip(" /.").lower() == "yr" else "month"), t)
    t = _sub_keep_dot(_PA_RE, "per year", t)

    t = _RANGE_DASH_RE.sub(lambda m: f"{m.group(1) or m.group(2)} to ", t)
    t = _RANGE_GROUPED_RE.sub(r"\1 to ", t)

    t = _sub_keep_dot(_APPROX_RE, "approximately", t)
    t = _EG_RE.sub("for example", t)
    t = _IE_RE.sub("that is", t)
    t = _VS_RE.sub("versus", t)
    t = _sub_keep_dot(_PVT_LTD_RE, "Private Limited", t)
    t = _sub_keep_dot(_LTD_RE, "Limited", t)
    t = _sub_keep_dot(_GOVT_RE, "government", t)
    t = _sub_keep_dot(_DEPT_RE, "department", t)
    t = _sub_keep_dot(_APPT_RE, "appointment", t)
    t = re.sub(r"\betc\.(?=\s+[a-z])", "and so on", t)
    t = re.sub(r"\betc\b\.?", "and so on.", t)
    t = re.sub(r"and so on\.(?=[,;:])", "and so on", t)
    t = _NUMBER_NO_RE.sub(lambda m: f"{m.group(1)} number ", t)
    t = _HASH_NUMBER_RE.sub("number ", t)
    t = _247_RE.sub("24 by 7", t)
    t = _PERCENT_RE.sub(r"\1 percent", t)
    t = _TILDE_RE.sub("about ", t)
    t = _PLUS_RE.sub(r"\1 plus", t)
    t = _AMP_RE.sub(" and ", t)

    t = _DAY_MONTH_RE.sub(lambda m: f"{m.group(1)} {_MONTHS[m.group(2)]}", t)
    t = _MONTH_DAY_RE.sub(lambda m: f"{_MONTHS[m.group(1)]} {m.group(2)}", t)
    t = _WEEKDAY_RE.sub(lambda m: _DAYS[m.group(1)], t)

    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r" +([,.;:!?])", r"\1", t)
    return t.strip() if text == text.strip() else t


# --- self-check -------------------------------------------------------------------------
def _selftest() -> None:
    cases = {
        # area — the reported complaint
        "The 3 BHK is 1,560 sq ft.": "The 3 BHK is 1,560 square feet.",
        "The 3 BHK is 1,560 sq. ft. Shall I book a visit?":
            "The 3 BHK is 1,560 square feet. Shall I book a visit?",
        "units of 1,239 sq.ft and 1,390 sqft": "units of 1,239 square feet and 1,390 square feet",
        "1,239 sft, west facing": "1,239 square feet, west facing",
        "about 1,560 square ft in size": "about 1,560 square feet in size",
        "it is 1560sqft": "it is 1560 square feet",
        "1,560 sq feet": "1,560 square feet",
        # rates
        "The rate is ₹11,499/sq ft.": "The rate is 11,499 rupees per square foot.",
        "11,000 per sq ft on super built-up": "11,000 per square foot on super built-up",
        "at 11,499 psf": "at 11,499 per square foot",
        "3.50 rupees per sq. ft. per month": "3.50 rupees per square foot per month",
        # currency
        "starting at ₹1.54 crore": "starting at 1.54 crore rupees",
        "starting at ₹1.54 Cr.": "starting at 1.54 crore rupees.",
        "legal fees of Rs. 30,000/-": "legal fees of 30,000 rupees",
        "Rs 5 lakh": "5 lakh rupees",
        "₹30,000 rupees": "30,000 rupees",
        "about 96 L all in": "about 96 lakh all in",
        "around 1.2 cr": "around 1.2 crore",
        # ranges
        "sizes of 1,239–1,560 square feet": "sizes of 1,239 to 1,560 square feet",
        "1,239-1,560 sq ft": "1,239 to 1,560 square feet",
        "between 10 - 12 lakh": "between 10 to 12 lakh",
        # units after a number
        "RGA Tech Park is 2.2 km away.": "RGA Tech Park is 2.2 kilometres away.",
        "about 15 mins from Wipro": "about 15 minutes from Wipro",
        "the visit takes 1 hr.": "the visit takes 1 hour.",
        "80cm x 80cm vitrified tiles": "80 centimetres by 80 centimetres vitrified tiles",
        "maintenance for 2 yrs": "maintenance for 2 years",
        # words and symbols
        "approx. 1,560 square feet": "approximately 1,560 square feet",
        "amenities e.g. a gym": "amenities for example a gym",
        "pool, gym, etc. are included": "pool, gym, and so on are included",
        "a pool, a gym etc.": "a pool, a gym and so on.",
        "DSR Infra Projects Pvt. Ltd. builds it": "DSR Infra Projects Private Limited builds it",
        "Plot No. 221, Road 17": "Plot number 221, Road 17",
        "100% power backup": "100 percent power backup",
        "40+ amenities": "40 plus amenities",
        "~1,560 square feet": "about 1,560 square feet",
        "CCTV & intercom": "CCTV and intercom",
        "security 24/7": "security 24 by 7",
        "3.50 rupees/month": "3.50 rupees per month",
        # dates
        "Saturday, 10 Oct at 5 PM": "Saturday, 10 October at 5 PM",
        "on Sat, 10 October": "on Saturday, 10 October",
        "from Oct 10": "from October 10",
    }
    for raw, want in cases.items():
        got = to_spoken_form(raw)
        assert got == want, f"\n  in:   {raw!r}\n  want: {want!r}\n  got:  {got!r}"
        assert to_spoken_form(got) == got, f"not idempotent: {raw!r} -> {got!r}"

    untouched = [
        "The 3 BHK is 1,560 square feet, and the rate is 11,499 rupees per square foot.",
        "Your visit is booked for 5 PM. The site office is off Sarjapur Road.",
        "It is a 3 BHK with RERA registration and GST extra.",
        "I said no. Five works better.",
        "Call 080 4912 3000 for the office.",
        "The sun sets late; give it a min of fuss.",
        "मैं वह ईमेल पर नहीं भेज सकता।",
        "Would you like the brochure on WhatsApp?",
        "",
    ]
    for t in untouched:
        assert to_spoken_form(t) == t, f"changed but should not: {t!r} -> {to_spoken_form(t)!r}"
    print("spoken_form selftest: all passed")


if __name__ == "__main__":
    _selftest()
