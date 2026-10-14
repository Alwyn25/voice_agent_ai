"""Guard: the agent may not tell a caller that our own documents disagree.

THE BUG THIS FIXES. Asked about 3 BHK sizes, a real call answered "There's a slight size-range
discrepancy between the brochure and cost sheet", and when the caller asked "what is the
discrepancy?" it explained it: "The brochure lists 3 BHK sizes between 1,239 and 1,422 sq ft,
but the cost sheet shows a 1,560 sq ft unit, which falls outside that range." That is an
internal paperwork problem read out to a buyer. It sounds like the developer does not know its
own numbers, and it puts a figure the developer does not stand behind (1,422) in the caller's
head.

Three layers, because each one alone has already failed:
  1. DATA — the knowledge base states one figure (seed_knowledge_dsr.py resolves the conflict
     in favour of the cost sheet; the "Where Our Documents Disagree" note is gone).
  2. PROMPT — PROMPT_RULE below, in BOTH the planner (prompts.py rule 2b) and the responder
     (responder.py rule 17). The first fix (8587bac) put a rule in the planner only; the
     responder writes every word the caller hears and never saw it.
  3. THIS GUARD — a prompt rule is a request. Re-running the reported call after layers 1-2 of
     the earlier fix still produced "The brochure might be outdated" and "There might be an
     inconsistency in the documents" whenever the caller pushed back with the other figure.

WHAT IT CATCHES. A sentence that names an internal source (brochure, cost sheet, price sheet,
"our documents/records/paperwork") AND, in the same sentence, says it conflicts / is outdated /
wrong / outside a range. A bare mention of a document is fine — "I can send you the brochure on
WhatsApp" is a real offer. When the CALLER raised a document or a discrepancy this turn, a
confession with no document named ("Yes, there are differences.", "there might be an
inconsistency") is caught too — without that context the same words can be innocent ("there's
a conflict with your existing visit"), so they are only judged against the caller's question.

WHAT IT DOES. Removes only the offending part: a clause split off by ";" or a dash, else the
whole sentence. "The brochure might be outdated; the largest 3 BHK is 1,560 square feet."
keeps "The largest 3 BHK is 1,560 square feet." If nothing is left, an honest fixed line in the
caller's language replaces the reply. Pure and self-tested
(``python -m app.engine.planner.source_conflict_claim``); called on each streamed sentence
(responder.py) and on the final text (turn_runner.py), the same as email_claim.
"""
import re

from app.engine.planner.text_normalize import normalize_quotes

# The rule given to BOTH models that write customer-facing words — one string, so the planner's
# draft and the responder's reply can never be told different things again.
PROMPT_RULE = (
    "YOUR KNOWLEDGE MATERIAL IS INTERNAL. If two passages give different figures for the same "
    "thing (a size, a price, a count, a date), state the figure from the most authoritative "
    "source, plainly, as the fact: a VERIFIED RESULT or inventory lookup first, then a cost or "
    "price sheet, then policy, then brochure or marketing text. Never tell the customer that "
    "documents disagree, differ, conflict, don't match, are outdated or have a discrepancy, and "
    "never name an internal document (brochure, cost sheet, price sheet, our records) as the "
    "reason a figure is uncertain. If the customer quotes a different figure, give the correct "
    "one without debating where theirs came from. Do not offer a callback because of such a "
    "difference — only when they ask for something you genuinely do not have.")

# Internal sources. ASCII names use \b; Devanagari is matched without \b (Python's \b breaks
# on Devanagari vowel signs, which are not \w).
_SOURCE_RE = re.compile(
    r"\b(?:brochures?|e-?brochures?|cost[\s-]?sheets?|price[\s-]?sheets?|price[\s-]?lists?|"
    r"rate[\s-]?cards?|paperwork|"
    r"(?:our|the|these|those|two|both|different|internal|official|company)\s+"
    r"(?:documents?|documentation|records|materials?|sources?|sheets?|data|files|literature))\b"
    r"|ब्रोशर|ब्रॉशर|कॉस्ट शीट|कास्ट शीट|प्राइस शीट|दस्तावेज",
    re.I)

# Conflict / unreliability language. "different" is deliberately absent ("different floor
# plans for each tower" is ordinary); "differ"/"difference" are present.
_CONFLICT_RE = re.compile(
    r"\b(?:discrepanc\w*|inconsisten\w*|differs?|differed|differing|differences?|"
    r"mismatch\w*|conflict\w*|contradict\w*|disagree\w*|(?:do|does|did)(?:n't| not) match|"
    r"not match(?:ing)?|outdated|out of date|older (?:figure|version|number|size|edition|"
    r"info\w*|data|copy)|old (?:figure|version|number|edition|copy)|"
    r"outside (?:that|the|this) range|errors?|mistakes?|typos?|wrong|incorrect|inaccura\w*|"
    r"not (?:accurate|correct|up to date|updated)|"
    r"farak|fark|alag alag|mel nahi|galat|purana|purani)\b"
    r"|फ़र्क|फर्क|अंतर|मेल नहीं|गलत|ग़लत|पुराना|पुरानी",
    re.I)

# A confession with no document named — judged only when the caller raised the subject.
_CONFESSION_RE = re.compile(
    r"\b(?:discrepanc\w*|inconsisten\w*|differences?|mismatch\w*|conflict\w*|"
    r"contradict\w*|disagree\w*)\b|फ़र्क|फर्क|अंतर", re.I)

# Split points for keeping the clean part of a sentence.
_CLAUSE_SPLIT_RE = re.compile(r"\s*;\s*|\s+[—–-]\s+")


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.?!।])\s+|\n+", text or "") if s.strip()]


def caller_raised_conflict(asked: str | None) -> bool:
    """True when the caller's own words this turn bring up a document or a discrepancy."""
    low = normalize_quotes(asked or "")
    return bool(_SOURCE_RE.search(low) or _CONFESSION_RE.search(low))


def _leaks(fragment: str, raised: bool) -> bool:
    low = normalize_quotes(fragment)
    if _SOURCE_RE.search(low) and _CONFLICT_RE.search(low):
        return True
    return raised and bool(_CONFESSION_RE.search(low))


def _scrub_sentence(sentence: str, raised: bool) -> str:
    """The sentence with any leaking clause removed; "" when nothing clean is left."""
    if not _leaks(sentence, raised):
        return sentence
    clauses = [c for c in _CLAUSE_SPLIT_RE.split(sentence) if c.strip()]
    if len(clauses) > 1:
        # Keep the clean clauses only if none of them still names a source — a kept "the
        # brochure lists 1,239 to 1,422" would repeat the very figure being withdrawn.
        clean = [c for c in clauses if not _leaks(c, raised)]
        if clean and len(clean) < len(clauses) and not any(
                _SOURCE_RE.search(normalize_quotes(c)) for c in clean):
            kept = " ".join(clean).strip()
            kept = kept[0].upper() + kept[1:]
            if kept[-1] not in ".?!।":
                kept += "." if sentence.rstrip()[-1:] not in "?" else "?"
            return kept
    return ""


def scrub(text: str, *, asked: str | None = None) -> str:
    """`text` with every sentence/clause that exposes an internal-document conflict removed.
    Returns the text unchanged when nothing leaks, and "" when nothing clean is left (the
    caller then substitutes honest_fallback)."""
    if not text:
        return text or ""
    raised = caller_raised_conflict(asked)
    parts = _sentences(text)
    kept = [s for s in (_scrub_sentence(p, raised) for p in parts) if s]
    if len(kept) == len(parts) and all(a == b for a, b in zip(kept, parts)):
        return text
    return " ".join(kept)


def is_source_conflict_leak(text: str, *, asked: str | None = None) -> bool:
    return scrub(text, asked=asked) != (text or "")


# Honest replacement when the whole reply was the leak. States the figures given stand, without
# naming any document, and moves the call forward — mirrors PROMPT_RULE.
FALLBACK = {
    "english": ("The figures I've shared are the current ones for this project. Would you like "
                "the price for a particular size?"),
    "hindi": ("मैंने जो आंकड़े बताए हैं, वही इस प्रोजेक्ट के मौजूदा आंकड़े हैं। क्या आप किसी खास साइज़ "
              "की कीमत जानना चाहेंगे?"),
    "hinglish": ("Maine jo figures bataye hain, wahi is project ke current figures hain. Kya aap "
                 "kisi particular size ka price jaanna chahenge?"),
}


def honest_fallback(*, language: str = "english") -> str:
    """`language` should come from responder.detect_language() on the caller's own words."""
    return FALLBACK.get(language, FALLBACK["english"])


def _selftest() -> None:
    leaks = [
        # the reported call, both turns
        ("We have 3 BHK units ranging from about 1,239 to 1,560 sq ft. There's a slight "
         "size-range discrepancy between the brochure and cost sheet, so our sales team can "
         "confirm the exact details.", "What sizes do you have in 3 BHK?"),
        ("The brochure lists 3 BHK sizes between 1,239 and 1,422 sq ft, but the cost sheet "
         "shows a 1,560 sq ft unit, which falls outside that range.",
         "sorry what is the discrepancy"),
        # the live replay after the first fix (caller pushed back with the brochure figure)
        ("The brochure might be outdated; the largest 3 BHK available is indeed 1,560 square "
         "feet.", "The brochure says 1,422, why are you saying 1,560?"),
        ("There might be an inconsistency in the documents. The biggest 3 BHK unit is indeed "
         "1,560 square feet.", "The brochure says 1,422, why are you saying 1,560?"),
        ("The brochure might have an older size listed. The largest 3 BHK is currently 1,560 "
         "square feet.", "The brochure says 1,422, why 1,560?"),
        ("Yes, there are differences. The cost sheet has the exact sizes.",
         "Is there any difference between your brochure and the cost sheet?"),
        ("Our documents don't match on that, so the sales team will confirm.", None),
        ("Brochure aur cost sheet mein thoda farak hai.", None),
    ]
    for text, asked in leaks:
        assert is_source_conflict_leak(text, asked=asked), text

    assert scrub(leaks[2][0], asked=leaks[2][1]) == \
        "The largest 3 BHK available is indeed 1,560 square feet."
    assert scrub(leaks[3][0], asked=leaks[3][1]) == \
        "The biggest 3 BHK unit is indeed 1,560 square feet."
    assert scrub(leaks[1][0], asked=leaks[1][1]) == ""            # nothing clean -> fallback
    assert scrub(leaks[0][0], asked=leaks[0][1]) == \
        "We have 3 BHK units ranging from about 1,239 to 1,560 sq ft."

    safe = [
        ("I can send you the brochure on WhatsApp right now.", "send me the brochure"),
        ("The 3 BHK sizes range from 1,239 to 1,560 square feet.", "what is the discrepancy"),
        ("We have different floor plans for each tower.", None),
        ("Prices differ by floor, from the 4th floor onwards.", None),
        ("You already have a visit that day, so there's a conflict — shall I move it?", None),
        ("There's a difference of about 18 lakh between the 2 BHK and 3 BHK.", None),
        ("The cost sheet total includes car parking and clubhouse charges.", "what's included?"),
        ("", None),
    ]
    for text, asked in safe:
        assert not is_source_conflict_leak(text, asked=asked), text
        assert scrub(text, asked=asked) == text

    assert caller_raised_conflict("sorry what is the discrepancy")
    assert caller_raised_conflict("the brochure says 1,422")
    assert not caller_raised_conflict("what's the price of the 3 BHK")
    assert honest_fallback(language="klingon") == FALLBACK["english"]
    print("source_conflict_claim selftest: all passed")


if __name__ == "__main__":
    _selftest()
