"""Consent: what the caller meant in reply to a staged action ("Shall I book …?").

WHY THIS EXISTS. A confirmation-required write (book, reschedule, cancel, ticket) is parked
and read back, and the caller's NEXT words decide whether it runs. That decision used to be
``is_affirmative``: a yes/no word list, and when neither list matched, the MODEL'S intent label
("affirm"). A real call went:

    Agent:  Shall I book your visit for today, Tuesday 13 October at 10:00 AM?
    Caller: i'm sorry what time / ja por time again        (STT garble of "just repeat the time")
    Agent:  Your 3 BHK visit is set for 10 AM on 13 October.   ← booked, never agreed to

Neither list matched, the planner guessed ``intent: affirm`` for the garbled sentence, and the
label alone released the booking. A label is exactly the signal that is least reliable when
the audio is worst, so here it no longer approves anything.

WHAT IT DOES. One decision with seven outcomes, each with one fixed engine response:

    ACCEPT        "yes", "sure let's do it", "go for it", "haan kar do"   → run the action
    DECLINE       "no", "I changed my mind", "rehne do", "not now"        → drop it, never run
    CHANGE        "make it 11 instead", "can we do tomorrow?"             → re-plan the details
    REPEAT        "what time?", "say that again", "phir se bolo"          → read it back again
    SIDE_QUESTION "how much is parking?"                                  → answer, then re-ask
    HOLD          "wait", "let me think", "not sure"                      → keep it, don't push
    UNCLEAR       garble, noise, both polarities with nothing to settle it → ask again

Two layers decide together:

  * Layer A — ``read_lexical``: deterministic phrase/sentence matching in English, romanised
    Hindi and Devanagari, with negation scope ("no problem, go ahead" is a yes; "don't book
    it" is a no; "mat kijiye" is a no even though "kijiye" alone is a yes), context-dependent
    verbs ("cancel it" accepts a staged cancellation and declines a staged booking), and a
    check that a time or day the caller names actually matches the staged slot.
  * Layer B — the fast understanding call (understanding.py) reads the reply against the exact
    question asked and returns ``{decision, evidence}``. It costs no extra model call.

``decide`` fuses them with an asymmetric safety rule: a wrong booking is far worse than one
more "shall I book it?". Any non-accept reading wins; the model alone may ACCEPT only when the
words it quotes as evidence are really in the transcript and the transcript is not low-
confidence; when the layers disagree the answer is UNCLEAR, which re-asks.

Pure (no I/O, no engine imports) and self-tested: ``python -m app.engine.planner.consent``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from app.engine.planner.text_normalize import normalize_quotes
from app.engine.planner.unclear_input import looks_unrecognizable

# ------------------------------------------------------------------ decisions
ACCEPT = "accept"
DECLINE = "decline"
CHANGE = "change"
REPEAT = "repeat"
SIDE_QUESTION = "side_question"
HOLD = "hold"
UNCLEAR = "unclear"

DECISIONS = (ACCEPT, DECLINE, CHANGE, REPEAT, SIDE_QUESTION, HOLD, UNCLEAR)

# Outcomes after which the staged action stays parked (nothing executes, nothing is dropped).
KEEPS_PENDING = (REPEAT, SIDE_QUESTION, HOLD, UNCLEAR)

# Devanagari has combining vowel signs that Python's \b treats as word boundaries, so a plain
# \b would split "नहीं" in the middle. These lookarounds bound a match on the script itself.
_DV = r"ऀ-ॿ"
_DB = rf"(?<![{_DV}])"      # Devanagari word start
_DE = rf"(?![{_DV}])"       # Devanagari word end


def _dv(*words: str) -> str:
    """Alternation of Devanagari phrases, each bounded on the script (not on \\b)."""
    return "|".join(f"{_DB}{w}{_DE}" for w in words)


def _rx(*alts: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{a})" for a in alts), re.I)


# ------------------------------------------------------------------ Layer A vocabularies
#
# Matched in THIS order, and each match is blanked out before the next group runs, so an
# idiom is consumed before its parts can be misread: "no problem" never reaches the "no"
# rule, "mat kijiye" never reaches the "kijiye" rule, "samajh nahi aaya" never reaches "nahi".

# Idioms that contain a negation word but mean YES.
_ACCEPT_IDIOMS = _rx(
    r"\bno (?:problem|problems|worries|issues?|objection)\b", r"\bnot a problem\b",
    r"\bwhy not\b", r"\bi (?:don'?t|do not) mind\b", r"\bno doubt\b",
    r"\bkoi (?:dikkat|problem|issue) nahi\b", r"\bkyun nahi\b|\bkyon nahi\b",
    _dv("कोई दिक्कत नहीं", "क्यों नहीं"),
)

# Requests to hear the staged details again, or plain "pardon?". Checked before any yes/no
# word so "okay, what time?" and "no, what was the time again?" are both a REPEAT.
_REPEAT = _rx(
    r"\b(?:say|tell|read|repeat|give)\b(?:\s+\S+){0,4}?\s+(?:again|once more|one more time)\b",
    r"\brepeat\b", r"\bpardon\b", r"\bcome again\b", r"\bone more time\b",
    r"\bonce (?:more|again)\b",
    r"\b(?:time|timing|day|date|slot|that|it|address|place)\s+again\b", r"^\s*again\s*$",
    r"\bwhat\b(?:\s+\S+){0,3}?\s+(?:time|timing|day|date|slot)\b",
    r"\bwhich\s+(?:\S+\s+){0,2}?(?:time|day|date|slot|one)\b",
    r"\bwhen\s+(?:is|was|did|will|would|'s)\b", r"\bwhen\s+exactly\b", r"^\s*when\s*$",
    r"\bwhat\s+(?:did|do)\s+you\s+(?:say|mean)\b", r"\bwhat\s+was\s+that\b",
    r"\byou\s+said\s+what\b",
    r"\b(?:didn'?t|did not|couldn'?t|could not|can'?t|cannot)\s+(?:quite\s+)?"
    r"(?:catch|hear|get|understand|follow)\b",
    r"\bmissed\s+(?:that|it|the|what)\b", r"\bnot\s+clear\b", r"\bunclear\b",
    r"^\s*(?:sorry|what|huh|eh|excuse me|pardon me|sorry what|i'?m sorry|im sorry)\s*$",
    r"\bsorry\s+what\b",
    # romanised Hindi
    r"\b(?:phir|fir)\s*se\b", r"\b(?:phirse|firse|dobara|dubara|dobaara)\b",
    r"\bkitne\s+baje\b", r"\bkab\b", r"\bkya\s+(?:bola|kaha|bataya|boli|bole)\b",
    r"\bsamajh?(?:a|i)?\s+nahi?n?\b", r"\bsuna\s+nahi?n?\b", r"\bsunai\s+nahi?n?\b",
    r"\b(?:kaun|kon|kaunsa|konsa)\s*(?:sa\s+)?(?:time|din|slot|samay)\b",
    r"\bkya\s+(?:time|samay)\b", r"\b(?:time|samay)\s+kya\b",
    r"\b(?:time|samay)\s+(?:bata|batao|bataiye|boliye|bolo)\b",
    r"\bek\s+(?:baar|bar)\s+(?:aur|phir|fir)\b",
    # Devanagari
    _dv("फिर से", "दोबारा", "कितने बजे", "कब", "क्या बोला", "क्या कहा", "क्या बताया",
        "समझ नहीं", "समझा नहीं", "समझी नहीं", "सुना नहीं", "सुनाई नहीं", "कौन सा टाइम",
        "कौन सा समय", "क्या टाइम", "क्या समय", "टाइम क्या", "समय क्या", "एक बार और",
        "टाइम बताइए", "टाइम बताओ"),
)

# A different time/day/arrangement — the caller wants the details changed, not run as staged.
_CHANGE = _rx(
    r"\binstead\b", r"\brather\b", r"\bhow about\b", r"\bwhat about\b",
    r"\b(?:can|could|shall)\s+(?:we|i|you)\s+(?:do|make it|have|get|go for|change|move|"
    r"shift|switch)\b",
    r"\b(?:make|change|switch|push|prepone|postpone)\s+(?:it|that|this|the\s+(?:time|slot|"
    r"visit|booking|day|date))\b",
    r"\b(?:another|different|other|some other|a later|an earlier|later|earlier|next)\s+"
    r"(?:time|timing|day|date|slot|week|one)s?\b",
    r"\bsomething\s+(?:else|later|earlier|in the)\b", r"\bprefer\b",
    r"\bwould be better\b", r"\bpostpone\b", r"\bprepone\b",
    r"\bki\s+jagah\b", r"\bke\s+(?:bajaye|badle)\b", r"\bbadal\s*(?:do|dijiye|dijie|de)\b",
    r"\bbadlo\b", r"\b(?:dusra|doosra|dusre|doosre)\b", r"\b(?:koi\s+aur|aur\s+koi)\b",
    _dv("की जगह", "के बजाय", "के बदले", "बदल दो", "बदल दीजिए", "दूसरा", "दूसरे", "कोई और"),
)

# Context verbs. Their meaning depends on WHICH action is staged, so they are resolved by
# _context_verbs() rather than sitting in a fixed list: "cancel it" agrees to a staged
# cancellation and refuses a staged booking; "move it" agrees to a staged reschedule and asks
# to change a staged booking.
_CANCEL_VERB = _rx(r"\bcancel(?:\s+(?:it|that|this|the\s+(?:visit|booking|appointment)|"
                   r"kar|karo|kar\s+do|kardo|kar\s+dijiye|kar\s+dijie))?\b",
                   _dv("कैंसिल", "रद्द"))
_MOVE_VERB = _rx(r"\b(?:move|shift|reschedule)\s+(?:it|that|this|the\s+(?:visit|booking|"
                 r"appointment)|kar|karo|kar\s+do|kardo)\b",
                 _dv("शिफ्ट कर"))

# A plain refusal or change of mind. Negated agreement comes first in the group so "don't
# book it", "not okay", "mat kijiye" are refusals and never leave a "book it"/"okay"/"kijiye"
# behind for the accept group to find.
_DECLINE_STRONG = _rx(
    r"\b(?:don'?t|do not|dont)\s+(?:book|do|want|need|bother|proceed|confirm|schedule|"
    r"go ahead|lock|fix|reserve|cancel|move|shift)\b",
    r"\bnot\s+(?:ok|okay|fine|good|possible|convenient|suitable|happening|interested|now|"
    r"today|needed|required|necessary|yet|right now|at the moment|this time)\b",
    r"\bchanged?\s+(?:my|of)\s+mind\b", r"\bchange of mind\b", r"\bon second thoughts?\b",
    r"\b(?:leave|forget|drop|skip|scrap|ditch)\s+(?:it|that|this|the\s+(?:booking|visit|"
    r"plan|appointment))\b",
    r"\bnever\s*mind\b", r"\bno need\b", r"\bno,?\s+thanks?\b", r"\bno thank you\b",
    r"\bnope\b", r"\bnah\b", r"\bnot really\b", r"\bi\s+(?:don'?t|do not)\s+think so\b",
    r"\brather not\b", r"\bi'?ll pass\b", r"\bmaybe later\b", r"\bsome other time\b",
    r"\b(?:can'?t|cannot|won'?t|will not)\s+(?:make it|come|go|do it|attend|be there|visit)\b",
    r"\bnot coming\b", r"\bno way\b", r"\b(?:absolutely|definitely|of course)\s+not\b",
    r"\b(?:theek|thik|sahi)\s+nahi?n?\b",
    r"\bmat\s+(?:karo|kijiye|kijie|karna|kar|karein|kare)\b", r"\bmat\b",
    r"\bnahi?n?\s+(?:karna|chahiye|chahie|karo|kijiye)\b", r"\bnai\s+chahiye\b",
    r"\b(?:rehne|rahne|rehene)\s+(?:do|dijiye|dijie|de)\b", r"\bbaad\s+(?:mein|me|main)\b",
    r"\bbad\s+me\b", r"\babhi\s+nahi?n?\b", r"\b(?:chhodo|chodo|chhod\s+do|chod\s+do)\b",
    r"\b(?:zarurat|zaroorat|jarurat|jaroorat)\s+nahi?n?\b", r"\b(?:mann|man)\s+badal\b",
    _dv("ठीक नहीं", "सही नहीं", "मत कीजिए", "मत कीजिये", "मत करो", "मत करिए", "मत",
        "नहीं करना", "नहीं चाहिए", "रहने दो", "रहने दीजिए",
        "बाद में", "अभी नहीं", "छोड़ो", "छोड़ दो", "छोड़िए", "ज़रूरत नहीं", "जरूरत नहीं",
        "मन बदल"),
)

# The caller asked for time, not refused: keep the action parked, don't push.
_HOLD = _rx(
    r"\bwait\b", r"\bhold on\b", r"\bhang on\b", r"\bone (?:sec|second|minute|moment|min)\b",
    r"\b(?:give me|just) a (?:sec|second|minute|moment|min)\b", r"\bnot sure\b", r"\bunsure\b",
    r"\blet me (?:think|check|see|ask|confirm|discuss|talk)\b",
    r"\bi'?ll (?:think|check|let you know|get back|call back|confirm later|discuss|ask)\b",
    r"\bneed (?:to|some) (?:think|time|check|ask|discuss)\b", r"\bthink about it\b",
    r"\bmaybe\b", r"\bperhaps\b", r"\bi (?:don'?t|do not) know\b", r"\bnot decided\b",
    r"\b(?:ruko|rukiye|rukie)\b", r"\bek\s+(?:minute|min|second|sec|minat)\b",
    r"\bsoch(?:ta|ti|ke|kar|k|unga|ungi|enge)?\b", r"\b(?:puch|pooch)\s*(?:ke|kar|ta|ti)\b",
    _dv("रुको", "रुकिए", "एक मिनट", "सोच", "सोचता", "सोचती", "सोचकर", "पूछ के", "पूछकर"),
)

# A question about something else. Wh-words and "is there / do you have" — deliberately NOT
# modal requests: "could you book it for me" is an agreement, not a question.
_QUESTION = _rx(
    r"\b(?:how|where|why|who|whose)\b", r"\bwhat\b", r"\bwhich\b",
    r"\b(?:is|are)\s+there\b", r"\bdo(?:es)?\s+(?:you|it|they|i)\s+(?:have|need|get|include|"
    r"cost|charge|take)\b", r"\bis\s+(?:it|that|this|parking|there)\b",
    r"\b(?:kitna|kitne|kitni|kaha|kahan|kaise|kyun|kyon|kaun|kon)\b", r"^\s*kya\b",
    r"\bkya\b.*\b(?:hai|hain|milega|milegi|hoga|hogi)\s*$",
    _dv("कितना", "कितने", "कितनी", "कहाँ", "कहां", "कैसे", "क्यों", "कौन"),
)

# Agreement. A bare "ya" is only a yes on its own — inside a sentence it is Hindi for "or".
_ACCEPT = _rx(
    r"\b(?:yes|yeah|yea|yep|yup|yess+|sure|ok|okay|okey|okk+|alright|all right|fine|"
    r"correct|absolutely|definitely|certainly|of course|perfect|great|excellent|wonderful|"
    r"awesome|cool|agreed|deal|confirmed?|proceed)\b",
    r"^\s*(?:ya|yaa|done|please|right|good)\s*$",
    r"\bthat'?s\s+(?:right|correct|fine|good|perfect|great)\b", r"\bsounds\s+(?:good|great|"
    r"perfect|fine)\b", r"\b(?:that|it)\s+works\b", r"\bworks for me\b", r"\bfine by me\b",
    r"\bgood to go\b", r"\ball good\b",
    r"\bgo\s+ahead\b", r"\bgo\s+for\s+it\b", r"\bdo\s+it\b", r"\blet'?s\s+(?:do|go|book)\b",
    r"\b(?:i'?ll|i will|we'?ll)\s+take\s+(?:it|that|this|the)\b", r"^\s*take it\s*$",
    r"\bgo\s+with\s+(?:it|that|this|the)\b", r"\bthat one\b(?!\s+or\b)",
    r"\bplease\s+(?:do|go ahead|book|proceed|confirm)\b", r"\byes\s+please\b",
    r"\bbook\b(?!\s*(?:let|later))", r"\b(?:lock|fix|reserve|schedule)\s+(?:it|that|this)\b",
    r"\bsign me up\b", r"\bcount me in\b", r"\bi'?m in\b", r"\bi'?d (?:like|love) (?:that|to)\b",
    r"\bi\s+(?:want|would like)\s+(?:it|that|to book|to visit|to come)\b",
    r"\bi'?ll (?:come|be there)\b", r"\bi will (?:come|be there)\b", r"\bsee you then\b",
    # romanised Hindi
    r"\b(?:haan|haa|han|haanji|haan\s+ji|ha\s+ji|ji\s+haan|ji\s+ha|ji)\b",
    r"\b(?:theek|thik|tik)\s*(?:hai|h|he)?\b", r"\bchalega\b", r"\b(?:chalo|chaliye)\b",
    r"\bkar\s*(?:do|dijiye|dijie|dijiyega|dena)\b", r"\bkardo\b", r"\bkaro\b",
    r"\bkarwa\s+do\b", r"\bho\s+(?:jaye|jaaye|jayega|jaega)\b", r"\bbilkul\b",
    r"\bsahi\s*(?:hai)?\b", r"\b(?:zaroor|zarur|jaroor|jarur)\b", r"\b(?:achha|accha|acha)\b",
    r"\bpakka\b",
    # Devanagari
    _dv("हां", "हाँ", "हा", "जी", "जी हां", "जी हाँ", "ठीक है", "ठीक", "बिल्कुल", "सही है",
        "कर दो", "कर दीजिए", "कर दीजिये", "करवा दो", "करवा दीजिए", "करवा दीजिये", "करिए",
        "कीजिए", "कीजिये", "चलेगा", "चलिए", "ज़रूर", "जरूर", "अच्छा", "पक्का", "बुक कर",
        "हो जाए", "हो जाएगा", "ओके"),
)

# A bare refusal — weakest signal, checked last. "nahi nahi, kar do" is MIXED, not a no.
_DECLINE_WEAK = _rx(r"\bno\b", r"\b(?:nahi|nahin|nai|nahee)\b", _dv("नहीं", "नही"))

# A day or time the caller named. Captured for the staged-slot comparison in _names_slot().
_HOUR_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.?m\.?|p\.?m\.?|baje|bje|o'?clock)\b"
    # "at 10" only when no am/pm follows — "at 10 pm" must reach the branch above, or the
    # bare 10 is compared on the 12-hour clock and 10 PM passes for a staged 10 AM.
    r"|\b(?:at|by|around)\s+(\d{1,2})(?::(\d{2}))?\b"
    r"(?!\s*(?::\d|am\b|pm\b|a\.?m\b|p\.?m\b|baje|bje|o'?clock))"
    r"|\b(\d{1,2}):(\d{2})\b"
    r"|(\d{1,2})\s*बजे", re.I)
_WORD_HOURS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
               "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
_WORD_HOUR_RE = re.compile(
    r"\b(" + "|".join(_WORD_HOURS) + r")\s*(am|pm|o'?clock|baje)\b", re.I)
_WEEKDAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4,
             "saturday": 5, "sunday": 6,
             "somvar": 0, "mangalvar": 1, "budhvar": 2, "guruvar": 3, "shukravar": 4,
             "shanivar": 5, "ravivar": 6,
             "सोमवार": 0, "मंगलवार": 1, "बुधवार": 2, "गुरुवार": 3, "शुक्रवार": 4, "शनिवार": 5,
             "रविवार": 6}
_WEEKDAY_RE = re.compile(
    r"\b(" + "|".join(k for k in _WEEKDAYS if k.isascii()) + r")\b"
    r"|(" + "|".join(k for k in _WEEKDAYS if not k.isascii()) + r")", re.I)
_REL_DAY_RE = re.compile(
    r"\b(today|tonight|tomorrow|day after tomorrow|day after|aaj|kal|parso|parson)\b"
    r"|" + _dv("आज", "कल", "परसों"), re.I)
_REL_DAY_OFFSET = {"today": 0, "tonight": 0, "aaj": 0, "आज": 0, "tomorrow": 1, "kal": 1,
                   "कल": 1, "day after tomorrow": 2, "day after": 2, "parso": 2, "parson": 2,
                   "परसों": 2}
_PART_RE = re.compile(
    r"\b(morning|afternoon|evening|night|noon|subah|dopahar|shaam|sham|raat)\b"
    r"|" + _dv("सुबह", "दोपहर", "शाम", "रात"), re.I)
_PART_HOURS = {"morning": (5, 12), "subah": (5, 12), "सुबह": (5, 12),
               "noon": (11, 14), "afternoon": (12, 17), "dopahar": (12, 17), "दोपहर": (12, 17),
               "evening": (16, 21), "shaam": (16, 21), "sham": (16, 21), "शाम": (16, 21),
               "night": (19, 24), "raat": (19, 24), "रात": (19, 24)}
_ORDINAL_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)\b", re.I)


@dataclass(frozen=True)
class LexicalRead:
    """Layer A's verdict. ``decision`` is None when the words alone settle nothing, and
    ``mixed`` marks the specific case of both polarities with nothing to choose between them
    ("nahi nahi, kar do") — left to the semantic layer, never guessed."""
    decision: str | None
    signals: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    mixed: bool = False
    slot_conflict: bool = False


def normalize(text: str | None) -> str:
    """Lower-case, straight quotes, punctuation to spaces, single-spaced. Apostrophes and the
    colon inside a clock time are kept so "don't" and "10:30" survive."""
    t = normalize_quotes(text or "").lower()
    t = re.sub(r"(?<!\d):|:(?!\d)", " ", t)
    t = re.sub(r"[.,!?;\"()\[\]{}|/\\—–\-…।]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _consume(pattern: re.Pattern, text: str) -> tuple[str, list[str]]:
    """Find every match, return the text with each match blanked out plus what matched."""
    found = [m.group(0).strip() for m in pattern.finditer(text) if m.group(0).strip()]
    if not found:
        return text, []
    return pattern.sub(" ", text), found


def _parse_staged(staged_start: str | None) -> datetime | None:
    if not staged_start:
        return None
    try:
        return datetime.fromisoformat(str(staged_start).replace("Z", "+00:00"))
    except ValueError:
        return None


def _hour_24(h: int, ampm: str | None) -> int:
    ampm = (ampm or "").lower().replace(".", "")
    if ampm.startswith("p") and h < 12:
        return h + 12
    if ampm.startswith("a") and h == 12:
        return 0
    return h


def _names_slot(t: str) -> bool:
    """Whether the caller named a time or day at all."""
    return bool(_HOUR_RE.search(t) or _WORD_HOUR_RE.search(t) or _WEEKDAY_RE.search(t)
                or _REL_DAY_RE.search(t) or _PART_RE.search(t) or _ORDINAL_RE.search(t))


def _slot_conflicts(t: str, staged: datetime, today: date | None) -> bool:
    """True when a time/day the caller named does NOT match the staged slot. "ok, 11 am" to a
    staged 10:00 is a change, never a yes to 10:00. Anything named must agree with the slot."""
    hours = []
    for m in _HOUR_RE.finditer(t):
        g = m.groups()
        if g[0] is not None:
            hours.append((int(g[0]), int(g[1] or 0), g[2]))
        elif g[3] is not None:
            hours.append((int(g[3]), int(g[4] or 0), None))
        elif g[5] is not None:
            hours.append((int(g[5]), int(g[6] or 0), None))
        elif g[7] is not None:
            hours.append((int(g[7]), 0, None))
    for m in _WORD_HOUR_RE.finditer(t):
        hours.append((_WORD_HOURS[m.group(1).lower()], 0, m.group(2)))
    for h, mins, ampm in hours:
        if ampm and not ampm.lower().startswith(("baje", "bje", "o")):
            if (_hour_24(h, ampm), mins) != (staged.hour, staged.minute):
                return True
        # No am/pm: "10" matches 10:00 or 22:00 — compare on the 12-hour clock.
        elif (h % 12, mins) != (staged.hour % 12, staged.minute):
            return True
    for m in _WEEKDAY_RE.finditer(t):
        key = (m.group(1) or m.group(2) or "").lower()
        if key in _WEEKDAYS and _WEEKDAYS[key] != staged.weekday():
            return True
    rel = _REL_DAY_RE.search(t)
    if rel:
        key = rel.group(0).strip().lower()
        offset = _REL_DAY_OFFSET.get(key)
        # Without a reference "today" a relative day cannot be checked — treat it as a
        # conflict so the slot is read back rather than assumed.
        if offset is None or today is None or staged.date() != today + timedelta(days=offset):
            return True
    for m in _PART_RE.finditer(t):
        lo, hi = _PART_HOURS.get(m.group(0).strip().lower(), (0, 24))
        if not lo <= staged.hour < hi:
            return True
    for m in _ORDINAL_RE.finditer(t):
        if int(m.group(1)) != staged.day:
            return True
    return False


def read_lexical(text: str | None, *, pending_tool: str | None = None,
                 staged_start: str | None = None, today: date | None = None) -> LexicalRead:
    """Layer A: the caller's words, read deterministically.

    ``pending_tool`` — the staged tool, so context verbs resolve correctly.
    ``staged_start`` — the staged slot (ISO), so a named time/day is checked against it.
    ``today`` — the call's "today", so "today"/"tomorrow" can be checked against the slot.
    """
    t = f" {normalize(text)} "
    if not t.strip():
        return LexicalRead(None)
    signals: list[str] = []
    evidence: list[str] = []

    def take(sig: str, pattern: re.Pattern) -> None:
        nonlocal t
        t, found = _consume(pattern, t)
        if found:
            signals.append(sig)
            evidence.extend(found)

    # Detect a named slot on the untouched text — every later step blanks words out.
    named_slot = _names_slot(t)
    staged = _parse_staged(staged_start)
    slot_conflict = bool(named_slot and staged is not None and _slot_conflicts(t, staged, today))

    take("accept", _ACCEPT_IDIOMS)
    take("repeat", _REPEAT)
    # Context verbs before the fixed groups, so "cancel it" / "move it" resolve against the
    # staged action instead of a fixed meaning.
    if pending_tool == "cancel_appointment":
        take("decline", _rx(r"\b(?:don'?t|do not|dont)\s+cancel\b", r"\bmat\s+cancel\b",
                            _dv("कैंसिल मत")))
        take("accept", _CANCEL_VERB)
    elif pending_tool == "reschedule_appointment":
        take("decline", _rx(r"\b(?:don'?t|do not|dont)\s+(?:move|shift|reschedule)\b"))
        take("accept", _MOVE_VERB)
    else:
        take("decline", _CANCEL_VERB)
    # Refusals before changes so "rather not" / "can't make it" are consumed whole and never
    # leave "rather" / "make it" behind to read as a change. When both are present the change
    # still wins in the decision below ("changed my mind, make it 11" is a change).
    take("decline", _DECLINE_STRONG)
    take("change", _CHANGE)
    take("hold", _HOLD)
    take("question", _QUESTION)
    take("accept", _ACCEPT)
    take("decline_weak", _DECLINE_WEAK)

    s = set(signals)
    ev = tuple(dict.fromkeys(evidence))
    sigs = tuple(dict.fromkeys(signals))

    def out(decision: str | None, mixed: bool = False) -> LexicalRead:
        return LexicalRead(decision, sigs, ev, mixed, slot_conflict)

    if "change" in s:
        return out(CHANGE)
    if "decline" in s:
        return out(None, mixed=True) if "accept" in s else out(DECLINE)
    if "repeat" in s:
        return out(REPEAT)
    if slot_conflict:
        return out(CHANGE)
    if "question" in s:
        return out(SIDE_QUESTION)
    if "hold" in s:
        return out(HOLD)
    if "accept" in s:
        if "decline_weak" in s:
            return out(None, mixed=True)
        return out(ACCEPT)
    if "decline_weak" in s:
        return out(DECLINE)
    # A day/time and nothing else: with no staged slot to compare it is a new choice; with a
    # staged slot it matched (a conflict returned CHANGE above), so it is an echo — not a yes
    # on its own, left to the semantic layer.
    if named_slot and staged is None:
        return out(CHANGE)
    return out(None)


# ------------------------------------------------------------------ fusion

@dataclass(frozen=True)
class ConsentResult:
    decision: str
    source: str                         # lexical | semantic | conflict | default
    evidence: tuple[str, ...] = ()
    lexical: str | None = None
    semantic: str | None = None
    semantic_evidence: str | None = None
    semantic_verified: bool = False
    note: str = ""
    signals: tuple[str, ...] = field(default_factory=tuple)

    @property
    def accepted(self) -> bool:
        return self.decision == ACCEPT

    def as_trace(self) -> dict:
        return {"decision": self.decision, "source": self.source,
                "evidence": list(self.evidence), "lexical": self.lexical,
                "lexical_signals": list(self.signals), "semantic": self.semantic,
                "semantic_evidence": self.semantic_evidence,
                "semantic_verified": self.semantic_verified, "note": self.note}


def normalize_semantic(value) -> str | None:
    """The understanding model's decision, or None when absent/off-list."""
    v = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return v if v in DECISIONS else None


def evidence_in_text(evidence: str | None, text: str | None) -> bool:
    """The words the model quoted as consent really are in what the caller said. This is what
    stops a model ACCEPT on a garble: there are no consent words in "ja por time again" for it
    to quote."""
    ev = normalize(evidence)
    if len(ev) < 2:
        return False
    return f" {ev} " in f" {normalize(text)} "


def decide(text: str | None, *, lexical: LexicalRead | None = None,
           semantic: dict | None = None, stt_confidence: float | None = None,
           stt_floor: float = 0.0, pending_tool: str | None = None,
           staged_start: str | None = None, today: date | None = None) -> ConsentResult:
    """Fuse both layers into one decision. The asymmetry is deliberate: every rule that can
    end in ACCEPT needs positive evidence in the caller's own words."""
    lex = lexical or read_lexical(text, pending_tool=pending_tool,
                                  staged_start=staged_start, today=today)
    sem = normalize_semantic((semantic or {}).get("decision"))
    sem_ev = (semantic or {}).get("evidence") or None
    verified = bool(sem == ACCEPT and evidence_in_text(sem_ev, text))
    low_stt = stt_confidence is not None and stt_floor and stt_confidence < stt_floor

    def res(decision: str, source: str, note: str = "") -> ConsentResult:
        return ConsentResult(decision=decision, source=source, evidence=lex.evidence,
                             lexical=lex.decision, semantic=sem,
                             semantic_evidence=sem_ev, semantic_verified=verified,
                             note=note, signals=lex.signals)

    # 1. The words say something other than yes → that stands. Deterministic and safe.
    if lex.decision is not None and lex.decision != ACCEPT:
        return res(lex.decision, "lexical")
    # 2. The words say yes.
    if lex.decision == ACCEPT:
        if sem is None or sem == ACCEPT:
            return res(ACCEPT, "lexical")
        # The model read the same words as something else. Ask, don't guess either way.
        return res(UNCLEAR, "conflict", f"lexical accept vs semantic {sem}")
    # 3. The words settle nothing (or both polarities) → the model's reading, if it holds up.
    if sem is None:
        return res(UNCLEAR, "default",
                   "mixed signals, no semantic read" if lex.mixed else "no consent signal")
    if sem != ACCEPT:
        return res(sem, "semantic")
    if not verified:
        return res(UNCLEAR, "semantic", "semantic accept rejected: evidence not in transcript")
    # Quoting the garble itself ("ja por") passes the substring check, so a short Latin-script
    # reply with no recognisable word at all cannot be a model-only yes. (Devanagari is
    # exempt: STT emits real words there, and the recogniser is English-only.)
    if (text or "").isascii() and looks_unrecognizable(text):
        return res(UNCLEAR, "semantic", "semantic accept rejected: unrecognisable utterance")
    if low_stt:
        return res(UNCLEAR, "semantic", "semantic accept rejected: low STT confidence")
    return res(ACCEPT, "semantic")


# ------------------------------------------------------------------ the lines the caller hears

# Per staged tool: (english verb, preposition before the time, hinglish verb, hindi verb).
_VERBS = {
    "book_appointment": ("book your visit", "for", "aapki visit book kar doon",
                         "आपकी विज़िट बुक कर दूँ"),
    "reschedule_appointment": ("move your visit", "to", "aapki visit shift kar doon",
                               "आपकी विज़िट शिफ्ट कर दूँ"),
    "cancel_appointment": ("cancel your visit", "on", "aapki visit cancel kar doon",
                           "आपकी विज़िट कैंसिल कर दूँ"),
    "create_ticket": ("log that for you", None, "ye request log kar doon",
                      "ये रिक्वेस्ट दर्ज कर दूँ"),
}
_DEFAULT_VERB = ("go ahead with that", None, "aage badhoon", "आगे बढ़ूँ")


def _verb(tool: str | None, english_verb: str | None) -> tuple[str, str | None, str, str]:
    v = _VERBS.get(tool or "", _DEFAULT_VERB)
    return (english_verb or v[0], v[1], v[2], v[3])


def readback(tool: str | None, when: str | None, *, kind: str, language: str = "english",
             attempt: int = 0, english_verb: str | None = None) -> str:
    """The deterministic line for a turn that keeps the staged action waiting.

    kind="repeat"   — they asked to hear it again: say the details, then ask.
    kind="unclear"  — we could not read the reply: say so, the details, then ask; from the
                      second attempt on, invite a plain yes or no.
    kind="reask"    — appended after answering a side question: just the question.
    Built only from the staged data (``when`` comes from the offered-slot list), so it can
    never state a time the caller was not offered.
    """
    en_verb, prep, hl_verb, hi_verb = _verb(tool, english_verb)
    lang = language if language in ("english", "hindi", "hinglish") else "english"
    if lang == "hindi":
        at = f" — {when}" if when else ""
        if kind == "repeat":
            return f"जी, समय है {when}। क्या मैं {hi_verb}?" if when else f"जी। क्या मैं {hi_verb}?"
        if kind == "reask":
            return f"और क्या मैं {hi_verb}{at}?"
        if attempt >= 1:
            return f"बस कन्फर्म कर लूँ — क्या मैं {hi_verb}{at}? हाँ या ना बता दीजिए।"
        return f"माफ़ कीजिए, मैं ठीक से सुन नहीं पायी। क्या मैं {hi_verb}{at}?"
    if lang == "hinglish":
        at = f" — {when}" if when else ""
        if kind == "repeat":
            return (f"Ji, time hai {when}. Kya main {hl_verb}?" if when
                    else f"Ji. Kya main {hl_verb}?")
        if kind == "reask":
            return f"Aur kya main {hl_verb}{at}?"
        if attempt >= 1:
            return f"Bas confirm kar loon — kya main {hl_verb}{at}? Haan ya na bata dijiye."
        return f"Sorry, theek se sunai nahi diya. Kya main {hl_verb}{at}?"
    at = f" {prep} {when}" if (when and prep) else (f" — {when}" if when else "")
    if kind == "repeat":
        if when:
            return f"Sure — that's {when}. Shall I go ahead and {en_verb}?"
        return f"Sure. Shall I go ahead and {en_verb}?"
    if kind == "reask":
        return f"And shall I go ahead and {en_verb}{at}?"
    if attempt >= 1:
        return f"Just to confirm — should I {en_verb}{at}? A simple yes or no is fine."
    return f"Sorry, I didn't quite catch that. Shall I go ahead and {en_verb}{at}?"


# ---------------------------------------------------------------------------------------
# Built-in self-test — runs without pytest.
#   python -m app.engine.planner.consent
# ---------------------------------------------------------------------------------------
def _selftest() -> int:
    failures: list[str] = []

    def check(name, cond):
        if not cond:
            failures.append(name)

    def lx(text, **kw):
        return read_lexical(text, **kw).decision

    # --- the reported bug: neither line is consent ----------------------------------------
    for t in ("i'm sorry what time", "ja por time again", "i'm sorry what time ja por time again"):
        check(f"bug_is_repeat::{t}", lx(t, pending_tool="book_appointment") == REPEAT)
        r = decide(t, semantic={"decision": "accept", "evidence": "time"},
                   pending_tool="book_appointment")
        check(f"bug_never_accept::{t}", r.decision != ACCEPT)

    # --- plain and sentence-shaped agreement ----------------------------------------------
    for t in ("yes", "yeah", "yep", "sure", "ok", "okay", "alright", "fine", "correct",
              "go ahead", "go for it", "do it", "sure let's do it", "let's do that",
              "yes please", "please do", "book it", "sounds good", "that works", "perfect",
              "no problem, go ahead", "why not", "i don't mind", "please",
              "i'll take it", "let's go with that one",
              "could you book it for me", "haan", "haan ji", "ji", "theek hai", "thik hai",
              "chalega", "kar do", "haan kar do", "book kar do", "bilkul", "zaroor", "achha",
              "हाँ", "हां जी", "जी", "ठीक है", "कर दीजिए", "हाँ बुक कर दीजिए", "बिल्कुल"):
        check(f"accept::{t}", lx(t, pending_tool="book_appointment") == ACCEPT)

    # --- refusals and changes of mind -----------------------------------------------------
    for t in ("no", "nope", "no thanks", "not now", "I changed my mind", "on second thought",
              "leave it", "forget it", "never mind", "don't book it", "do not book",
              "not interested", "maybe later", "I can't make it", "nahi", "rehne do",
              "abhi nahi", "baad mein", "mat karo", "nahi chahiye", "theek nahi hai",
              "नहीं", "रहने दो", "मत कीजिए", "अभी नहीं", "बाद में", "cancel it"):
        check(f"decline::{t}", lx(t, pending_tool="book_appointment") == DECLINE)

    # --- repeat / clarify -----------------------------------------------------------------
    for t in ("what time", "what time was it", "okay what time?", "yes, which day was it",
              "say that again", "can you repeat", "pardon", "sorry?", "come again",
              "I didn't catch that", "when is it", "what did you say", "one more time",
              "phir se bolo", "dobara bataiye", "kitne baje", "kab hai", "samajh nahi aaya",
              "फिर से बताइए", "कितने बजे", "समझ नहीं आया", "क्या बोला"):
        check(f"repeat::{t}", lx(t, pending_tool="book_appointment") == REPEAT)

    # --- change the details ---------------------------------------------------------------
    staged = "2026-10-13T10:00:00+05:30"
    today = date(2026, 10, 13)
    for t in ("make it 11 instead", "can we do tomorrow", "another day", "how about 4 pm",
              "yes but at 11 am", "ok 11 am", "yes at 10 pm", "yes tomorrow",
              "something later", "kal ka time do",
              "dusra time", "11 baje", "tomorrow morning", "yes see you on friday"):
        check(f"change::{t}", lx(t, pending_tool="book_appointment", staged_start=staged,
                                  today=today) == CHANGE)
    # a named slot that MATCHES the staged one is not a change
    for t in ("yes 10 am is fine", "ok today at 10", "haan 10 baje theek hai",
              "yes see you today morning"):
        check(f"matching_slot_accept::{t}",
              lx(t, pending_tool="book_appointment", staged_start=staged,
                 today=today) == ACCEPT)

    # --- side questions and holds ---------------------------------------------------------
    for t in ("how much is parking", "is there parking", "ok and where is the site",
              "yes and is it ready to move", "parking kitna hai", "kahan hai site"):
        check(f"side::{t}", lx(t, pending_tool="book_appointment") == SIDE_QUESTION)
    for t in ("wait", "hold on", "let me think", "not sure", "maybe", "one second",
              "ruko", "soch ke batata hoon", "रुकिए"):
        check(f"hold::{t}", lx(t, pending_tool="book_appointment") == HOLD)

    # --- context verbs --------------------------------------------------------------------
    check("cancel_accepts_staged_cancel",
          lx("yes cancel it", pending_tool="cancel_appointment") == ACCEPT)
    check("cancel_karo_accepts_staged_cancel",
          lx("haan cancel kar do", pending_tool="cancel_appointment") == ACCEPT)
    check("dont_cancel_declines_staged_cancel",
          lx("no don't cancel", pending_tool="cancel_appointment") == DECLINE)
    check("move_accepts_staged_reschedule",
          lx("yes move it", pending_tool="reschedule_appointment") == ACCEPT)

    # --- mixed polarity → semantic decides; without it → UNCLEAR --------------------------
    r = read_lexical("nahi nahi, kar do", pending_tool="book_appointment")
    check("mixed_flag", r.decision is None and r.mixed)
    check("mixed_no_semantic_unclear",
          decide("nahi nahi, kar do", pending_tool="book_appointment").decision == UNCLEAR)
    check("mixed_semantic_accept_verified",
          decide("nahi nahi, kar do", semantic={"decision": "accept", "evidence": "kar do"},
                 pending_tool="book_appointment").decision == ACCEPT)

    # --- fusion safety --------------------------------------------------------------------
    check("semantic_accept_needs_evidence",
          decide("blah blah", semantic={"decision": "accept", "evidence": "yes"}).decision
          == UNCLEAR)
    check("semantic_accept_verified",
          decide("let us carry on with this plan",
                 semantic={"decision": "accept", "evidence": "carry on with this plan"}).decision
          == ACCEPT)
    check("semantic_accept_low_stt",
          decide("let us carry on with this plan", stt_confidence=0.3, stt_floor=0.55,
                 semantic={"decision": "accept",
                           "evidence": "carry on with this plan"}).decision == UNCLEAR)
    check("lexical_accept_semantic_decline_conflict",
          decide("yes", semantic={"decision": "decline", "evidence": "yes"}).decision
          == UNCLEAR)
    check("lexical_repeat_beats_semantic_accept",
          decide("ok what time", semantic={"decision": "accept",
                                           "evidence": "ok"}).decision == REPEAT)
    check("semantic_side_question",
          decide("the clubhouse thing", semantic={"decision": "side_question",
                                                  "evidence": ""}).decision == SIDE_QUESTION)
    check("nothing_is_unclear", decide("ja por").decision == UNCLEAR)
    check("garble_quoting_itself_rejected",
          decide("ja por", semantic={"decision": "accept", "evidence": "ja por"}).decision
          == UNCLEAR)
    check("evidence_curly_quote", evidence_in_text("let’s do it", "ok let's do it"))
    check("off_list_semantic_ignored", normalize_semantic("affirm") is None)

    # --- the lines --------------------------------------------------------------------------
    w = "Tuesday 13 October at 10:00 AM"
    check("readback_repeat_en", readback("book_appointment", w, kind="repeat") ==
          "Sure — that's Tuesday 13 October at 10:00 AM. Shall I go ahead and book your visit?")
    check("readback_unclear_en", "didn't quite catch" in readback(
        "book_appointment", w, kind="unclear"))
    check("readback_unclear_second", "yes or no" in readback(
        "book_appointment", w, kind="unclear", attempt=1))
    check("readback_hinglish", readback("book_appointment", w, kind="repeat",
                                        language="hinglish").startswith("Ji, time hai"))
    check("readback_no_when", readback("cancel_appointment", None, kind="reask") ==
          "And shall I go ahead and cancel your visit?")

    if failures:
        print(f"consent selftest: {len(failures)} FAILED")
        for f in failures:
            print("  -", f)
        return 1
    print("consent selftest: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
