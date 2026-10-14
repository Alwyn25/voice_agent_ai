"""Flag words in a voice transcript that SOUND LIKE a term this call expects but were
transcribed as something else -- so the planner reads "position" as "possession" instead of
routing the turn to a booking-status lookup.

WHY HINTS, NOT REWRITES. "When is the position" almost certainly means possession; "what
position is the flat facing" does not. Only the conversation can tell, so this module never
changes what the caller said: it finds candidates deterministically and the planner -- which
has the conversation -- decides (app/engine/planner/prompts.py renders the hint). A false
hint costs a few prompt tokens; a false rewrite would put words in the caller's mouth.

HOW. Each transcript window (1..n+1 words, for a term of n words) is compared with each term
of the call lexicon (app/engine/voice/lexicon.py) on two signals:
  * phonetic_key -- a small consonant-skeleton key (vowels dropped, voicing pairs and common
    spellings of one sound merged), so "rare"/"RERA", "position"/"possession" and
    "mokai visit"/"book a visit" land close even though their spelling does not;
  * spelling similarity, so the key alone cannot match unrelated words.
Spelled-out acronyms with no vowels ("BHK", "EMI" is not one) are matched letter-by-letter
instead: same length, same first letter, at most one letter different ("bsk" -> "BHK").
Pure, dependency-free and domain-agnostic: everything domain-specific is in the lexicon.
"""
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

# Score at or above which a window is reported as a likely mishearing of a term. Tuned on
# real transcripts (tests/test_asr_hints.py): the recorded mishearings score 0.78-0.96, while
# real phrases that merely resemble a term ("site office"/"site visit" 0.744, "rate"/"RERA",
# "area"/"RERA") stay below it.
THRESHOLD = 0.75
_KEY_WEIGHT, _CHAR_WEIGHT = 0.6, 0.4
# At most this many hints per turn -- the strongest ones; a transcript that "matches"
# more than this is noise, not a few misheard words.
MAX_HINTS = 3
# Terms shorter than this (letters only) are too short to match safely by sound alone.
_MIN_TERM_LETTERS = 3

# Ordered spelling -> sound rewrites applied before the skeleton is taken.
_SPELLINGS = (
    ("ssion", "sn"), ("sion", "sn"), ("tion", "sn"), ("cion", "sn"),
    ("ph", "f"), ("gh", "g"), ("ck", "k"), ("qu", "k"), ("q", "k"), ("x", "ks"),
    ("ce", "se"), ("ci", "si"), ("cy", "sy"), ("c", "k"),
)
# Sounds STT confuses with each other (voicing pairs, v/w, s/z) collapse to one symbol.
_SAME_SOUND = {"b": "p", "d": "t", "g": "k", "v": "f", "w": "f", "z": "s", "j": "s"}
_VOWELS = set("aeiouy")

# Function words never reported on their own: "is" is not a mishearing of anything.
_STOPWORDS = frozenset("""
a an the is it its it's to of in on at for and or but my me i you your we our us he she they
them this that these those be am are was were do does did can could will would should shall
may might must have has had not no yes ok okay so if then there here what when where who how
why which with from by as up out about please hello hi hey thanks thank just very really
""".split())


@dataclass(frozen=True)
class AsrHint:
    heard: str     # the words as transcribed
    term: str      # the lexicon term they most likely were
    score: float


def phonetic_key(text: str) -> str:
    """Consonant skeleton of ``text`` (letters only; spaces and digits ignored)."""
    s = re.sub(r"[^a-z]", "", (text or "").lower())
    if not s:
        return ""
    for spelled, sound in _SPELLINGS:
        s = s.replace(spelled, sound)
    out: list[str] = []
    for i, ch in enumerate(s):
        if ch in _VOWELS:
            if i == 0:
                out.append("a")          # a leading vowel is audible; mark it once
            continue
        if ch == "h":                    # "bh", "dh", "kh", "sh", a silent h: not distinctive
            continue
        sym = _SAME_SOUND.get(ch, ch)
        if not out or out[-1] != sym:    # doubled letters / merged sounds collapse
            out.append(sym)
    return "".join(out)


def _letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", (text or "").lower())


def _words(text: str) -> list[str]:
    # Latin-script words and numbers only; other scripts (Devanagari etc.) are left alone --
    # a sound key over a different script would be meaningless.
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _is_spelled_acronym(term: str) -> bool:
    return term.isupper() and term.isalpha() and not (set(term.lower()) & _VOWELS)


def _acronym_match(window: str, term: str) -> float:
    """'bsk' / 'b h k' vs 'BHK': same letters count, same first letter, <= 1 letter off."""
    w, t = _letters(window), term.lower()
    if len(w) != len(t) or w[0] != t[0]:
        return 0.0
    diff = sum(1 for a, b in zip(w, t) if a != b)
    return 1.0 if diff == 0 else (0.8 if diff == 1 else 0.0)


def _score(window: str, term: str) -> float:
    if _is_spelled_acronym(term):
        return _acronym_match(window, term)
    wk, tk = phonetic_key(window), phonetic_key(term)
    if not wk or not tk:
        return 0.0
    # One word for one word must SOUND the same -- identical keys. Every real single-word
    # mishearing does ("position"/"possession", "rare"/"RERA", "consultashun"); ordinary
    # words that only resemble a term do not ("person" prsn vs psn, "session" sn vs psn), and
    # without this they scored 0.76-0.81 on spelling overlap alone. Phrases keep the graded
    # score: STT legitimately swaps a sound inside one ("mokai visit" -> "book a visit").
    if " " not in window.strip() and " " not in term.strip() and wk != tk:
        return 0.0
    key_sim = SequenceMatcher(None, wk, tk).ratio()
    char_sim = SequenceMatcher(None, _letters(window), _letters(term)).ratio()
    return _KEY_WEIGHT * key_sim + _CHAR_WEIGHT * char_sim


def find_hints(transcript: str, lexicon: list[str] | None) -> list[AsrHint]:
    """The strongest likely mishearings of lexicon terms in ``transcript``, best first.

    A term the transcript already contains correctly is never hinted; a window that is
    itself a lexicon term is never treated as a mishearing of another; windows made only of
    function words are skipped; and each span of the transcript is attributed to at most
    one term (its best match)."""
    words = _words(transcript)
    if not words or not lexicon:
        return []
    padded = f" {' '.join(words)} "
    known = {" ".join(_words(t)) for t in lexicon}
    # Word positions already covered by a term heard CORRECTLY. A window overlapping one is
    # never a mishearing -- "to book a visit" must not be read as a garbled "book a site visit".
    correct: set[int] = set()
    for k in known:
        kw = k.split()
        for i in range(0, len(words) - len(kw) + 1):
            if kw and words[i:i + len(kw)] == kw:
                correct.update(range(i, i + len(kw)))

    best_by_span: dict[tuple[int, int], AsrHint] = {}
    for term in lexicon:
        t_words = _words(term)
        if not t_words or len(_letters(term)) < _MIN_TERM_LETTERS:
            continue
        if f" {' '.join(t_words)} " in padded:
            continue                                   # heard correctly already
        n = len(t_words)
        sizes = [1] if _is_spelled_acronym(term) else sorted({max(1, n - 1), n, n + 1})
        for size in sizes:
            for i in range(0, len(words) - size + 1):
                span = words[i:i + size]
                window = " ".join(span)
                if (window in known or all(w in _STOPWORDS for w in span)
                        or correct.intersection(range(i, i + size))):
                    continue
                if size == 1 and (len(span[0]) < 3 or span[0].isdigit()):
                    continue
                score = _score(window, term)
                if score < THRESHOLD:
                    continue
                key = (i, i + size)
                prior = best_by_span.get(key)
                if prior is None or score > prior.score:
                    best_by_span[key] = AsrHint(heard=window, term=term, score=round(score, 3))

    # Keep the best non-overlapping spans, one hint per term.
    chosen: list[AsrHint] = []
    taken: set[int] = set()
    used_terms: set[str] = set()
    for (start, end), hint in sorted(best_by_span.items(), key=lambda kv: -kv[1].score):
        positions = set(range(start, end))
        if positions & taken or hint.term.lower() in used_terms:
            continue
        taken |= positions
        used_terms.add(hint.term.lower())
        chosen.append(hint)
        if len(chosen) >= MAX_HINTS:
            break
    return chosen
