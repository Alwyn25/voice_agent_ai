"""Split a compound question into the separate questions it contains.

THE BUG THIS FIXES. "What is the per square foot rate for the 3 BHK, and how far is the
project from Wipro?" is two questions. Retrieval scored it as ONE query, and the price half
("per square foot") marked the whole thing a price question: every pricing document got the
price boost, four of them filled all eight window slots, and the Location document — which
states "Wipro Corporate Office 2.5 km" — never reached the model. The responder then invented
a distance ("approximately 5 kilometres"). Asked on its own, the distance question retrieves
Location first and is answered correctly. Nothing about pricing or Wipro is special here: any
question-specific boost (price, policy/scope, a metadata lift, the focus anchor) earned by one
half of a compound question can crowd out the other half the same way.

The fix is structural, not a re-weighting: each sub-question is retrieved on its own — with
the boosts IT earns — and the windows are merged by taking turns (retrieval.retrieve). So every
question the caller asked gets its best evidence into the window, whatever the other asked.

SPLITTING IS CONSERVATIVE, because a wrong split is worse than none (it can strip the subject
off a half: "the price of the 2 BHK and the 3 BHK" must stay one query). A split happens only
  * at a "?" that ends one question with another following, or
  * at a conjunction ("and", "also", "plus", "aur", …) followed by a NEW QUESTION HEAD — a
    wh-word, an auxiliary that opens a question ("is there", "do you", "can I"), or the
    Hinglish heads (kya, kitna, kahan, kab, kaise, …).
Each part must still carry at least one content word; anything shorter is folded back into
its neighbour. A single question returns [query] unchanged, and so takes the exact pre-split
retrieval path.
"""
import re

# Words that open a new question when they follow a conjunction.
_QUESTION_HEADS = (
    r"what|what's|whats|how|how's|where|where's|when|which|who|whom|whose|why|"
    r"is there|are there|is it|is the|are the|is this|are they|"
    r"do you|does it|does the|do i|did you|can i|can you|could you|could i|"
    r"will it|will you|would it|would you|should i|"
    r"tell me|let me know|any idea|"
    # Hinglish (romanised) heads
    r"kya|kitna|kitni|kitne|kahan|kahaan|kab|kaise|kaun|kaunsa|kaunsi|konsa|konsi"
)
_CONJ = r"(?:and|also|plus|as well as|then|aur|or)"
# ", and how far …" / " and also what …" / " aur kitna …"
_CONJ_SPLIT_RE = re.compile(
    rf"\s*[,;]?\s+{_CONJ}(?:\s+also)?\s+(?=(?:{_QUESTION_HEADS})\b)", re.IGNORECASE)
# One question ends, another begins.
_QMARK_SPLIT_RE = re.compile(r"(?<=\?)\s+(?=\S)")

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "can", "could", "will",
    "would", "should", "have", "has", "had", "i", "you", "we", "my", "your", "me", "it", "to",
    "of", "for", "on", "in", "at", "and", "or", "if", "so", "that", "this", "there", "then",
    "what", "how", "why", "when", "which", "who", "where", "much", "many", "any", "some",
    "tell", "know", "please", "also", "plus", "about", "aur", "kya", "hai", "ka", "ki", "ke",
}


def _content_words(text: str) -> int:
    return sum(1 for w in re.findall(r"[\w']+", text.lower()) if w not in _STOP)


def split_compound_query(query: str, *, max_parts: int = 3) -> list[str]:
    """The sub-questions in `query`, in order. `[query]` when it is a single question."""
    q = (query or "").strip()
    if not q:
        return [q]
    pieces: list[str] = []
    for sentence in _QMARK_SPLIT_RE.split(q):
        pieces.extend(p for p in _CONJ_SPLIT_RE.split(sentence) if p and p.strip())

    # Fold fragments too thin to search on back into the previous piece ("…, and why?").
    parts: list[str] = []
    for p in (x.strip(" ,;") for x in pieces):
        if parts and _content_words(p) < 1:
            parts[-1] = f"{parts[-1]} {p}".strip()
        elif not parts and _content_words(p) < 1 and len(pieces) > 1:
            parts.append(p)               # leading fragment: merged forward below
        else:
            if parts and _content_words(parts[-1]) < 1:
                parts[-1] = f"{parts[-1]} {p}".strip()
            else:
                parts.append(p)
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return [q]
    if len(parts) > max_parts:            # keep the first max_parts-1, fold the rest together
        parts = parts[:max_parts - 1] + [" ".join(parts[max_parts - 1:])]
    return parts


def _selftest() -> None:
    cases = {
        # the reported bug
        "What is the per square foot rate for the 3 BHK, and how far is the project from Wipro?":
            ["What is the per square foot rate for the 3 BHK",
             "how far is the project from Wipro?"],
        "How long does a site visit take and what are the maintenance charges?":
            ["How long does a site visit take", "what are the maintenance charges?"],
        "What's the price of the 3 BHK? Is there a gym?":
            ["What's the price of the 3 BHK?", "Is there a gym?"],
        "Tell me the carpet area and also is there a creche in the clubhouse":
            ["Tell me the carpet area", "is there a creche in the clubhouse"],
        "3 BHK ka price kya hai aur kitna door hai Wipro se":
            ["3 BHK ka price kya hai", "kitna door hai Wipro se"],
        "what is the rate, where is the site office and when can I visit?":
            ["what is the rate, where is the site office", "when can I visit?"],
    }
    for q, want in cases.items():
        got = split_compound_query(q)
        assert got == want, f"\n  q:    {q!r}\n  want: {want!r}\n  got:  {got!r}"

    single = [
        "What is the price of the 2 BHK and the 3 BHK?",       # one question, two subjects
        "Is there a swimming pool and a gym?",
        "how far is the project from Wipro?",
        "I want a 3 BHK and I can pay up to 1.6 crore",       # statement, no question head after "and"
        "What are the sizes and prices for 3 BHK?",
        "Can you send the floor plan and the brochure?",
        "and what?",
        "",
    ]
    for q in single:
        assert split_compound_query(q) == [q.strip()], f"split but should not: {q!r} -> " \
                                                         f"{split_compound_query(q)!r}"
    assert len(split_compound_query(
        "What is the rate? Is there a gym? How far is Wipro? When is possession?",
        max_parts=3)) == 3
    print("query_split selftest: all passed")


if __name__ == "__main__":
    _selftest()
