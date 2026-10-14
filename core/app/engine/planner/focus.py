"""Active-focus entity slots: what the call is currently about.

THE PROBLEM. Retrieval searched on the raw current utterance and nothing else. When a
customer said "I want a 2.5 BHK" and then "can you give me both cost breakdowns", the
second string contains no "2.5", no "BHK" — so the retriever had no way to prefer the
2.5 BHK document, and the long, number-dense 2 BHK and 3 BHK cost sheets won the ranking
as they always do. The prompt then tells the model those chunks are the only material it
may quote, so it answered about the wrong flat. The history still held "2.5 BHK" and the
model could see it; the retriever is stateless and overrode it.

WHY THIS AND NOT QUERY REWRITING. An LLM rewrite step ("condense the follow-up into a
standalone query") also fixes it, and adds a serial model call on the retrieval path —
150 to 400ms on a system already at 4-8 seconds a turn. This adds none, because the
planner ALREADY extracts entities every turn under its `understood` contract. We were
throwing them away. Rewriting can be layered on later for messy coreference like "the
cheaper one"; the two are complementary.

Nothing here is domain-specific. The slot names come from whatever the planner emits, the
same way the understanding classifier (planner.understanding) routes by meaning rather than
a keyword list — an engine that sells anything cannot enumerate one industry's nouns.
"""
import re

# A BLOCKLIST, not an allowlist.
#
# The first version listed the slot names worth searching on — "configuration", "variant",
# "tier" and so on. That was the same mistake as the old FACTUAL_TOPICS keyword list: the
# model is never told what to call its entities (the contract just says
# {"<name>": "<value>"}), so it emits whatever it likes — flat_type, bhk, unit_type,
# property_type. None of those were on the list, so "2.5 BHK" was captured, stored, shown
# to the model, and never reached the search query. The customer asked about a 2.5 BHK and
# got the 3 BHK price.
#
# Blocking the handful of things that are genuinely not subjects is both shorter and
# domain-neutral. Anything else the model considered worth extracting is worth searching on.
NON_SUBJECT_SLOTS = frozenset({
    "sentiment", "emotion", "tone", "mood", "urgency", "priority", "intent",
    "action", "confidence", "language", "name", "full_name", "customer_name",
    "phone", "phone_number", "mobile", "email", "date", "time", "day",
    "slot_time", "appointment_time", "datetime", "when", "yes_no", "answer",
})

# An explicit change of subject. Without this, "actually show me the 3 BHK instead" would
# keep dragging 2.5 BHK into every subsequent search.
_RESET = re.compile(r"\b(?:actually|instead|different|other one|change|forget (?:that|it)|"
                    r"never ?mind|not that|rather)\b", re.I)

_MAX_SLOTS = 12
_MAX_VALUE = 60


def _clean(value) -> str:
    """A short, printable string, or empty."""
    if value is None or isinstance(value, (dict, list, bool)):
        return ""
    text = " ".join(str(value).split())
    # Placeholders the model emits when it has no real value — the same class of thing
    # that once registered a customer as "<customer's phone number>".
    if not text or text.startswith(("<", "{", "[")) or text.lower() in (
            "none", "null", "n/a", "unknown", "string", "unspecified"):
        return ""
    return text[:_MAX_VALUE]


def merge_entities(prior: dict | None, incoming: dict | None, user_input: str) -> dict:
    """Carry the focus forward, overwriting only where they stated something new.

    - a new value for a slot replaces the old one (they changed their mind)
    - a slot they did not mention this turn is KEPT — it is still what the call is about
    - an explicit subject change clears the carried slots before merging
    """
    merged = {} if _RESET.search(user_input or "") else dict(prior or {})
    for name, value in (incoming or {}).items():
        key = str(name).strip().lower()[:40]
        val = _clean(value)
        if key and val:
            merged[key] = val
    if len(merged) > _MAX_SLOTS:
        # Dicts keep insertion order, so this drops the oldest.
        merged = dict(list(merged.items())[-_MAX_SLOTS:])
    return merged


# Generic English cues that a turn LEANS ON PRIOR CONTEXT instead of naming its own subject:
# pronouns and demonstratives standing in for an earlier noun ("both of them", "the cheaper
# one", "that"), and back-references ("the one I selected", "the same"). Deliberately NOT
# domain nouns — "flat", "unit", "generator", "daycare" are subjects a turn states outright,
# and must never trigger the carry-over. When one of these fires the utterance needs the
# carried focus folded into its search; when none does and the turn names its own subject, the
# carried unit must stay OUT of the query text or it poisons an unrelated question ("is there
# a daycare?" searched as "3 BHK … daycare" buried the amenities chunk under the cost sheets).
#
# The Hindi/Hinglish set below is the exact same idea in Roman-script transliteration — "usme"
# is "it/that" ("usme kitna square feet hai?" = "how much square feet is in it/that"), "wala"
# is the "one" in "the cheaper one" ("sasta wala"). Without these, a Hindi/Hinglish follow-up
# never matched the English-only list, so the carried subject (e.g. "3 BHK") never got folded
# in and the turn was searched broadly — the agent answered about every configuration instead
# of the one just discussed.
_ANAPHORA = re.compile(
    r"\b(?:it|its|that|those|these|them|they|one|ones|both|either|neither|same|"
    r"selected|chosen|picked|above|previous|latter|former|earlier|"
    r"usme(?:in)?|isme(?:in)?|uska|uski|uske|iska|iski|iske|voh|vo|woh|wo|"
    r"yeh|ye|vahi|wahi|wala|wali|dono|inn|unn)\b", re.I)

# Words that carry no subject signal — used only to judge whether an utterance is "subject-thin"
# (a bare follow-up like "and the price?"), which also needs the carried focus. Kept tiny and
# domain-neutral; it is a heuristic backstop to the anaphora cue, not a parser. The Hindi/
# Hinglish additions are the equivalent common function words (is/are, what/how, possessive
# particles, "also", "tell me") — without them a Hindi bare follow-up like "aur price?" counted
# every word as "content" and never registered as thin.
_THIN_NOISE = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "will", "would", "should", "have", "has", "had", "i", "you", "we", "my", "your",
    "me", "it", "to", "of", "for", "on", "in", "at", "and", "or", "if", "so", "that",
    "this", "there", "then", "what", "how", "why", "when", "which", "who", "much",
    "many", "any", "some", "get", "got", "tell", "know", "want", "need", "please",
    "about", "give", "show", "them", "both", "also", "too", "us", "am", "be",
    "hai", "hain", "ho", "tha", "thi", "ka", "ki", "ke", "ko", "se", "mein",
    "kya", "kaisa", "kaisi", "kaise", "kitna", "kitni", "kitne", "bhi", "aur",
    "hoga", "hogi", "chahiye", "bata", "batao", "bataye", "karna", "kare", "karo",
    "aap", "aapka", "aapki", "aapke", "mujhe", "hume", "humein",
})


def needs_context_anchor(user_input: str) -> bool:
    """Should the carried focus (and the standing unit preference) be folded into THIS turn's
    retrieval query text?

    Yes only when the turn does not stand on its own — it refers back with a pronoun or
    demonstrative ("give me both", "the cheaper one", "details of the one I selected"), or it
    is too thin to carry a subject at all ("and the price?"). A turn that names its own subject
    ("is a daycare available at the clubhouse?", "are bricks used?") returns False and is
    searched on exactly what the caller said — so the call's standing unit never contaminates a
    question that was never about the unit. Pure string work; the focus_terms RANKING signal is
    applied regardless (it can only reorder, never evict), so a False here still disambiguates
    variants when a cost sheet is genuinely in the window."""
    text = (user_input or "").strip()
    if not text:
        return False
    if _ANAPHORA.search(text):
        return True
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    content = [w for w in words if len(w) > 2 and w not in _THIN_NOISE]
    # At most one real content word → a bare follow-up ("and the price?", "ok") that leans on
    # the carried subject. The threshold is deliberately tight: a two-content-word turn like
    # "how many dog parks" or "2 bhk for sale" names its own subject and must be searched clean,
    # so anaphora (above) is the primary signal and this is only the ultra-thin backstop.
    return len(content) <= 1


def contextual_query(active_entities: dict | None, user_input: str) -> str:
    """The utterance with the carried focus prepended. Pure string work, no model call."""
    low = (user_input or "").lower()
    carried = []
    for slot, raw in (active_entities or {}).items():
        if str(slot).strip().lower() in NON_SUBJECT_SLOTS:
            continue
        val = _clean(raw)
        # Skip anything they just said themselves — repeating it would only skew the
        # ranking towards a term that is already weighted. And skip a value already
        # carried, since the model may name the same thing twice under two keys.
        if val and val.lower() not in low and val not in carried:
            carried.append(val)
    if not carried:
        return user_input or ""
    return f"{' '.join(carried)} {user_input or ''}".strip()


def focus_terms(active_entities: dict | None) -> list[str]:
    """The subject values, for ranking rather than for the query string.

    Same values contextual_query prepends, but handed to retrieval separately. A term
    inside the query competes with every other word in it; as a ranking signal it is a
    direct statement of which variant the customer means — which is what settles a
    0.992-versus-1.000 race between the right chunk and the wrong one.
    """
    out = []
    for slot, raw in (active_entities or {}).items():
        if str(slot).strip().lower() in NON_SUBJECT_SLOTS:
            continue
        val = _clean(raw)
        if val and val not in out:
            out.append(val)
    return out