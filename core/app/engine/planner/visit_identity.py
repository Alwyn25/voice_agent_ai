"""Gate: a returning caller asking about an existing visit is asked to confirm their name
BEFORE we read their booking status back to them.

WHY THIS EXISTS. An inbound caller is matched to a customer record by caller ID at connect
time — that is why the greeting already says "Hello abc". Booking status is then read ONLY
against that matched record (turn_context.load_upcoming_appointments, keyed on customer_id).
When that record has no live visit, the agent flatly said "there isn't a visit booked" and
pivoted to offering a new one — even though the caller plainly had one, just under a
slightly different record, or the caller ID matched the wrong person entirely. A caller who
knows they booked a visit hears their own booking denied.

THE POLICY (operator decision). For a returning caller, ALWAYS ask for the name first, then
answer from THEIR OWN record — no cross-record search. The name is a soft identity
confirmation: if it matches the record we hold, we answer the status honestly (which may
still be "no visit"); if it does not, the caller ID matched the wrong record and we must not
read that record's booking back as theirs.

This module is pure and self-tested (``python -m app.engine.planner.visit_identity``). The
wiring in turn_runner supplies the turn's routing flags, the matched customer, and the name
the caller just spoke; this module decides only whether to gate and what to say.
"""
import re

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Status-question intents the planner may hand us, in addition to the fast pre-pass flag. A
# status turn is one asking about the state of the caller's OWN existing booking — never a
# request to make a new one, which is a booking turn and must not be gated behind a name.
_STATUS_INTENTS = ("booking_status", "appointment_status", "visit_status")


def is_visit_status_turn(routing: dict | None, intent: str | None) -> bool:
    """True when this turn asks about the caller's own existing visit/booking.

    Read primarily from the fast understanding pre-pass (``is_status_question``); the planner's
    own intent label is a secondary signal for the case where the pre-pass was skipped."""
    if routing and bool(routing.get("is_status_question")):
        return True
    return str(intent or "").strip().lower() in _STATUS_INTENTS


_EXISTING_VISIT_RE = re.compile(
    r"\b(my|our)\s+(site\s+)?(visit|appointment|appt|booking|slot|meeting)\b"
    r"|\bi\s+(had|have|booked|made|got|scheduled)\b[^.?!]{0,20}"
    r"\b(visit|appointment|appt|booking|slot|meeting)\b"
    r"|\b(visit|appointment|booking|slot|meeting)\b[^.?!]{0,20}"
    r"\b(scheduled|booked|assigned|confirmed)\b"
    # "who is assigned to my visit" — direct assignment verb
    r"|\bwho(?:'s|s| is| will be)?\s+(?:assigned|handling|coming|meeting|taking|showing)\b"
    # P1-F12: "who is the salesperson for tomorrow's visit?" — salesperson noun phrase.
    # "who is the salesperson/agent/rep" doesn't contain an assignment verb, so the old
    # pattern missed it. Matching on the noun phrase after "who is [the/my]".
    r"|\bwho(?:'s|s| is| will be)?\s+(?:(?:the|my|our)\s+)?(?:assign\w*\s+)?(?:"
    r"sales\s*-?\s*person|salesperson|sales\s*rep\w*|sales\s*exec\w*|"
    r"sales\s*-?\s*agent|sales\s*team\s*member|"
    r"(?:assigned\s+)?(?:agent|representative|advisor|executive|rep)\b)",
    re.I)


def mentions_existing_visit(text: str | None) -> bool:
    """Deterministic signal that the caller is asking about an EXISTING visit of their own —
    "my visit", "I had a visit scheduled", "who is assigned" — as opposed to requesting a new
    one ("book a visit tomorrow").

    This is the signal that does not depend on which tool the planner happened to run. The
    gate's other triggers fail exactly when the planner mis-routes a returning caller's status
    question: it runs locate_centre toward a NEW booking (so my_appointments never ran) or
    labels the intent as a product question (so is_status_question is false), and the caller's
    own visit is then denied without ever confirming who they are. Reading the caller's words
    directly makes the always-ask policy hold regardless of that routing."""
    return bool(_EXISTING_VISIT_RE.search(text or ""))


def should_ask_name(*, routing: dict | None, intent: str | None,
                    ran_my_appointments: bool, is_write_turn: bool,
                    asked_about_visit: bool = False) -> bool:
    """Whether to gate this turn behind a name confirmation.

    The fast classifier's ``is_status_question`` is the primary signal, but it misreads a
    question like "who is the sales person on my visit?" as a contact lookup and leaves the
    flag false. So two more deterministic signals back it up: the planner ran
    ``my_appointments`` (it read the turn as about the caller's own bookings, whatever it
    labelled the intent), and ``asked_about_visit`` — the caller's own words name an existing
    visit even when the planner ran no lookup at all (or routed toward a new booking).

    ``is_write_turn`` short-circuits all three: a caller booking, rescheduling or cancelling
    is ACTING on the visit, not asking about it, and must never be interrupted to recite a
    name. It is read from the caller's actual command (a book/reschedule/cancel phrase or an
    executed write), NOT from speculative scheduling state — a phantom reschedule spun up from
    a status question must not suppress the gate."""
    if is_write_turn:
        return False
    return (is_visit_status_turn(routing, intent) or bool(ran_my_appointments)
            or bool(asked_about_visit))


_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def _romanize_word(word: str) -> str:
    """One Devanagari word to a Roman approximation good enough for name matching (not a
    general-purpose translation).

    ITRANS is a Sanskrit-oriented scheme: every bare consonant carries an implicit trailing
    'a' unless another vowel is explicitly marked, so "रमेश" -> "ramesha" and "राहुल" ->
    "rAhula" — grammatically consistent, but not how a caller spells "Ramesh" or "Rahul" in
    Roman script; that inherent 'a' is silent in spoken/written Hindi names. Stripping ONE
    trailing lowercase 'a' undoes exactly that — and only that. An explicit long-A matra
    (ITRANS capitalizes it, "प्रिया" -> "priyA") is a real, pronounced vowel and is left
    alone, as are explicit short i/u endings ("अंजलि" -> "amjali") which are written vowels
    in the original, not the inherent one this scheme invents.
    """
    out = transliterate(word, sanscript.DEVANAGARI, sanscript.ITRANS)
    if out and out[-1] == "a":
        out = out[:-1]
    return out.lower()


# Punctuation that rides along with a spoken or typed name — a period in initials ("R.K."),
# a hyphen ("Al-Rashid"), a slash ("Rahul/Kumar"), an apostrophe ("D'Souza"), stray quotes or
# brackets from STT/formatting — and must NEVER decide whether a visit is theirs. Stripped on
# BOTH sides (the caller's spoken name AND the stored record) so the two compare in one space.
_NAME_PUNCT_RE = re.compile(r"[.,\-/'’`\"“”()\[\]{}|_\\]+")


def _clean(value: str | None) -> str:
    """Lowercase, romanize Devanagari, and remove name punctuation — leaving only words. The
    single normalization every name (given or on-record) passes through before it is compared,
    so `.`, `-`, `/`, `'` and the like can never make an otherwise-identical name mismatch."""
    # Punctuation → space (a separator) FIRST, so a Devanagari word never reaches the
    # transliterator with a period or hyphen glued to it, and initials like "R.K." split cleanly.
    text = _NAME_PUNCT_RE.sub(" ", value or "")
    if _DEVANAGARI_RE.search(text):
        text = " ".join(_romanize_word(w) if _DEVANAGARI_RE.search(w) else w
                        for w in text.split())
    return text.lower()


def _tokens(value: str | None) -> list[str]:
    """Lowercased alphabetic name tokens, order preserved. Honorifics and filler that ride
    along with a spoken name ("mr", "this is") are dropped so they never count as a match, and
    name punctuation (see _clean) is removed first so it never splits or corrupts a token.

    Devanagari words are romanized first, so a name given in either script compares in the
    same space. Until this, a Devanagari value tokenized to nothing at all —
    re.findall(r"[a-z]+", ...) matches zero characters in Devanagari text — so
    names_match() below hit its `if not g or not r: return False` guard immediately and a
    returning Hindi caller's own name ("रमेश", from an earlier Hindi call) could never match
    their own record, in EITHER script: not against itself in Devanagari, and not against
    "Ramesh" typed in a later chat session.
    """
    raw = re.findall(r"[a-z]+", _clean(value))
    return [t for t in raw if t not in _NAME_NOISE]


_NAME_NOISE = {"mr", "mrs", "ms", "dr", "the", "this", "is", "im", "name", "my"}


_PHONETIC_SUBS = [
    # K/C interchange — the single most common Indian-name STT error. Deepgram transcribes
    # "Kapil" as "Capel" / "Capelle". Normalise initial C → K so both land in the same
    # phonetic space. Only at the word start (the most reliable match point for given names).
    (re.compile(r"^c"), "k"),
    # ph → f (aspiration: "Phoolan" → "Foolan")
    (re.compile(r"ph"), "f"),
    # double consonant → single ("Kapelle" → "Kapele", "Subbash" → "Subash")
    (re.compile(r"([bcdfghjklmnpqrstvwxyz])\1"), r"\1"),
    # strip trailing weak syllables STT silently appends to Indian names:
    # -le/-el/-il/-al ("Capelle"→"kap", "Capel"→"kap", "Kapil"→"kap")
    (re.compile(r"(le|el|il|al)$"), ""),
    # strip any remaining trailing vowel (e.g. "kape" after removing -le from "kapele")
    (re.compile(r"[aeiou]$"), ""),
]


def _phonetic(tok: str) -> str:
    """Phonetic reduction for Indian name STT noise (K/C, aspirates, double letters,
    trailing weak syllables). Returns a stemmed form used ONLY as a secondary fallback
    when the normal token comparison already failed — never replaces it."""
    t = tok
    for pattern, repl in _PHONETIC_SUBS:
        t = pattern.sub(repl, t)
    return t


def names_match(given: str | None, on_record: str | None) -> bool:
    """A soft match between the name the caller spoke and the name on their record.

    Deliberately lenient — a caller says their first name ("Rahul") against a record of
    "Rahul Kumar", or the full name against a first-name record. Matches when the first
    spoken token agrees, or when one token set is contained in the other. A bare shared
    surname is NOT enough on its own, so "Priya Kumar" does not pass for "Rahul Kumar".

    Case- and punctuation-insensitive by construction: both names pass through _clean/_tokens,
    which lowercase, romanize Devanagari and strip name punctuation. A final collapsed-form
    check (all separators removed) rescues names that differ ONLY in how their parts were split
    — "R.K." vs "R K" vs "RK", "Al-Rashid" vs "Alrashid" — using equality, never substring, so
    unrelated names ("an" vs "anil") can never collide.

    A phonetic fallback (_phonetic) then catches the K/C + trailing-syllable class of STT
    errors: "Kapil" transcribed as "Capelle" or "Capel" passes because both reduce to "kap"
    after the normaliser runs."""
    g = _tokens(given)
    r = _tokens(on_record)
    if not g or not r:
        return False
    if g[0] == r[0]:
        return True
    gs, rs = set(g), set(r)
    if gs <= rs or rs <= gs:
        return True
    # Same name, differently split: compare the whole thing with every separator removed.
    if "".join(g) == "".join(r):
        return True
    # Phonetic fallback: reduce each first token and compare. Guards against the common
    # Indian-name ASR error class (K→C, double letters, trailing -le/-el/-il/-a/-e) where
    # exact and collapsed-form checks both fail. Only the FIRST token (the given name) is
    # compared — a shared phonetic stem of a SURNAME is never enough on its own.
    if len(g[0]) >= 3 and len(r[0]) >= 3 and _phonetic(g[0]) == _phonetic(r[0]):
        return True
    return False


# ---- the lines the caller hears ---------------------------------------------------------

def ask_for_name() -> str:
    """Turn A: defer the status answer and ask for the name to pull the visit up under."""
    return ("Of course — may I have the name the visit is booked under, so I can pull it up "
            "for you?")


def _first(name: str | None) -> str:
    toks = _tokens(name)
    return toks[0].capitalize() if toks else ""


# The caller's visit question is specifically about WHO is assigned — a sales person / agent
# / representative coming to the visit. The appointment record does not store an assignee, so
# this is answered honestly ("not listed") rather than by inventing a name.
_ASSIGNEE_RE = re.compile(
    r"\b(sales\s*-?\s*person|salesperson|sales\s*-?\s*rep\w*|\bagent\b|representative|"
    r"executive|assigned\s+to|who(?:'s|s| is| will be)?\s+"
    r"(?:assigned|handling|coming|meeting|taking|showing))\b", re.I)


def asks_about_assignee(text: str | None) -> bool:
    """True when the caller is asking who is assigned to their visit (sales person / agent).

    That fact is not on the appointment record, so the confirmed reply says so plainly. Kept
    deterministic and unit-tested — it decides only how honestly to phrase the answer, never
    whether to gate."""
    return bool(_ASSIGNEE_RE.search(text or ""))


def confirmed_reply(given: str, *, has_visit: bool, when: str | None,
                    asked_assignee: bool = False) -> str:
    """Turn B, name matches the record: answer the status honestly from THEIR record."""
    hi = _first(given)
    lead = f"Thanks, {hi}. " if hi else "Thank you. "
    if has_visit:
        when_part = f" for {when}" if when else ""
        if asked_assignee:
            # The visit is real, but the sales person assigned to it is not stored on the
            # record. Say so honestly and offer the team follow-up — never invent a name.
            return (f"{lead}Yes — I can see your visit on record{when_part}. The specific "
                    "sales person assigned isn't listed in my system, but I'll have the team "
                    "confirm and reach out. Is there anything else I can help with?")
        # Confirm the visit exists and invite the specific follow-up, which the normal flow
        # then answers now that identity is confirmed.
        return (f"{lead}Yes — I can see your visit on record{when_part}. "
                "What would you like to know about it?")
    return (f"{lead}I've checked your record and I don't see a visit booked at the moment. "
            "Would you like me to arrange one?")


def mismatch_reply(given: str | None) -> str:
    """Turn B, the spoken name does NOT match the record on this number. We will not read a
    booking that may not be theirs. Offers the last-4-digits check as the alternative — not
    "try the name again", since a name that was mis-heard once is often mis-heard the same
    way twice, and digits are the more reliable thing for STT to get right."""
    who = _first(given)
    tail = f" under {who}" if who else ""
    return (f"I'm not finding a visit{tail} on this number. Could you tell me the last 4 "
            "digits of the mobile number it's booked under instead?")


def _selftest() -> None:
    assert is_visit_status_turn({"is_status_question": True}, None)
    assert is_visit_status_turn({}, "booking_status")
    assert not is_visit_status_turn({"is_status_question": False}, "book_appointment")
    assert not is_visit_status_turn(None, None)

    # should_ask_name: the my_appointments signal catches what the classifier missed
    # ("who is the sales person on my visit?"), but a write turn is never gated.
    assert should_ask_name(routing={"is_status_question": True}, intent=None,
                           ran_my_appointments=False, is_write_turn=False)
    assert should_ask_name(routing={"is_status_question": False}, intent="product_question",
                           ran_my_appointments=True, is_write_turn=False)
    assert not should_ask_name(routing={"is_status_question": True}, intent="reschedule",
                              ran_my_appointments=True, is_write_turn=True)
    assert not should_ask_name(routing={}, intent=None,
                              ran_my_appointments=False, is_write_turn=False)
    # The caller's words alone fire the gate even when no lookup ran and the classifier was
    # silent — the case where the planner routed toward a NEW booking (locate_centre).
    assert should_ask_name(routing={"is_status_question": False}, intent="product_question",
                           ran_my_appointments=False, is_write_turn=False,
                           asked_about_visit=True)
    # ...but a genuine change command (is_write_turn) still short-circuits it.
    assert not should_ask_name(routing={}, intent=None, ran_my_appointments=False,
                               is_write_turn=True, asked_about_visit=True)

    # mentions_existing_visit: an existing visit is recognised; a NEW-booking request is not.
    assert mentions_existing_visit("i had a visit scheduled tomorrow, who is assigned?")
    assert mentions_existing_visit("what time is my appointment?")
    assert mentions_existing_visit("who is the sales person assigned to my visit")
    # P1-F12: salesperson-noun queries without an assignment verb after "who is"
    assert mentions_existing_visit("who is the salesperson for tomorrow's visit?")
    assert mentions_existing_visit("who is the agent for my appointment?")
    assert mentions_existing_visit("who is the sales rep coming to the visit?")
    assert mentions_existing_visit("who is my assigned sales executive?")
    assert not mentions_existing_visit("i want to book a visit for tomorrow")
    assert not mentions_existing_visit("can I see the 3 BHK price")

    # Matching: first-name, full-name, subset either way.
    assert names_match("Rahul", "Rahul Kumar")
    assert names_match("Rahul Kumar", "Rahul")
    assert names_match("rahul kumar", "Rahul Kumar")
    assert names_match("this is Rahul", "Rahul Kumar")
    assert names_match("abc", "abc")
    # A different first name with only a shared surname must NOT pass.
    assert not names_match("Priya Kumar", "Rahul Kumar")
    assert not names_match("Sunil", "Rahul Kumar")
    assert not names_match("", "Rahul")
    assert not names_match("Rahul", None)

    # Case + punctuation + spacing insensitivity — a visit is matched to its record however the
    # caller's name arrives (STT/formatting adds periods, hyphens, slashes, quotes, mixed case).
    assert names_match("RAHUL", "rahul kumar")                 # pure case
    assert names_match("Rahul.", "Rahul Kumar")                # trailing period
    assert names_match("rahul, kumar", "Rahul Kumar")          # comma between tokens
    assert names_match("Rahul-Kumar", "Rahul Kumar")           # hyphen as separator
    assert names_match("Rahul/Kumar", "Rahul Kumar")           # slash as separator
    assert names_match("  Mr. Rahul  Kumar ", "rahul kumar")   # honorific + stray spaces
    assert names_match("D'Souza", "dsouza")                    # apostrophe removed → one name
    assert names_match("R.K.", "R K")                          # initials, differently split
    assert names_match("R.K.", "RK")                           # initials vs joined (collapsed)
    assert names_match("Al-Rashid", "Alrashid")                # hyphen-split vs joined
    assert names_match("प्रिया.", "Priya")                       # Devanagari + punctuation
    # The collapsed fallback is equality-only, so it never lets an unrelated shorter name in.
    assert not names_match("An", "Anil")
    assert not names_match("Priya", "Rahul")

    # Phonetic fallback: K/C interchange + trailing STT syllables — the real bug that
    # exposed this: Deepgram transcribed "Kapil" as "Capelle" then "Capel" on two
    # consecutive attempts, so the visit lookup failed both times.
    assert names_match("Capelle", "Kapil")    # c→k, double-l, -le strip, -e strip → kap == kap
    assert names_match("Capel", "Kapil")      # c→k, -el strip → kap == kap
    assert not names_match("Priya", "Kapil")  # phonetic stems differ

    # Devanagari matches itself, and a Roman spelling matches the Devanagari record it came
    # from originally — the real call that exposed this bug: a customer registered as
    # "रमेश" on a Hindi voice call, then typed "Ramesh" in a later chat session.
    assert names_match("रमेश", "रमेश")
    assert names_match("Ramesh", "रमेश")
    assert names_match("रमेश", "Ramesh Kumar")
    assert names_match("राहुल", "Rahul")
    assert not names_match("Priya", "रमेश")

    assert "name" in ask_for_name().lower()
    assert "don't see" in confirmed_reply("Rahul", has_visit=False, when=None)
    m = confirmed_reply("Rahul Kumar", has_visit=True, when="tomorrow at 5:00 PM")
    assert "Rahul" in m and "tomorrow at 5:00 PM" in m
    assert "last 4" in mismatch_reply("Rahul")

    # asks_about_assignee: the sales-person / agent question is recognised; a plain
    # "is my visit still on?" is not.
    assert asks_about_assignee("who is the sales person assigned?")
    assert asks_about_assignee("which agent will be coming?")
    assert asks_about_assignee("who's handling my visit")
    assert not asks_about_assignee("is my visit still on for tomorrow?")
    assert not asks_about_assignee("what time is my appointment?")
    # The honest assignee reply confirms the visit but does not invent a name.
    a = confirmed_reply("Rahul", has_visit=True, when="tomorrow at 10:00 AM",
                        asked_assignee=True)
    assert "tomorrow at 10:00 AM" in a and "isn't listed" in a
    print("visit_identity selftest: all passed")


if __name__ == "__main__":
    _selftest()
