"""Phone normalization to E.164. Default country: India (+91)."""
import re

# A real Indian mobile number is 10 digits starting 6-9 -- landlines, short codes and
# placeholder strings like "1234567890" don't match this and shouldn't become a customer
# record. Deliberately not applied inside normalize_phone(): that function also formats
# numbers on lookup/matching paths (an existing call, a console test session) where a
# weird number must still resolve to *something* rather than throw.
INDIAN_MOBILE_RE = re.compile(r"^[6-9]\d{9}$")


def normalize_phone(raw: str, default_country: str = "91") -> str:
    digits = re.sub(r"[^\d+]", "", raw or "")
    if digits.startswith("+"):
        return "+" + re.sub(r"\D", "", digits)
    digits = re.sub(r"\D", "", digits)
    if len(digits) == 10:
        return f"+{default_country}{digits}"
    if digits.startswith("0") and len(digits) == 11:
        return f"+{default_country}{digits[1:]}"
    if digits.startswith(default_country) and len(digits) == 12:
        return f"+{digits}"
    return f"+{digits}"


_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    # Hindi/Hinglish digit words (Roman script) -- a caller reading a phone number aloud
    # switches language mid-number as often as mid-sentence; common spelling variants
    # included since STT/transliteration is not consistent turn to turn. Some of these
    # (do/char/teen) are also ordinary English words -- acceptable here the same way "one"
    # already was: this function is only ever called in a context already known to be a
    # phone-number reading, never on free-form text.
    "shunya": "0", "sunya": "0",
    "ek": "1",
    "do": "2",
    "teen": "3", "tin": "3",
    "char": "4", "chaar": "4",
    "paanch": "5", "panch": "5",
    "chhe": "6", "che": "6", "chhah": "6",
    "saat": "7",
    "aath": "8", "aaath": "8",
    "nau": "9",
}
_REPEAT_WORDS = {"double": 2, "triple": 3, "treble": 3}


def extract_spoken_digits(text: str) -> str:
    """Pull every digit out of free text the way a caller reads a phone number aloud: bare
    digit characters ("8 1 4 1"), digit words ("eight one four one"), or the "double X"/
    "triple X" shorthand for a repeated digit ("double one" -> "11").

    Deterministic on purpose. A real call had the planner LLM extract this pattern inline,
    and it was inconsistent turn to turn within the SAME call: it correctly expanded
    "double 1" to "11" once, then to a single "1" a few turns later once the surrounding
    sentence had more digits in it. A digit is either there or it isn't; there is no
    judgment call for a model to make, so this never asks one.
    """
    tokens = re.findall(r"[a-zA-Z]+|\d+", text or "")
    digits = []
    i = 0
    while i < len(tokens):
        word = tokens[i].lower()
        repeat = _REPEAT_WORDS.get(word)
        if repeat and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            digit = nxt if (nxt.isdigit() and len(nxt) == 1) else _DIGIT_WORDS.get(nxt.lower())
            if digit:
                digits.append(digit * repeat)
                i += 2
                continue
        if word.isdigit():
            digits.append(word)
        elif word in _DIGIT_WORDS:
            digits.append(_DIGIT_WORDS[word])
        i += 1
    return "".join(digits)


def is_valid_indian_mobile(raw: str) -> bool:
    """True if raw looks like a real Indian mobile number, in any of the forms a customer
    or a CSV row might supply it: bare 10-digit, a leading 0, or a +91/91 prefix."""
    digits = re.sub(r"\D", "", raw or "")
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]
    return bool(INDIAN_MOBILE_RE.match(digits))
