"""Normalise typographic punctuation before a guard regex matches AGENT-generated text.

THE BUG THIS FIXES: every deterministic guard that checks the AGENT's own drafted reply for
a contraction ("I'll", "we've", "you'd") was written with a straight ASCII apostrophe in its
regex. A live production call had the responder (OpenAI) write "Sure, we’ll email the
brochure to you shortly." with a CURLY apostrophe (U+2019, the one real prose typically
uses) -- email_claim.is_fabricated_email_claim's "(?:i|we)(?:'ll|...)" pattern silently never
matched, and the exact fabrication the guard exists to catch reached a live caller.

Caller input (STT transcription) is plain ASCII and was never at risk; this is specifically
about text an LLM wrote. Apply to any guard's input text before matching, not to the regex
patterns themselves -- changing one normalization point is far less error-prone than finding
every apostrophe in every pattern across every guard module.

Pure stdlib, self-tested: ``python -m app.engine.planner.text_normalize``.
"""
_QUOTE_MAP = {
    "‘": "'", "’": "'", "‚": "'", "′": "'",  # single quotes/apostrophes
    "“": '"', "”": '"', "„": '"', "″": '"',  # double quotes
}
_TRANSLATE = str.maketrans(_QUOTE_MAP)


def normalize_quotes(text: str) -> str:
    """Replace curly/typographic quotes and apostrophes with their ASCII equivalents."""
    return (text or "").translate(_TRANSLATE)


def _selftest() -> None:
    assert normalize_quotes("we’ll email it") == "we'll email it"
    assert normalize_quotes("the “brochure”") == 'the "brochure"'
    assert normalize_quotes("plain text") == "plain text"
    assert normalize_quotes("") == ""
    assert normalize_quotes(None) == ""
    print("text_normalize selftest: all passed")


if __name__ == "__main__":
    _selftest()
