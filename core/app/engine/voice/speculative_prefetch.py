"""Pure decision logic for P3.06 speculative retrieval prefetch.

TurnBridge (bridge.py) uses the otherwise-idle STT endpointing window to warm the retrieval
query embedding from a confident interim transcript, so the real turn's embedding is a free
in-memory cache hit. The *decision* of whether a given interim is worth embedding -- enough
words, confident enough, not a backchannel over the bot, not a repeat of one already warmed --
is factored out here as a single pure function.

Kept free of pipecat, asyncio and DB imports on purpose: the TurnBridge that calls it pulls in
pipecat's whole frame pipeline, which a unit test must not need. This mirrors
entity_continuation.transcript_looks_unfinished, whose gate logic is likewise a pure function
tested in isolation (see tests/test_speculative_prefetch.py).
"""


# Skip reasons are returned as plain strings so the bridge can log exactly WHY a given interim
# was not prefetched (P3.06 debuggability). "fire" is the one value that means "go embed this".
FIRE = "fire"
SKIP_EMPTY = "empty"
SKIP_BOT_SPEAKING = "bot_speaking"
SKIP_FEW_WORDS = "too_few_words"
SKIP_LOW_CONFIDENCE = "low_confidence"
SKIP_UNCHANGED = "unchanged"


def should_prefetch(text: str,
                    stt_confidence: float | None,
                    *,
                    normalized_key: str,
                    last_prefetched_key: str | None,
                    bot_speaking: bool,
                    min_words: int,
                    min_confidence: float) -> str:
    """Decide whether an interim transcript is worth embedding speculatively.

    Returns FIRE to embed it, or one of the SKIP_* reasons. Pure: no I/O, no side effects, so
    the caller owns launching the actual embed and recording the reason.

    The guards, in the order a skip is cheapest to detect:

    * empty / whitespace-only interim -> nothing to embed.
    * bot is speaking -> a caller talking over the bot is almost always a backchannel
      ("okay", "hm"), not a question to retrieve for; embedding it wastes a call and its
      vector would never match a real turn. Same reasoning as the backchannel gate in
      bridge.py.
    * fewer than `min_words` -> too partial for its embedding to resemble the final turn.
    * confidence below `min_confidence` -> a shaky interim will likely be revised before it
      finalises, so its embedding would miss the cache. Applied only when a confidence is
      reported: `None` means the STT provider gave none (best-effort), and word count still
      gates, so it is not treated as a failure.
    * `normalized_key` unchanged since the last prefetch -> already warmed (or in flight); the
      embedding is a pure function of the text, so re-embedding the identical key is pointless.
      This is what keeps a stream of near-identical interims from firing repeatedly.
    """
    if not (text or "").strip():
        return SKIP_EMPTY
    if bot_speaking:
        return SKIP_BOT_SPEAKING
    if len(text.split()) < min_words:
        return SKIP_FEW_WORDS
    # None = provider reported no confidence: do not block on it (word count already gated).
    if stt_confidence is not None and stt_confidence < min_confidence:
        return SKIP_LOW_CONFIDENCE
    if normalized_key and normalized_key == last_prefetched_key:
        return SKIP_UNCHANGED
    return FIRE
