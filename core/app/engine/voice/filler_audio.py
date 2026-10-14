"""Pre-generated filler TTS audio, cached in memory at startup (P1.10).

Why this exists
---------------
A handful of the agent's spoken lines are *fillers*: short, non-committal
acknowledgements the engine speaks purely to cover a pause with a voice instead of
silence. The clearest example is the perceived-latency filler in turn_runner.py -- the
moment the planner decides to run a tool, it speaks "Let me check that for you..." (read-
only) or "Give me just a moment..." over the same TTS path the real reply uses, so the
1-2 second tool round-trip doesn't read as a dropped line.

These lines are *deterministic*: the exact same words, every time, chosen by code rather
than written by the LLM. Yet every filler turn still pays the full ~280ms ElevenLabs
first-audio latency to synthesise words that never change. The Sept-2026 benchmark put
filler turns at ~25% of all turns and TTS first-audio at ~280ms, making this ~280ms of
avoidable latency on a quarter of all turns.

What this does
--------------
At server startup (`prewarm_filler_cache`, called from main.py's lifespan) each phrase in
`ALL_CACHED_PHRASES` is synthesised ONCE via ElevenLabs' REST API -- using the *same*
voice, model and voice-settings the live pipeline uses, so a cached clip is
indistinguishable from a live one -- and the resulting raw PCM is held in memory.

The catalog is two kinds of line (`ALL_CACHED_PHRASES` = both). The tool-preamble FILLERS
(`_FILLER_TABLE`, "Let me check that for you...") the engine emits itself before a tool runs;
and a small set of COMMON short phrases (`_COMMON_PHRASES`, "Okay.", "Theek hai.", "Thank
you!") the responder speaks verbatim on many ordinary turns. Both are cached and served the
same way; the only difference is that fillers are also the bridge's de-dup exclusion set
(`is_filler_phrase`) while common phrases are NOT (they are real reply segments). This is the
task's "pre-generate the top-N common phrases, serve the cached bytes on a match" coverage.

At call time `FillerAudioProcessor` sits in the voice pipeline just upstream of the TTS
service. When a `TTSSpeakFrame` carries text that matches a cached phrase, it replays the
cached PCM as the exact frame sequence a TTS service would emit
(`TTSStartedFrame` -> `TTSTextFrame` -> `TTSAudioRawFrame` chunks -> `TTSStoppedFrame`) and
swallows the speak frame so it never reaches the real TTS. The `TTSTextFrame` is what lets
RTVI report the line as a `bot-tts-text` event, so a cached filler shows up in a live
transcript exactly like a live one -- omitting it would play real audio the caller hears
with no matching line in the UI. Every other line passes straight through to live TTS
untouched. Nothing about correctness depends on a cache hit -- a miss, an empty cache, or a
disabled feature simply means the line is spoken live exactly as before.

Why match verbatim rather than "any filler-ish line"
----------------------------------------------------
The saving is only safe because these specific strings are emitted by *our* code, not the
model. Matching is therefore an exact, normalised equality against a small curated set --
never a fuzzy or semantic match, which could replace a subtly-different real reply with the
wrong cached clip. `_FILLER_TABLE` is the single source of truth: turn_runner.py builds its
filler by calling `filler_phrase(intent, language)` against the same table, so the emitted
text and the cached text cannot drift apart.

Language
--------
The filler is spoken in the SAME language the caller is using this turn -- English, Hindi
(Devanagari) or Hinglish (Roman) -- so it never precedes an English reply with a Hindi
"one moment" or vice-versa. turn_runner.py picks the language with the very same
`responder.detect_language(user_input)` the responder itself uses to choose the reply
language, so the filler and the reply that follows it always agree. Every language variant
of every filler is pre-generated at startup, so a Hindi or Hinglish filler is served from
cache exactly like the English one.

Audio format
------------
The voice transport (app/api/voice_ws.py) plays 16-bit signed little-endian mono PCM at
`VOICE_OUTPUT_SAMPLE_RATE` (matching its `audio_out_sample_rate`). ElevenLabs' `pcm_16000`
output format is exactly that -- headerless raw PCM -- which is also what the live
websocket TTS service pushes as `TTSAudioRawFrame` payloads, so the cached and live paths
produce byte-identical frame shapes.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from typing import Iterable, Iterator

import httpx
from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.text.base_text_aggregator import AggregationType

logger = logging.getLogger(__name__)

# The pipeline's audio_out_sample_rate (app/api/voice_ws.py). Kept here as a named constant
# because the startup build runs in main.py, far from that wiring; if voice_ws ever changes
# its output sample rate, this must change with it (and the assertion in the processor will
# flag a mismatch loudly rather than playing chipmunk/slowed audio silently).
VOICE_OUTPUT_SAMPLE_RATE = 16000

# --- Canonical filler phrases -------------------------------------------------------------
# The engine speaks a filler in TWO shapes, by what it is about to do:
#   * READ_ONLY -- it is about to LOOK SOMETHING UP (slots, price, inventory, knowledge):
#     "let me check". These map to _READ_ONLY_PIPELINE_TOOLS in turn_runner.py.
#   * GENERIC   -- it is about to DO/WRITE something (book, ticket, callback): "one moment".
FILLER_INTENT_READ_ONLY = "read_only"
FILLER_INTENT_GENERIC = "generic"

# The reply language for a turn, exactly as responder.detect_language() reports it -- so the
# filler is spoken in the SAME language the responder will then reply in. Kept as named
# constants (rather than bare strings) so the table below and detect_language cannot drift.
LANG_ENGLISH = "english"
LANG_HINDI = "hindi"        # Devanagari script
LANG_HINGLISH = "hinglish"  # Roman-script Hindi-English code-mixing
_DEFAULT_LANGUAGE = LANG_ENGLISH

# The single source of truth for every filler the engine speaks: intent -> language -> the
# EXACT string. turn_runner.py builds its filler by calling filler_phrase(intent, language)
# against this same table, so the words spoken and the words pre-generated at startup are
# literally the same object and cannot drift (a drift would silently turn every cache hit
# into a miss). Every entry must be a real, verbatim line the engine emits -- a phrase the
# LLM paraphrases would never match and only wastes a startup synthesis.
#
# The Hindi lines are Devanagari and the Hinglish lines are Roman, matching how the responder
# replies in each (prompts.py) and how the live TTS already renders each -- no number-words
# are involved here, so the Devanagari-number caveat in responder.py does not apply. Phrasing
# is short (a filler only covers a ~1-2s gap) and in the agent's own first person ("karti
# hoon" / "देखती हूँ").
_FILLER_TABLE: dict[str, dict[str, str]] = {
    FILLER_INTENT_READ_ONLY: {
        LANG_ENGLISH: "Let me check that for you...",
        LANG_HINDI: "एक सेकंड, मैं अभी देखती हूँ...",
        LANG_HINGLISH: "Ek second, main abhi check karti hoon...",
    },
    FILLER_INTENT_GENERIC: {
        LANG_ENGLISH: "Give me just a moment...",
        LANG_HINDI: "बस एक मिनट रुकिए...",
        LANG_HINGLISH: "Bas ek minute rukiye...",
    },
}


def filler_phrase(intent: str, language: str) -> str:
    """The exact filler string for (intent, language).

    Falls back defensively -- an unknown intent to GENERIC, an unknown/absent language to
    English -- so the lookup ALWAYS returns a real, pre-generated string and the caller never
    has to handle a missing entry. `language` is exactly what responder.detect_language()
    returns ("english" | "hindi" | "hinglish").
    """
    table = _FILLER_TABLE.get(intent) or _FILLER_TABLE[FILLER_INTENT_GENERIC]
    return table.get(language) or table[_DEFAULT_LANGUAGE]


# Every FILLER phrase = every value in the table, de-duplicated preserving order. This is the
# tool-preamble set specifically: it drives filler_phrase() AND is_filler_phrase() (the bridge
# de-dup exclusion). It is deliberately NOT the full pre-generation catalog -- see below.
CACHED_FILLER_PHRASES: tuple[str, ...] = tuple(dict.fromkeys(
    phrase
    for by_language in _FILLER_TABLE.values()
    for phrase in by_language.values()))

# --- Common short phrases (P1.10 "top-N" catalog) -----------------------------------------
# Beyond the tool-preamble fillers above, the agent very often speaks a handful of short,
# CLOSED-CLASS, context-independent lines verbatim -- plain acknowledgements and closings
# ("Okay.", "Theek hai.", "Thank you!"). Unlike a full reply, each of these renders the same
# way every time regardless of surrounding context, so pre-generating it once and serving the
# cached PCM whenever the responder happens to stream exactly that line is safe: it can only
# ever replace a line with its OWN identical audio. This is the task's "top-10 common filler
# responses" coverage -- the responder is never forced to emit these; the cache simply serves
# them, opportunistically, on the turns it naturally does, by the same exact-normalised match.
#
# CRITICAL -- these are NOT fillers. They are the responder's own reply segments, so they must
# remain visible to the bridge's response de-dup: is_filler_phrase() stays False for them (it
# is built from CACHED_FILLER_PHRASES only, below), and only the audio cache ever knows them.
# If a common phrase were treated as a filler, the bridge would drop it from its _prepushed
# de-dup and re-speak the real first sentence -- the exact double-speak is_filler_phrase exists
# to prevent. Keep this list conservative: include a line only when its single cached rendering
# is acceptable for every variant _normalize() folds together (it strips a trailing "?", so
# never add a word whose question intonation must sound different from its statement form).
#
# The Hindi lines are Devanagari and the Hinglish lines Roman, matching how the responder
# renders each; first-person gendered forms ("gayi"/"गई") match the agent's feminine voice,
# consistent with the fillers above.
_COMMON_PHRASES: dict[str, tuple[str, ...]] = {
    LANG_ENGLISH: ("Okay.", "Sure.", "Got it.", "Thank you!"),
    LANG_HINDI: ("ठीक है।", "जी बिल्कुल।", "समझ गई।", "धन्यवाद!"),
    LANG_HINGLISH: ("Theek hai.", "Ji bilkul.", "Samajh gayi.", "Shukriya!"),
}

# Everything pre-generated at startup: every filler + every common phrase, de-duplicated while
# preserving order. build() synthesises this full catalog; the cache can then serve a hit for
# any of them. dict.fromkeys keeps it in sync as either source grows.
ALL_CACHED_PHRASES: tuple[str, ...] = tuple(dict.fromkeys((
    *CACHED_FILLER_PHRASES,
    *(phrase for variants in _COMMON_PHRASES.values() for phrase in variants),
)))


def _normalize(text: str) -> str:
    """Fold a phrase to its cache key.

    Matching must tolerate the incidental differences between how a phrase is written in
    source and how it arrives on a frame -- surrounding whitespace, doubled spaces, case,
    and a trailing run of "...", ".", "!" etc. (the same line appears both with and without
    its trailing ellipsis across the codebase). It deliberately does NOT touch *internal*
    punctuation or words: the phrase set is tiny and curated, so there is no collision to
    resolve by stripping more, and stripping more only raises the risk of two distinct lines
    folding together and one being spoken with the other's audio.
    """
    text = (text or "").strip().casefold()
    text = re.sub(r"\s+", " ", text)
    # Trailing sentence-final punctuation / ellipsis only, from the right edge.
    text = re.sub(r"[\s.…!?]+$", "", text)
    return text


# Normalised set of every filler string the engine can emit (all intents, all languages),
# built once. Independent of whether the audio cache actually built -- a filler is a filler
# whether or not its audio was pre-generated -- because it comes from the same _FILLER_TABLE
# that both the cache and turn_runner.filler_phrase() draw from.
_FILLER_KEYS = frozenset(_normalize(p) for p in CACHED_FILLER_PHRASES)


def is_filler_phrase(text: str) -> bool:
    """True if `text` is one of the engine's deterministic filler lines (any language).

    The voice bridge (app/engine/voice/bridge.py) uses this to keep fillers OUT of its
    post-turn response-segment de-duplication. A filler is spoken to cover the pause before a
    tool result, but it is NOT one of the reply's segments; if it were compared against them,
    the bridge's "skip the sentence already streamed" check would compare the wrong entry and
    replay the real first sentence, so the caller hears it twice. This is deliberately
    independent of the audio cache: the de-dup must be correct even when filler audio caching
    is disabled and the filler is spoken via live TTS."""
    return _normalize(text) in _FILLER_KEYS


def _iter_pcm_chunks(pcm: bytes, chunk_bytes: int) -> Iterator[bytes]:
    """Yield `pcm` in `chunk_bytes`-sized pieces (the final piece may be shorter)."""
    for start in range(0, len(pcm), chunk_bytes):
        yield pcm[start:start + chunk_bytes]


class FillerAudioCache:
    """Process-wide store of pre-synthesised filler PCM, keyed by normalised phrase.

    One instance is built at startup and shared read-only by every live call (see
    `get_filler_cache`). `build` is the only writer and runs once at boot; `get` is the hot
    read on every spoken line. Both run on the single app event loop, so no locking is
    needed. A cache that failed to build, was disabled, or simply missed is never an error --
    every reader treats "no bytes" as "speak it live".
    """

    def __init__(self, sample_rate: int = VOICE_OUTPUT_SAMPLE_RATE):
        self._sample_rate = sample_rate
        # normalised phrase -> raw PCM bytes.
        self._entries: dict[str, bytes] = {}
        # Original (un-normalised) text for each key, for readable diagnostics only.
        self._labels: dict[str, str] = {}
        self._built = False
        # Live hit/miss tallies, surfaced by describe() for the health endpoint / logs so the
        # feature's real-world effectiveness is observable rather than guessed at.
        self.hits = 0
        self.misses = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def ready(self) -> bool:
        """True once at least one phrase is cached and available to serve."""
        return self._built and bool(self._entries)

    def get(self, text: str) -> bytes | None:
        """Return cached PCM for `text`, or None to fall back to live TTS.

        Records a hit or miss for observability. A miss is the normal, expected outcome for
        every non-filler line, so it is intentionally cheap and silent here (the processor
        logs a hit; misses would be far too noisy at INFO).
        """
        if not self._entries:
            return None
        pcm = self._entries.get(_normalize(text))
        if pcm is None:
            self.misses += 1
            return None
        self.hits += 1
        return pcm

    async def build(self, settings, *, phrases: Iterable[str] | None = None) -> None:
        """Synthesise every phrase once and populate the cache. Idempotent-ish: safe to call
        again, it simply rebuilds. Best-effort per phrase -- one failure never aborts the
        rest, and a wholly failed build leaves an empty (harmless) cache.

        Skipped entirely, leaving the cache empty, when the feature is off, the TTS provider
        is not ElevenLabs (only ElevenLabs can produce the matching pcm_16000 audio here), or
        the ElevenLabs credentials/voice needed to synthesise are missing.
        """
        self._built = True  # "we attempted a build"; ready still requires actual entries.

        if not getattr(settings, "filler_audio_cache_enabled", True):
            logger.info("filler audio cache disabled by config -- fillers use live TTS")
            return
        if settings.tts_provider != "elevenlabs":
            logger.info("filler audio cache skipped (tts_provider=%s, not elevenlabs)",
                        settings.tts_provider)
            return
        if not settings.elevenlabs_api_key or not settings.elevenlabs_voice_id:
            # A voice call needs these anyway; without them the feature just stays inert.
            logger.info("filler audio cache skipped (missing ElevenLabs api key or voice id)")
            return

        wanted = tuple(phrases) if phrases is not None else ALL_CACHED_PHRASES
        timeout_s = max(0.5, settings.filler_audio_build_timeout_ms / 1000)
        t0 = time.perf_counter()

        # A small concurrency cap: generating ~10 short clips in parallel keeps boot fast
        # without hammering ElevenLabs hard enough to risk a rate-limit rejection.
        sem = asyncio.Semaphore(3)
        entries: dict[str, bytes] = {}
        labels: dict[str, str] = {}

        async def _one(phrase: str) -> None:
            key = _normalize(phrase)
            if not key or key in entries:
                return  # empty or duplicate-after-normalisation -- nothing to do.
            async with sem:
                try:
                    pcm = await _synthesize_pcm(settings, phrase, self._sample_rate, timeout_s)
                except Exception:
                    # Per-phrase failure is logged and swallowed: that phrase falls back to
                    # live TTS, every other phrase still gets cached.
                    logger.warning("filler audio: failed to pre-generate %r -- will use live TTS",
                                   phrase, exc_info=True)
                    return
                if not pcm:
                    logger.warning("filler audio: empty audio for %r -- skipping", phrase)
                    return
                entries[key] = pcm
                labels[key] = phrase

        await asyncio.gather(*(_one(p) for p in wanted))

        self._entries = entries
        self._labels = labels
        total_bytes = sum(len(b) for b in entries.values())
        logger.info(
            "filler audio cache built: %d/%d phrases, %.1f KB, in %d ms (sample_rate=%d)",
            len(entries), len(wanted), total_bytes / 1024,
            int((time.perf_counter() - t0) * 1000), self._sample_rate)

    def describe(self) -> dict:
        """A JSON-friendly snapshot for the health endpoint / diagnostics. Lists exactly
        which phrases are cached (so a wording drift between source and cache shows up as a
        phrase that isn't listed) alongside the live hit/miss tally."""
        return {
            "ready": self.ready,
            "sample_rate": self._sample_rate,
            "phrase_count": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "phrases": [
                {"text": self._labels[k], "bytes": len(self._entries[k])}
                for k in self._entries
            ],
        }


async def _synthesize_pcm(settings, text: str, sample_rate: int, timeout_s: float) -> bytes:
    """One-shot ElevenLabs REST synthesis returning headerless 16-bit mono PCM.

    Deliberately a plain REST call, not the pipeline's streaming websocket service: this runs
    once at boot with no caller waiting, so the simplest possible request is right, and it
    keeps startup wholly independent of the live pipeline's connection state. Voice, model and
    voice-settings mirror app/api/voice_ws.py's ElevenLabsTTSService config so a cached clip
    sounds identical to a live one.
    """
    # pcm_16000 etc. -- ElevenLabs' headerless raw-PCM formats, matching the transport's
    # output sample rate. Anything else would need resampling before it could be played.
    output_format = f"pcm_{sample_rate}"
    url = (f"https://api.elevenlabs.io/v1/text-to-speech/{settings.elevenlabs_voice_id}"
           f"?output_format={output_format}")

    # Only include voice-setting keys that are actually set, matching pipecat's own
    # build_elevenlabs_voice_settings so the rendering is the same as the live service's.
    voice_settings: dict[str, float | bool] = {}
    for key, val in (("stability", settings.elevenlabs_stability),
                     ("similarity_boost", settings.elevenlabs_similarity_boost),
                     ("style", settings.elevenlabs_style),
                     ("use_speaker_boost", settings.elevenlabs_use_speaker_boost)):
        if val is not None:
            voice_settings[key] = val

    payload: dict = {"text": text, "model_id": settings.elevenlabs_model}
    if voice_settings:
        payload["voice_settings"] = voice_settings

    headers = {"xi-api-key": settings.elevenlabs_api_key, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            # Surface a short, safe slice of the body -- enough to see "voice not found" or a
            # quota error without dumping a huge response into the log.
            raise RuntimeError(
                f"ElevenLabs REST {resp.status_code}: {resp.text[:200]}")
        return resp.content


# --- Pipeline processor -------------------------------------------------------------------

class FillerAudioProcessor(FrameProcessor):
    """Serves cached filler PCM in place of a live TTS call.

    Placed in the voice pipeline immediately upstream of the TTS service (see
    app/api/voice_ws.py). It watches downstream `TTSSpeakFrame`s -- the frames the bridge
    pushes to have something spoken. On a cache hit it emits the cached audio as the same
    frame sequence a TTS service emits and does NOT forward the speak frame, so the real TTS
    never sees it and makes no network call. On a miss (the common case for real replies) it
    forwards the frame untouched and the live TTS speaks it exactly as before.

    Every other frame -- audio, control, interruption, transcripts -- passes straight through;
    this processor only ever acts on `TTSSpeakFrame`.
    """

    def __init__(self, cache: FillerAudioCache | None, *,
                 conversation_id=None, chunk_ms: int = 40, turn_timing=None):
        super().__init__()
        self._cache = cache
        self._conversation_id = conversation_id
        # Optional TTFA tracker (app/engine/voice/ttfa.py). When present, this processor
        # classifies WHAT produces each turn's first audio (cached filler / live filler /
        # responder) so the TTFAProbe downstream can attribute the measured latency. None on
        # any path that doesn't care about the measurement.
        self._turn_timing = turn_timing
        # Bytes per emitted audio frame: chunk_ms of 16-bit (2-byte) mono PCM. Forced even so
        # a frame boundary never splits a sample down the middle.
        sr = cache.sample_rate if cache is not None else VOICE_OUTPUT_SAMPLE_RATE
        self._sample_rate = sr
        chunk_bytes = int(sr * (max(1, chunk_ms) / 1000)) * 2
        self._chunk_bytes = max(2, chunk_bytes - (chunk_bytes % 2))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Contract: every pipecat FrameProcessor must call super().process_frame first.
        await super().process_frame(frame, direction)

        # Only downstream speak frames are candidates; a cache hit short-circuits live TTS.
        if isinstance(frame, TTSSpeakFrame) and direction == FrameDirection.DOWNSTREAM:
            pcm = self._cache.get(frame.text) if self._cache is not None else None
            # Attribute this turn's first-audio source for the TTFA measurement (first speak
            # frame of the turn wins -- see TurnTiming.note_source_if_first). A cache hit is a
            # win either way, but we keep the two cached sources distinct: a tool-preamble
            # FILLER served from cache (filler_cache) vs a common responder phrase served from
            # cache (phrase_cache). A filler with no cached bytes still went out via live TTS
            # (filler_live); anything else is the responder's own live speech (responder).
            if self._turn_timing is not None:
                if pcm is not None:
                    self._turn_timing.note_source_if_first(
                        "filler_cache" if is_filler_phrase(frame.text) else "phrase_cache")
                elif is_filler_phrase(frame.text):
                    self._turn_timing.note_source_if_first("filler_live")
                else:
                    self._turn_timing.note_source_if_first("responder")
            if pcm is not None:
                await self._emit_cached_audio(frame.text, pcm)
                return  # swallow: do NOT forward the speak frame to the real TTS.

        await self.push_frame(frame, direction)

    async def _emit_cached_audio(self, text: str, pcm: bytes) -> None:
        """Replay `pcm` as the frame sequence the TTS service would have produced.

        The output transport (pipecat base_output) brackets bot speech off these frames: the
        first `TTSAudioRawFrame` makes it emit `BotStartedSpeakingFrame`, and `TTSStoppedFrame`
        makes it emit `BotStoppedSpeakingFrame`. Producing exactly this sequence is what lets a
        cached filler behave -- for turn-taking, interruption and RTVI -- identically to a live
        one. A fresh context_id per utterance mirrors what the real TTS assigns.

        Also pushes a `TTSTextFrame` -- a live TTS call emits one of these alongside its audio
        (see pipecat's tts_service.py), and it is what RTVI turns into the `bot-tts-text`
        server event the console/dashboard transcript is built from. Without it here, a cache
        hit played real, audible speech that never appeared in the transcript: the caller heard
        "Let me check that for you..." but the UI showed nothing, because there was no text
        frame for the transport to report -- the audio and the transcript silently diverged.
        """
        context_id = str(uuid.uuid4())
        chunk_count = 0
        # These frames flow through the downstream TTS service, which forwards frames it did
        # not itself generate in order (via its serialization queue) and stamps them with the
        # transport destination -- so they reach the output transport correctly ordered.
        #
        # THE BUG THIS FIXES: every cached filler was spoken TWICE on a real call -- once from
        # this cache, once more live, right after. TTSTextFrame is a TextFrame subclass (via
        # AggregatedTextFrame), and the live TTS service's own process_frame dispatches ANY
        # incoming AggregatedTextFrame to _push_tts_frames -- i.e. "synthesize this" -- with no
        # check for whether the audio was already produced elsewhere. Pushing this text frame
        # downstream to reach the transport (so RTVI can report it) was mistaken by the TTS
        # service for a fresh, unspoken line to convert to speech, so it synthesized the exact
        # same phrase live immediately after the cached audio played. `skip_tts` is pipecat's
        # own escape hatch for exactly this (TTSService.process_frame checks it FIRST, before
        # the AggregatedTextFrame branch, and just forwards the frame when set) -- this is text
        # already spoken as audio, never text waiting to be spoken.
        await self.push_frame(TTSStartedFrame(context_id=context_id), FrameDirection.DOWNSTREAM)
        _text_frame = TTSTextFrame(text, aggregated_by=AggregationType.SENTENCE)
        _text_frame.skip_tts = True
        await self.push_frame(_text_frame, FrameDirection.DOWNSTREAM)
        for chunk in _iter_pcm_chunks(pcm, self._chunk_bytes):
            await self.push_frame(
                TTSAudioRawFrame(audio=chunk, sample_rate=self._sample_rate,
                                 num_channels=1, context_id=context_id),
                FrameDirection.DOWNSTREAM)
            chunk_count += 1
        await self.push_frame(TTSStoppedFrame(context_id=context_id), FrameDirection.DOWNSTREAM)

        # A hit is worth an INFO line: it is the whole point of the feature, rare enough per
        # call not to be noisy, and names the phrase + size so a hit can be confirmed in the
        # logs of a real call.
        logger.info(
            "filler audio served from cache (conversation=%s text=%r bytes=%d chunks=%d) "
            "-- skipped live TTS", self._conversation_id, text, len(pcm), chunk_count)


# --- Process-global singleton -------------------------------------------------------------
# One cache per process, built once at startup and read by every call. A module global rather
# than app state so both the startup hook (main.py) and each websocket handler (voice_ws.py)
# reach the same instance without threading it through.
_CACHE: FillerAudioCache | None = None


def get_filler_cache() -> FillerAudioCache:
    """The process-wide cache, created empty on first access. Empty is safe: every reader
    falls back to live TTS until (and unless) `prewarm_filler_cache` fills it."""
    global _CACHE
    if _CACHE is None:
        _CACHE = FillerAudioCache()
    return _CACHE


async def prewarm_filler_cache(settings) -> None:
    """Build the process-wide cache at startup. Best-effort: any failure is logged and the
    app serves normally with fillers going through live TTS. Called from main.py's lifespan,
    in the background, so it never delays readiness."""
    try:
        await get_filler_cache().build(settings)
    except Exception:
        logger.exception("filler audio prewarm failed -- fillers will use live TTS")
