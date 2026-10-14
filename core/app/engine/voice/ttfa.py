"""Time-to-first-audio (TTFA) instrumentation for voice turns (P1.10 verification).

Purpose
-------
The filler-audio cache is meant to reduce how long a caller waits, after they finish
speaking, before they hear ANY audio back. The engine's own per-turn timing (turn_runner's
`debug.timing.total_ms`) does NOT measure that -- it measures how long the whole reply takes
to compute, which the filler deliberately does not change. This module measures the thing the
filler actually improves: the moment the FIRST audio frame of a turn reaches the output.

What is measured
----------------
For each turn, `ttfa_ms` = time from when the finalized transcript is handed to the engine
(`TurnBridge._run_turn` start = t0) until the first `TTSAudioRawFrame` for that turn leaves
the TTS stage. It is tagged with the `source` that produced that first audio:

  * ``filler_cache``  -- a pre-generated tool-preamble filler was served from memory (the win)
  * ``phrase_cache``  -- a common responder phrase ("Okay.", "Theek hai.") was served from
                         memory -- the P1.10 "top-N common phrases" coverage, also a win
  * ``filler_live``   -- a filler was spoken, but via live TTS (cache miss/disabled)
  * ``responder``     -- no cache hit this turn; first audio is the responder's own live speech

Scope note: t0 is the transcript-handed-to-engine moment, so `ttfa_ms` covers the
engine + TTS portion of TTFA. It deliberately EXCLUDES the upstream VAD silence window and
STT finalize (they happen before the bridge sees the transcript) and the final network hop
to the browser. That is exactly the portion the filler affects, which is what makes this the
right yardstick for evaluating the feature. To read the full mic-to-ear number, add the
fixed upstream costs (VAD ~700 ms + STT) to this.

How to use it
-------------
Watch the per-turn lines during a live call:

    docker compose logs -f core | grep "voice TTFA"

    voice TTFA (conversation=… ttfa_ms=3120 source=filler_cache text='What are the slots?')
    voice TTFA (conversation=… ttfa_ms=5280 source=responder    text='I'm looking for a flat')

Or read the rolling per-source summary any time:

    curl -s http://localhost:8000/health/ttfa

Interpreting it for the P1.10 goal:
  * ``filler_cache`` vs ``filler_live``  -> the pure cache saving (skipping the ~280 ms live
    ElevenLabs first-audio hop on filler turns).
  * ``filler_*`` vs ``responder``        -> the bigger win: the filler puts audio in the
    caller's ear as soon as the planner decides to run a tool, instead of after the whole
    tool + responder + TTS chain finishes.

Output-buffer ("wire") timing (P3.07 verification)
--------------------------------------------------
`ttfa_ms` stops at the TTS stage, BEFORE `transport.output()` -- so any time the first audio
spends waiting in the output transport's buffer was invisible. P3.07 ("reduce jitter buffer
depth to 40 ms") is about exactly that buffer, so it needs its own number:

  * ``wire_ttfa_ms``  -- t0 until the first audio packet of the turn is handed to the socket.
  * ``out_buffer_ms`` -- ``wire_ttfa_ms - ttfa_ms``: how long the first audio sat in the
                         output transport before going out. THIS is the time P3.07 is about.

How it is observed: pipecat's output transport pushes a ``BotStartedSpeakingFrame`` UPSTREAM
immediately before it writes the first audio packet of an utterance (base_output.py:
``_handle_bot_speech`` -> ``_bot_started_speaking``, then ``write_audio_frame``). TTFAProbe
sits directly upstream of ``transport.output()``, so that frame passes through it -- no
pipecat subclassing or monkey-patching needed. The frame reaches the probe via pipecat's frame
queue, so the probe observes it a fraction of a millisecond AFTER the write (measured ~0.2 ms
against the real transport in tests/test_ttfa_wire.py) -- negligible at this resolution.

The transport here is a WebSocket (FastAPIWebsocketTransport), NOT WebRTC/Daily, so there is
no WebRTC jitter buffer. The only buffer on this path is pipecat's output packet size,
``audio_out_10ms_chunks`` x 10 ms (pipecat default 4 -> 40 ms), logged as ``out_chunk_ms``.
Expected: ``out_buffer_ms`` ~0 whenever TTS hands over >= one packet of audio at once, i.e.
the 40 ms target is already met and shrinking it further buys nothing. The numbers below are
what prove (or disprove) that on real calls:

    voice TTFA wire (conversation=… ttfa_ms=812 wire_ttfa_ms=813 out_buffer_ms=1
                     out_chunk_ms=40 source=responder)

A turn gets NO wire sample -- deliberately, never a wrong one -- when its first audio arrives
while the bot is still speaking from an earlier utterance (pipecat does not re-announce
"started speaking", so there is no packet-out moment to attribute to this turn), or when the
turn is interrupted before its audio goes out. Those skips are logged at DEBUG with a reason.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict

from pipecat.frames.frames import (BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame,
                                   InterruptionFrame, TTSAudioRawFrame)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

logger = logging.getLogger(__name__)

# What can produce the first audio of a turn (the two *_cache sources are the P1.10 wins).
SOURCE_FILLER_CACHE = "filler_cache"
SOURCE_PHRASE_CACHE = "phrase_cache"
SOURCE_FILLER_LIVE = "filler_live"
SOURCE_RESPONDER = "responder"


class TTFAStats:
    """Process-global rolling TTFA samples, grouped by source, for the health endpoint.

    Keeps only the last `max_samples` per source so a long-running process cannot grow this
    without bound; that is plenty to see a stable p50/avg during a testing session. Single
    event loop, so no locking needed."""

    def __init__(self, max_samples: int = 200):
        self._samples: dict[str, list[int]] = defaultdict(list)
        # P3.07: output-transport hold time of the first audio, per source. Kept apart from
        # `_samples` so the existing ttfa_ms numbers (and anything already reading them) are
        # untouched; surfaced as an extra nested key per source in describe().
        self._out_buffer_samples: dict[str, list[int]] = defaultdict(list)
        self._max = max_samples

    def _append(self, bucket: list[int], ms: int) -> None:
        bucket.append(ms)
        if len(bucket) > self._max:
            del bucket[0]

    def record(self, source: str, ms: int) -> None:
        self._append(self._samples[source], ms)

    def record_out_buffer(self, source: str, ms: int) -> None:
        """P3.07: how long this turn's first audio waited in the output transport."""
        self._append(self._out_buffer_samples[source], ms)

    @staticmethod
    def _summary(xs: list[int]) -> dict:
        ss = sorted(xs)
        return {
            "count": len(ss),
            "min_ms": ss[0],
            "p50_ms": ss[len(ss) // 2],
            # Nearest-rank p95; with few samples this is simply the max, which is the honest
            # answer for a small test session.
            "p95_ms": ss[min(len(ss) - 1, int(len(ss) * 0.95))],
            "avg_ms": int(sum(ss) / len(ss)),
            "max_ms": ss[-1],
        }

    def describe(self) -> dict:
        """Per-source count / min / p50 / p95 / avg / max, JSON-friendly for /health/ttfa.
        A source with P3.07 wire samples also carries an ``out_buffer`` sub-summary."""
        out: dict[str, dict] = {}
        for source, xs in self._samples.items():
            if not xs:
                continue
            out[source] = self._summary(xs)
            buffered = self._out_buffer_samples.get(source)
            if buffered:
                out[source]["out_buffer"] = self._summary(buffered)
        return out


# One collector per process, shared across all calls (see get_ttfa_stats).
_STATS = TTFAStats()


def get_ttfa_stats() -> TTFAStats:
    return _STATS


class TurnTiming:
    """Per-call TTFA tracker, shared by three collaborators in one voice pipeline:

      * the bridge calls `start_turn(text)` the instant a finalized transcript begins
        processing (t0);
      * the filler processor calls `note_source_if_first(...)` as it sees the turn's first
        spoken line, recording WHAT will produce the first audio;
      * the TTFAProbe (just after the TTS stage) calls `mark_first_audio()` when that first
        audio frame actually comes out, which logs and records the sample.

    All three run on the one event loop, in this order per turn, so no locking is needed.
    A turn that never produces audio (or audio that arrives before any turn starts, e.g. the
    opening greeting) is simply never recorded -- every method is a no-op until `start_turn`.

    P3.07 adds a fourth step, also driven by the TTFAProbe: `on_bot_started_speaking()` when
    the output transport announces it is about to put the first packet on the socket, which
    closes the turn's wire timing (see the module docstring). The rule that keeps it honest:
    a wire sample is only taken when the bot was SILENT at the moment the turn's first audio
    passed the probe -- only then is the next "started speaking" unambiguously this turn's
    first packet. Anything else is skipped (DEBUG log with the reason), never guessed.
    """

    def __init__(self, conversation_id, *, enabled: bool = True,
                 out_chunk_ms: int | None = None):
        self._conv = conversation_id
        self._enabled = enabled
        # Output packet size the transport was built with (audio_out_10ms_chunks x 10 ms).
        # Only used to label the wire log line, so a log alone says which setting it measured.
        self._out_chunk_ms = out_chunk_ms
        self._t0: float | None = None
        self._text: str | None = None
        self._source: str | None = None
        self._done = False
        # --- P3.07 wire timing -------------------------------------------------------------
        # Call-level, NOT reset per turn: whether the output transport is currently playing
        # audio, as last announced by Bot{Started,Stopped}SpeakingFrame. Tracked even before
        # the first turn, so the opening greeting's playback is accounted for.
        self._bot_speaking = False
        # Per turn: set when the first audio passed the probe while the bot was silent, i.e.
        # the next BotStartedSpeakingFrame will be this turn's first packet going out.
        self._wire_pending = False
        self._first_audio_at: float | None = None
        self._ttfa_ms: int | None = None

    def start_turn(self, text: str) -> None:
        """Mark t0 for a new turn and reset per-turn state."""
        if not self._enabled:
            return
        if self._wire_pending:
            # The previous turn's audio never reached the socket (e.g. dropped by an
            # interruption that did not pass the probe) -- say so rather than silently lose it.
            self._skip_wire("superseded_by_next_turn")
        self._t0 = time.monotonic()
        self._text = text
        self._source = None
        self._done = False
        self._first_audio_at = None
        self._ttfa_ms = None

    def note_source_if_first(self, source: str) -> None:
        """Record what produces this turn's first audio -- only the first call per turn wins,
        so the filler (pushed before the responder's own speech) is what gets attributed."""
        if self._t0 is not None and self._source is None and not self._done:
            self._source = source

    def mark_first_audio(self) -> None:
        """First audio frame of the turn is out: compute ttfa_ms, log it, record the sample.
        Idempotent per turn (only the first audio frame counts)."""
        if not self._enabled or self._t0 is None or self._done:
            return
        self._done = True
        now = time.monotonic()
        # round(), not int(): int() truncates float error (0.39999 s -> 399 ms), which also made
        # ttfa_ms + out_buffer_ms disagree with wire_ttfa_ms by 1 ms in the wire log line.
        ttfa_ms = round((now - self._t0) * 1000)
        source = self._source or "unknown"
        _STATS.record(source, ttfa_ms)
        logger.info("voice TTFA (conversation=%s ttfa_ms=%d source=%s text=%r)",
                    self._conv, ttfa_ms, source, self._text)
        # P3.07: arm the wire measurement for this turn -- unless the bot is still playing an
        # earlier utterance, in which case pipecat will not announce a new "started speaking"
        # for this audio and there is no packet-out moment that belongs to this turn.
        self._first_audio_at = now
        self._ttfa_ms = ttfa_ms
        if self._bot_speaking:
            self._skip_wire("bot_already_speaking")
        else:
            self._wire_pending = True

    # --- P3.07 wire timing ------------------------------------------------------------------

    def on_bot_started_speaking(self) -> None:
        """Output transport is writing the first packet of an utterance to the socket. If this
        turn armed a wire measurement, this is its packet-out moment: log + record it."""
        self._bot_speaking = True
        if not self._wire_pending:
            return
        self._wire_pending = False
        now = time.monotonic()
        wire_ttfa_ms = round((now - self._t0) * 1000)
        # max(0, ...) is purely defensive: monotonic time cannot go backwards, so the hold is
        # never negative, but a negative number in this log would send someone chasing a ghost.
        out_buffer_ms = max(0, round((now - self._first_audio_at) * 1000))
        source = self._source or "unknown"
        _STATS.record_out_buffer(source, out_buffer_ms)
        logger.info("voice TTFA wire (conversation=%s ttfa_ms=%d wire_ttfa_ms=%d "
                    "out_buffer_ms=%d out_chunk_ms=%s source=%s)",
                    self._conv, self._ttfa_ms, wire_ttfa_ms, out_buffer_ms,
                    self._out_chunk_ms, source)

    def on_bot_stopped_speaking(self) -> None:
        """Output transport finished (or was cut off from) playing an utterance."""
        self._bot_speaking = False

    def on_interruption(self) -> None:
        """Caller barged in: queued output audio is discarded, so an armed turn's first packet
        will never go out -- drop the measurement instead of pairing it with a later packet."""
        if self._wire_pending:
            self._skip_wire("interrupted")

    def _skip_wire(self, reason: str) -> None:
        self._wire_pending = False
        logger.debug("voice TTFA wire skipped (conversation=%s reason=%s source=%s)",
                     self._conv, reason, self._source or "unknown")


class TTFAProbe(FrameProcessor):
    """Observes the first audio frame of each turn to time TTFA. Placed immediately AFTER the
    TTS stage so it sees EVERY turn's first audio uniformly -- both cached filler audio
    (emitted upstream, which passes through the TTS service) and live TTS audio (produced by
    the TTS service itself). It only observes; every frame is forwarded unchanged.

    P3.07: must ALSO sit immediately BEFORE ``transport.output()`` -- that adjacency is what
    lets it see the output transport's UPSTREAM Bot{Started,Stopped}SpeakingFrames, which time
    the first packet actually leaving the server (see the module docstring)."""

    def __init__(self, turn_timing: TurnTiming):
        super().__init__()
        self._timing = turn_timing

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Contract: every pipecat FrameProcessor must call super().process_frame first.
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, TTSAudioRawFrame):
                self._timing.mark_first_audio()
            elif isinstance(frame, InterruptionFrame):
                self._timing.on_interruption()
        # The output transport pushes these both ways; only the UPSTREAM copy passes through
        # this processor (the downstream copy leaves the pipeline's far end), so no double count.
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._timing.on_bot_started_speaking()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._timing.on_bot_stopped_speaking()
        await self.push_frame(frame, direction)
