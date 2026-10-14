"""Tests for the P3.07 output-buffer ("wire") timing in app/engine/voice/ttfa.py.

Two layers:

  * TurnTiming state machine, on a fake clock -- deterministic, covers every skip path
    (bot already speaking, interruption, superseded turn, disabled) so a wire sample is
    never attributed to the wrong turn.
  * TTFAProbe in front of pipecat's REAL BaseOutputTransport (only the socket write is
    stubbed) -- proves the assumption the whole measurement rests on: the transport's UPSTREAM
    BotStartedSpeakingFrame reaches the probe within a millisecond of the first packet write. It also proves the measurement can SEE buffering when there is some: TTS audio
    trickled in 10 ms frames must show ~30 ms of hold in a 40 ms packet, while one large
    frame shows ~0.

Needs pipecat installed (ttfa.py imports it), i.e. run inside the core container:

    docker exec ai-voice-agent-core-1 sh -c "cd /app/core && python -m pytest tests/test_ttfa_wire.py -q"
"""
import asyncio
import logging
import sys
import time
from pathlib import Path

import pytest

# core/ on the path so `import app...` resolves the same way the app does at runtime.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.voice import ttfa  # noqa: E402
from app.engine.voice.ttfa import TTFAProbe, TTFAStats, TurnTiming  # noqa: E402


class _FakeClock:
    """Stands in for the `time` module inside ttfa.py; only monotonic() is used there."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def advance_ms(self, ms):
        self.now += ms / 1000


@pytest.fixture
def clock(monkeypatch):
    c = _FakeClock()
    monkeypatch.setattr(ttfa, "time", c)
    return c


@pytest.fixture
def stats(monkeypatch):
    # Fresh process-global collector per test so samples never leak between tests.
    s = TTFAStats()
    monkeypatch.setattr(ttfa, "_STATS", s)
    return s


def _wire_lines(caplog):
    return [r.getMessage() for r in caplog.records if "voice TTFA wire (" in r.getMessage()]


def _skip_lines(caplog):
    return [r.getMessage() for r in caplog.records if "voice TTFA wire skipped" in r.getMessage()]


# --- TurnTiming: the happy path ----------------------------------------------------------------

def test_wire_sample_measures_hold_between_first_audio_and_first_packet(clock, stats, caplog):
    caplog.set_level(logging.DEBUG, logger=ttfa.__name__)
    t = TurnTiming("conv-1", out_chunk_ms=40)
    t.start_turn("kal ka slot")
    t.note_source_if_first(ttfa.SOURCE_RESPONDER)
    clock.advance_ms(800)
    t.mark_first_audio()           # ttfa_ms = 800
    clock.advance_ms(12)
    t.on_bot_started_speaking()    # first packet out 12 ms later

    [line] = _wire_lines(caplog)
    assert "ttfa_ms=800" in line
    assert "wire_ttfa_ms=812" in line
    assert "out_buffer_ms=12" in line
    assert "out_chunk_ms=40" in line
    assert "source=responder" in line
    assert stats.describe()["responder"]["out_buffer"]["p50_ms"] == 12


def test_only_first_packet_of_a_turn_counts(clock, stats, caplog):
    # Filler, then a pause long enough for the bot to stop, then the responder: the second
    # "started speaking" belongs to the same turn but is NOT its first audio.
    caplog.set_level(logging.INFO, logger=ttfa.__name__)
    t = TurnTiming("conv-1")
    t.start_turn("book it")
    t.note_source_if_first(ttfa.SOURCE_FILLER_CACHE)
    clock.advance_ms(50)
    t.mark_first_audio()
    t.on_bot_started_speaking()
    t.on_bot_stopped_speaking()
    clock.advance_ms(900)
    t.mark_first_audio()           # idempotent within the turn
    t.on_bot_started_speaking()

    assert len(_wire_lines(caplog)) == 1
    assert stats.describe()["filler_cache"]["out_buffer"]["count"] == 1


# --- TurnTiming: every skip path yields NO sample, never a wrong one ---------------------------

def test_skip_when_bot_still_speaking_from_earlier_utterance(clock, stats, caplog):
    caplog.set_level(logging.DEBUG, logger=ttfa.__name__)
    t = TurnTiming("conv-1")
    t.on_bot_started_speaking()    # e.g. greeting still playing
    t.start_turn("hello")
    clock.advance_ms(300)
    t.mark_first_audio()           # queued behind audio already playing
    t.on_bot_started_speaking()    # (would not happen in pipecat, but must not count if it did)

    assert _wire_lines(caplog) == []
    assert "reason=bot_already_speaking" in _skip_lines(caplog)[0]
    assert "out_buffer" not in stats.describe()["unknown"]
    # ttfa_ms itself is still recorded exactly as before P3.07.
    assert stats.describe()["unknown"]["count"] == 1


def test_skip_when_interrupted_before_first_packet(clock, stats, caplog):
    caplog.set_level(logging.DEBUG, logger=ttfa.__name__)
    t = TurnTiming("conv-1")
    t.start_turn("book it")
    t.mark_first_audio()
    t.on_interruption()            # queued audio discarded, never goes out
    clock.advance_ms(2000)
    t.on_bot_started_speaking()    # some later audio -- must NOT be paired with this turn

    assert _wire_lines(caplog) == []
    assert "reason=interrupted" in _skip_lines(caplog)[0]


def test_interruption_with_nothing_pending_is_silent(clock, stats, caplog):
    caplog.set_level(logging.DEBUG, logger=ttfa.__name__)
    t = TurnTiming("conv-1")
    t.on_interruption()            # caller barge-in before any turn: nothing to skip
    assert _skip_lines(caplog) == []


def test_pending_measurement_superseded_by_next_turn(clock, stats, caplog):
    caplog.set_level(logging.DEBUG, logger=ttfa.__name__)
    t = TurnTiming("conv-1")
    t.start_turn("first")
    t.mark_first_audio()
    t.start_turn("second")         # first turn's packet never went out
    assert "reason=superseded_by_next_turn" in _skip_lines(caplog)[0]

    clock.advance_ms(400)
    t.mark_first_audio()
    clock.advance_ms(3)
    t.on_bot_started_speaking()
    [line] = _wire_lines(caplog)
    assert "ttfa_ms=400" in line and "out_buffer_ms=3" in line


def test_greeting_before_any_turn_records_nothing(clock, stats, caplog):
    caplog.set_level(logging.DEBUG, logger=ttfa.__name__)
    t = TurnTiming("conv-1")
    t.on_bot_started_speaking()
    t.on_bot_stopped_speaking()
    assert _wire_lines(caplog) == [] and _skip_lines(caplog) == []
    assert stats.describe() == {}


def test_disabled_timing_records_nothing(clock, stats, caplog):
    caplog.set_level(logging.DEBUG, logger=ttfa.__name__)
    t = TurnTiming("conv-1", enabled=False)
    t.start_turn("x")
    t.mark_first_audio()
    t.on_bot_started_speaking()
    assert _wire_lines(caplog) == []
    assert stats.describe() == {}


# --- TTFAStats: backward-compatible shape ------------------------------------------------------

def test_describe_keeps_existing_keys_and_nests_out_buffer():
    s = TTFAStats()
    for ms in (100, 200, 300):
        s.record("responder", ms)
    s.record("filler_cache", 50)
    s.record_out_buffer("responder", 0)
    s.record_out_buffer("responder", 2)

    d = s.describe()
    # Pre-P3.07 keys still present with the same meaning.
    assert {"count", "min_ms", "p50_ms", "avg_ms", "max_ms"} <= d["responder"].keys()
    assert d["responder"]["p50_ms"] == 200
    assert d["responder"]["out_buffer"] == {"count": 2, "min_ms": 0, "p50_ms": 2, "p95_ms": 2,
                                            "avg_ms": 1, "max_ms": 2}
    # A source without wire samples gets no out_buffer key at all.
    assert "out_buffer" not in d["filler_cache"]


# --- Integration: real pipecat output transport ------------------------------------------------

def _run_through_real_output_transport(audio_frames_ms: list[int], gap_ms: int):
    """Push TTS audio through [TTFAProbe -> real BaseOutputTransport] and return
    (first socket-write time, probe's measured out_buffer_ms, probe's first-audio time).

    Only write_audio_frame (the socket send) is stubbed; chunking, the audio queue and the
    BotStartedSpeaking announcement are pipecat's own code, unchanged."""
    from pipecat.frames.frames import TTSAudioRawFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.tests.utils import SleepFrame, run_test
    from pipecat.transports.base_output import BaseOutputTransport
    from pipecat.transports.base_transport import TransportParams

    sample_rate = 16000
    first_write_at: list[float] = []

    class _RecordingOutput(BaseOutputTransport):
        async def start(self, frame):
            await super().start(frame)
            await self.set_transport_ready(frame)

        async def write_audio_frame(self, frame):
            first_write_at.append(time.monotonic())
            return True

    marks: dict[str, float] = {}

    class _SpyTiming(TurnTiming):
        # Capture the exact instants the probe acted on, to compare with the real write.
        def mark_first_audio(self):
            marks.setdefault("first_audio", time.monotonic())
            super().mark_first_audio()

        def on_bot_started_speaking(self):
            marks.setdefault("bot_started", time.monotonic())
            super().on_bot_started_speaking()

    timing = _SpyTiming("it-conv", out_chunk_ms=40)
    timing.start_turn("integration")
    output = _RecordingOutput(TransportParams(audio_out_enabled=True,
                                              audio_out_sample_rate=sample_rate,
                                              audio_out_10ms_chunks=4))
    frames = []
    for i, ms in enumerate(audio_frames_ms):
        if i:
            frames.append(SleepFrame(sleep=gap_ms / 1000))
        silence = bytes(int(sample_rate * ms / 1000) * 2)
        frames.append(TTSAudioRawFrame(audio=silence, sample_rate=sample_rate, num_channels=1))
    frames.append(SleepFrame(sleep=0.2))   # let the audio task drain before EndFrame

    asyncio.run(run_test(Pipeline([TTFAProbe(timing), output]), frames_to_send=frames))
    assert first_write_at, "transport never wrote audio"
    assert "bot_started" in marks, "BotStartedSpeakingFrame never reached the probe upstream"
    return first_write_at[0], marks


def test_real_transport_announces_before_first_write_and_large_frame_is_not_held():
    # One 200 ms TTS frame (typical ElevenLabs chunk size) >= one 40 ms packet: the first
    # packet can go out immediately, so the hold should be ~0.
    first_write, marks = _run_through_real_output_transport([200], gap_ms=0)
    # The property the wire timing relies on: the moment the probe hears "started speaking" is
    # effectively the moment of the first socket write. pipecat announces just before writing,
    # but the frame reaches the probe through a queue, so the probe observes it ~0.2 ms AFTER
    # the write -- hence a small tolerance either side rather than a strict ordering.
    assert abs(marks["bot_started"] - first_write) < 0.005
    assert (marks["bot_started"] - marks["first_audio"]) * 1000 < 20


def test_real_transport_hold_is_visible_when_tts_trickles_small_frames():
    # Four 10 ms frames, 10 ms apart: pipecat cannot emit a 40 ms packet until the 4th arrives,
    # so the first audio is held ~30 ms. The measurement must SEE that -- otherwise a ~0
    # reading on real calls would prove nothing.
    _, marks = _run_through_real_output_transport([10, 10, 10, 10], gap_ms=10)
    held_ms = (marks["bot_started"] - marks["first_audio"]) * 1000
    assert 20 <= held_ms < 120, held_ms
