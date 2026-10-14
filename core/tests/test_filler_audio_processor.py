"""FillerAudioProcessor in front of a real pipecat TTSService (no network).

A cached filler is replayed as TTSStarted -> TTSText -> audio -> TTSStopped and its speak
frame is swallowed, so the live TTS must synthesise nothing. The TTSTextFrame (needed for
the transcript's bot-tts-text) travels through the TTS service too, and pipecat speaks any
TextFrame unless skip_tts is set: before that flag, every cached filler was spoken twice --
the cached clip, then the same line again from live TTS.

    python -m pytest tests/test_filler_audio_processor.py -q
"""
import asyncio

import pytest

pytest.importorskip("pipecat")

from pipecat.frames.frames import (  # noqa: E402
    TTSAudioRawFrame, TTSSpeakFrame, TTSStartedFrame, TTSStoppedFrame, TTSTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.services.tts_service import TTSService  # noqa: E402
from pipecat.tests.utils import run_test  # noqa: E402

from app.engine.voice.filler_audio import FillerAudioCache, FillerAudioProcessor, _normalize  # noqa: E402

FILLER = "Let me check that for you..."


class RecordingTTS(TTSService):
    """A TTS service that records what it was asked to speak and returns no audio."""

    def __init__(self):
        super().__init__(sample_rate=16000)
        self.spoken: list[str] = []

    def can_generate_metrics(self) -> bool:
        return False

    async def run_tts(self, text: str, context_id: str):
        self.spoken.append(text)
        if False:  # an async generator that yields nothing
            yield None


def _cache_with(phrase: str) -> FillerAudioCache:
    cache = FillerAudioCache()
    cache._entries = {_normalize(phrase): b"\x00\x01" * 1600}   # 100 ms of 16 kHz PCM
    cache._labels = {_normalize(phrase): phrase}
    cache._built = True
    return cache


def _run(frames):
    tts = RecordingTTS()
    pipeline = Pipeline([FillerAudioProcessor(_cache_with(FILLER)), tts])
    down, _ = asyncio.run(run_test(pipeline, frames_to_send=frames))
    return tts, down


def test_cached_filler_is_not_spoken_again_by_live_tts():
    tts, down = _run([TTSSpeakFrame(FILLER)])
    assert tts.spoken == []
    texts = [f for f in down if isinstance(f, TTSTextFrame)]
    assert [f.text for f in texts] == [FILLER]          # still reaches the transcript, once
    assert any(isinstance(f, TTSAudioRawFrame) for f in down)
    assert any(isinstance(f, TTSStartedFrame) for f in down)
    assert any(isinstance(f, TTSStoppedFrame) for f in down)


def test_uncached_line_still_goes_to_live_tts():
    tts, _ = _run([TTSSpeakFrame("Your visit is confirmed for Wednesday.")])
    assert tts.spoken == ["Your visit is confirmed for Wednesday."]
