"""Signal the moment the caller first hears the agent.

A GTM demo call's time limit (app/demo/limits.py) runs from the agent's first audio, not
from the button press: connecting, the handshake and synthesising the greeting aren't the
visitor's time. The output transport pushes BotStartedSpeakingFrame downstream as it starts
playing a turn's audio, so this sits after transport.output() and fires once on the first.
The browser starts its countdown on the same event (RTVI BotStartedSpeaking), so the page's
timer and the server's cut-off agree.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from pipecat.frames.frames import BotStartedSpeakingFrame, Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class FirstAudioSignal(FrameProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.event = asyncio.Event()
        self.at: datetime | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame) and not self.event.is_set():
            self.at = datetime.now(timezone.utc)
            self.event.set()
        await self.push_frame(frame, direction)
