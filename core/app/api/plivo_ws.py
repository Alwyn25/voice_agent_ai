"""Plivo telephony media stream WebSocket endpoint.

Plivo opens this WebSocket immediately after the Answer-URL webhook returns the
<Stream> XML directive. The path parameter {call_uuid} is the Plivo CallUUID, stored as
Call.provider_call_id. No JWT is required — the connection is carrier-initiated, secured
by the fact that the call_uuid is a random Plivo-generated identifier.

Audio format: Plivo sends/expects mulaw 8 kHz, base64-encoded in JSON frames.
Internally the pipeline uses linear PCM 16 kHz (same as the browser WebSocket path).
PlivoFrameSerializer converts at the boundary, with stateful audioop.ratecv so resampling
is continuous across chunk edges rather than restarting at each frame.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import warnings
from datetime import datetime, timezone

from fastapi import APIRouter, Path, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.api.sessions import _default_greeting, with_recording_disclosure
from app.config import get_settings
from app.db.base import get_session_factory
from app.db.models import Call, Conversation, Customer, Message, Tenant
from app.engine.memory import working
from app.engine.memory.pipeline import run_post_call_task

# audioop is deprecated in Python 3.11+ and removed in 3.13.  It ships with 3.11/3.12;
# on 3.13+ the deployment would need to pip-install audioop2 which provides the same API.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop  # type: ignore[import]

router = APIRouter(tags=["voice"])
logger = logging.getLogger(__name__)

_BACKCHANNEL_MIN_WORDS = 3


class PlivoFrameSerializer:
    """Converts between pipecat AudioRawFrames and Plivo's JSON WebSocket protocol.

    Plivo → server: {"event": "media", "media": {"payload": "<base64 mulaw 8kHz>"}}
    Server → Plivo: {"event": "playAudio",
                     "media": {"contentType": "audio/x-mulaw;rate=8000",
                               "payload": "<base64 mulaw 8kHz>"}}
    """

    def __init__(self) -> None:
        from pipecat.serializers.base_serializer import FrameSerializerType
        self.type = FrameSerializerType.TEXT
        self._in_state: object = None    # ratecv state for 8 kHz → 16 kHz (inbound)
        self._out_state: object = None   # ratecv state for 16 kHz → 8 kHz (outbound)

    def serialize(self, frame: object) -> str | None:
        from pipecat.frames.frames import AudioRawFrame
        if not isinstance(frame, AudioRawFrame):
            return None
        pcm_8k, self._out_state = audioop.ratecv(
            frame.audio, 2, 1, 16000, 8000, self._out_state)
        ulaw = audioop.lin2ulaw(pcm_8k, 2)
        return json.dumps({
            "event": "playAudio",
            "media": {
                "contentType": "audio/x-mulaw;rate=8000",
                "payload": base64.b64encode(ulaw).decode(),
            },
        })

    def deserialize(self, data: str | bytes) -> object | None:
        from pipecat.frames.frames import AudioRawFrame
        try:
            msg = json.loads(data)
        except Exception:
            return None
        if msg.get("event") != "media":
            return None
        media = msg.get("media") or {}
        payload = media.get("payload", "")
        if not payload:
            return None
        ulaw = base64.b64decode(payload)
        pcm_8k = audioop.ulaw2lin(ulaw, 2)
        pcm_16k, self._in_state = audioop.ratecv(
            pcm_8k, 2, 1, 8000, 16000, self._in_state)
        return AudioRawFrame(audio=pcm_16k, sample_rate=16000, num_channels=1)


@router.websocket("/ws/plivo/{call_uuid}")
async def plivo_media_stream(websocket: WebSocket, call_uuid: str = Path(...)):
    """Handle Plivo's bidirectional audio stream for an outbound telephony call.

    Path: /ws/plivo/{call_uuid}  where call_uuid == Plivo CallUUID == Call.provider_call_id
    """
    settings = get_settings()
    if settings.stt_provider == "deepgram" and not settings.deepgram_api_key:
        await websocket.close(code=4500)
        return
    if settings.tts_provider == "elevenlabs" and not settings.elevenlabs_api_key:
        await websocket.close(code=4500)
        return

    # Lazy pipecat imports — same pattern as voice_ws.py so this module does not pull the
    # full dependency tree at import time and block unrelated routes from starting.
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.transports.websocket.fastapi import (FastAPIWebsocketParams,
                                                       FastAPIWebsocketTransport)
    from pipecat.workers.runner import WorkerRunner

    async with get_session_factory()() as db:
        call = (await db.execute(
            select(Call).where(
                Call.provider == "plivo",
                Call.provider_call_id == call_uuid,
            )
        )).scalar_one_or_none()

        if call is None:
            logger.warning("plivo ws: no call found for uuid=%s", call_uuid)
            await websocket.close(code=4404)
            return

        tenant_id = call.tenant_id
        tenant = (await db.execute(
            select(Tenant).where(Tenant.id == tenant_id)
        )).scalar_one_or_none()

        customer = None
        if call.customer_id:
            customer = (await db.execute(
                select(Customer).where(Customer.id == call.customer_id)
            )).scalar_one_or_none()

        await websocket.accept()

        conv = Conversation(
            tenant_id=tenant_id,
            customer_id=call.customer_id,
            channel="voice",
            direction="outbound",
            status="active",
            started_at=datetime.now(timezone.utc),
        )
        db.add(conv)
        await db.flush()
        # Plain value, safe to read even after a rollback has expired `conv` (see close-out).
        conv_id = conv.id

        call.conversation_id = conv.id
        if not call.answered_at:
            call.answered_at = datetime.now(timezone.utc)

        # The tenant's recording disclosure, spoken before the caller says anything.
        opening = with_recording_disclosure(
            _default_greeting(getattr(tenant, "name", None), customer), tenant)
        db.add(Message(
            tenant_id=tenant_id, conversation_id=conv.id, role="agent", content=opening))

        await db.commit()
        # Same call lexicon the browser voice path builds (app/engine/voice/lexicon.py): STT
        # keyterm boosting plus the per-turn misheard-word check. This telephony path used to
        # pass Deepgram no keyterms at all -- not even the tenant's own project names.
        from app.engine.voice.lexicon import build_call_lexicon
        lexicon = await build_call_lexicon(db, tenant_id)
        await working.save(conv.id, {"asr_lexicon": lexicon.terms()})
        logger.info("plivo call started (call=%s conversation=%s tenant=%s)",
                    call_uuid, conv.id, tenant_id)

        from app.api.voice_ws import _build_stt, _build_tts
        from app.engine.voice.bridge import TurnBridge
        from app.engine.voice.filler_audio import FillerAudioProcessor, get_filler_cache
        from app.engine.voice.ttfa import TTFAProbe, TurnTiming
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.processors.audio.vad_processor import VADProcessor
        from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
        from pipecat.turns.user_turn_processor import UserTurnProcessor
        from pipecat.turns.user_turn_strategies import UserTurnStrategies

        stt = await _build_stt(settings, lexicon.keyterms())
        tts = await _build_tts(settings)

        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                serializer=PlivoFrameSerializer(),
                audio_in_sample_rate=16000,
                audio_out_sample_rate=16000))

        _endpointing_ms = settings.deepgram_endpointing_ms
        turn_timing = TurnTiming(conv.id, enabled=settings.ttfa_logging_enabled)
        bridge = TurnBridge(
            db=db, tenant_id=tenant_id, conversation=conv, actor="plivo",
            opening_line=opening,
            min_words_while_bot_speaking=_BACKCHANNEL_MIN_WORDS,
            smart_turn_analyzer=None,
            turn_timing=turn_timing,
            endpointing_window_ms=_endpointing_ms)

        vad_params = VADParams(
            stop_secs=_endpointing_ms / 1000,
            min_volume=settings.vad_min_volume)

        stages = [
            transport.input(),
            VADProcessor(vad_analyzer=SileroVADAnalyzer(params=vad_params)),
            stt,
            UserTurnProcessor(user_turn_strategies=UserTurnStrategies(
                start=[MinWordsUserTurnStartStrategy(
                    min_words=_BACKCHANNEL_MIN_WORDS, use_interim=True)])),
            bridge,
            FillerAudioProcessor(
                get_filler_cache(), conversation_id=conv.id,
                chunk_ms=settings.filler_audio_chunk_ms, turn_timing=turn_timing),
            tts,
            TTFAProbe(turn_timing),
            transport.output(),
        ]

        pipeline = Pipeline(stages)
        worker = PipelineWorker(pipeline)
        runner = WorkerRunner(handle_sigint=False)

        # The carrier closing the stream (caller hung up) must end the pipeline -- pipecat only
        # fires this event. Without it the call kept running (STT/TTS sockets open, the
        # conversation never closed out), same as voice_ws.py found on the browser path.
        @transport.event_handler("on_client_disconnected")
        async def _on_client_disconnected(_transport, _ws):
            await worker.cancel(reason="client disconnected")

        try:
            await runner.run(worker)
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("plivo call crashed (call=%s conversation=%s)", call_uuid, conv_id)
        finally:
            # Close out the CONVERSATION only. The Call row is deliberately left alone: its
            # lifecycle belongs to apply_call_event (app/services/telephony/events.py, "nothing
            # else mutates Call.status"), driven by Plivo's own hangup webhook, which carries the
            # authoritative duration, cost and hangup cause and runs the retry/campaign side
            # effects. That path is idempotent on terminal state, so marking the call completed
            # here first would make it IGNORE the real webhook -- cost, hangup cause and campaign
            # progress lost. (voice_ws.py's browser path closes its Call itself only because no
            # carrier webhook ever arrives for a browser_test call.)
            #
            # Refreshed first: a barge-in rollback (bridge.py) expires every ORM object on this
            # shared session, and a plain attribute read would then lazy-load outside an awaited
            # context and raise MissingGreenlet, aborting the whole close-out.
            try:
                await db.refresh(conv)
            except Exception:  # noqa: BLE001 — close out with what we have
                logger.exception("plivo close-out: refresh failed (conversation=%s)", conv_id)
            try:
                conv.status = "completed"
                conv.ended_at = datetime.now(timezone.utc)
                await db.commit()
            except Exception:  # noqa: BLE001 — still clear memory and run the post-call task
                logger.exception("plivo call close-out failed (conversation=%s)", conv_id)
                try:
                    await db.rollback()
                except Exception:  # noqa: BLE001
                    logger.exception("plivo close-out rollback failed (conversation=%s)", conv_id)
            await working.clear(conv_id)
            logger.info("plivo call ended (call=%s conversation=%s)", call_uuid, conv_id)
            # Memory extraction never holds up the line -- same as voice_ws.py.
            asyncio.create_task(run_post_call_task(tenant_id, conv_id))
