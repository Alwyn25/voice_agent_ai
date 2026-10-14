"""Voice channel entry point.

A browser mic (and, later, a telephony media stream — see the TwilioFrameSerializer note
below) connects here. Pipecat owns the call itself: capturing audio, running it through
Deepgram, speaking the reply through ElevenLabs, turn-taking and interruptions.
TurnBridge (app.engine.voice.bridge) is the one seam back into process_turn — the same
function /sessions/{id}/turns already calls for a chat turn. Nothing about the engine's
rules, tools, retrieval or postconditions changes because the words arrived spoken.

Authentication matches app.api.events_ws: a token query parameter, since a browser cannot
set an Authorization header on a WebSocket handshake.

STT/TTS providers are chosen by STT_PROVIDER/TTS_PROVIDER (app/config.py), each defaulting
to the cloud engine (deepgram/elevenlabs) production runs on. Setting either to the local
alternative (whisper/piper) needs no API key and costs nothing, at the price of noticeably
worse latency and quality -- meant for development and testing only. The socket refuses the
call outright if a selected cloud provider's key is missing, rather than accepting a
connection that can only fail once real audio arrives.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import jwt as pyjwt
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from app.api.sessions import _default_greeting, with_recording_disclosure
from app.config import get_settings
from app.db.base import get_session_factory
from app.demo.voice import (DemoVoiceRefused, authenticate_demo_voice, finish_demo_call,
                            looks_like_demo_token)
from app.demo.voice import opening_line as demo_opening_line
from app.db.models import Call, Conversation, Customer, Message, Tenant, User
from app.engine.memory import working
from app.engine.memory.pipeline import run_post_call_task
from app.engine.planner import outbound_flow
from app.engine.voice.bridge import TurnBridge
from app.engine.voice.lexicon import build_call_lexicon
from app.security.auth import decode_token
from app.services.outbound import place_outbound_call

router = APIRouter(tags=["voice"])
logger = logging.getLogger(__name__)

# core/.voice_models -- gitignored. Piper auto-downloads its voice model on first use;
# without an explicit download_dir it would default to the process cwd, which in local dev
# is this repo's bind-mounted core/ directory, and the .onnx file would land in the working
# tree by accident.
_PIPER_MODELS_DIR = Path(__file__).resolve().parents[2] / ".voice_models"

# How many words a caller needs to say, while the bot is mid-reply, before it counts as an
# actual interruption rather than a backchannel ("okay", "hm", "go on"). Below this while the
# bot is silent, one word is always enough -- see MinWordsUserTurnStartStrategy below for the
# interruption side and TurnBridge (app/engine/voice/bridge.py) for the matching turn-side
# check; both need the SAME number; kept here once, shared, rather than duplicated.
_BACKCHANNEL_MIN_WORDS = 3
# A demo call's clock starts at the agent's first audio, or after this long without any.
_DEMO_FIRST_AUDIO_GRACE_S = 20


async def _build_stt(settings, keyterms: list[str] | None = None):
    """deepgram (default, production) or whisper (local, no key, dev/test only --
    noticeably higher latency since faster-whisper transcribes per VAD-detected segment
    rather than Deepgram's true streaming).

    ``keyterms`` (app/engine/voice/keyterms.py) boosts recognition of this call's own
    tenant's proper nouns -- project names, site locations -- confirmed via a real
    Deepgram connection test to work alongside language="multi"/model="nova-3" (the
    older ``keywords`` parameter does not: HTTP 400). Whisper has no equivalent, so this
    is a no-op there -- the dev/test fallback path is unaffected either way.
    """
    if settings.stt_provider == "whisper":
        from pipecat.services.whisper.stt import WhisperSTTService
        # WhisperSTTService.__init__ downloads (first run) and loads the model
        # synchronously -- on this app's single event loop, calling it directly here
        # froze the ENTIRE server, every route, for as long as the load took, not just
        # this one call. asyncio.to_thread keeps that blocking work off the loop.
        return await asyncio.to_thread(
            WhisperSTTService, settings=WhisperSTTService.Settings(model=settings.whisper_model))
    from pipecat.services.deepgram.stt import DeepgramSTTService
    return DeepgramSTTService(
        api_key=settings.deepgram_api_key,
        settings=DeepgramSTTService.Settings(
            model=settings.deepgram_model,
            language=settings.deepgram_language,
            endpointing=settings.deepgram_endpointing_ms,
            smart_format=settings.deepgram_smart_format,
            punctuate=settings.deepgram_punctuate,
            numerals=settings.deepgram_numerals,
            **({"keyterm": keyterms} if keyterms else {})))


async def _build_tts(settings):
    """elevenlabs (default, production) or piper (local, no key, dev/test only --
    noticeably lower voice quality). Note: the piper-tts package is GPL-3.0 licensed."""
    if settings.tts_provider == "piper":
        from pipecat.services.piper.tts import PiperTTSService
        _PIPER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        # Same reasoning as WhisperSTTService above: the voice-model download/load in
        # __init__ is blocking and must not run on the event loop directly.
        return await asyncio.to_thread(
            PiperTTSService, download_dir=_PIPER_MODELS_DIR,
            settings=PiperTTSService.Settings(voice=settings.piper_voice))
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
    return ElevenLabsTTSService(
        api_key=settings.elevenlabs_api_key,
        settings=ElevenLabsTTSService.Settings(
            model=settings.elevenlabs_model,
            voice=settings.elevenlabs_voice_id or None,
            stability=settings.elevenlabs_stability,
            similarity_boost=settings.elevenlabs_similarity_boost,
            style=settings.elevenlabs_style,
            use_speaker_boost=settings.elevenlabs_use_speaker_boost))


async def _authenticate(token: str) -> tuple[uuid.UUID | None, str | None]:
    try:
        payload = decode_token(token, expected_type="access")
    except pyjwt.PyJWTError:
        return None, None
    async with get_session_factory()() as db:
        user = (await db.execute(select(User).where(
            User.id == uuid.UUID(payload["sub"])))).scalar_one_or_none()
        if user is None or not user.is_active:
            return None, None
        return user.tenant_id, user.email


@router.websocket("/ws/voice")
async def voice_call(websocket: WebSocket, token: str = Query(...),
                     customer_id: uuid.UUID | None = Query(None),
                     direction: str = Query("inbound"),
                     purpose: str = Query("marketing")):
    settings = get_settings()
    if settings.stt_provider == "deepgram" and not settings.deepgram_api_key:
        await websocket.close(code=4500)
        return
    if settings.tts_provider == "elevenlabs" and not settings.elevenlabs_api_key:
        await websocket.close(code=4500)
        return

    if direction not in ("inbound", "outbound"):
        await websocket.close(code=4400)
        return

    # GTM demo links (docs/gtm-demo-links.md §8.3): a prospect's browser connects with a
    # short-lived, single-use demo voice token instead of a staff login. It always runs
    # inbound, on the conversation its demo session prepared, with no debug envelope.
    demo_call = None
    if looks_like_demo_token(token):
        async with get_session_factory()() as db:
            try:
                demo_call = await authenticate_demo_voice(db, token, websocket.headers.get("origin"))
                await db.commit()
            except DemoVoiceRefused as e:
                logger.info("demo voice refused (%s): %s", e.code, e.reason)
                await websocket.close(code=e.code)
                return
        tenant_id, actor, direction, customer_id = demo_call.tenant_id, "demo", "inbound", None
    else:
        tenant_id, actor = await _authenticate(token)
        if tenant_id is None:
            await websocket.close(code=4401)
            return

    # Imported here, not at module level: these pull in pipecat's full dependency tree
    # (~heavy), and every other route in this app should keep starting up even if voice's
    # dependencies are unavailable for some reason.
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineWorker
    from pipecat.serializers.protobuf import ProtobufFrameSerializer
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
    from pipecat.workers.runner import WorkerRunner

    async with get_session_factory()() as db:
        customer = None
        if customer_id:
            customer = (await db.execute(select(Customer).where(
                Customer.tenant_id == tenant_id, Customer.id == customer_id))).scalar_one_or_none()

        # Outbound goes through the SAME dial gate a campaign or a real call would:
        # place_outbound_call runs the DNC / calling-hours / open-escalation rules and, when
        # they pass, writes the queued Call row. Voice must not be a way around that gate, so
        # a blocked dial refuses the handshake (close before accept) rather than opening a
        # call it should never have placed.
        call_id = None
        if direction == "outbound":
            if customer is None:
                # Compliance is evaluated against a customer record (DNC, calling hours,
                # attempt caps all live there), so an unknown number has nothing to check.
                await websocket.close(code=4400)
                return
            decision = await place_outbound_call(db, tenant_id=tenant_id, customer=customer,
                                                 purpose=purpose, actor=actor or "voice")
            if not decision.allowed:
                await db.commit()   # keep the call_blocked analytics event
                logger.info("outbound voice refused by policy (customer=%s rule=%s)",
                            customer_id, decision.blocked_by)
                await websocket.close(code=4403)
                return
            call_id = decision.call_id

        await websocket.accept()

        if demo_call is not None:
            # Prepared by POST /public/demo/{token}/session: the sample lead, the conversation
            # and the demo greeting already exist.
            conv = await db.get(Conversation, demo_call.conversation_id)
            conv.channel, conv.started_at = "voice", datetime.now(timezone.utc)
        else:
            conv = Conversation(tenant_id=tenant_id, customer_id=customer.id if customer else None,
                                channel="voice", direction=direction, status="active",
                                started_at=datetime.now(timezone.utc))
            db.add(conv)
        await db.flush()
        # Plain value, safe to read even after a rollback has expired `conv` (see close-out).
        conv_id = conv.id

        tenant = (await db.execute(select(Tenant).where(Tenant.id == tenant_id))).scalar_one_or_none()

        # The queued Call becomes an answered one the moment the media stream is up. The
        # browser mic stands in for the carrier here, exactly as the chat harness does in
        # app/api/sessions.py; the real telephony provider will set this from its own
        # 'answered' webhook in a later stage.
        call = None
        if call_id is not None:
            call = (await db.execute(select(Call).where(Call.id == call_id))).scalar_one()
            call.conversation_id = conv.id
            call.status = "answered"
            call.answered_at = datetime.now(timezone.utc)
        else:
            # Inbound (and any other path that didn't go through the outbound dial gate
            # above) has no pre-existing Call row to attach to. Without one, this session
            # would show up in Conversation/Message (chat history) but never on the Calls
            # page, which reads only from the calls table. provider="browser_test" marks it
            # as a browser-mic test call at a glance, distinct from a real carrier call
            # (plivo/twilio) -- everything below (finalize, duration, cost roll-ups) treats
            # it identically to an outbound call's Call row since `call` is never None again.
            # customer_id comes from the conversation: the same customer on every other path,
            # and for a GTM demo call the sample lead its demo session prepared.
            call = Call(tenant_id=tenant_id, customer_id=conv.customer_id,
                       conversation_id=conv.id, provider="browser_test", purpose=purpose,
                       status="answered", answered_at=datetime.now(timezone.utc))
            db.add(call)

        # Inbound opens with the standard greeting; outbound opens by confirming it reached
        # the right person (voice-pipeline spec §3.5). Either way the agent speaks first, so
        # a live line is never silently waiting for the other side to talk.
        if demo_call is not None:
            opening = await demo_opening_line(db, conv.id)      # already stored, just spoken
        else:
            opening = (outbound_flow.identity_opener(
                           getattr(tenant, "name", None), getattr(customer, "full_name", None))
                       if direction == "outbound"
                       else _default_greeting(getattr(tenant, "name", None), customer))
            # The tenant's recording disclosure, spoken before the caller says anything.
            opening = with_recording_disclosure(opening, tenant)
            db.add(Message(tenant_id=tenant_id, conversation_id=conv.id, role="agent",
                           content=opening))

        await db.commit()
        # The call lexicon (app/engine/voice/lexicon.py): every term this call's caller is
        # likely to say -- the tenant's proper nouns, the pack's vocabulary and the action
        # phrases derived from it, the tenant's own intents. Built once per call, used twice:
        # Deepgram's keyterm boost below, and (via working memory) the per-turn check for
        # words still misheard (app/engine/voice/asr_hints.py). Best-effort throughout.
        lexicon = await build_call_lexicon(db, tenant_id)
        await working.save(conv.id, {"asr_lexicon": lexicon.terms()})
        logger.info("voice call started (conversation=%s tenant=%s direction=%s)",
                    conv.id, tenant_id, direction)

        # serializer must match the client: @pipecat-ai/websocket-transport's
        # WebSocketTransport defaults to protobuf framing, not raw PCM, so the server has to
        # ask for the same thing or the two sides cannot decode each other's audio at all.
        # Sample rate likewise has to match the client's recorderSampleRate/playerSampleRate.
        #
        # audio_out_10ms_chunks is deliberately left at pipecat's default (4 -> 40 ms output
        # packets). That packet is the ONLY output-side buffer on this path -- this is a
        # WebSocket transport, not WebRTC/Daily, so there is no jitter buffer to tune, and the
        # browser player (@pipecat-ai/websocket-transport) starts on the first 128 samples with
        # no pre-roll. Task P3.07's 40 ms target is therefore already the default; TTFAProbe's
        # wire timing (app/engine/voice/ttfa.py, "voice TTFA wire" log) measures what that
        # packet actually costs per turn, as out_buffer_ms.
        transport_params = FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True,
                                                  add_wav_header=False,
                                                  serializer=ProtobufFrameSerializer(),
                                                  audio_in_sample_rate=16000,
                                                  audio_out_sample_rate=16000)
        transport = FastAPIWebsocketTransport(websocket=websocket, params=transport_params)
        # Keyterms come from the call lexicon built above -- see app/engine/voice/keyterms.py
        # for why they are fetched per-call rather than hardcoded anywhere. Empty for a tenant
        # with nothing on file yet, which is a normal, harmless case -- no boosting.
        stt = await _build_stt(settings, lexicon.keyterms())
        tts = await _build_tts(settings)
        # Smart Turn (config.py's smart_turn_enabled, default off): a small local ML model,
        # bundled with pipecat, that judges from the raw audio whether an utterance sounds
        # complete -- see bridge.py's _handle_vad_stopped_speaking/_handle_final_transcript
        # for what it's used for and why it lives in TurnBridge rather than pipecat's own
        # UserTurnStrategies (that path exists too, but only drives barge-in interruption
        # timing, not when a turn reaches the planner -- verified by reading TurnBridge's
        # actual trigger, which is the finalized transcript alone).
        #
        # Effective VAD silence window (stop_secs), lowest number = most aggressive. Computed
        # once, HERE, because both the Smart Turn analyzer (its own internal stop_secs) and the
        # VADParams below must use the SAME value or they disagree about when silence ended:
        #   semantic_endpointing (task P2.01): 120ms  -- silence wait collapsed; Smart Turn is
        #                                                 the real end-of-turn judge (see below).
        #   smart_turn:                        350ms  -- conservative first cut, timer + semantics.
        #   neither (default):                 700ms  -- silence duration alone ends the turn.
        _endpointing_ms = (settings.semantic_endpointing_ms if settings.semantic_endpointing_enabled
                          else settings.smart_turn_endpointing_ms if settings.smart_turn_enabled
                          else settings.deepgram_endpointing_ms)
        # Build the Smart Turn analyzer when EITHER semantic endpointing or the plain smart-turn
        # mode is on. Semantic endpointing REQUIRES it: the 120ms window is only safe because the
        # analyzer, not the clock, decides whether the caller actually finished -- so enabling the
        # short window without the semantic check would just be the old "shorten stop_secs" bug
        # (mid-sentence pauses split into two turns) the 700ms default was tuned to avoid.
        smart_turn_analyzer = None
        if settings.smart_turn_enabled or settings.semantic_endpointing_enabled:
            from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
            from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
            smart_turn_analyzer = LocalSmartTurnAnalyzerV3(
                params=SmartTurnParams(stop_secs=_endpointing_ms / 1000))
        # TTFA (time-to-first-audio) tracker (P1.10 verification). One per call, shared by the
        # bridge (stamps t0 at turn start), the filler processor (attributes the first-audio
        # source) and the TTFAProbe after the TTS stage (records first audio). Lets a live call
        # be measured -- filler_cache vs filler_live vs responder -- to confirm the filler
        # actually cuts time-to-first-audio. See app/engine/voice/ttfa.py.
        from app.engine.voice.ttfa import TTFAProbe, TurnTiming
        # out_chunk_ms is read back from the params actually handed to the transport (not
        # hard-coded), so the wire log always reports the real packet size, even if pipecat's
        # default or this file's params change later.
        turn_timing = TurnTiming(conv.id, enabled=settings.ttfa_logging_enabled,
                                 out_chunk_ms=transport_params.audio_out_10ms_chunks * 10)
        bridge = TurnBridge(db=db, tenant_id=tenant_id, conversation=conv, actor=actor or "voice",
                           opening_line=opening,
                           min_words_while_bot_speaking=_BACKCHANNEL_MIN_WORDS,
                           smart_turn_analyzer=smart_turn_analyzer,
                           turn_timing=turn_timing,
                           # The active VAD silence window (stop_secs), passed in so the bridge's
                           # endpointing timing debug can report it as the single source of truth
                           # rather than re-deriving the 120/200/350/700 precedence itself.
                           endpointing_window_ms=_endpointing_ms,
                           # A prospect on a demo link never gets the per-turn debug envelope
                           # (tools run, rules fired, retrieval, cost).
                           send_debug=demo_call is None)

        # VAD runs for every provider, not just whisper. WhisperSTTService (a
        # SegmentedSTTService) needs it to know where segments start/end -- Deepgram
        # doesn't, it does its own endpointing server-side. VAD is also where
        # VADUserStartedSpeakingFrame comes from -- but that frame alone does NOT
        # interrupt anything: it's a raw activity signal, and nothing downstream reacts
        # to it directly. A caller talking over the bot was heard (VAD fired every time,
        # confirmed in logs) but never interrupted anything -- the bot talked over them
        # instead, and the caller's audio just queued up unused behind whatever the bot
        # was already saying. UserTurnProcessor below is the piece that was actually
        # missing: it is what turns "VAD noticed activity" into a real
        # StartInterruptionFrame the output transport and TTS service act on.
        #
        # VADUserStoppedSpeakingFrame does more than signal "caller went quiet" downstream --
        # DeepgramSTTService reacts to it directly by sending a Finalize RPC that forces
        # Deepgram to emit whatever transcript it currently has, immediately, regardless of
        # deepgram_endpointing_ms above. SileroVADAnalyzer's own silence threshold (stop_secs)
        # was left unset here, defaulting to 0.2s -- far shorter than the endpointing value
        # this file already tunes to 700ms. That default, not the tuned endpointing, was the
        # thing actually ending every turn: an ordinary mid-sentence breath easily exceeds
        # 0.2s of silence, so "I have [breath] booked an appointment" was forced through as
        # two separate finalized transcripts -- two unrelated turns to the planner -- no
        # matter how generous endpointing_ms was set, because VAD finalized it first. Setting
        # stop_secs to match endpointing_ms makes the forced finalize a backstop that agrees
        # with Deepgram's own silence detection instead of pre-empting it.
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        from pipecat.audio.vad.vad_analyzer import VADParams
        from pipecat.processors.audio.vad_processor import VADProcessor
        # stop_secs = _endpointing_ms computed above (semantic 120ms / smart-turn 350ms / 700ms).
        # This is the number that actually ends turns (see the VADUserStoppedSpeakingFrame note
        # above: it triggers Deepgram's Finalize RPC immediately, regardless of the STT service's
        # own deepgram_endpointing param, which stays at 700ms as a slower backstop). Dropping
        # this to 120ms for semantic endpointing is the -580ms TTFA win; the smart-turn hold-and-
        # merge in bridge.py is what stops that short window from splitting mid-sentence pauses.
        # min_volume (P1.07) rejects sub-threshold background noise as silence, independent of
        # the window length above.
        vad_params = VADParams(stop_secs=_endpointing_ms / 1000,
                               min_volume=settings.vad_min_volume)
        stages = [transport.input(),
                 VADProcessor(vad_analyzer=SileroVADAnalyzer(params=vad_params))]
        stages.append(stt)
        # The actual barge-in fix. UserTurnProcessor watches the live transcript and, the
        # moment it decides a real user turn has started, calls broadcast_interruption() --
        # which is what an InterruptionFrame reaching ElevenLabsTTSService/the output
        # transport actually means: stop talking now. Positioned after `stt` because its
        # start strategy needs the live transcript, not just the VAD signal.
        #
        # NOT pipecat's own default start strategy (VADUserTurnStartStrategy) -- that fires
        # on raw VAD activity alone, no word count or content, so a one-word backchannel
        # ("okay", "hm", "go on" -- said to mean "I'm listening, keep going", not "stop") cut
        # the bot off mid-sentence exactly as hard as an actual interruption. On a real call
        # the bot's "Would you like to" never got to finish because the caller said "Okay."
        # MinWordsUserTurnStartStrategy is pipecat's own answer to this: it requires
        # _BACKCHANNEL_MIN_WORDS spoken words to start a turn WHILE THE BOT IS SPEAKING, but
        # drops to a single word the instant the bot goes quiet -- so an interruption needs
        # the caller to actually be saying something, while a normal reply (bot already
        # silent) starts exactly as fast as before. 3 was picked because every backchannel
        # example so far ("okay", "hm", "go on", "right") is 1-2 words and a genuine
        # interruption reads as a real clause ("wait, actually—", "no hold on") by the third
        # word. This ONLY stops the bot's AUDIO from being interrupted -- it does not, and
        # cannot, stop Deepgram from finalizing a transcript for the backchannel anyway. That
        # transcript still reaches TurnBridge downstream regardless of this strategy's
        # decision, which is why TurnBridge has its own matching min-words check (same
        # constant, passed in below) before it treats a transcript as an actual turn.
        #
        # This REPLACES VADUserTurnStartStrategy rather than adding alongside it --
        # UserTurnController tries start strategies in list order and stops at the first
        # match, and the VAD frame always arrives before the transcript does, so leaving
        # both in the list would mean VAD keeps winning the race and MinWords never gets a
        # turn to gate anything.
        #
        # Its default STOP strategy is a bonus, not the reason this is here: a local
        # "smart turn" model that judges sentence completion from the audio itself, rather
        # than a fixed silence timeout. TurnBridge's OWN turn-boundary decision still keys off
        # Deepgram's own finalized transcript, not this strategy's stop verdict, so it changes
        # nothing about existing behavior -- left at pipecat's default because disabling it
        # buys nothing and a later change to use it needs no pipeline surgery.
        from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
        from pipecat.turns.user_turn_processor import UserTurnProcessor
        from pipecat.turns.user_turn_strategies import UserTurnStrategies
        stages.append(UserTurnProcessor(user_turn_strategies=UserTurnStrategies(
            start=[MinWordsUserTurnStartStrategy(
                min_words=_BACKCHANNEL_MIN_WORDS, use_interim=True)])))
        # Filler-audio cache (P1.10). Sits between the bridge (which pushes the TTSSpeakFrames)
        # and the live TTS service: a TTSSpeakFrame whose text is a known deterministic filler
        # is served from pre-generated PCM here and never reaches ElevenLabs, saving the ~280ms
        # live first-audio latency on filler turns. Every other line -- and every line at all
        # when the cache is empty/disabled -- passes straight through to `tts` unchanged. The
        # cache is the process-global one built at startup (main.py); an unbuilt cache is a safe
        # no-op (all misses). See app/engine/voice/filler_audio.py.
        from app.engine.voice.filler_audio import FillerAudioProcessor, get_filler_cache
        filler_processor = FillerAudioProcessor(
            get_filler_cache(), conversation_id=conv.id,
            chunk_ms=settings.filler_audio_chunk_ms, turn_timing=turn_timing)
        # TTFAProbe sits AFTER the TTS stage so it observes the first audio frame of every turn
        # uniformly -- cached filler audio (which passes through TTS) and live TTS audio alike.
        stages += [bridge, filler_processor, tts, TTFAProbe(turn_timing), transport.output()]
        # A demo call's clock starts when the visitor first hears the agent (see below).
        first_audio = None
        if demo_call is not None:
            from app.engine.voice.first_audio import FirstAudioSignal
            first_audio = FirstAudioSignal()
            stages.append(first_audio)
        pipeline = Pipeline(stages)
        # PipelineWorker/WorkerRunner, not the deprecated PipelineTask/PipelineRunner
        # (removed in pipecat 2.0) -- same constructor shape, just the current names.
        worker = PipelineWorker(pipeline)
        # handle_sigint=False: this runs inside the long-lived FastAPI/uvicorn process
        # alongside every other request, not as pipecat's own dedicated process — one call
        # must not try to install a process-wide signal handler.
        runner = WorkerRunner(handle_sigint=False)

        # The caller closing the tab or hanging up must end the pipeline. pipecat only fires
        # this event; without the handler the call kept running (STT/TTS sockets open, the
        # conversation never closed out) until the process restarted.
        @transport.event_handler("on_client_disconnected")
        async def _on_client_disconnected(_transport, _ws):
            await worker.cancel(reason="client disconnected")

        # A demo call is cut once the visitor's remaining time runs out, whatever the caller
        # does. The clock starts at the agent's first audio, which is also when the page starts
        # its countdown; if no audio ever plays (TTS down) it starts after a grace period so a
        # silent call can't stay open.
        demo_end = {"reason": "user_hangup"}
        demo_clock = {"started_at": None}
        demo_timer = None
        if demo_call is not None:
            async def _cut_demo_call():
                try:
                    await asyncio.wait_for(first_audio.event.wait(), _DEMO_FIRST_AUDIO_GRACE_S)
                    demo_clock["started_at"] = first_audio.at
                except asyncio.TimeoutError:
                    demo_clock["started_at"] = datetime.now(timezone.utc)
                    logger.warning("demo call: no agent audio after %ss, starting its clock "
                                   "anyway (conversation=%s)", _DEMO_FIRST_AUDIO_GRACE_S, conv.id)
                await asyncio.sleep(demo_call.max_seconds)
                demo_end["reason"] = "max_duration"
                logger.info("demo call reached its %ss limit (conversation=%s)",
                            demo_call.max_seconds, conv.id)
                # Closing from our side doesn't fire on_client_disconnected: end the pipeline too.
                await worker.cancel(reason="demo call limit")
                try:
                    await websocket.close(code=4408)
                except Exception:  # noqa: BLE001 — already closing
                    pass
            demo_timer = asyncio.create_task(_cut_demo_call())

        try:
            await runner.run(worker)
        except WebSocketDisconnect:
            pass
        except Exception:
            demo_end["reason"] = "error"
            logger.exception("voice call crashed (conversation=%s)", conv_id)
        finally:
            if demo_timer is not None:
                demo_timer.cancel()
            ended = datetime.now(timezone.utc)
            # Reload the call's rows before reading them. A barge-in rollback (bridge.py's
            # cancelled-turn cleanup) EXPIRES every ORM object on this shared session, and the
            # bridge only refreshes the Conversation it holds. Reading call.answered_at below then
            # tried a synchronous lazy load outside an awaited context -> MissingGreenlet, which
            # aborted this whole block: the call stayed "answered", the conversation "active",
            # working memory was never cleared and the post-call task never ran. Refreshing here,
            # awaited, makes every attribute read below safe whatever happened mid-call.
            for _row in (conv, call):
                if _row is not None:
                    try:
                        await db.refresh(_row)
                    except Exception:  # noqa: BLE001 — close out with what we have
                        logger.exception("voice close-out: refresh failed (conversation=%s)",
                                         conv_id)
            try:
                conv.status = "completed"
                conv.ended_at = ended
                # Close out the Call row so the call log, duration and cost roll-ups are right.
                # answered_at is always set above, for both the outbound and inbound paths.
                if call is not None:
                    call.status = "completed"
                    call.ended_at = ended
                    if call.answered_at is not None:
                        call.duration_seconds = max(
                            0, int((ended - call.answered_at).total_seconds()))
                if demo_call is not None:
                    try:
                        await finish_demo_call(db, demo_call, end_reason=demo_end["reason"],
                                               audio_started_at=demo_clock["started_at"] or first_audio.at)
                    except Exception:  # noqa: BLE001 — never block closing the call
                        logger.exception("demo session close-out failed (conversation=%s)", conv_id)
                await db.commit()
            except Exception:  # noqa: BLE001 — still clear memory and run the post-call task
                logger.exception("voice call close-out failed (conversation=%s)", conv_id)
                try:
                    await db.rollback()
                except Exception:  # noqa: BLE001
                    logger.exception("voice close-out rollback failed (conversation=%s)", conv_id)
            await working.clear(conv_id)
            logger.info("voice call ended (conversation=%s)", conv_id)
            # A real caller must never wait for memory extraction — same reasoning as
            # SessionEnd.analyse="background" in app/api/sessions.py.
            asyncio.create_task(run_post_call_task(tenant_id, conv_id))
