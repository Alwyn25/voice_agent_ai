"""The one seam between Pipecat's frame pipeline and this engine's own planner.

Pipecat owns everything about a live voice call — capturing audio, running it through
Deepgram, streaming the reply through ElevenLabs, turn-taking and interruptions. None of
that is where the agent's actual behaviour lives. TurnBridge is deliberately thin: a
finalized transcription comes in, process_turn() (the exact function a chat turn already
calls through /sessions/{id}/turns) decides what happens, and the reply text goes out as
something ElevenLabs can speak. Every rule, tool, retrieval and postcondition already
built runs identically whether the words arrived typed or spoken.

Every other frame (audio, interruption/turn-taking control frames, interim
transcriptions) passes through untouched — Pipecat's own machinery depends on those
reaching every processor in the pipeline.
"""
import asyncio
import logging
import time
from datetime import date

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Conversation
from app.engine.memory import working
from app.engine.planner.turn_runner import TurnCancelledAfterCommit, process_turn
from app.engine.voice import barge_in
from app.engine.voice.entity_continuation import (
    looks_like_partial_entity,
    transcript_looks_unfinished,
)
from app.engine.voice.filler_audio import is_filler_phrase
from app.services.spoken_form import to_spoken_form

logger = logging.getLogger(__name__)


class TurnBridge(FrameProcessor):
    """One instance per live call. Holds the DB session and Conversation for the
    call's lifetime, same as a single HTTP request holds them for one chat turn."""

    def __init__(self, db: AsyncSession, tenant_id, conversation: Conversation,
                actor: str = "voice", opening_line: str | None = None,
                min_words_while_bot_speaking: int = 3,
                smart_turn_analyzer: BaseTurnAnalyzer | None = None,
                turn_timing=None,
                endpointing_window_ms: int = 0,
                send_debug: bool = True):
        super().__init__()
        # Whether each turn's debug envelope (tools, rules, retrieval, cost) is pushed to the
        # client. True for the staff Test Console; False for prospects on GTM demo links.
        self._send_debug = send_debug
        self._db = db
        self._tenant_id = tenant_id
        self._conversation = conversation
        self._actor = actor
        self._opening_line = opening_line
        # Optional TTFA tracker (app/engine/voice/ttfa.py). The bridge is where a turn's clock
        # starts -- the instant a finalized transcript begins processing -- so it stamps t0.
        # None (default) makes every TTFA call below a no-op, unchanged behaviour.
        self._turn_timing = turn_timing
        # Endpointing timing debug (task P2.01, config.py's endpointing_trace_enabled). The
        # active VAD silence window (stop_secs: 200ms with semantic endpointing on, 700ms off --
        # voice_ws.py owns the precedence and passes the chosen value in). _vad_stopped_at stamps
        # the moment VAD reports silence for the MOST RECENT utterance; _run_turn measures from it
        # to turn start, so a live call shows the real endpointing cost -- window + finalize +
        # any hold -- enabled vs not, instead of only the configured number.
        self._endpointing_window_ms = endpointing_window_ms
        self._vad_stopped_at: float | None = None
        # Must match the min_words the pipeline's MinWordsUserTurnStartStrategy (voice_ws.py)
        # uses to decide whether to interrupt the bot -- see the backchannel check below for
        # why this bridge needs its own copy of the same number.
        self._min_words_while_bot_speaking = min_words_while_bot_speaking
        self._bot_speaking = False
        # Smart Turn (config.py's smart_turn_enabled): None unless voice_ws.py constructed
        # one, which it only does when the setting is on -- every smart-turn code path below
        # is a no-op while this stays None, so existing behavior is unchanged by default.
        self._smart_turn_analyzer = smart_turn_analyzer
        self._vad_user_speaking = False
        # Cached verdict from the analyzer, read at VAD-stop time (see the docstring on
        # _handle_vad_stopped_speaking for why then and not later) and consumed once by the
        # next finalized transcript. None means "no verdict yet" -- treated as COMPLETE
        # (i.e. behave exactly as if Smart Turn were disabled) so a transcript arriving
        # before any VAD-stop event (shouldn't normally happen, but no data to gate on if it
        # did) is never held for a reason that was never actually checked.
        self._last_smart_turn_state: EndOfTurnState | None = None
        # Hold-and-merge state for Smart Turn, deliberately separate from entity_continuation's
        # fields below rather than shared -- the two gates catch different-shaped problems
        # (a bare digit fragment vs. audio that doesn't sound finished) and can fire on the
        # same turn in sequence; keeping their state independent avoids one gate's timeout
        # silently cancelling the other's.
        self._pending_smart_turn_text: str | None = None
        self._pending_smart_turn_stt_confidence = None
        self._pending_smart_turn_started_at: float | None = None
        self._pending_smart_turn_task: asyncio.Task | None = None
        # Deepgram doesn't hand back an exact per-turn billed duration, so this is an
        # estimate for cost tracking (app/observability/pricing.py), not an invoice figure --
        # elapsed wall-clock time between VAD's own start/stop for the utterance that becomes
        # the next finalized transcript. Each VAD start/stop cycle overwrites the previous
        # value, so it always reflects the MOST RECENT utterance by the time a transcript
        # (real turn or dropped backchannel) arrives -- good enough for cost trends, not
        # meant to be billing-exact.
        self._speech_started_at: float | None = None
        self._last_speech_seconds: float | None = None
        # Hold-and-merge state for entity_continuation (config.py) -- a finalized
        # transcript that looks like a bare, still-incomplete number waits here briefly
        # for a continuation instead of running the planner on half a phone number. See
        # _handle_final_transcript.
        self._pending_fragment: str | None = None
        self._pending_fragment_stt_confidence = None
        self._pending_fragment_started_at: float | None = None
        self._pending_fragment_task: asyncio.Task | None = None
        # Per-call embedding cache: maps query text → embedding vector for the duration
        # of this voice session. Repeated questions within one call (or the speculative +
        # tool retrieval pair hitting the same query) skip the embedding network call
        # entirely. Freed automatically when the session closes (GC collects the bridge).
        self._session_embed_cache: dict[str, list[float]] = {}
        # Words of a turn the caller's barge-in cancelled before it was saved, waiting to be
        # merged into the next transcript -- see app/engine/voice/barge_in.py and _run_turn.
        self._carryover: barge_in.CarryOver | None = None
        # Serializes every use of self._db across turns. process_turn() runs on TurnBridge's
        # own pipecat frame-processing task, which pipecat cancels (frame_processor.py's
        # _start_interruption) the instant the caller starts talking again -- barge-in is
        # normal mid-call, and ordinarily harmless, but a caller reading a phone number in
        # quick bursts (chunked phone-capture mode) can land the next burst while the
        # previous chunk's turn is still writing to the DB. That cancellation can land mid
        # `db.flush()`/`db.commit()`, and pipecat immediately creates a FRESH frame-processing
        # task that starts the next turn right away -- on the SAME shared AsyncSession.
        # AsyncSession is not safe for concurrent use from two tasks at once; without this
        # lock, that overlap is exactly what produced "greenlet_spawn has not been called"
        # once the rollback-on-cancel below was added (asyncio.shield protected the rollback
        # from a second cancellation, but did nothing to stop the NEXT turn from touching the
        # same session before that rollback finished). Holding this for the full turn --
        # including the cleanup in the except block -- guarantees the next turn's first `await`
        # blocks until this one's session use, successful or cancelled, is fully settled.
        self._db_lock = asyncio.Lock()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # Every processor sees StartFrame once, at pipeline start -- it must still
            # propagate downstream, same as any other frame this bridge doesn't act on.
            # A voice call that just stays silent until the caller speaks first reads as
            # a dead line, not an agent waiting politely, so it opens the same way a
            # chat session's opening_line already does (app/api/sessions.py).
            await self.push_frame(frame, direction)
            if self._smart_turn_analyzer is not None:
                self._smart_turn_analyzer.set_sample_rate(frame.audio_in_sample_rate)
            if self._opening_line:
                # The one spoken line that never passes through process_turn -- and it is
                # tenant-authored text, so it gets the same abbreviation/symbol rewrite every
                # reply gets (app/services/spoken_form.py) before it reaches TTS.
                await self.push_frame(TTSSpeakFrame(text=to_spoken_form(self._opening_line)),
                                      FrameDirection.DOWNSTREAM)
            return

        if isinstance(frame, (BotStartedSpeakingFrame, BotStoppedSpeakingFrame)):
            # Tracked so a short backchannel while the bot is talking can be recognised below
            # -- these already flow through this bridge untouched; this just also reads them.
            self._bot_speaking = isinstance(frame, BotStartedSpeakingFrame)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InputAudioRawFrame):
            # Keeps the analyzer's audio buffer filled so a verdict is ready the moment VAD
            # reports the user stopped -- see _handle_vad_stopped_speaking. No-op (cheap
            # isinstance check only) when smart_turn_enabled is off.
            if self._smart_turn_analyzer is not None:
                self._smart_turn_analyzer.append_audio(frame.audio, self._vad_user_speaking)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame)):
            # STT cost estimate -- see _speech_started_at's docstring in __init__.
            if isinstance(frame, VADUserStartedSpeakingFrame):
                self._speech_started_at = time.monotonic()
                self._vad_user_speaking = True
            else:
                self._vad_user_speaking = False
                if self._speech_started_at is not None:
                    self._last_speech_seconds = time.monotonic() - self._speech_started_at
                # Endpointing timing debug: stamp when VAD reported silence. The configured
                # stop_secs window has already elapsed by the time this frame fires (VAD waits
                # out the silence before emitting it), so the window is added back in _run_turn;
                # the span from here to turn start is the finalize + hold-and-merge tail.
                self._vad_stopped_at = time.monotonic()
                await self._handle_vad_stopped_speaking()
            await self.push_frame(frame, direction)
            return

        # TranscriptionFrame IS the final result -- Deepgram's service pushes
        # InterimTranscriptionFrame for interim guesses and only ever constructs a
        # TranscriptionFrame once a result is final, so the frame type is the whole
        # signal. (Its own `finalized` field means something unrelated -- whether a
        # finalize() RPC was confirmed -- and Deepgram never sets it, so it defaults
        # to False on every real transcript. Checking it here filtered out every
        # single turn: STT still showed up client-side because the RTVI processor
        # reads transcripts independently of this bridge, but process_turn never ran.)
        is_final_transcript = (
            isinstance(frame, TranscriptionFrame)
            and direction == FrameDirection.DOWNSTREAM
            and frame.text.strip())

        if not is_final_transcript:
            await self.push_frame(frame, direction)
            return

        # A short backchannel ("okay", "hm", "no") while the bot is mid-reply correctly does
        # NOT interrupt the bot's audio any more -- MinWordsUserTurnStartStrategy upstream
        # (voice_ws.py) suppresses that. But it does not, and cannot, stop this frame: Deepgram
        # still finalizes a transcript for whatever the caller said, and that transcript
        # reaches this bridge regardless of the interruption decision. Without this check, that
        # finalized "No." ran the full planner as an unrelated new turn WHILE the bot's current
        # reply was still queued to keep speaking -- the caller heard the bot's sentence
        # trail off, then heard an answer to a question they never meant to ask, its own reply
        # segments interleaved with the sentence that was still finishing. Same threshold,
        # same reasoning as the interruption gate: below min_words while the bot is talking is
        # an acknowledgment, not a turn, so it is heard and silently dropped, not answered.
        if self._bot_speaking and len(frame.text.split()) < self._min_words_while_bot_speaking:
            logger.info(
                "voice backchannel ignored (conversation=%s text=%r) -- bot still speaking",
                self._conversation.id, frame.text)
            await self.push_frame(frame, direction)
            return

        # Deepgram attaches its own response (channel.alternatives[0].confidence) to
        # `result` on every final TranscriptionFrame; other STT services (e.g. local
        # whisper dev path) don't share that shape, so this stays best-effort. Without
        # this, the low-STT-confidence guardrail in confidence.py/prompts.py -- which
        # chat callers already exercise via body.stt_confidence -- never fired for a
        # real phone call, no matter how badly Deepgram misheard the audio.
        stt_confidence = None
        try:
            stt_confidence = frame.result.channel.alternatives[0].confidence
        except (AttributeError, IndexError):
            pass

        await self._handle_final_transcript(frame.text, stt_confidence)

    async def _handle_vad_stopped_speaking(self) -> None:
        """Ask Smart Turn (config.py's smart_turn_enabled) whether the audio just captured
        actually sounds like a finished thought, and cache the verdict for the finalized
        transcript that's about to arrive from Deepgram.

        Run HERE, at VAD-stop, rather than when the transcript itself lands: the analyzer
        judges the raw audio, which is already fully buffered the instant VAD reports
        speech stopped, whereas Deepgram's own finalize can take a little longer to come
        back. Asking now means the verdict is already in hand by the time the transcript
        arrives, instead of adding the model's own inference time on top of the wait.
        """
        if self._smart_turn_analyzer is None:
            return
        state, _ = await self._smart_turn_analyzer.analyze_end_of_turn()
        self._last_smart_turn_state = state

    async def _handle_final_transcript(self, text: str, stt_confidence,
                                       _grace_flushed: bool = False) -> None:
        """Hold-and-merge gates run in sequence before a transcript is treated as a genuine,
        complete turn -- each catching a different-shaped problem, so no gate's timeout can
        silently swallow another's:

        1. Smart Turn (config.py's smart_turn_enabled / semantic_endpointing_enabled): the
           shortened VAD silence window that makes this worth having at all means Deepgram
           can finalize on a pause that isn't really the end of the thought -- e.g. "I have
           [breath] booked an appointment". Smart Turn judges the actual audio rather than
           trusting silence duration alone; a NON-complete verdict holds the transcript briefly
           for a likely continuation instead of running the planner on an unfinished sentence.

           CORROBORATION (config.py's semantic_require_incomplete_corroboration, task P2.01):
           at a 120ms window the model is starved of trailing silence and misfires INCOMPLETE on
           turns that are actually finished ("Yes.", "registered? Can you give that again?") --
           each wasting a full continuation grace. So in semantic mode an INCOMPLETE verdict is
           only trusted enough to HOLD when the transcript ITSELF corroborates it -- it looks
           unfinished (transcript_looks_unfinished -- dangling connector, bare digits, partial
           entity). A complete-looking clause the model merely misflagged, a bare "yes"/"ok"
           answer included, is OVERRULED and run at once. In plain smart-turn mode (semantic off)
           the raw verdict is trusted as before -- no change.
        2. entity_continuation (config.py): a finalized transcript that's nothing but a
           short run of digits ("598") may be half a phone number or price Deepgram cut off
           on an ordinary pause between groups, not a genuine complete turn -- see
           app/engine/voice/entity_continuation.py for why this specific shape is the one
           proven to actually happen.

        Both hold briefly and merge with whatever arrives next, rather than running the
        planner on the fragment as-is. Everything else runs immediately, exactly as before
        any of these existed.

        _grace_flushed is set True only on the internal re-entry from a hold's grace-timer
        flush (or the max-hold safety ceiling): the hold window already elapsed with no
        continuation, so gate 1 MUST NOT hold again -- doing so would re-arm the timer every
        grace window and never run the turn. The value is intentionally not exposed to callers.
        """
        settings = get_settings()

        # Read-then-clear: consumed once by this transcript regardless of outcome below, so
        # a verdict never lingers to be (mis)applied to a later, unrelated transcript.
        smart_turn_state = self._last_smart_turn_state
        self._last_smart_turn_state = None

        # Smart Turn's hold-and-merge is active for BOTH the plain smart-turn mode and the
        # aggressive semantic-endpointing mode (task P2.01) -- either one builds the analyzer
        # (voice_ws.py), and the shortened window both introduce is exactly what makes holding
        # a non-complete verdict worthwhile.
        smart_turn_active = (self._smart_turn_analyzer is not None
                             and (settings.smart_turn_enabled
                                  or settings.semantic_endpointing_enabled))

        # Merge with a smart-turn-held fragment from an EARLIER transcript first, before
        # anything else below -- this transcript is the continuation Deepgram just finalized
        # for it, arriving through the normal path above, not back through
        # _hold_for_smart_turn_continuation. Doing the merge here, unconditionally, is what
        # makes that continuation actually reach the held text rather than being judged and
        # run on its own.
        smart_turn_held_since = None
        if self._pending_smart_turn_text is not None:
            if self._pending_smart_turn_task is not None:
                self._pending_smart_turn_task.cancel()
                self._pending_smart_turn_task = None
            smart_turn_held_since = self._pending_smart_turn_started_at
            text = f"{self._pending_smart_turn_text} {text.strip()}".strip()
            self._pending_smart_turn_text = None
            self._pending_smart_turn_stt_confidence = None
            self._pending_smart_turn_started_at = None

        word_count = len(text.split())

        # Gate 1: Smart Turn judged the audio NOT complete. Whether that verdict is trusted
        # enough to HOLD depends on corroboration (task P2.01). _grace_flushed skips the whole
        # gate on a flush re-entry so a held fragment eventually runs rather than re-holding
        # forever (on that re-entry smart_turn_state is already None anyway, but the guard is
        # explicit for clarity and safety).
        incomplete_verdict = (smart_turn_active and smart_turn_state is not None
                              and smart_turn_state != EndOfTurnState.COMPLETE)
        if incomplete_verdict and not _grace_flushed:
            corroborate = (settings.semantic_endpointing_enabled
                           and settings.semantic_require_incomplete_corroboration)
            # Plain smart-turn mode (no corroboration) trusts the raw INCOMPLETE and always holds
            # -- unchanged behaviour. Semantic mode holds only when the WORDS corroborate the
            # verdict (transcript_looks_unfinished: dangling connector, bare digits, partial
            # entity); a complete-looking clause the model misflagged is overruled and runs. A
            # bare backchannel ("yes"/"ok") is NOT unfinished, so it runs immediately here rather
            # than being delayed -- the "yes/ok false trigger" case the task raises is handled
            # upstream by MinWordsUserTurnStartStrategy (1-word-during-silence) on the interruption
            # path, not by delaying genuine short answers at the turn boundary.
            if (not corroborate) or transcript_looks_unfinished(text):
                self._trace_endpointing("smart_turn_hold", text, word_count, smart_turn_state)
                await self._hold_for_smart_turn_continuation(
                    text, stt_confidence, smart_turn_held_since)
                return
            # Corroboration failed: model says incomplete, but the transcript reads as a finished
            # clause -- trust the words, fall through and run (this is the false-INCOMPLETE fix).
            self._trace_endpointing("incomplete_overruled", text, word_count, smart_turn_state)

        if not settings.entity_continuation_enabled:
            self._trace_endpointing("ran", text, word_count, smart_turn_state)
            await self._run_turn(text, stt_confidence)
            return

        if self._pending_fragment_task is not None:
            self._pending_fragment_task.cancel()
            self._pending_fragment_task = None

        if self._pending_fragment is not None:
            combined = f"{self._pending_fragment} {text.strip()}".strip()
            held_since = self._pending_fragment_started_at or time.monotonic()
        else:
            combined = text.strip()
            held_since = time.monotonic()

        # Semantic mode lengthens the entity grace/ceiling too (semantic_continuation_*): the
        # dictation pauses that shatter a number at a 120ms window are the same 0.9-2s gaps the
        # smart-turn hold now waits out, so both holds share one dictation-appropriate window.
        entity_grace_ms, entity_max_hold_ms = (
            (settings.semantic_continuation_grace_ms, settings.semantic_continuation_max_hold_ms)
            if settings.semantic_endpointing_enabled
            else (settings.entity_continuation_grace_ms, settings.entity_continuation_max_hold_ms))

        within_max_hold = (time.monotonic() - held_since) < (entity_max_hold_ms / 1000)

        if looks_like_partial_entity(combined) and within_max_hold:
            logger.info(
                "voice turn held for possible continuation (conversation=%s text=%r)",
                self._conversation.id, combined)
            self._trace_endpointing("entity_hold", combined,
                                    len(combined.split()), smart_turn_state)
            self._pending_fragment = combined
            self._pending_fragment_stt_confidence = stt_confidence
            self._pending_fragment_started_at = held_since
            self._pending_fragment_task = asyncio.create_task(
                self._flush_pending_fragment(entity_grace_ms / 1000))
            return

        self._pending_fragment = None
        self._pending_fragment_stt_confidence = None
        self._pending_fragment_started_at = None
        self._trace_endpointing("ran", combined, len(combined.split()), smart_turn_state)
        await self._run_turn(combined, stt_confidence)

    async def _hold_for_smart_turn_continuation(
            self, text: str, stt_confidence, held_since: float | None) -> None:
        """Hold `text` (already merged with any earlier fragment by the caller) for a
        possible continuation, unless the safety ceiling has already been reached."""
        settings = get_settings()
        held_since = held_since or time.monotonic()

        # Semantic mode uses the longer, dictation-sized continuation window (semantic_continuation_*)
        # so a name or number dictated in groups with 0.9-2s gaps actually accumulates into one turn
        # instead of flushing between fragments; plain smart-turn mode keeps its own 900ms grace.
        grace_ms, max_hold_ms = (
            (settings.semantic_continuation_grace_ms, settings.semantic_continuation_max_hold_ms)
            if settings.semantic_endpointing_enabled
            else (settings.smart_turn_grace_ms, settings.smart_turn_max_hold_ms))

        within_max_hold = (time.monotonic() - held_since) < (max_hold_ms / 1000)

        if within_max_hold:
            logger.info(
                "voice turn held for smart-turn continuation (conversation=%s text=%r)",
                self._conversation.id, text)
            self._pending_smart_turn_text = text
            self._pending_smart_turn_stt_confidence = stt_confidence
            self._pending_smart_turn_started_at = held_since
            self._pending_smart_turn_task = asyncio.create_task(
                self._flush_pending_smart_turn(grace_ms / 1000))
            return

        # Safety ceiling reached (config.py's smart_turn_max_hold_ms): stop holding and let
        # the transcript through -- re-entering _handle_final_transcript rather than calling
        # _run_turn directly so entity_continuation still gets a chance to check it. Both
        # state reads at the top (_last_smart_turn_state, _pending_smart_turn_text) are
        # already clear by this point, so this re-entry falls straight through to
        # entity_continuation's own gate, not back into this one. _grace_flushed=True so gate 1
        # also stands down -- the ceiling means "run it now", and re-holding a still-short "yes"
        # here would defeat that ceiling.
        await self._handle_final_transcript(text, stt_confidence, _grace_flushed=True)

    def _trace_endpointing(self, decision: str, text: str, word_count: int,
                           smart_turn_state) -> None:
        """One structured line per end-of-turn decision, for debugging semantic endpointing
        (config.py's endpointing_trace_enabled, task P2.01). A cheap flag check and nothing
        else when disabled, so it is free to leave in the hot path.

        `decision` names which gate fired: 'smart_turn_hold' (INCOMPLETE + corroborated) |
        'incomplete_overruled' (model said INCOMPLETE but the words read as finished, so it ran) |
        'entity_hold' | 'ran'. `smart_turn_state` is the Smart Turn verdict for this transcript
        -- i.e. what the audio itself said about completeness -- or None if none was captured.
        Together with word_count and the (possibly merged) text, this lets a specific call be
        replayed after the fact to see exactly why a turn started, or was held, when it did.
        """
        if not get_settings().endpointing_trace_enabled:
            return
        logger.info(
            "endpointing-trace conversation=%s decision=%s words=%d smart_turn=%s text=%r",
            self._conversation.id, decision, word_count,
            getattr(smart_turn_state, "name", smart_turn_state), text)

    def _log_endpointing_timing(self, text: str) -> None:
        """Measure and log the REAL VAD-endpointing cost of the turn about to run, so the
        latency with semantic endpointing on vs off can be compared from a live call rather
        than inferred from the configured number (task P2.01 timing debug). Gated on
        endpointing_trace_enabled, same as the decision trace.

        Breakdown (all ms):
          window       -- the configured VAD silence window (stop_secs) VAD waited out BEFORE it
                          reported silence: 200 with semantic endpointing on, 700 off. This is the
                          part the P2.01 task shortens, and the dominant saving.
          finalize_hold-- measured span from that VAD-silence report to this turn actually starting:
                          STT finalize + any smart-turn / entity hold-and-merge. Small for a turn
                          that runs straight through; larger when a fragment was held to merge.
          total        -- window + finalize_hold ~= the full time from the caller falling silent to
                          the planner starting. This is the apples-to-apples number to compare
                          across the two configs.

        Consumes _vad_stopped_at once (clears it) so a turn with no preceding VAD-stop this cycle
        (shouldn't normally happen) logs window-only rather than a stale span.
        """
        if not get_settings().endpointing_trace_enabled:
            return
        window_ms = self._endpointing_window_ms
        stopped_at = self._vad_stopped_at
        self._vad_stopped_at = None
        if stopped_at is None:
            logger.info(
                "endpointing-timing conversation=%s window_ms=%d finalize_hold_ms=n/a "
                "total_ms=%d text=%r", self._conversation.id, window_ms, window_ms, text)
            return
        finalize_hold_ms = round((time.monotonic() - stopped_at) * 1000)
        logger.info(
            "endpointing-timing conversation=%s window_ms=%d finalize_hold_ms=%d total_ms=%d "
            "text=%r", self._conversation.id, window_ms, finalize_hold_ms,
            window_ms + finalize_hold_ms, text)

    async def _flush_pending_smart_turn(self, grace_seconds: float) -> None:
        try:
            await asyncio.sleep(grace_seconds)
        except asyncio.CancelledError:
            return
        text = self._pending_smart_turn_text
        stt_confidence = self._pending_smart_turn_stt_confidence
        self._pending_smart_turn_text = None
        self._pending_smart_turn_stt_confidence = None
        self._pending_smart_turn_started_at = None
        self._pending_smart_turn_task = None
        if text:
            # Re-enters _handle_final_transcript (not _run_turn directly) so the
            # entity_continuation gate still gets a chance to run on the merged text --
            # Smart Turn having settled doesn't mean the result can't ALSO be a bare digit
            # fragment. _grace_flushed=True: the continuation grace window elapsed with no
            # continuation, so gate 1 must let this through now rather than re-holding the same
            # (still-unfinished-looking) fragment for another grace window.
            await self._handle_final_transcript(text, stt_confidence, _grace_flushed=True)

    async def _flush_pending_fragment(self, grace_seconds: float) -> None:
        try:
            await asyncio.sleep(grace_seconds)
        except asyncio.CancelledError:
            return
        text = self._pending_fragment
        stt_confidence = self._pending_fragment_stt_confidence
        self._pending_fragment = None
        self._pending_fragment_stt_confidence = None
        self._pending_fragment_started_at = None
        self._pending_fragment_task = None
        if text:
            await self._run_turn(text, stt_confidence)

    async def _run_turn(self, text: str, stt_confidence) -> None:
        # Words from a turn the caller's own barge-in cancelled before it was saved (see
        # app/engine/voice/barge_in.py) are folded in FIRST, so this turn plans the caller's
        # whole request rather than just the fragment that interrupted it. Consumed here
        # whatever the outcome; a cancellation of THIS turn re-carries the merged text below.
        carried, self._carryover = self._carryover, None
        if carried is not None:
            text, stt_confidence, outcome = barge_in.merge(
                carried, text, stt_confidence,
                max_age_s=get_settings().voice_barge_in_carryover_max_age_s)
            logger.info("voice barge-in carry-over %s (conversation=%s carried=%r)",
                        outcome, self._conversation.id, carried.text)

        logger.info("voice turn started (conversation=%s text=%r stt_confidence=%s)",
                   self._conversation.id, text, stt_confidence)

        self._log_endpointing_timing(text)

        # Start this turn's TTFA clock (app/engine/voice/ttfa.py): t0 is now, the moment the
        # finalized transcript begins processing. The first audio frame that reaches the output
        # (a cached filler, a live filler, or the responder's own speech) is measured against
        # this. No-op when TTFA tracking is off.
        if self._turn_timing is not None:
            self._turn_timing.start_turn(text)

        # Captures every sentence the responder streams (P3.02: not just the first) so
        # the post-turn segment loop below can skip re-speaking whichever ones it already
        # played. Perceived-latency FILLERS (P1.10 -- "Let me check that for you...") also
        # arrive through this same callback, but they are NOT response segments and must
        # be kept out of _prepushed: if a filler landed here, the dedup below would compare
        # the filler (not the real first sentence) against segments[0], fail to match, and
        # replay the first sentence -- the caller would hear it twice. is_filler_phrase
        # recognises every filler in every language, independent of whether the audio
        # cache is enabled. See app/engine/voice/filler_audio.py.
        _prepushed: list[str] = []

        async def _sentence_callback(text: str) -> None:
            if not is_filler_phrase(text):
                _prepushed.append(text)
            await self.push_frame(TTSSpeakFrame(text=text), FrameDirection.DOWNSTREAM)

        try:
            session = await working.load(self._conversation.id)
            raw_as_of = session.get("as_of_date")
            as_of = date.fromisoformat(raw_as_of) if raw_as_of else None

            # Held for the full turn, including the cancellation cleanup below. pipecat
            # cancels THIS task (process_turn runs inline on TurnBridge's own
            # frame-processing task, not a separate one) whenever the caller starts talking
            # again before this turn finished: broadcast_interruption() -> every processor's
            # _start_interruption() cancels and recreates its current frame task unless that
            # frame is marked UninterruptibleFrame, and a plain TranscriptionFrame isn't one.
            # Landing mid `db.flush()`/`db.commit()`, that leaves the shared per-call
            # AsyncSession's transaction broken and never rolled back -- SQLAlchemy then
            # refuses every later query on it, so the agent goes silent for the rest of the
            # call. A real call exposed this: chunked phone-number capture has the caller
            # reading digits in quick bursts with short gaps, which (unlike normal
            # back-and-forth, where the bot's full reply gives process_turn plenty of time to
            # finish) can land a new utterance while the previous chunk's turn is still
            # writing to the DB.
            #
            # The lock matters even though pipecat's own cancel_task() already awaits this
            # task to completion before creating the next one (task_manager.py: a single
            # task.cancel(), no timeout, no second cancellation on this path) -- an EARLIER
            # version of this fix used asyncio.shield() around the rollback, which spawns a
            # separate, untracked Task; pipecat's `await task` returned as soon as THIS task
            # finished, not the shielded one, so the next turn's task started touching
            # self._db while that orphaned rollback was still running -- AsyncSession is not
            # safe for concurrent use from two tasks, and that race produced a *different*
            # failure ("greenlet_spawn has not been called"). Plain (unshielded) rollback
            # here is correct given pipecat's actual cancellation behaviour; the lock makes
            # the ordering structural instead of depending on it.
            async with self._db_lock:
                try:
                    result = await process_turn(
                        self._db, tenant_id=self._tenant_id, conversation=self._conversation,
                        user_input=text, actor=self._actor, as_of_date=as_of,
                        stt_confidence=stt_confidence, stt_seconds=self._last_speech_seconds,
                        sentence_callback=_sentence_callback,
                        session_embed_cache=self._session_embed_cache)
                except TurnCancelledAfterCommit:
                    # Cancelled only after process_turn's persistence had begun -- which it
                    # then finished (turn_runner._persist_turn). The turn is saved: nothing
                    # to roll back, and its words are already in history.
                    logger.info("voice turn cancelled after commit (conversation=%s)",
                                self._conversation.id)
                    raise
                except asyncio.CancelledError:
                    try:
                        await self._db.rollback()
                        # rollback() EXPIRES every ORM object the session still tracks --
                        # including self._conversation, which TurnBridge holds for the whole
                        # call and every turn touches synchronously (_run_turn's very first
                        # line is `self._conversation.id`). Left expired, that next plain
                        # attribute access can't silently refresh itself outside an awaited
                        # context and raises its own unrelated-looking crash
                        # ("greenlet_spawn has not been called") on every turn from here on --
                        # which is exactly what shipped before this line was added. Refresh it
                        # back to a usable state immediately, inside this same awaited
                        # context, so nothing downstream ever touches an expired attribute.
                        await self._db.refresh(self._conversation)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.exception(
                            "rollback after cancelled voice turn failed (conversation=%s)",
                            self._conversation.id)
                    raise
        except TurnCancelledAfterCommit:
            raise
        except asyncio.CancelledError:
            # Barge-in cancelled this turn before it was saved: everything it did is rolled
            # back above, including the customer's message. Keep their words for the next
            # transcript instead of losing them (app/engine/voice/barge_in.py). CancelledError
            # is a BaseException, so the generic handler below never sees it; re-raised so
            # pipecat's task cancellation completes exactly as before.
            if get_settings().voice_barge_in_carryover_enabled:
                self._carryover = barge_in.carry(carried, text, stt_confidence)
                logger.info("voice turn cancelled by barge-in; carrying words forward "
                            "(conversation=%s text=%r)", self._conversation.id, text)
            else:
                logger.info("voice turn cancelled by barge-in; words dropped, carry-over "
                            "disabled (conversation=%s text=%r)", self._conversation.id, text)
            raise
        except Exception:
            # A turn failure must not kill the call -- the customer is on the line. Log it
            # server-side and say something rather than going silent, which on a phone call
            # reads as a dropped connection.
            logger.exception("voice turn failed (conversation=%s)", self._conversation.id)
            await self.push_frame(
                TTSSpeakFrame(text="Sorry, I'm having trouble on my side — let me get a "
                                   "colleague to help you."),
                FrameDirection.DOWNSTREAM)
            return

        # result.debug["timing"] already breaks this turn down by stage (planner_ms,
        # retrieval_ms, tool_ms, responder_ms, total_ms -- see turn_runner.process_turn)
        # for every chat turn; voice turns compute the exact same breakdown but it was
        # being thrown away here, logged only as a single opaque line with no numbers.
        logger.info("voice turn done (conversation=%s response=%r end_call=%s timing=%s "
                   "confidence=%s)", self._conversation.id, result.response_text,
                   result.end_call, (result.debug or {}).get("timing"), result.confidence)

        # Same debug envelope /sessions/{id}/turns already returns for a chat turn (tools
        # run, rules fired, retrieval, timing, trace_id) -- sent here as an RTVI server
        # message so the SAME "tool called / intent / latency" chips the text console
        # renders under a chat reply can render under a voice turn too. This used to be
        # deliberately left out (see useVoiceCall.js's old comment): the evidence still
        # landed in the decision trace server-side, but nothing streamed it live to a
        # caller-facing test console, so a voice call gave no visibility into what the
        # engine actually did on each turn while it was happening.
        # TEMPORARY DIAGNOSTIC: wrapped and logged explicitly, rather than trusting the push
        # to succeed silently, because a client-side counter showed the message sometimes
        # never arriving with no error visible anywhere -- this confirms from the SERVER side
        # whether the push itself completed, or raised (e.g. a value in `result.debug` that
        # isn't cleanly serializable), which would otherwise be swallowed inside pipecat's own
        # frame-processing error handling and look identical to "never sent" from the outside.
        # Never for a prospect on a GTM demo link (send_debug=False): the envelope carries
        # internals (tools, rules, retrieval, cost) that only the staff console may see.
        if self._send_debug:
            try:
                await self.push_frame(
                    RTVIServerMessageFrame(data={
                        "type": "turn_debug",
                        "response_text": result.response_text,
                        "response_segments": result.response_segments,
                        "action": result.action,
                        "executed_tools": result.executed_tools,
                        "state_changed_to": result.state_changed_to,
                        "escalation": result.escalation,
                        "end_call": result.end_call,
                        "trace_id": str(result.trace_id) if result.trace_id else None,
                        "confidence": result.confidence,
                        "pending_confirmation": result.pending_confirmation,
                        "debug": result.debug,
                    }),
                    FrameDirection.DOWNSTREAM)
                logger.info("voice turn_debug RTVI server message pushed (conversation=%s)",
                           self._conversation.id)
            except Exception:
                logger.exception("voice turn_debug RTVI server message FAILED to push "
                                "(conversation=%s)", self._conversation.id)

        # Validation/grounding checks (booking-date/time verification, feature-grounding,
        # scope-claim checks -- see turn_runner.py) all run against the FULL response text
        # before this point and can rewrite it outright, so nothing unverified is ever
        # spoken. Only once that's settled do we split it for playback: pushing each
        # already-validated sentence as its own TTSSpeakFrame (instead of one frame for
        # the whole reply) lets ElevenLabs start producing audio for each sentence as
        # soon as it's validated, rather than only ever handing it one giant block to
        # speak from a standing start.
        #
        # P3.02: for read-only-tool/no-tool-respond turns, EVERY sentence (not just the
        # first) may already have been pushed to TTS during the responder's streaming
        # phase (_sentence_callback above, _prepushed). Skip however many of them match
        # the validated segments in order, so a well-behaved reply -- the common case --
        # is never re-spoken at all here; this loop just plays whatever's left. The
        # moment a prepushed sentence diverges from its validated counterpart (a
        # validator rewrote it), stop skipping and replay from there onward, so the
        # caller only ever hears the validated text.
        segments = result.response_segments or ([result.response_text]
                                                if result.response_text else [])
        if not _prepushed:
            # Nothing streamed early for this turn -- a write-tool confirmation (never
            # streamed to begin with), a phone-capture fast-path reply, a templated
            # reply, or any other non-streamed path (including the no-tool "respond"
            # case, which stopped streaming -- see turn_runner.py's _no_tool_respond_ok
            # fix). Every sentence is already fully known and validated at once here, so
            # splitting them into separate TTSSpeakFrames buys no latency -- each becomes
            # its own ElevenLabs TTS context, and THE BUG THIS WORKS AROUND is that
            # pipecat's own word-alignment (aggregated_frame_sequencer) can corrupt the
            # boundary between two back-to-back contexts. A real call had "I can send you
            # the location pin..." come back as "...cation pin..." with the "lo" silently
            # dropped -- confirmed from pipecat's own logs ("Word 'cation' not recognised
            # by any slot, emitting as passthrough"), entirely inside pipecat's TTS
            # machinery, not this engine's generated text (voice turn done logged the
            # full, correct sentence). One combined frame has no context boundary to
            # corrupt.
            if result.response_text:
                await self.push_frame(
                    TTSSpeakFrame(text=result.response_text), FrameDirection.DOWNSTREAM)
        else:
            _skip_n = 0
            for _a, _b in zip(_prepushed, segments):
                if _a.strip() != _b.strip():
                    break
                _skip_n += 1
            for i, segment in enumerate(segments):
                if i < _skip_n:
                    continue
                await self.push_frame(TTSSpeakFrame(text=segment), FrameDirection.DOWNSTREAM)

        if result.end_call:
            self._conversation.status = "completed"
            await self._db.commit()
