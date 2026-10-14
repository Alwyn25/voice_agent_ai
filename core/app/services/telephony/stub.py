"""The stub carrier: the default until a real telephony integration is configured.

It places NO real call. What it does is faithfully simulate a carrier's event timeline so every
downstream path — ringing/answered/busy/no-answer/voicemail, retries, campaign stats, cost,
the dashboard events — is exercised end to end with no account, no credentials and no phone.

  * ``originate`` returns immediately as 'ringing', then schedules an in-process task that,
    after a short delay, emits the outcome through the SAME apply_call_event the real webhook
    uses. An 'answered' call emits 'completed' after a simulated talk time, with a simulated
    duration and cost.
  * The outcome is normally 'answered'. A number can force a specific outcome by its last two
    digits (…00 no-answer, …11 busy, …22 voicemail, …99 failed) so a demo can show a connect
    and a miss deterministically; otherwise telephony_stub_outcome decides.

Because it never faces a real carrier, verify_signature accepts (there is no signature to
check) — but the stub webhook route is not mounted for real traffic; the stub drives events
itself. Swapping in a real provider changes nothing about how the dialer calls originate().
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select

from app.config import Settings
from app.db.base import get_session_factory
from app.db.models import Call
from app.services.telephony.base import (CallEvent, DialRequest, OriginateResult,
                                         STATUS_ANSWERED, STATUS_COMPLETED, STATUS_RINGING,
                                         TERMINAL_STATUSES, TelephonyProvider)
from app.services.telephony.events import apply_call_event

log = logging.getLogger(__name__)

# Last-two-digits convention so a demo can force an outcome deterministically.
_SUFFIX_OUTCOMES = {"00": "no_answer", "11": "busy", "22": "voicemail", "99": "failed"}


class StubTelephonyProvider(TelephonyProvider):
    name = "stub"

    def __init__(self, settings: Settings):
        self._s = settings
        # Hold references so the fire-and-forget simulator tasks are not garbage-collected
        # mid-flight (asyncio only keeps weak references to tasks).
        self._tasks: set[asyncio.Task] = set()

    def _outcome_for(self, to_number: str) -> str:
        digits = "".join(ch for ch in (to_number or "") if ch.isdigit())
        forced = _SUFFIX_OUTCOMES.get(digits[-2:]) if len(digits) >= 2 else None
        return forced or (self._s.telephony_stub_outcome or "answered")

    async def originate(self, req: DialRequest) -> OriginateResult:
        provider_call_id = f"stub-{uuid.uuid4()}"
        outcome = self._outcome_for(req.to_number)
        log.info("[telephony:stub] SIMULATED originate to=%s purpose=%s -> will be %r (id=%s)",
                 req.to_number, req.purpose, outcome, provider_call_id)
        task = asyncio.create_task(self._run(provider_call_id, outcome))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return OriginateResult(ok=True, provider=self.name,
                               provider_call_id=provider_call_id, status=STATUS_RINGING)

    async def _run(self, provider_call_id: str, outcome: str) -> None:
        """The simulated timeline: wait out the ring, emit the outcome; if answered, wait out
        the talk time and emit completed. Best-effort — a call hung up elsewhere just stops."""
        try:
            await asyncio.sleep(max(0.0, self._s.telephony_stub_ring_seconds))
            if outcome == STATUS_ANSWERED:
                await self._emit(provider_call_id, CallEvent(
                    provider=self.name, provider_call_id=provider_call_id,
                    status=STATUS_ANSWERED))
                await asyncio.sleep(max(0.0, self._s.telephony_stub_talk_seconds))
                talk = max(0, int(self._s.telephony_stub_talk_seconds))
                cost = round(self._s.telephony_stub_cost_per_minute * (talk / 60.0), 4)
                await self._emit(provider_call_id, CallEvent(
                    provider=self.name, provider_call_id=provider_call_id,
                    status=STATUS_COMPLETED, duration_seconds=talk, cost=cost,
                    recording_url=f"stub://recording/{provider_call_id}",
                    hangup_cause="normal_clearing"))
            else:
                # busy | no_answer | voicemail | failed — a single terminal event, no talk time.
                await self._emit(provider_call_id, CallEvent(
                    provider=self.name, provider_call_id=provider_call_id,
                    status=outcome if outcome in TERMINAL_STATUSES else "failed",
                    hangup_cause=outcome))
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("[telephony:stub] simulator failed for %s", provider_call_id)

    async def _emit(self, provider_call_id: str, event: CallEvent) -> None:
        """Open a fresh session (the dialer's is long gone by now) and apply the event."""
        event.occurred_at = event.occurred_at or datetime.now(timezone.utc)
        async with get_session_factory()() as db:
            call = (await db.execute(select(Call).where(
                Call.provider_call_id == provider_call_id))).scalar_one_or_none()
            if call is None:
                log.warning("[telephony:stub] no Call for %s (event %s dropped)",
                            provider_call_id, event.status)
                return
            await apply_call_event(db, call, event)
            await db.commit()

    def verify_signature(self, *, headers, body) -> bool:
        return True
