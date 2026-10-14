"""What every telephony carrier must implement, and the canonical shapes around it.

Kept small and carrier-neutral on purpose, mirroring app/services/messaging/base.py. A carrier
adapter (Plivo, Exotel, Twilio) does exactly two things:

  * ``originate`` — ask the carrier to place a call. It returns an OriginateResult and NEVER
    raises for an expected failure (a rejected number, a 4xx, a timeout) — that comes back as
    ok=False so the dialer records it and moves on. TelephonyError is reserved for a genuine
    programming fault.
  * ``parse_webhook`` / ``verify_signature`` — turn the carrier's own call-event callbacks into
    the single canonical CallEvent the rest of the system understands, after proving the
    request actually came from the carrier.

Everything downstream — the Call row, retries, campaign stats, cost — is expressed in the
canonical vocabulary here, so no call site ever needs to know which carrier is active.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Call.status vocabulary (also in the model comment). Every carrier maps its own status names
# onto these; the dialer and analytics only ever see these.
STATUS_QUEUED = "queued"        # accepted by the gate, not yet handed to a carrier
STATUS_RINGING = "ringing"      # carrier is dialling / the phone is ringing
STATUS_ANSWERED = "answered"    # the callee picked up; media stream is (or would be) live
STATUS_BUSY = "busy"            # line busy
STATUS_NO_ANSWER = "no_answer"  # rang out, nobody answered
STATUS_VOICEMAIL = "voicemail"  # answering machine detected
STATUS_FAILED = "failed"        # carrier could not place the call at all
STATUS_COMPLETED = "completed"  # answered call ended normally

# A call in one of these is finished — the dialer never touches it again, and it no longer
# counts against the live-call concurrency cap.
TERMINAL_STATUSES = frozenset({STATUS_BUSY, STATUS_NO_ANSWER, STATUS_VOICEMAIL,
                               STATUS_FAILED, STATUS_COMPLETED})
# Outcomes that mean "the customer was not reached" — candidates for a retry.
RETRYABLE_STATUSES = frozenset({STATUS_BUSY, STATUS_NO_ANSWER, STATUS_VOICEMAIL})
# In-flight states that occupy a concurrency slot.
ACTIVE_STATUSES = frozenset({STATUS_RINGING, STATUS_ANSWERED})


class TelephonyError(RuntimeError):
    """A genuine fault (misconfiguration, programming error) — NOT an expected dial failure,
    which is returned as OriginateResult(ok=False)."""


@dataclass
class OriginateResult:
    """The immediate outcome of asking a carrier to place a call. The real call progression
    (ringing -> answered/…) arrives later as CallEvents via the webhook."""
    ok: bool
    provider: str
    provider_call_id: str | None = None
    # The status to record right after a successful originate — usually "ringing".
    status: str = STATUS_RINGING
    error: str | None = None


@dataclass
class CallEvent:
    """One carrier call-event, normalised. Produced by a provider's parse_webhook (real) or by
    the stub simulator, and consumed by apply_call_event — the single writer of call state."""
    provider: str
    provider_call_id: str
    status: str                                   # one of the STATUS_* above
    occurred_at: datetime | None = None
    hangup_cause: str | None = None
    recording_url: str | None = None
    duration_seconds: int | None = None
    cost: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class DialRequest:
    """Everything a carrier needs to place one call. Assembled by the dialer from the Call row
    and tenant/campaign context, so a provider never reaches back into the database."""
    call_id: Any                    # uuid.UUID of the Call row
    tenant_id: Any
    to_number: str
    from_number: str
    purpose: str = "marketing"
    # Where the carrier should stream the answered call's media (real providers only).
    media_stream_url: str | None = None


class TelephonyProvider(abc.ABC):
    """Place calls and interpret carrier callbacks. One small interface so a carrier is trivial
    to add and hard to get subtly wrong."""

    #: short, stable identifier recorded on Call.provider ("stub", "plivo", "exotel", "twilio")
    name: str = "base"

    @abc.abstractmethod
    async def originate(self, req: DialRequest) -> OriginateResult:
        """Ask the carrier to dial ``req.to_number``. Never raises for an expected failure —
        returns OriginateResult(ok=False, error=...)."""
        raise NotImplementedError

    async def hangup(self, provider_call_id: str) -> None:
        """Best-effort tear-down of a live call. Optional; the default is a no-op so a carrier
        without a hangup API (or the stub) needs no override."""
        return None

    def verify_signature(self, *, headers: dict[str, str], body: bytes) -> bool:
        """Prove a webhook request came from this carrier. The default rejects everything, so a
        real adapter must opt in explicitly — a webhook that mutates call state must never be
        callable by an unauthenticated third party (PRD §9: 'secured by provider signature
        verification, not by API key'). The stub overrides this to accept, as it is never
        exposed to a real carrier."""
        return False

    def parse_webhook(self, *, payload: dict[str, Any]) -> CallEvent | None:
        """Translate one carrier webhook body into a canonical CallEvent, or None if the body
        is a keep-alive / event we do not act on. Real adapters implement this; the stub does
        not receive webhooks (it drives events in-process), so the base returns None."""
        return None
