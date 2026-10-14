"""Demo voice calls (docs/gtm-demo-links.md §8.3).

POST /public/demo/{token}/session issues a demo voice token; /ws/voice accepts it in place
of a staff login. The token is:

  * its own JWT type ("demo_voice"), so it can never pass for a login token anywhere else
    (decode_token checks the type) and a login token can never open a demo call
  * short-lived (DEMO_VOICE_TOKEN_TTL_S) and single-use (its jti is burned at the handshake)
  * bound to one demo session: one link, one demo tenant, one prepared conversation

At the handshake the link is checked again (still active, not expired), the browser's Origin
must be a demo origin, and the call is inbound only. When the call ends, its length, end
reason and outcome (did the sample lead book?) are written to the session.
"""
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt as pyjwt
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import Appointment, Conversation, DemoLink, DemoProspect, DemoSession, Message
from app.demo.limits import VOICE_TOKEN_TTL_S, burn_once, link_limit
from app.demo.links import effective_status
from app.demo.settings import csv_list
from app.security.auth import decode_token

log = logging.getLogger(__name__)
TOKEN_TYPE = "demo_voice"
TOKEN_TTL_S = VOICE_TOKEN_TTL_S


class DemoVoiceRefused(Exception):
    """The handshake must be refused with WebSocket close `code`."""

    def __init__(self, code: int, reason: str):
        super().__init__(reason)
        self.code, self.reason = code, reason


@dataclass(frozen=True)
class DemoCall:
    session_id: uuid.UUID
    link_id: uuid.UUID
    tenant_id: uuid.UUID
    conversation_id: uuid.UUID
    max_seconds: int


def issue_voice_token(*, session: DemoSession, link: DemoLink, prospect: DemoProspect,
                      max_seconds: int) -> str:
    s = get_settings()
    now = datetime.now(timezone.utc)
    payload = {"type": TOKEN_TYPE, "jti": uuid.uuid4().hex, "sid": str(session.id),
               "lid": str(link.id), "tid": str(prospect.tenant_id),
               "cid": str(session.conversation_id), "max": int(max_seconds),
               "iat": now, "exp": now + timedelta(seconds=TOKEN_TTL_S)}
    return pyjwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def looks_like_demo_token(token: str) -> bool:
    """Cheap pre-check (no verification) so a login token never takes the demo path."""
    try:
        return pyjwt.decode(token, options={"verify_signature": False}).get("type") == TOKEN_TYPE
    except pyjwt.PyJWTError:
        return False


def origin_allowed(origin: str | None) -> bool:
    allowed = {o.rstrip("/").lower() for o in csv_list(get_settings().demo_allowed_origins)}
    return bool(origin) and origin.rstrip("/").lower() in allowed


async def authenticate_demo_voice(db: AsyncSession, token: str, origin: str | None) -> DemoCall:
    """Verify a demo voice token at the WebSocket handshake. Raises DemoVoiceRefused."""
    if not get_settings().demo_enabled:
        raise DemoVoiceRefused(4404, "demo calls are not enabled")
    try:
        claims = decode_token(token, expected_type=TOKEN_TYPE)
    except pyjwt.ExpiredSignatureError:
        raise DemoVoiceRefused(4401, "demo voice token expired") from None
    except pyjwt.PyJWTError:
        raise DemoVoiceRefused(4401, "invalid demo voice token") from None
    if not origin_allowed(origin):
        raise DemoVoiceRefused(4403, "origin not allowed")
    if not await burn_once(f"jti:{claims['jti']}", TOKEN_TTL_S + 60):
        raise DemoVoiceRefused(4401, "demo voice token already used")

    session = await db.get(DemoSession, uuid.UUID(claims["sid"]))
    link = await db.get(DemoLink, uuid.UUID(claims["lid"]))
    if session is None or link is None or session.demo_link_id != link.id:
        raise DemoVoiceRefused(4401, "unknown demo session")
    if effective_status(link) != "active":
        raise DemoVoiceRefused(4410, "demo link is no longer active")
    if session.started_at is not None or session.ended_at is not None:
        raise DemoVoiceRefused(4409, "demo session already used")
    prospect = await db.get(DemoProspect, link.prospect_id)
    conv = await db.get(Conversation, session.conversation_id) if session.conversation_id else None
    if (prospect is None or conv is None or str(prospect.tenant_id) != claims["tid"]
            or conv.tenant_id != prospect.tenant_id or str(conv.id) != claims["cid"]):
        raise DemoVoiceRefused(4401, "demo session does not match its link")
    session.started_at = datetime.now(timezone.utc)
    await db.flush()
    return DemoCall(session_id=session.id, link_id=link.id, tenant_id=prospect.tenant_id,
                    conversation_id=conv.id,
                    max_seconds=min(int(claims["max"]), link_limit(link, "max_call_seconds")))


async def opening_line(db: AsyncSession, conversation_id: uuid.UUID) -> str | None:
    """The greeting already stored when the session was prepared — spoken, not re-stored."""
    return (await db.execute(select(Message.content).where(
        Message.conversation_id == conversation_id, Message.role == "agent")
        .order_by(Message.created_at).limit(1))).scalar_one_or_none()


async def finish_demo_call(db: AsyncSession, call: DemoCall, *, end_reason: str,
                           audio_started_at: datetime | None = None) -> None:
    """Record how the call ended. Outcome: goal_completed when the sample lead booked.

    duration_seconds runs from the agent's first audio (audio_started_at), the same moment the
    call's time limit starts, so connecting and the greeting's synthesis aren't counted. A
    call that never played audio counts as 0."""
    session = await db.get(DemoSession, call.session_id)
    if session is None or session.ended_at is not None:
        return
    ended = datetime.now(timezone.utc)
    session.ended_at = ended
    session.end_reason = end_reason
    session.duration_seconds = (max(0, round((ended - audio_started_at).total_seconds()))
                                if audio_started_at else 0)
    conv = await db.get(Conversation, call.conversation_id)
    booked = 0
    if conv is not None and conv.customer_id is not None:
        booked = (await db.execute(select(func.count()).select_from(Appointment).where(
            Appointment.tenant_id == call.tenant_id, Appointment.customer_id == conv.customer_id,
            Appointment.status.in_(("booked", "rescheduled"))))).scalar_one()
    session.outcome = "goal_completed" if booked else "none"
    await db.flush()
