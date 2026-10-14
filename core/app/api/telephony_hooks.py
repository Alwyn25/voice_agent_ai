"""Carrier callbacks: POST /webhooks/telephony/{provider}.

A real carrier (Plivo/Exotel/Twilio) POSTs call events here — ringing, answered, hangup,
recording ready. This endpoint is deliberately NOT behind the JWT auth every other route uses:
a carrier cannot present a user token. It is secured instead by the provider's own request
signature (PRD §9), verified by the adapter before a single byte of the body is trusted.

The flow is intentionally thin: verify → translate to a canonical CallEvent (the adapter's job)
→ apply_call_event (the one writer of call state). The stub does not use this path — it drives
events in-process — so in a stub deployment this endpoint simply has nothing to act on.
"""
import logging

from fastapi import APIRouter, HTTPException, Path, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import select

from app.config import get_settings
from app.db.base import get_session_factory
from app.db.models import Call
from app.services.call_runtime import CallRuntimeError, resolve_tenant_and_agent_by_number, run_call
from app.services.telephony import apply_call_event, get_telephony_provider
from app.services.telephony.base import STATUS_ANSWERED

router = APIRouter(tags=["telephony"])
logger = logging.getLogger(__name__)


class InboundCallIn(BaseModel):
    to_number: str
    from_number: str
    # A real carrier delivers audio, not text; there is no STT transcription pass in
    # front of this endpoint yet (Vocal Backend Tasksheet P6's own scope decision —
    # see app/services/call_runtime.py's module docstring), so the caller's opening
    # line is supplied directly. Defaults to a generic opener so the endpoint is
    # still exercisable with nothing but the two numbers above.
    opening_utterance: str = "Hello"


@router.post("/webhooks/telephony/inbound", status_code=status.HTTP_201_CREATED)
async def inbound_call(body: InboundCallIn, request: Request):
    """Task 140: "Resolve agent by to_number, start graph." Unauthenticated like the
    carrier-status webhook below (a carrier cannot present a user token) — the dialled
    number IS the tenant lookup, resolved fresh from the DB rather than trusted from
    anything the caller supplied, exactly like that webhook resolves a Call's tenant
    from the matched row rather than from the request.

    Registered BEFORE the generic `/webhooks/telephony/{provider}` route below: FastAPI
    matches routes in registration order, and `{provider}` is a path parameter that
    would otherwise swallow a literal "inbound" segment first."""
    # Verify inbound webhook secret when configured (Copilot review #3).
    from app.config import get_settings
    secret = getattr(get_settings(), "telephony_webhook_secret", "")
    if secret:
        provided = request.headers.get("x-webhook-secret", "")
        if not provided or provided != secret:
            logger.warning("inbound webhook rejected: invalid or missing X-Webhook-Secret")
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid webhook secret")

    async with get_session_factory()() as db:
        resolved = await resolve_tenant_and_agent_by_number(db, body.to_number)
        if resolved is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND,
                                "No agent is assigned to that number")
        tenant_id, agent = resolved
        try:
            call = await run_call(db, tenant_id=tenant_id, agent=agent,
                                  from_number=body.from_number, to_number=body.to_number,
                                  opening_utterance=body.opening_utterance)
        except CallRuntimeError as e:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))
        await db.commit()
    return {"call_id": str(call.id), "agent_id": str(agent.id), "status": call.status,
           "handoff": call.hangup_cause == "handoff",
           "summary": call.summary, "transcript": call.transcript}


@router.post("/webhooks/telephony/{provider}")
async def telephony_webhook(request: Request, provider: str = Path(...)):
    configured = get_telephony_provider()
    # The URL names a carrier; only the one this deployment is configured for may drive state.
    if provider.lower() != configured.name:
        logger.warning("telephony webhook for %r but provider is %r — ignored",
                       provider, configured.name)
        return Response(status_code=status.HTTP_404_NOT_FOUND)

    body = await request.body()
    if not configured.verify_signature(headers=dict(request.headers), body=body):
        logger.warning("telephony webhook signature rejected (provider=%s)", provider)
        return Response(status_code=status.HTTP_403_FORBIDDEN)

    ct = request.headers.get("content-type", "")
    if "application/x-www-form-urlencoded" in ct or "multipart/form-data" in ct:
        # Plivo (and most telcos) POST call events as form-encoded bodies, not JSON.
        form = await request.form()
        payload = dict(form)
    else:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
    event = configured.parse_webhook(payload=payload)
    if event is None:
        # A keep-alive or an event we do not act on — acknowledged, nothing to do.
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    async with get_session_factory()() as db:
        call = (await db.execute(select(Call).where(
            Call.provider == event.provider,
            Call.provider_call_id == event.provider_call_id))).scalar_one_or_none()
        if call is None:
            logger.warning("telephony webhook for unknown call (provider=%s id=%s)",
                           event.provider, event.provider_call_id)
            return Response(status_code=status.HTTP_404_NOT_FOUND)
        await apply_call_event(db, call, event)
        await db.commit()

    if (event.status == STATUS_ANSWERED
            and configured.name == "plivo"
            and get_settings().telephony_media_stream_base_url):
        base = get_settings().telephony_media_stream_base_url.rstrip("/")
        stream_url = f"{base}/{event.provider_call_id}"
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            f'<Stream keepCallAlive="true" bidirectional="true"'
            f' contentType="audio/x-mulaw;rate=8000">{stream_url}</Stream>'
            "</Response>"
        )
        return Response(content=xml, media_type="application/xml",
                        status_code=status.HTTP_200_OK)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
