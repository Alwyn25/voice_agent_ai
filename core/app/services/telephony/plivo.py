"""Plivo telephony adapter — places real PSTN calls via Plivo's REST API.

Uses httpx directly (ADR-001: no Plivo SDK, httpx already in requirements and
async-native). Handles the full call lifecycle:

  * originate  — POST /Account/{auth_id}/Call/
  * hangup     — DELETE /Account/{auth_id}/Call/{uuid}/
  * verify_signature — Plivo V2 HMAC-SHA256 (url + nonce + sha256(body))
  * parse_webhook    — maps Plivo's CallStatus/Event → canonical STATUS_*

The media streaming side (Plivo WebSocket Audio → pipecat) is handled separately
in app/api/voice_ws.py; this adapter only concerns itself with the call control
plane and lifecycle events.
"""
from __future__ import annotations

import hashlib
import hmac as _hmac
import logging
from typing import Any

import httpx

from app.config import Settings
from app.services.telephony.base import (
    CallEvent,
    DialRequest,
    OriginateResult,
    STATUS_ANSWERED,
    STATUS_BUSY,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_NO_ANSWER,
    STATUS_RINGING,
    STATUS_VOICEMAIL,
    TelephonyProvider,
)

logger = logging.getLogger(__name__)

_PLIVO_API_BASE = "https://api.plivo.com/v1"

# Plivo CallStatus / Event values → canonical status.
# "canceled" is Plivo's spelling for a call that didn't connect (caller hung up
# before answer, or line not reachable) — maps to NO_ANSWER, not FAILED, so the
# retry logic still runs on it.
_STATUS_MAP: dict[str, str] = {
    "ringing":     STATUS_RINGING,
    "in-progress": STATUS_ANSWERED,
    "answered":    STATUS_ANSWERED,
    "busy":        STATUS_BUSY,
    "no-answer":   STATUS_NO_ANSWER,
    "canceled":    STATUS_NO_ANSWER,
    "machine":     STATUS_VOICEMAIL,
    "completed":   STATUS_COMPLETED,
    "failed":      STATUS_FAILED,
}


class PlivoTelephonyProvider(TelephonyProvider):
    """Plivo carrier adapter.  One instance per process (cached by provider.py)."""

    name = "plivo"

    def __init__(self, settings: Settings) -> None:
        self._auth_id     = settings.plivo_auth_id
        self._auth_token  = settings.plivo_auth_token
        self._caller_id   = settings.telephony_caller_id
        self._webhook_url = settings.telephony_webhook_url
        self._media_base  = settings.telephony_media_stream_base_url

    # ------------------------------------------------------------------ originate

    async def originate(self, req: DialRequest) -> OriginateResult:
        payload: dict[str, Any] = {
            "from":          self._caller_id,
            "to":            req.to_number,
            "answer_url":    self._webhook_url,
            "hangup_url":    self._webhook_url,
            "answer_method": "POST",
            "hangup_method": "POST",
            "caller_name":   req.purpose or "",
        }
        if req.media_stream_url:
            payload["stream_url"] = req.media_stream_url

        async with httpx.AsyncClient(
            auth=(self._auth_id, self._auth_token), timeout=10.0
        ) as client:
            resp = await client.post(
                f"{_PLIVO_API_BASE}/Account/{self._auth_id}/Call/",
                json=payload,
            )

        if resp.status_code not in (200, 201, 202):
            logger.warning(
                "plivo originate failed (to=%s status=%d body=%s)",
                req.to_number, resp.status_code, resp.text[:200],
            )
            return OriginateResult(ok=False, provider="plivo", error=resp.text)

        data = resp.json()
        call_id = data.get("request_uuid") or data.get("call_uuid") or ""
        return OriginateResult(ok=True, provider="plivo", provider_call_id=call_id)

    # ------------------------------------------------------------------ hangup

    async def hangup(self, provider_call_id: str) -> None:
        async with httpx.AsyncClient(
            auth=(self._auth_id, self._auth_token), timeout=10.0
        ) as client:
            await client.delete(
                f"{_PLIVO_API_BASE}/Account/{self._auth_id}/Call/{provider_call_id}/",
            )

    # ------------------------------------------------------------------ verify_signature

    def verify_signature(self, *, headers: dict[str, str], body: bytes) -> bool:
        """Plivo V2 signature: HMAC-SHA256 of (webhook_url + nonce + sha256hex(body)).

        Header names are lowercase in FastAPI's request.headers dict, which is what
        the caller (telephony_hooks.py) passes here.
        """
        nonce = headers.get("x-plivo-signature-v2-nonce", "")
        sig   = headers.get("x-plivo-signature-v2", "")
        if not sig:
            return False

        body_sha256 = hashlib.sha256(body).hexdigest()
        message     = self._webhook_url + nonce + body_sha256
        expected    = _hmac.new(
            self._auth_token.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        return _hmac.compare_digest(expected, sig)

    # ------------------------------------------------------------------ parse_webhook

    def parse_webhook(self, *, payload: dict[str, Any]) -> CallEvent | None:
        """Translate one Plivo webhook body into a canonical CallEvent.

        Plivo sends CallUUID + CallStatus for most events. Some async-completion
        callbacks use a different shape (Event field); both are handled here.
        Returns None for keep-alives and events we do not act on.
        """
        call_uuid   = payload.get("CallUUID") or payload.get("call_uuid")
        call_status = (
            payload.get("CallStatus") or payload.get("Event") or ""
        ).lower().strip()

        if not call_uuid or not call_status:
            return None

        status = _STATUS_MAP.get(call_status)
        if status is None:
            logger.debug("plivo webhook: unrecognised status %r — ignored", call_status)
            return None

        duration_str = payload.get("Duration") or payload.get("BillDuration") or ""
        duration_secs = int(duration_str) if str(duration_str).isdigit() else None

        return CallEvent(
            provider="plivo",
            provider_call_id=str(call_uuid),
            status=status,
            duration_seconds=duration_secs,
            raw=payload,
        )
