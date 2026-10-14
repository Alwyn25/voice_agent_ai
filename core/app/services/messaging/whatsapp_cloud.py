"""The real provider: Meta's WhatsApp Cloud API (Graph API).

Fully implemented so that going live is purely a matter of putting the access token and phone
number id in .env — there is no further code to write. Until those exist the factory hands back
the stub instead, so importing this module is always safe (httpx is imported lazily, inside the
one method that needs it, so a stub-only deployment never depends on it).

Endpoint:  POST {base}/{version}/{phone_number_id}/messages
Auth:      Authorization: Bearer {access_token}
Body:      a "document" message pointing at a publicly reachable link; the Cloud API fetches the
           file itself, which is why ShareableDocument.file_url must be reachable by Meta.

Every expected failure — a 4xx from Meta, an unreachable host, a timeout — is caught and
returned as SendResult(ok=False, STATUS_FAILED, error=...). The tool decides what to say; the
provider never takes the turn down.

Reference: https://developers.facebook.com/docs/whatsapp/cloud-api/reference/messages
"""
from __future__ import annotations

import logging

from app.services.messaging.base import (OutboundDocument, SendResult, STATUS_FAILED,
                                          STATUS_SENT, WhatsAppProvider)

log = logging.getLogger(__name__)


class WhatsAppCloudProvider(WhatsAppProvider):
    name = "whatsapp_cloud"

    def __init__(self, *, access_token: str, phone_number_id: str,
                 base_url: str = "https://graph.facebook.com", api_version: str = "v21.0",
                 timeout_seconds: float = 8.0) -> None:
        if not access_token or not phone_number_id:
            # A programming error (the factory should never build this without credentials),
            # so it raises rather than returning a failed send.
            raise ValueError("WhatsAppCloudProvider requires access_token and phone_number_id")
        self._access_token = access_token
        self._phone_number_id = phone_number_id
        self._endpoint = (f"{base_url.rstrip('/')}/{api_version.strip('/')}/"
                          f"{phone_number_id}/messages")
        self._timeout = timeout_seconds

    async def send_document(self, *, to: str, document: OutboundDocument) -> SendResult:
        # Lazy import so the stub-only path never requires httpx to be installed/importable.
        try:
            import httpx
        except ImportError as e:  # pragma: no cover - httpx ships with openai/uvicorn
            log.error("[whatsapp:cloud] httpx not available: %s", e)
            return SendResult(ok=False, status=STATUS_FAILED, provider=self.name,
                              error="httpx is not installed")

        doc: dict = {"link": document.url}
        if document.file_name:
            doc["filename"] = document.file_name
        if document.caption:
            doc["caption"] = document.caption
        payload = {"messaging_product": "whatsapp", "recipient_type": "individual",
                   "to": to, "type": "document", "document": doc}
        headers = {"Authorization": f"Bearer {self._access_token}",
                   "Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._endpoint, json=payload, headers=headers)
        except httpx.TimeoutException:
            log.warning("[whatsapp:cloud] timeout sending to=%s file=%r", to, document.file_name)
            return SendResult(ok=False, status=STATUS_FAILED, provider=self.name,
                              error="WhatsApp API timed out")
        except httpx.HTTPError as e:
            log.warning("[whatsapp:cloud] transport error to=%s: %s", to, e)
            return SendResult(ok=False, status=STATUS_FAILED, provider=self.name,
                              error=f"WhatsApp API unreachable: {e.__class__.__name__}")

        if resp.status_code // 100 != 2:
            # Meta returns a structured error object; surface its message for the logs without
            # letting a parse failure mask the real HTTP status.
            detail = _error_detail(resp)
            log.warning("[whatsapp:cloud] send rejected to=%s status=%s detail=%s",
                        to, resp.status_code, detail)
            return SendResult(ok=False, status=STATUS_FAILED, provider=self.name,
                              error=f"WhatsApp API {resp.status_code}: {detail}")

        message_id = _message_id(resp)
        log.info("[whatsapp:cloud] sent document to=%s file=%r id=%s",
                 to, document.file_name, message_id)
        return SendResult(ok=True, status=STATUS_SENT, provider=self.name,
                          provider_message_id=message_id)


def _error_detail(resp) -> str:
    try:
        body = resp.json()
    except Exception:
        return (resp.text or "")[:200]
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return str(err.get("message") or err.get("type") or err)[:200]
    return str(body)[:200]


def _message_id(resp) -> str | None:
    try:
        body = resp.json()
        messages = body.get("messages") or []
        if messages and isinstance(messages[0], dict):
            return messages[0].get("id")
    except Exception:
        pass
    return None
