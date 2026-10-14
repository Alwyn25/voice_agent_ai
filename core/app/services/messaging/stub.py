"""The stub provider: the default until a real WhatsApp integration is configured.

It performs NO network I/O. It logs the send at INFO with everything a real one would carry —
recipient, document, url — and returns a success marked ``STATUS_SIMULATED`` with a synthetic
message id. Two deliberate properties:

  * The status is ``simulated``, never ``sent`` — so the audit row (tool_executions.output) and
    anyone reading it can always tell a real delivery from a rehearsed one, even though the
    caller-facing sentence says the document is on its way. Honesty lives in the record; the
    spoken line is the product's choice.
  * The message id is prefixed ``stub-`` so it can never be mistaken for a provider id in a
    later reconciliation.

Swapping in the real provider (credentials in .env) changes nothing about how the tool calls
this — that is the whole point of the abstraction.
"""
from __future__ import annotations

import logging
import uuid

from app.services.messaging.base import (OutboundDocument, SendResult, STATUS_SIMULATED,
                                          WhatsAppProvider)

log = logging.getLogger(__name__)


class StubWhatsAppProvider(WhatsAppProvider):
    name = "stub"

    async def send_document(self, *, to: str, document: OutboundDocument) -> SendResult:
        message_id = f"stub-{uuid.uuid4()}"
        log.info("[whatsapp:stub] SIMULATED document send to=%s file=%r url=%s id=%s "
                 "(no WhatsApp credentials configured — set WHATSAPP_ACCESS_TOKEN and "
                 "WHATSAPP_PHONE_NUMBER_ID to send for real)",
                 to, document.file_name, document.url, message_id)
        return SendResult(ok=True, status=STATUS_SIMULATED, provider=self.name,
                          provider_message_id=message_id)
