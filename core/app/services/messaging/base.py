"""The messaging abstraction: what every provider must accept and return.

Kept deliberately small and channel-agnostic in shape so a second channel (SMS, email) can
reuse SendResult unchanged. A provider NEVER raises for an expected failure — a bad number, a
provider 4xx, a timeout — it returns SendResult(ok=False, ...). The caller (the tool) turns
that into something honest to say; an exception is reserved for a genuine programming fault.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

# Delivery status vocabulary, shared by every provider so the tool and the audit row read the
# same words whoever sent the message:
#   sent      — the provider accepted it and reports it on its way
#   simulated — the stub "sent" it; nothing left the building (see StubWhatsAppProvider)
#   failed    — the provider rejected it or was unreachable; error is set
STATUS_SENT = "sent"
STATUS_SIMULATED = "simulated"
STATUS_FAILED = "failed"


@dataclass
class OutboundDocument:
    """One document to deliver. ``url`` must be reachable by the provider — the Cloud API
    fetches the file itself from this link."""
    url: str
    file_name: str
    caption: str | None = None
    mime_type: str | None = None


@dataclass
class SendResult:
    ok: bool
    status: str                       # STATUS_SENT | STATUS_SIMULATED | STATUS_FAILED
    provider: str
    provider_message_id: str | None = None
    error: str | None = None

    @property
    def simulated(self) -> bool:
        return self.status == STATUS_SIMULATED


class WhatsAppProvider(abc.ABC):
    """Send a document to a WhatsApp number. One method, so a provider is trivial to add and
    impossible to get subtly wrong."""

    #: short, stable identifier recorded on the delivery ("stub", "whatsapp_cloud")
    name: str = "base"

    @abc.abstractmethod
    async def send_document(self, *, to: str, document: OutboundDocument) -> SendResult:
        """Deliver ``document`` to E.164-ish ``to``. Never raises for an expected failure —
        returns SendResult(ok=False, status=STATUS_FAILED, error=...)."""
        raise NotImplementedError
