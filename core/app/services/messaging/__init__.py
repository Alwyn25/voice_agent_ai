"""Outbound messaging — provider-pluggable delivery of documents to a caller.

The engine talks only to the abstractions here (``WhatsAppProvider``, ``OutboundDocument``,
``SendResult``); which concrete provider runs is a config decision resolved by
``get_whatsapp_provider``. Today that is the Meta WhatsApp Cloud API or a stub; a new channel
is a new provider class, no change to the tool that calls it.
"""
from app.services.messaging.base import (OutboundDocument, SendResult, WhatsAppProvider)
from app.services.messaging.provider import get_whatsapp_provider, reset_provider_cache

__all__ = ["OutboundDocument", "SendResult", "WhatsAppProvider",
           "get_whatsapp_provider", "reset_provider_cache"]
