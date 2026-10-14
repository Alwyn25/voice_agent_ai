"""Provider selection — the one place that turns config into a concrete provider.

``whatsapp_provider`` (in .env) decides:
  * "auto"     — the default. Use the Cloud API when credentials are present, otherwise the
                 stub. This is what makes the feature "add the key and it works": the moment a
                 token and phone number id exist, real sends begin, with no code change.
  * "cloud_api" — force the real API. Missing credentials is a configuration error and raises,
                 so a production box that must never silently stub fails loudly at first send.
  * "stub"      — force the stub even if credentials exist (staging, demos, load tests).

The chosen provider is cached per (provider, token, phone-number-id) so a rotated credential in
a fresh process is picked up while a hot path does not rebuild an httpx-less object every call.
"""
from __future__ import annotations

import logging

from app.config import Settings, get_settings
from app.services.messaging.base import WhatsAppProvider
from app.services.messaging.stub import StubWhatsAppProvider
from app.services.messaging.whatsapp_cloud import WhatsAppCloudProvider

log = logging.getLogger(__name__)

_cache: dict[tuple, WhatsAppProvider] = {}


def _has_cloud_credentials(s: Settings) -> bool:
    return bool(s.whatsapp_access_token and s.whatsapp_phone_number_id)


def get_whatsapp_provider(settings: Settings | None = None) -> WhatsAppProvider:
    s = settings or get_settings()
    mode = (s.whatsapp_provider or "auto").strip().lower()
    key = (mode, s.whatsapp_access_token, s.whatsapp_phone_number_id,
           s.whatsapp_api_base_url, s.whatsapp_api_version)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    provider = _build(mode, s)
    _cache[key] = provider
    return provider


def _build(mode: str, s: Settings) -> WhatsAppProvider:
    if mode == "stub":
        return StubWhatsAppProvider()

    if mode == "cloud_api":
        if not _has_cloud_credentials(s):
            # Explicit request for the real API with nothing to authenticate it — a config
            # mistake worth failing on rather than quietly degrading to the stub.
            raise RuntimeError(
                "whatsapp_provider=cloud_api but WHATSAPP_ACCESS_TOKEN / "
                "WHATSAPP_PHONE_NUMBER_ID are not set")
        return _cloud(s)

    # "auto" (and any unrecognised value, treated as auto): cloud when we can, stub otherwise.
    if mode not in ("auto", ""):
        log.warning("unknown whatsapp_provider %r — treating as 'auto'", mode)
    if _has_cloud_credentials(s):
        return _cloud(s)
    log.info("whatsapp_provider=auto with no credentials — using the stub provider "
             "(set WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_NUMBER_ID to send for real)")
    return StubWhatsAppProvider()


def _cloud(s: Settings) -> WhatsAppCloudProvider:
    return WhatsAppCloudProvider(
        access_token=s.whatsapp_access_token,
        phone_number_id=s.whatsapp_phone_number_id,
        base_url=s.whatsapp_api_base_url,
        api_version=s.whatsapp_api_version,
        timeout_seconds=s.whatsapp_send_timeout_seconds)


def reset_provider_cache() -> None:
    """Drop the cached provider(s). For tests that flip settings between cases."""
    _cache.clear()
