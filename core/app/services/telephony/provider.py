"""Provider selection — config into a concrete carrier, the one place that decision lives.

``telephony_provider`` (in .env) decides, exactly like app/services/messaging/provider.py:

  * "stub"   — the default. Simulate the full call lifecycle in-process, no carrier account.
  * "plivo" / "exotel" / "twilio" — a real carrier. Its adapter reads its own credentials and
    raises loudly if they are missing, so a production box never silently stubs.
  * "auto"   — a real carrier when its credentials are present, else the stub.

Cached per (provider, credentials) so a rotated credential in a fresh process is picked up
while the dialer's hot path does not rebuild the provider every tick.

Real carrier adapters are intentionally not implemented in this stage (stub-first, per the
build plan): selecting one raises a clear NotImplementedError rather than pretending to dial.
Adding one is a new class implementing TelephonyProvider and a branch here — no other change.
"""
from __future__ import annotations

import logging

from app.config import Settings, get_settings
from app.services.telephony.base import TelephonyError, TelephonyProvider
from app.services.telephony.stub import StubTelephonyProvider

log = logging.getLogger(__name__)

_cache: dict[tuple, TelephonyProvider] = {}

# Which config fields must be present for each real carrier's credentials to count as "set".
_CREDENTIAL_FIELDS = {
    "plivo": ("plivo_auth_id", "plivo_auth_token"),
    "exotel": ("exotel_api_key", "exotel_api_token", "exotel_subdomain"),
    "twilio": ("twilio_account_sid", "twilio_auth_token"),
}


def _has_credentials(s: Settings, name: str) -> bool:
    return all(getattr(s, f, "") for f in _CREDENTIAL_FIELDS.get(name, ()))


def get_telephony_provider(settings: Settings | None = None) -> TelephonyProvider:
    s = settings or get_settings()
    mode = (s.telephony_provider or "stub").strip().lower()
    key = (mode, s.telephony_stub_outcome,
           s.plivo_auth_id, s.exotel_api_key, s.twilio_account_sid)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    provider = _build(mode, s)
    _cache[key] = provider
    return provider


def _build(mode: str, s: Settings) -> TelephonyProvider:
    if mode == "stub":
        return StubTelephonyProvider(s)

    if mode in _CREDENTIAL_FIELDS:
        if not _has_credentials(s, mode):
            raise TelephonyError(
                f"telephony_provider={mode} but its credentials are not set "
                f"({', '.join(_CREDENTIAL_FIELDS[mode])})")
        return _real_adapter(mode, s)

    if mode == "auto":
        for name in ("plivo", "exotel", "twilio"):
            if _has_credentials(s, name):
                return _real_adapter(name, s)
        log.info("telephony_provider=auto with no carrier credentials — using the stub")
        return StubTelephonyProvider(s)

    log.warning("unknown telephony_provider %r — treating as 'stub'", mode)
    return StubTelephonyProvider(s)


def _real_adapter(name: str, s: Settings) -> TelephonyProvider:
    if name == "plivo":
        from app.services.telephony.plivo import PlivoTelephonyProvider
        return PlivoTelephonyProvider(s)

    raise NotImplementedError(
        f"The {name} adapter is not implemented yet. Set TELEPHONY_PROVIDER=stub to exercise "
        f"the full outbound lifecycle without a carrier, or implement {name} as a "
        f"TelephonyProvider and wire it into app/services/telephony/provider._real_adapter.")


def reset_provider_cache() -> None:
    """Drop the cached provider(s). For tests that flip settings between cases."""
    _cache.clear()
