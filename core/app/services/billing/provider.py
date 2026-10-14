"""Provider selection — mirrors app/services/telephony/provider.py exactly:

  * "stub"   — the default. No processor account needed.
  * "stripe" — a real processor. Not implemented yet (stub-first, per the build
    plan); selecting it raises a clear NotImplementedError rather than pretending
    to charge a card. Adding it is a new class implementing BillingProvider and a
    branch here — no other change (app/api/billing.py never imports a concrete
    provider directly).
"""
from __future__ import annotations

import logging

from app.config import Settings, get_settings
from app.services.billing.base import BillingError, BillingProvider
from app.services.billing.stub import StubBillingProvider

log = logging.getLogger(__name__)

_cache: dict[str, BillingProvider] = {}


def get_billing_provider(settings: Settings | None = None) -> BillingProvider:
    s = settings or get_settings()
    mode = (s.billing_provider or "stub").strip().lower()
    cached = _cache.get(mode)
    if cached is not None:
        return cached
    provider = _build(mode, s)
    _cache[mode] = provider
    return provider


def _build(mode: str, s: Settings) -> BillingProvider:
    if mode == "stub":
        return StubBillingProvider()
    if mode == "stripe":
        if not s.stripe_secret_key:
            raise BillingError(
                "billing_provider=stripe but stripe_secret_key is not set. Set "
                "BILLING_PROVIDER=stub to exercise the checkout flow without a "
                "processor account, or configure STRIPE_SECRET_KEY.")
        raise NotImplementedError(
            "The stripe billing provider is not implemented yet. Set "
            "BILLING_PROVIDER=stub, or implement StripeBillingProvider "
            "(app/services/billing/base.py's BillingProvider) and wire it in here.")
    log.warning("unknown billing_provider %r — treating as 'stub'", mode)
    return StubBillingProvider()


def reset_provider_cache() -> None:
    _cache.clear()
