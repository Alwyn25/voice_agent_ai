"""What every billing provider must implement (Vocal Backend Tasksheet P8), mirroring
app/services/telephony/base.py's split: a small carrier-neutral interface, with the
concrete choice (stub vs a real processor) made once in provider.py.

A provider does one thing — start a checkout for a plan/seat change and hand back
where to send the customer. Nothing here talks to a database; app/api/billing.py
owns the Subscription/Invoice rows, exactly like app/services/telephony/dialer.py
owns the Call row rather than a TelephonyProvider touching it directly.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass


class BillingError(RuntimeError):
    """A genuine fault (misconfiguration, programming error) — not a normal decline,
    which a real processor would report through its own checkout-session status."""


@dataclass
class CheckoutSession:
    provider: str
    session_id: str
    checkout_url: str
    # A real processor's checkout completes asynchronously (the customer pays on a
    # hosted page, then a webhook confirms it) — `immediately_active` tells the
    # caller whether THIS provider already knows the outcome (the stub does; a real
    # one wouldn't) so app/api/billing.py knows whether to activate the subscription
    # now or wait for a webhook that doesn't exist yet.
    immediately_active: bool = False


class BillingProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    async def create_checkout_session(self, *, tenant_id: str, plan: str, seats: int,
                                      success_url: str, cancel_url: str) -> CheckoutSession:
        """Start a checkout for a plan/seat change. Never raises for an expected
        decline — that would come back as a status on the session once a real
        provider's webhook confirms it; BillingError is reserved for a genuine
        programming/configuration fault, same split as TelephonyError."""
        raise NotImplementedError
