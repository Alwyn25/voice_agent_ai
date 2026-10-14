"""The stub billing provider: the default until a real payment processor is
configured — same "stub-first" stance as app/services/telephony/stub.py. Places no
real checkout: a synthetic session id/URL are returned, and `immediately_active=True`
tells the caller (app/api/billing.py) to activate the subscription right away rather
than waiting on a webhook that doesn't exist, since there is no hosted checkout page
for a customer to actually complete here.
"""
from __future__ import annotations

import uuid
from urllib.parse import urlencode, urlsplit, urlunsplit

from app.services.billing.base import BillingProvider, CheckoutSession


class StubBillingProvider(BillingProvider):
    name = "stub"

    async def create_checkout_session(self, *, tenant_id: str, plan: str, seats: int,
                                      success_url: str, cancel_url: str) -> CheckoutSession:
        session_id = f"stub_cs_{uuid.uuid4().hex[:24]}"
        parts = urlsplit(success_url)
        # `&`, never a second literal `?`, when success_url already carries a query
        # string (e.g. "...?success=1") — a plain f-string append got this wrong.
        extra = urlencode({"stub_session": session_id, "plan": plan, "seats": seats})
        query = f"{parts.query}&{extra}" if parts.query else extra
        checkout_url = urlunsplit((*parts[:3], query, parts.fragment))
        return CheckoutSession(provider=self.name, session_id=session_id,
                               checkout_url=checkout_url, immediately_active=True)
