"""Telephony: placing outbound PSTN calls and turning carrier events into Call-row state.

The public surface is deliberately small:

  * ``get_telephony_provider()`` — the configured carrier (stub by default), chosen the same
    way the WhatsApp messaging provider is (app/services/messaging/provider.py).
  * ``TelephonyProvider`` / ``OriginateResult`` / ``CallEvent`` — the abstraction every
    carrier adapter implements, and the canonical event every adapter (and the stub) funnels
    into ``apply_call_event`` so the webhook endpoint and the simulator share one code path.
  * ``apply_call_event`` — the ONE function that mutates a Call row from a carrier event and
    schedules retries; nothing else writes call lifecycle state.
  * ``dial_pending`` — the dialer worker tick: originate queued calls within the concurrency
    cap and campaign pacing, re-checking policy at dial time.
"""
from app.services.telephony.base import (CallEvent, OriginateResult, TelephonyError,
                                          TelephonyProvider, TERMINAL_STATUSES)
from app.services.telephony.events import apply_call_event
from app.services.telephony.provider import get_telephony_provider, reset_provider_cache

__all__ = ["CallEvent", "OriginateResult", "TelephonyError", "TelephonyProvider",
           "TERMINAL_STATUSES", "apply_call_event", "get_telephony_provider",
           "reset_provider_cache"]
