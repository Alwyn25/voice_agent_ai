#!/usr/bin/env python3
"""Adapter contract check (Vocal Backend Tasksheet P1 Definition of Done: "Adapter
contract tests — every adapter tested against a mocked provider response").

Every registered BaseToolAdapter subclass is exercised against every action the
tasksheet lists for its provider, with a representative payload, asserting a
successful ToolResult with the expected data keys. No DB, no network — these are
pure in-memory mock adapters (see app/integrations/adapters/), so this needs
nothing but the Python environment:

    docker compose exec -T core python scripts/verify_adapters.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.integrations  # noqa: E402 — registers every adapter
from app.integrations.base import ADAPTER_REGISTRY, ToolResult  # noqa: E402

PASS, WARN, FAIL, RESET = "\033[92mPASS\033[0m", "\033[93mWARN\033[0m", "\033[91mFAIL\033[0m", "\033[0m"

# (provider_name, action, payload, required_data_keys)
CASES: list[tuple[str, str, dict, list[str]]] = [
    ("shopify", "lookup_order", {"order_id": "1001"}, ["order_id", "status", "items"]),
    ("shopify", "track_order", {"order_id": "1001"}, ["carrier", "tracking_number"]),
    ("shopify", "reschedule_delivery", {"order_id": "1001", "new_date": "2026-10-01"}, ["rescheduled"]),
    ("shopify", "cancel_order", {"order_id": "1001"}, ["cancelled"]),

    ("stripe", "quote_refund", {"charge_id": "ch_1", "amount_usd": "20.00"}, ["refundable_amount_usd"]),
    ("stripe", "issue_refund", {"charge_id": "ch_1", "amount_usd": "20.00"}, ["refund_id", "status"]),
    ("stripe", "check_status", {"refund_id": "re_1"}, ["status"]),

    ("google_calendar", "check_availability", {"calendar_id": "primary"}, ["slots"]),
    ("google_calendar", "create", {"start_time": "2026-09-20T10:00:00Z"}, ["event_id", "status"]),
    ("google_calendar", "reschedule", {"event_id": "evt_1", "new_start_time": "2026-09-21T10:00:00Z"}, ["status"]),
    ("google_calendar", "cancel", {"event_id": "evt_1"}, ["status"]),

    ("zendesk", "create", {"subject": "Order issue"}, ["ticket_id", "status"]),
    ("zendesk", "update", {"ticket_id": "zd_1", "status": "pending"}, ["updated"]),
    ("zendesk", "escalate", {"ticket_id": "zd_1"}, ["escalated"]),

    ("hubspot", "find_contact", {"email": "a@example.com"}, ["contact_id", "found"]),
    ("hubspot", "update_contact", {"contact_id": "hs_1", "properties": {"stage": "won"}}, ["updated"]),
    ("hubspot", "log_activity", {"contact_id": "hs_1", "activity_type": "call"}, ["logged"]),

    ("shippo", "track", {"tracking_number": "TRK1"}, ["status", "eta"]),
    ("shippo", "re_route", {"tracking_number": "TRK1", "new_address": "123 Main St"}, ["rerouted"]),
    ("shippo", "hold", {"tracking_number": "TRK1"}, ["held"]),

    ("twilio", "transfer", {"call_id": "c1", "to": "+15550001111"}, ["transferred"]),
    ("twilio", "warm_transfer", {"call_id": "c1", "to": "+15550001111"}, ["briefing_delivered", "transferred"]),
    ("twilio", "callback", {"phone": "+15550001111", "requested_time": "2026-09-20T10:00:00Z"},
     ["callback_scheduled"]),
    ("twilio", "send_sms", {"to": "+15550001111"}, ["sent", "message_id"]),
    ("twilio", "send_email", {"to": "a@example.com"}, ["sent", "message_id"]),
    ("twilio", "provision_number", {"area_code": "415"}, ["e164", "provider"]),

    ("internal_api", "verify", {"customer_id": "cust_1"}, ["verified"]),
    ("internal_api", "fetch_history", {"customer_id": "cust_1"}, ["history"]),
    ("internal_api", "ask_rating", {"customer_id": "cust_1"}, ["prompted"]),
    ("internal_api", "save_comment", {"customer_id": "cust_1", "comment": "Great service"}, ["saved"]),

    ("recharge", "pause", {"subscription_id": "sub_1"}, ["status"]),
    ("recharge", "skip", {"subscription_id": "sub_1", "date": "2026-10-01"}, ["status"]),
    ("recharge", "cancel", {"subscription_id": "sub_1"}, ["status"]),
    ("recharge", "resume", {"subscription_id": "sub_1"}, ["status"]),

    ("custom_api", "call", {"foo": "bar"}, ["method", "endpoint_url", "response"]),
]

# Guardrail-specific checks that go beyond a plain execute() call.
GUARDRAIL_CASES = [
    ("stripe", "issue_refund", {"amount_usd": "500"}, {"refund_limit_usd": 200}, "exceeds"),
    ("stripe", "issue_refund", {"amount_usd": "50"}, {"refund_limit_usd": 200}, None),
]


def main() -> int:
    failures = 0
    tested_providers = set()

    for provider, action, payload, required_keys in CASES:
        cls = ADAPTER_REGISTRY.get(provider)
        if cls is None:
            print(f"{FAIL} {provider}.{action}: no adapter registered for '{provider}'")
            failures += 1
            continue
        tested_providers.add(provider)
        config = {"method": "GET", "endpoint_url": "https://example.com/hook"} if provider == "custom_api" else {}
        adapter = cls(config, {})
        result: ToolResult = adapter.execute(action, payload)
        if not result.success:
            print(f"{FAIL} {provider}.{action}: expected success, got error={result.error!r}")
            failures += 1
            continue
        missing = [k for k in required_keys if k not in result.data]
        if missing:
            print(f"{FAIL} {provider}.{action}: missing data keys {missing} (got {list(result.data)})")
            failures += 1
            continue
        print(f"{PASS} {provider}.{action}")

    unknown = ADAPTER_REGISTRY.get("shopify")
    if unknown is not None:
        result = unknown({}, {}).execute("not_a_real_action", {})
        if result.success:
            print(f"{FAIL} shopify.not_a_real_action: expected failure for an unknown action")
            failures += 1
        else:
            print(f"{PASS} shopify.not_a_real_action -> correctly rejected")

    for provider, action, payload, guardrails, expect_substr in GUARDRAIL_CASES:
        cls = ADAPTER_REGISTRY[provider]
        adapter = cls({}, {})
        denial = adapter.check_guardrails(action, payload, guardrails)
        if expect_substr is None:
            ok = denial is None
        else:
            ok = denial is not None and expect_substr in denial
        label = f"{provider}.{action} guardrail({guardrails})"
        if ok:
            print(f"{PASS} {label} -> {denial!r}")
        else:
            print(f"{FAIL} {label} -> unexpected result {denial!r}")
            failures += 1

    registered_not_tested = set(ADAPTER_REGISTRY) - tested_providers
    if registered_not_tested:
        print(f"{WARN} registered but not covered by any case: {sorted(registered_not_tested)}")

    print()
    if failures:
        print(f"{FAIL} {failures} check(s) failed")
        return 1
    print(f"{PASS} all adapter contract checks passed ({len(CASES)} actions across {len(tested_providers)} providers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
