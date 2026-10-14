"""Mock Twilio adapter — backs two tool_catalog entries that share one provider
(twilio_human_handoff: transfer/warm_transfer/callback; twilio_notification_manager:
send_sms/send_email — see scripts/seed.py's BUILTIN_TOOL_CATALOG), plus
provision_number, called by POST /agents/{id}/phone-numbers (Vocal Backend
Tasksheet P4) when the caller doesn't supply an existing E.164 number. No real
Twilio API call — deterministic fake data keyed off the input.
"""
import hashlib

from app.integrations.base import BaseToolAdapter, ToolResult, register_adapter


@register_adapter
class TwilioAdapter(BaseToolAdapter):
    provider_name = "twilio"

    def execute(self, action: str, payload: dict) -> ToolResult:
        if action == "provision_number":
            area_code = payload.get("area_code") or "555"
            # Copilot review #23: deterministic mock — same inputs always produce
            # the same number, making tests reproducible.
            seed = f"{area_code}:{payload.get('tenant_id', '')}".encode()
            digits = int(hashlib.sha256(seed).hexdigest()[:7], 16) % 9000000 + 1000000
            mock_e164 = f"+1{area_code}{digits}"
            return ToolResult(success=True, data={"e164": mock_e164, "provider": "twilio"})
        if action == "transfer":
            return ToolResult(success=True, data={
                "call_id": payload.get("call_id"), "to": payload.get("to"), "transferred": True,
            })
        if action == "warm_transfer":
            return ToolResult(success=True, data={
                "call_id": payload.get("call_id"), "to": payload.get("to"),
                "briefing_delivered": True, "transferred": True,
            })
        if action == "callback":
            return ToolResult(success=True, data={
                "phone": payload.get("phone"), "requested_time": payload.get("requested_time"),
                "callback_scheduled": True,
            })
        if action == "send_sms":
            return ToolResult(success=True, data={
                "to": payload.get("to"), "message_id": "sms_mock_" + str(payload.get("to", "0"))[-4:],
                "sent": True,
            })
        if action == "send_email":
            return ToolResult(success=True, data={
                "to": payload.get("to"), "message_id": "email_mock_" + str(payload.get("to", "0"))[-4:],
                "sent": True,
            })
        return ToolResult(success=False, error=f"Unknown action '{action}' for twilio")
