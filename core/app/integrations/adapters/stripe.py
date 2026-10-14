"""Mock Stripe adapter (Refund Manager). No real Stripe API call — deterministic
fake data. Implements the refund_limit_usd guardrail for real, since that is exactly
the example the tasksheet's guardrail_policies row names.
"""
from app.integrations.base import BaseToolAdapter, ToolResult, register_adapter


@register_adapter
class StripeAdapter(BaseToolAdapter):
    provider_name = "stripe"

    def check_guardrails(self, action: str, payload: dict, guardrails: dict) -> str | None:
        if action != "issue_refund":
            return None
        limit = guardrails.get("refund_limit_usd")
        amount = payload.get("amount_usd")
        if limit is not None and amount is not None and float(amount) > float(limit):
            return f"Refund amount ${amount} exceeds the configured limit of ${limit}"
        if guardrails.get("destructive_actions_enabled") is False:
            return "Destructive actions are disabled for this connection"
        return None

    def execute(self, action: str, payload: dict) -> ToolResult:
        if action == "quote_refund":
            return ToolResult(success=True, data={
                "charge_id": payload.get("charge_id"), "refundable_amount_usd": payload.get("amount_usd", "0.00"),
            })
        if action == "issue_refund":
            return ToolResult(success=True, data={
                "charge_id": payload.get("charge_id"), "amount_usd": payload.get("amount_usd"),
                "refund_id": "re_mock_" + str(payload.get("charge_id", "0"))[-8:], "status": "succeeded",
            })
        if action == "check_status":
            return ToolResult(success=True, data={"refund_id": payload.get("refund_id"), "status": "succeeded"})
        return ToolResult(success=False, error=f"Unknown action '{action}' for stripe")
