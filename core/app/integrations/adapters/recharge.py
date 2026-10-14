"""Mock Recharge adapter (Subscription Manager). No real Recharge API call —
deterministic fake data keyed off the input.
"""
from app.integrations.base import BaseToolAdapter, ToolResult, register_adapter


@register_adapter
class RechargeAdapter(BaseToolAdapter):
    provider_name = "recharge"

    def execute(self, action: str, payload: dict) -> ToolResult:
        if action == "pause":
            return ToolResult(success=True, data={
                "subscription_id": payload.get("subscription_id"), "status": "paused",
            })
        if action == "skip":
            return ToolResult(success=True, data={
                "subscription_id": payload.get("subscription_id"),
                "skipped_date": payload.get("date"), "status": "active",
            })
        if action == "cancel":
            return ToolResult(success=True, data={
                "subscription_id": payload.get("subscription_id"), "status": "cancelled",
            })
        if action == "resume":
            return ToolResult(success=True, data={
                "subscription_id": payload.get("subscription_id"), "status": "active",
            })
        return ToolResult(success=False, error=f"Unknown action '{action}' for recharge")
