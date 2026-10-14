"""Mock Shippo adapter (Delivery Manager). No real Shippo API call — deterministic
fake data keyed off the input.
"""
from app.integrations.base import BaseToolAdapter, ToolResult, register_adapter


@register_adapter
class ShippoAdapter(BaseToolAdapter):
    provider_name = "shippo"

    def execute(self, action: str, payload: dict) -> ToolResult:
        if action == "track":
            return ToolResult(success=True, data={
                "tracking_number": payload.get("tracking_number"), "carrier": "USPS",
                "status": "in_transit", "eta": "2026-09-22",
            })
        if action == "re_route":
            return ToolResult(success=True, data={
                "tracking_number": payload.get("tracking_number"),
                "new_address": payload.get("new_address"), "rerouted": True,
            })
        if action == "hold":
            return ToolResult(success=True, data={
                "tracking_number": payload.get("tracking_number"), "held": True,
            })
        return ToolResult(success=False, error=f"Unknown action '{action}' for shippo")
