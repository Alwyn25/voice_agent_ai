"""Mock Shopify adapter (Order Manager). No real Shopify API call — deterministic
fake data keyed off the input, so the same call always returns the same shape. Swap
the method bodies for real `shopify_api` calls once a tenant's real store token is
available; check_guardrails/execute's signatures do not need to change.
"""
from app.integrations.base import BaseToolAdapter, ToolResult, register_adapter


@register_adapter
class ShopifyAdapter(BaseToolAdapter):
    provider_name = "shopify"

    def execute(self, action: str, payload: dict) -> ToolResult:
        if action == "lookup_order":
            order_id = payload.get("order_id", "unknown")
            return ToolResult(success=True, data={
                "order_id": order_id, "status": "fulfilled",
                "items": [{"sku": "SKU-1001", "qty": 1, "name": "Sample Product"}],
                "total": "49.99", "currency": "USD",
            })
        if action == "track_order":
            return ToolResult(success=True, data={
                "order_id": payload.get("order_id"), "carrier": "UPS",
                "tracking_number": "1Z999AA10123456784", "eta": "2026-09-20",
            })
        if action == "reschedule_delivery":
            return ToolResult(success=True, data={
                "order_id": payload.get("order_id"), "new_date": payload.get("new_date"),
                "rescheduled": True,
            })
        if action == "cancel_order":
            return ToolResult(success=True, data={"order_id": payload.get("order_id"), "cancelled": True})
        return ToolResult(success=False, error=f"Unknown action '{action}' for shopify")
