"""Mock Internal API adapter — backs two tool_catalog entries that share one
provider (internal_customer_profile_manager: verify/fetch_history;
internal_feedback_manager: ask_rating/save_comment — see scripts/seed.py's
BUILTIN_TOOL_CATALOG). No real backend call — deterministic fake data keyed off
the input. A real implementation would call the tenant's own internal API using
this connection's stored base_url/api_key config.
"""
from app.integrations.base import BaseToolAdapter, ToolResult, register_adapter


@register_adapter
class InternalApiAdapter(BaseToolAdapter):
    provider_name = "internal_api"

    def execute(self, action: str, payload: dict) -> ToolResult:
        if action == "verify":
            return ToolResult(success=True, data={
                "customer_id": payload.get("customer_id"), "verified": True,
            })
        if action == "fetch_history":
            return ToolResult(success=True, data={
                "customer_id": payload.get("customer_id"),
                "history": [{"event": "order_placed", "at": "2026-08-01"}],
            })
        if action == "ask_rating":
            return ToolResult(success=True, data={
                "customer_id": payload.get("customer_id"), "prompted": True,
            })
        if action == "save_comment":
            return ToolResult(success=True, data={
                "customer_id": payload.get("customer_id"), "comment": payload.get("comment"), "saved": True,
            })
        return ToolResult(success=False, error=f"Unknown action '{action}' for internal_api")
