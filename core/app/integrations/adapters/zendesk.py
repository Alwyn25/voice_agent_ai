"""Mock Zendesk adapter (Ticket Manager). No real Zendesk API call — deterministic
fake data keyed off the input.
"""
from app.integrations.base import BaseToolAdapter, ToolResult, register_adapter


@register_adapter
class ZendeskAdapter(BaseToolAdapter):
    provider_name = "zendesk"

    def execute(self, action: str, payload: dict) -> ToolResult:
        if action == "create":
            return ToolResult(success=True, data={
                "ticket_id": "zd_mock_" + str(payload.get("subject", "ticket"))[:8],
                "subject": payload.get("subject"), "status": "open",
            })
        if action == "update":
            return ToolResult(success=True, data={
                "ticket_id": payload.get("ticket_id"), "status": payload.get("status", "open"),
                "updated": True,
            })
        if action == "escalate":
            return ToolResult(success=True, data={
                "ticket_id": payload.get("ticket_id"), "priority": "urgent", "escalated": True,
            })
        return ToolResult(success=False, error=f"Unknown action '{action}' for zendesk")
