"""Mock HubSpot adapter (CRM Manager). No real HubSpot API call — deterministic
fake data keyed off the input.
"""
from app.integrations.base import BaseToolAdapter, ToolResult, register_adapter


@register_adapter
class HubSpotAdapter(BaseToolAdapter):
    provider_name = "hubspot"

    def execute(self, action: str, payload: dict) -> ToolResult:
        if action == "find_contact":
            return ToolResult(success=True, data={
                "contact_id": "hs_mock_" + str(payload.get("email", "contact"))[:8],
                "email": payload.get("email"), "found": True,
            })
        if action == "update_contact":
            return ToolResult(success=True, data={
                "contact_id": payload.get("contact_id"), "properties": payload.get("properties", {}),
                "updated": True,
            })
        if action == "log_activity":
            return ToolResult(success=True, data={
                "contact_id": payload.get("contact_id"), "activity_type": payload.get("activity_type", "note"),
                "logged": True,
            })
        return ToolResult(success=False, error=f"Unknown action '{action}' for hubspot")
