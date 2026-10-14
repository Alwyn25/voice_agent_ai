"""Mock Google Calendar adapter (Appointment Manager). No real Calendar API call —
deterministic fake data keyed off the input. Swap the method bodies for real
`google-api-python-client` calls once a tenant's real OAuth grant is available;
check_guardrails/execute's signatures do not need to change.
"""
from app.integrations.base import BaseToolAdapter, ToolResult, register_adapter


@register_adapter
class GoogleCalendarAdapter(BaseToolAdapter):
    provider_name = "google_calendar"

    def execute(self, action: str, payload: dict) -> ToolResult:
        if action == "check_availability":
            return ToolResult(success=True, data={
                "calendar_id": payload.get("calendar_id", "primary"),
                "slots": ["2026-09-20T10:00:00Z", "2026-09-20T14:00:00Z"],
            })
        if action == "create":
            return ToolResult(success=True, data={
                "event_id": "evt_mock_" + str(payload.get("start_time", "0"))[-8:],
                "start_time": payload.get("start_time"), "end_time": payload.get("end_time"),
                "status": "confirmed",
            })
        if action == "reschedule":
            return ToolResult(success=True, data={
                "event_id": payload.get("event_id"), "start_time": payload.get("new_start_time"),
                "status": "confirmed",
            })
        if action == "cancel":
            return ToolResult(success=True, data={"event_id": payload.get("event_id"), "status": "cancelled"})
        return ToolResult(success=False, error=f"Unknown action '{action}' for google_calendar")
