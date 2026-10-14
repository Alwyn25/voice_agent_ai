"""Mock Custom API Tool adapter — backs a tenant's self-registered custom tool
(POST /integrations/tools/custom; see app/api/integrations_tools.py's
register_custom_tool, which stores {method, endpoint_url, description} as the
catalog entry's single action). No real HTTP call here — a real implementation
would issue `self.config["method"]` against `self.config["endpoint_url"]` with
`payload` as the body/query. Kept a mock like every other adapter in this pass:
deterministic fake data so a contract test never depends on network access.
"""
from app.integrations.base import BaseToolAdapter, ToolResult, register_adapter


@register_adapter
class CustomApiAdapter(BaseToolAdapter):
    provider_name = "custom_api"

    def execute(self, action: str, payload: dict) -> ToolResult:
        method = self.config.get("method", "GET")
        endpoint_url = self.config.get("endpoint_url")
        if not endpoint_url:
            return ToolResult(success=False, error="Custom tool connection has no endpoint_url configured")
        return ToolResult(success=True, data={
            "method": method, "endpoint_url": endpoint_url, "request_payload": payload,
            "response": {"mock": True, "status_code": 200},
        })
