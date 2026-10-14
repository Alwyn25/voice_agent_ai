"""BaseToolAdapter interface (Vocal Backend Tasksheet P1).

execute(action, payload) -> ToolResult, deterministic, no LLM inside. All 9 catalog
providers (Shopify, Stripe, Google Calendar, Zendesk, HubSpot, Shippo, Twilio,
Internal API, Recharge) plus the tenant-registrable Custom API Tool are implemented
as mock adapters — see app/integrations/adapters/. Every one returns deterministic
fake data with no real network call; swap a given adapter's method bodies for a real
provider SDK/API call once a tenant's real credentials are available.
`scripts/verify_adapters.py` runs a contract check against every registered adapter.
"""
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    success: bool
    data: dict = field(default_factory=dict)
    error: str | None = None


class BaseToolAdapter:
    """One instance per ToolConnection. Subclasses implement `execute`; nothing here
    ever calls an LLM — a tool adapter is a deterministic side-effect, not a decision."""
    provider_name: str = "base"

    def __init__(self, config: dict, credentials: dict):
        self.config = config
        self.credentials = credentials

    def execute(self, action: str, payload: dict) -> ToolResult:
        raise NotImplementedError

    def check_guardrails(self, action: str, payload: dict, guardrails: dict) -> str | None:
        """Returns a denial reason string, or None if the action is allowed. Adapters
        override this for provider-specific checks (e.g. Stripe's refund_limit_usd);
        the base implementation has no opinion."""
        return None


ADAPTER_REGISTRY: dict[str, type[BaseToolAdapter]] = {}


def register_adapter(cls: type[BaseToolAdapter]) -> type[BaseToolAdapter]:
    ADAPTER_REGISTRY[cls.provider_name] = cls
    return cls


def get_adapter(provider_name: str, config: dict, credentials: dict) -> BaseToolAdapter:
    cls = ADAPTER_REGISTRY.get(provider_name)
    if cls is None:
        raise ValueError(f"No adapter registered for provider '{provider_name}'")
    return cls(config, credentials)
