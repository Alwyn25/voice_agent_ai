"""Shared types for the tool framework."""
import uuid
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Customer

# Tools that only READ state -- they change no record and commit the customer to nothing.
# One definition shared by every consumer that must agree on it: the planner's confidence
# banding (a lookup is never demoted to "confirm" -- there is nothing to confirm) and
# turn_runner's TTS pipelining (safe to speak the first sentence before grounding settles).
READ_ONLY_TOOLS = frozenset({
    "search_slots", "slot_search", "inventory_query",
    "knowledge_lookup", "my_appointments", "my_tickets",
    "locate_centre", "verify_identity",
})


@dataclass
class ToolContext:
    """Everything a tool implementation or check may need."""
    db: AsyncSession
    tenant_id: uuid.UUID
    actor: str = "agent"
    customer: Customer | None = None
    conversation_id: uuid.UUID | None = None
    decision_trace_id: uuid.UUID | None = None
    identity_verified: bool = False
    # The project this call resolved to (multi-project-per-tenant), so a tool's own retrieval
    # or booking scopes to the same project the planner did. None for tenant-wide tenants and
    # while the project is still ambiguous.
    project_id: uuid.UUID | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckResult:
    ok: bool
    check: str
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolOutcome:
    """Result of an execution attempt. status is the single source of truth."""
    status: str                     # success | blocked | failed | skipped_duplicate
    tool_name: str
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    failed_check: str | None = None
    failure_action: dict[str, Any] = field(default_factory=dict)   # from failure_policy
    state_changed_to: str | None = None
    execution_id: uuid.UUID | None = None
    duration_ms: int = 0
    validation: dict[str, Any] = field(default_factory=dict)
