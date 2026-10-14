"""Tool registry: loads contracts from the DB and filters them to what is
executable right now. The planner (Stage 6) only ever sees the filtered list."""
import importlib
import uuid
from typing import Any, Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ToolDefinition

ImplFn = Callable[..., Awaitable[dict]]


async def get_tool(db: AsyncSession, tenant_id: uuid.UUID, name: str) -> ToolDefinition | None:
    return (await db.execute(select(ToolDefinition).where(
        ToolDefinition.tenant_id == tenant_id,
        ToolDefinition.name == name))).scalar_one_or_none()


async def list_tools(db: AsyncSession, tenant_id: uuid.UUID,
                     enabled_only: bool = True) -> list[ToolDefinition]:
    q = select(ToolDefinition).where(ToolDefinition.tenant_id == tenant_id)
    if enabled_only:
        q = q.where(ToolDefinition.is_enabled == True)  # noqa: E712
    return list((await db.execute(q.order_by(ToolDefinition.name))).scalars().all())


async def candidate_tools(db: AsyncSession, tenant_id: uuid.UUID, *,
                          current_state: str | None,
                          principal_kind: str = "agent") -> list[ToolDefinition]:
    """Planner step 4: only tools that could legally run right now.

    Filters: enabled, permission level allows the agent, and the customer's
    current journey state is in allowed_states (empty allowed_states = any state).
    """
    out = []
    for t in await list_tools(db, tenant_id, enabled_only=True):
        if t.permission_level in ("supervisor_only", "human_only") and principal_kind == "agent":
            continue
        if t.allowed_states and current_state and current_state not in t.allowed_states:
            continue
        out.append(t)
    return out


def load_implementation(spec: str) -> ImplFn:
    """spec: 'builtin:module.function' — resolved under app.engine.tools.builtin"""
    if not spec.startswith("builtin:"):
        raise ValueError(f"Unsupported implementation spec: {spec}")
    module_name, func_name = spec[len("builtin:"):].rsplit(".", 1)
    module = importlib.import_module(f"app.engine.tools.builtin.{module_name}")
    fn = getattr(module, func_name, None)
    if fn is None:
        raise ValueError(f"Implementation not found: {spec}")
    return fn


def tool_summary(t: ToolDefinition) -> dict[str, Any]:
    """Compact form handed to the LLM in the planner prompt."""
    return {"name": t.name, "description": t.description,
            "parameters": t.input_schema,
            "requires_confirmation": t.requires_confirmation}
