"""Tool registry admin: inspect contracts, enable/disable, dry-run test."""
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import AuditEvent, Customer
from app.engine.tools import contracts, registry
from app.engine.tools.executor import execute, validate_params
from app.engine.tools.types import ToolContext
from app.security.deps import Principal, require

router = APIRouter(prefix="/tools", tags=["tools"])


@router.get("")
async def list_tools(enabled_only: bool = False,
                     p: Principal = Depends(require("tools:read")),
                     db: AsyncSession = Depends(get_db)):
    rows = await registry.list_tools(db, p.tenant_id, enabled_only=enabled_only)
    return [{"name": t.name, "category": t.category, "description": t.description,
             "is_enabled": t.is_enabled, "requires_confirmation": t.requires_confirmation,
             "permission_level": t.permission_level, "allowed_states": t.allowed_states,
             "input_schema": t.input_schema,
             "preconditions": [c.get("check") for c in (t.preconditions or [])],
             "postconditions": [c.get("check") for c in (t.postconditions or [])],
             "effects": t.effects, "implementation": t.implementation} for t in rows]


@router.get("/checks")
async def list_checks(p: Principal = Depends(require("tools:read"))):
    """The check library available to tool contracts."""
    return {"checks": contracts.available_checks()}


@router.get("/candidates")
async def candidates(customer_id: uuid.UUID | None = None,
                     p: Principal = Depends(require("tools:read")),
                     db: AsyncSession = Depends(get_db)):
    """What the planner would see right now for this customer (step 4 preview)."""
    state = None
    if customer_id:
        c = (await db.execute(select(Customer).where(
            Customer.tenant_id == p.tenant_id, Customer.id == customer_id))).scalar_one_or_none()
        if c is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found")
        state = c.current_state
    tools = await registry.candidate_tools(db, p.tenant_id, current_state=state)
    return {"current_state": state, "candidate_tools": [registry.tool_summary(t) for t in tools],
            "excluded": [t.name for t in await registry.list_tools(db, p.tenant_id)
                         if t.name not in {x.name for x in tools}]}


class ToolPatch(BaseModel):
    is_enabled: bool | None = None
    requires_confirmation: bool | None = None
    permission_level: str | None = None
    allowed_states: list[str] | None = None
    description: str | None = None


@router.patch("/{tool_name}")
async def patch_tool(tool_name: str, body: ToolPatch,
                     p: Principal = Depends(require("tools:write")),
                     db: AsyncSession = Depends(get_db)):
    t = await registry.get_tool(db, p.tenant_id, tool_name)
    if t is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Tool not found")
    changes = body.model_dump(exclude_unset=True)
    before = {k: getattr(t, k) for k in changes}
    for k, v in changes.items():
        setattr(t, k, v)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="tools.update",
                      resource_type="tool", resource_id=tool_name,
                      before=_json(before), after=_json(changes)))
    await db.commit()
    return {"name": t.name, "is_enabled": t.is_enabled,
            "requires_confirmation": t.requires_confirmation,
            "permission_level": t.permission_level, "allowed_states": t.allowed_states}


class ToolTest(BaseModel):
    params: dict
    customer_id: uuid.UUID | None = None
    execute_for_real: bool = False


@router.post("/{tool_name}/test")
async def test_tool(tool_name: str, body: ToolTest,
                    p: Principal = Depends(require("tools:read")),
                    db: AsyncSession = Depends(get_db)):
    """Dry-run the full validation pipeline. Shows exactly which gate would stop it.
    Set execute_for_real=true to actually run it (requires tools:write)."""
    if body.execute_for_real and "tools:write" not in p.permissions and "*" not in p.permissions:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "execute_for_real requires tools:write")

    t = await registry.get_tool(db, p.tenant_id, tool_name)
    if t is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Tool not found")

    customer = None
    if body.customer_id:
        customer = (await db.execute(select(Customer).where(
            Customer.tenant_id == p.tenant_id,
            Customer.id == body.customer_id))).scalar_one_or_none()
        if customer is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found")

    ctx = ToolContext(db=db, tenant_id=p.tenant_id, actor=p.actor, customer=customer)
    outcome = await execute(ctx, tool_name, body.params,
                            principal_kind="agent", dry_run=not body.execute_for_real)
    if body.execute_for_real and outcome.status in ("success", "skipped_duplicate"):
        await db.commit()
    else:
        await db.rollback()

    return {"tool": tool_name, "mode": "executed" if body.execute_for_real else "dry_run",
            "status": outcome.status, "error": outcome.error,
            "failed_check": outcome.failed_check, "failure_action": outcome.failure_action,
            "output": outcome.output, "state_changed_to": outcome.state_changed_to,
            "validation": outcome.validation, "duration_ms": outcome.duration_ms}


class SchemaCheck(BaseModel):
    params: dict


@router.post("/{tool_name}/validate-params")
async def validate_only(tool_name: str, body: SchemaCheck,
                        p: Principal = Depends(require("tools:read")),
                        db: AsyncSession = Depends(get_db)):
    t = await registry.get_tool(db, p.tenant_id, tool_name)
    if t is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Tool not found")
    errors = validate_params(t.input_schema or {}, body.params)
    return {"valid": not errors, "errors": errors}


def _json(d: dict) -> dict:
    return {k: (str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v)
            for k, v in d.items()}
