"""Vocal Backend Tasksheet P1 (Tool Library) — the connectable third-party tool
marketplace. Served under /integrations/tools rather than the tasksheet's literal
/tools path, because /tools already exists and serves a different, currently-working
purpose (the voice engine's built-in tool admin API — see app/api/tools_admin.py).
See app/db/models/integrations.py for why the schema is kept separate too.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

import app.integrations  # noqa: F401 — registers every mock adapter (see app/integrations/adapters/)
from app.db.base import get_db
from app.db.models import GuardrailPolicy, ToolCatalogEntry, ToolConnection, User
from app.security.deps import Principal, require
from app.services.activity import append_activity_event
from app.services.credentials import encrypt_credentials

router = APIRouter(prefix="/integrations/tools", tags=["tool-marketplace"])


async def _connection_for(db: AsyncSession, tenant_id: uuid.UUID, catalog_id: uuid.UUID) -> ToolConnection | None:
    return (await db.execute(select(ToolConnection).where(
        ToolConnection.tenant_id == tenant_id,
        ToolConnection.tool_catalog_id == catalog_id))).scalars().first()


async def _other_admin_ids(db: AsyncSession, tenant_id: uuid.UUID, exclude_user_id: uuid.UUID | None) -> list[uuid.UUID]:
    """Other admins/owners/ops_leads in the org, so a tool-marketplace change one
    of them makes shows up as a bell notification for the others — the concrete
    trigger that exercises the notifications plumbing (P0) end-to-end, not just
    its schema. ops_lead is included alongside admin/owner because rbac.py's
    DEFAULT_ROLES grants it the same tools:write/knowledge:write permissions —
    an org whose only elevated users are ops_leads would otherwise never see
    these notifications at all."""
    q = select(User.id).where(User.tenant_id == tenant_id, User.role.in_(["admin", "owner", "ops_lead"]))
    if exclude_user_id is not None:
        q = q.where(User.id != exclude_user_id)
    return list((await db.execute(q)).scalars().all())


def _catalog_out(entry: ToolCatalogEntry, connection: ToolConnection | None) -> dict:
    return {
        "id": str(entry.id), "key": entry.key, "category": entry.category,
        "provider_name": entry.provider_name, "actions": entry.actions,
        "action_schema": entry.action_schema, "is_custom": entry.is_custom,
        "connection": None if connection is None else {
            "id": str(connection.id), "status": connection.status,
            "config": connection.config, "guardrails": connection.guardrails,
            "created_at": connection.created_at, "updated_at": connection.updated_at,
        },
    }


@router.get("")
async def list_tools(category: str | None = None, search: str | None = None,
                     p: Principal = Depends(require("tools:read")),
                     db: AsyncSession = Depends(get_db)):
    q = select(ToolCatalogEntry).where(ToolCatalogEntry.tenant_id == p.tenant_id)
    if category:
        q = q.where(ToolCatalogEntry.category == category)
    if search:
        q = q.where(ToolCatalogEntry.provider_name.ilike(f"%{search}%") |
                    ToolCatalogEntry.key.ilike(f"%{search}%"))
    entries = (await db.execute(q.order_by(ToolCatalogEntry.provider_name))).scalars().all()

    connections = {c.tool_catalog_id: c for c in (await db.execute(select(ToolConnection).where(
        ToolConnection.tenant_id == p.tenant_id))).scalars().all()}
    return [_catalog_out(e, connections.get(e.id)) for e in entries]


@router.get("/{tool_id}")
async def get_tool(tool_id: uuid.UUID, p: Principal = Depends(require("tools:read")),
                   db: AsyncSession = Depends(get_db)):
    entry = (await db.execute(select(ToolCatalogEntry).where(
        ToolCatalogEntry.tenant_id == p.tenant_id, ToolCatalogEntry.id == tool_id))).scalar_one_or_none()
    if entry is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Tool not found")
    return _catalog_out(entry, await _connection_for(db, p.tenant_id, tool_id))


class ConnectBody(BaseModel):
    credentials: dict = {}
    config: dict = {}


@router.post("/{tool_id}/connect", status_code=status.HTTP_201_CREATED)
async def connect_tool(tool_id: uuid.UUID, body: ConnectBody,
                       p: Principal = Depends(require("tools:write")),
                       db: AsyncSession = Depends(get_db)):
    """Mock OAuth/API-key entry — real credentials are encrypted at rest and never
    echoed back; a real OAuth redirect flow would replace this with a callback."""
    entry = (await db.execute(select(ToolCatalogEntry).where(
        ToolCatalogEntry.tenant_id == p.tenant_id, ToolCatalogEntry.id == tool_id))).scalar_one_or_none()
    if entry is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Tool not found")

    connection = await _connection_for(db, p.tenant_id, tool_id)
    encrypted = encrypt_credentials(body.credentials) if body.credentials else None
    if connection is None:
        connection = ToolConnection(tenant_id=p.tenant_id, tool_catalog_id=tool_id,
                                    status="connected", credentials_encrypted=encrypted,
                                    config=body.config)
        db.add(connection)
    else:
        connection.status = "connected"
        if encrypted:
            connection.credentials_encrypted = encrypted
        connection.config = body.config or connection.config
    try:
        await db.flush()
    except IntegrityError:
        # Two concurrent connect requests for the same tool raced each other; the
        # unique constraint on (tenant_id, tool_catalog_id) caught it. Whoever lost
        # the race just re-reads the winner's row rather than erroring.
        await db.rollback()
        connection = await _connection_for(db, p.tenant_id, tool_id)
        return _catalog_out(entry, connection)
    append_activity_event(db, p, "tools.connect", resource_type="tool_connection",
                          resource_id=str(connection.id), after={"provider": entry.provider_name},
                          notify_user_ids=await _other_admin_ids(db, p.tenant_id, p.user_id))
    await db.commit()
    return _catalog_out(entry, connection)


class CustomToolBody(BaseModel):
    key: str
    method: str  # GET|POST
    endpoint_url: str
    description: str = ""
    action_schema: dict = {}


@router.post("/custom", status_code=status.HTTP_201_CREATED)
async def register_custom_tool(body: CustomToolBody, p: Principal = Depends(require("tools:write")),
                               db: AsyncSession = Depends(get_db)):
    existing = (await db.execute(select(ToolCatalogEntry).where(
        ToolCatalogEntry.tenant_id == p.tenant_id, ToolCatalogEntry.key == body.key))).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "A tool with that key already exists")

    entry = ToolCatalogEntry(
        tenant_id=p.tenant_id, key=body.key, category="custom", provider_name="custom_api",
        actions=[{"method": body.method, "endpoint_url": body.endpoint_url, "description": body.description}],
        action_schema=body.action_schema, is_custom=True)
    db.add(entry)
    await db.flush()
    append_activity_event(db, p, "tools.custom_register", resource_type="tool_catalog",
                          resource_id=str(entry.id), after={"key": entry.key},
                          notify_user_ids=await _other_admin_ids(db, p.tenant_id, p.user_id))
    await db.commit()
    return _catalog_out(entry, None)


class ConnectionPatch(BaseModel):
    config: dict | None = None
    guardrails: dict | None = None


async def _get_connection(db: AsyncSession, tenant_id: uuid.UUID, connection_id: uuid.UUID) -> ToolConnection:
    row = (await db.execute(select(ToolConnection).where(
        ToolConnection.tenant_id == tenant_id, ToolConnection.id == connection_id))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Connection not found")
    return row


@router.patch("/connections/{connection_id}")
async def patch_connection(connection_id: uuid.UUID, body: ConnectionPatch,
                           p: Principal = Depends(require("tools:write")),
                           db: AsyncSession = Depends(get_db)):
    row = await _get_connection(db, p.tenant_id, connection_id)
    before = {"config": row.config, "guardrails": row.guardrails}
    if body.config is not None:
        row.config = body.config
    if body.guardrails is not None:
        row.guardrails = body.guardrails
        # Mirror into normalised guardrail_policies rows so they stay queryable/named —
        # add/update keys present in the new dict, and delete rows for keys removed
        # from it, so this table never drifts out of sync with the JSONB blob above.
        existing = {g.key: g for g in (await db.execute(select(GuardrailPolicy).where(
            GuardrailPolicy.tool_connection_id == row.id))).scalars().all()}
        for key, value in body.guardrails.items():
            if key in existing:
                existing[key].value = {"value": value}
            else:
                db.add(GuardrailPolicy(tenant_id=p.tenant_id, tool_connection_id=row.id,
                                       key=key, value={"value": value}))
        for key, policy in existing.items():
            if key not in body.guardrails:
                await db.delete(policy)
    append_activity_event(db, p, "tools.connection_update", resource_type="tool_connection",
                          resource_id=str(row.id), before=before,
                          after={"config": row.config, "guardrails": row.guardrails},
                          notify_user_ids=await _other_admin_ids(db, p.tenant_id, p.user_id))
    await db.commit()
    await db.refresh(row)
    return {"id": str(row.id), "status": row.status, "config": row.config, "guardrails": row.guardrails,
           "created_at": row.created_at, "updated_at": row.updated_at}


@router.delete("/connections/{connection_id}")
async def disconnect_tool(connection_id: uuid.UUID, p: Principal = Depends(require("tools:write")),
                          db: AsyncSession = Depends(get_db)):
    row = await _get_connection(db, p.tenant_id, connection_id)
    row.status = "disconnected"
    row.credentials_encrypted = None
    append_activity_event(db, p, "tools.disconnect", resource_type="tool_connection", resource_id=str(row.id),
                          notify_user_ids=await _other_admin_ids(db, p.tenant_id, p.user_id))
    await db.commit()
    return {"id": str(row.id), "status": "disconnected"}
