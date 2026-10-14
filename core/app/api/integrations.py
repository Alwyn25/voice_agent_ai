"""Vocal Backend Tasksheet P8 (Settings tier): GET/POST /integrations — the org-level
OAuth/API-key store `Integration` (app/db/models/integrations.py) was built for back
in P1, but never got a route of its own; every /integrations/tools/* endpoint
(app/api/integrations_tools.py) operates on ToolConnection, one level below this.

Connecting an org-level integration for a provider that already has ToolConnection
rows links them to it (`tool_connection.integration_id`) — "one Shopify OAuth grant
that several tool_connections can share," per that model's own docstring — rather
than leaving that column permanently null, which is all it has ever been until now.
"""
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import AuditEvent, Integration, ToolCatalogEntry, ToolConnection
from app.security.deps import Principal, require
from app.services.credentials import encrypt_credentials

router = APIRouter(prefix="/integrations", tags=["settings"])


def _integration_out(i: Integration) -> dict:
    return {"id": str(i.id), "provider_name": i.provider_name, "status": i.status,
           "connected_at": i.connected_at, "created_at": i.created_at}


@router.get("")
async def list_integrations(p: Principal = Depends(require("integrations:read")),
                            db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(Integration).where(
        Integration.tenant_id == p.tenant_id).order_by(Integration.provider_name))).scalars().all()
    return [_integration_out(i) for i in rows]


class IntegrationConnectIn(BaseModel):
    provider_name: str
    credentials: dict = {}


@router.post("", status_code=status.HTTP_201_CREATED)
async def connect_integration(body: IntegrationConnectIn,
                              p: Principal = Depends(require("integrations:write")),
                              db: AsyncSession = Depends(get_db)):
    """Connect (or reconnect) one provider's org-level credentials — one row per
    (tenant, provider), matching ToolConnection's own one-per-(tenant, tool) shape."""
    existing = (await db.execute(select(Integration).where(
        Integration.tenant_id == p.tenant_id,
        Integration.provider_name == body.provider_name))).scalar_one_or_none()
    encrypted = encrypt_credentials(body.credentials) if body.credentials else None
    now = datetime.now(timezone.utc)
    if existing is None:
        row = Integration(tenant_id=p.tenant_id, provider_name=body.provider_name,
                          status="connected", credentials_encrypted=encrypted, connected_at=now)
        db.add(row)
    else:
        row = existing
        row.status = "connected"
        # Copilot review #18: only overwrite credentials when new ones are provided,
        # so a reconnect without credentials doesn't wipe the existing grant.
        if encrypted is not None:
            row.credentials_encrypted = encrypted
        row.connected_at = now
    await db.flush()

    # Link any pre-existing tool_connections for this provider to the org-level grant.
    linked = (await db.execute(
        select(ToolConnection).join(ToolCatalogEntry,
                                    ToolConnection.tool_catalog_id == ToolCatalogEntry.id)
        .where(ToolConnection.tenant_id == p.tenant_id,
              ToolCatalogEntry.provider_name == body.provider_name))).scalars().all()
    for conn in linked:
        conn.integration_id = row.id

    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="integrations.connect",
                      resource_type="integration", resource_id=str(row.id),
                      after={"provider_name": body.provider_name, "linked_tool_connections": len(linked)}))
    await db.commit()
    return {**_integration_out(row), "linked_tool_connections": len(linked)}


@router.delete("/{integration_id}")
async def disconnect_integration(integration_id: uuid.UUID,
                                 p: Principal = Depends(require("integrations:write")),
                                 db: AsyncSession = Depends(get_db)):
    row = (await db.execute(select(Integration).where(
        Integration.tenant_id == p.tenant_id, Integration.id == integration_id
        ))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Integration not found")
    row.status = "disconnected"
    row.credentials_encrypted = None
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="integrations.disconnect",
                      resource_type="integration", resource_id=str(integration_id),
                      after={"provider_name": row.provider_name}))
    await db.commit()
    return {"id": str(integration_id), "status": row.status}
