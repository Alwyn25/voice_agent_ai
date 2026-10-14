"""Vocal Backend Tasksheet P9 (Analytics + Reports) — explicitly "deferred, not part
of design pass" in the tasksheet, the lowest-priority track in the whole build: a
stub CRUD over `saved_reports`. `config` is an opaque blob nothing server-side
interprets yet (no report-rendering engine exists) — see GET /analytics/capabilities
for what a future report could actually be built from.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import AuditEvent, SavedReport
from app.security.deps import Principal, require

router = APIRouter(prefix="/reports", tags=["analytics & audit"])


def _report_out(r: SavedReport) -> dict:
    return {"id": str(r.id), "name": r.name, "config": r.config,
           "created_by": str(r.created_by) if r.created_by else None,
           "created_at": r.created_at, "updated_at": r.updated_at}


@router.get("")
async def list_reports(p: Principal = Depends(require("analytics:read")),
                       db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(SavedReport).where(
        SavedReport.tenant_id == p.tenant_id).order_by(SavedReport.created_at.desc()))).scalars().all()
    return [_report_out(r) for r in rows]


class SavedReportIn(BaseModel):
    name: str
    config: dict = {}


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_report(body: SavedReportIn, p: Principal = Depends(require("analytics:write")),
                        db: AsyncSession = Depends(get_db)):
    row = SavedReport(tenant_id=p.tenant_id, name=body.name[:200], config=body.config,
                      created_by=p.user_id)
    db.add(row)
    await db.flush()
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="reports.create",
                      resource_type="saved_report", resource_id=str(row.id),
                      after={"name": row.name}))
    await db.commit()
    return _report_out(row)


@router.get("/{report_id}")
async def get_report(report_id: uuid.UUID, p: Principal = Depends(require("analytics:read")),
                     db: AsyncSession = Depends(get_db)):
    row = await _report(db, p.tenant_id, report_id)
    return _report_out(row)


@router.delete("/{report_id}")
async def delete_report(report_id: uuid.UUID, p: Principal = Depends(require("analytics:write")),
                        db: AsyncSession = Depends(get_db)):
    row = await _report(db, p.tenant_id, report_id)
    await db.delete(row)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="reports.delete",
                      resource_type="saved_report", resource_id=str(report_id)))
    await db.commit()
    return {"id": str(report_id), "deleted": True}


async def _report(db: AsyncSession, tenant_id: uuid.UUID, report_id: uuid.UUID) -> SavedReport:
    row = (await db.execute(select(SavedReport).where(
        SavedReport.tenant_id == tenant_id, SavedReport.id == report_id))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Report not found")
    return row
