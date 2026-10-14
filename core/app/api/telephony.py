"""Operator-facing telephony controls: run a dial tick on demand, and see live dialer state.

The dialer normally runs on a background loop (app/main.py), but a manual tick is invaluable
for testing and demos — queue a campaign or a call, hit POST /telephony/tick, and watch the
Call rows move queued → ringing → answered/…/completed without waiting on the interval.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.base import get_db
from app.db.models import Call
from app.security.deps import Principal, require
from app.services.telephony.base import ACTIVE_STATUSES, STATUS_QUEUED
from app.services.telephony.dialer import dial_pending
from app.services.telephony.provider import get_telephony_provider

router = APIRouter(prefix="/telephony", tags=["telephony"])


@router.post("/tick")
async def dial_tick(p: Principal = Depends(require("calls:write")),
                    db: AsyncSession = Depends(get_db)):
    """Run one dialer pass for this tenant now: pace campaigns, enqueue due callbacks, and
    originate queued calls within the concurrency cap. Returns exactly what happened."""
    report = await dial_pending(db, tenant_id=p.tenant_id)
    await db.commit()
    return report.as_json()


@router.get("/status")
async def telephony_status(p: Principal = Depends(require("calls:read")),
                           db: AsyncSession = Depends(get_db)):
    """Live dialer state for this tenant: the active carrier, the concurrency cap, and how many
    calls are in flight vs queued."""
    settings = get_settings()
    active = (await db.execute(select(func.count()).select_from(Call).where(
        Call.tenant_id == p.tenant_id, Call.status.in_(ACTIVE_STATUSES)))).scalar_one()
    queued = (await db.execute(select(func.count()).select_from(Call).where(
        Call.tenant_id == p.tenant_id, Call.status == STATUS_QUEUED))).scalar_one()
    by_status = dict((await db.execute(
        select(Call.status, func.count()).where(Call.tenant_id == p.tenant_id)
        .group_by(Call.status))).all())
    return {"provider": get_telephony_provider(settings).name,
            "dialer_enabled": settings.dialer_enabled,
            "max_concurrent_calls": settings.dialer_max_concurrent_calls,
            "active_calls": int(active), "queued_calls": int(queued),
            "capacity_now": max(0, settings.dialer_max_concurrent_calls - int(active)),
            "calls_by_status": {k: int(v) for k, v in by_status.items()}}
