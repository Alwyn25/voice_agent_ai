"""Read-only API over the tool_metrics table — per-tool success rate, latency and error
breakdown. Separate router from tools_admin.py so the write-side hook in the executor
and this read side never need to share a file."""
from datetime import date, datetime, time, timedelta, timezone

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.observability import queries
from app.security.deps import Principal, require

router = APIRouter(prefix="/tools/metrics", tags=["tools"])


def _since(date_from: date | None) -> datetime:
    if date_from:
        return datetime.combine(date_from, time.min, tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - timedelta(days=7)


@router.get("")
async def stats(tool_name: str | None = None,
                date_from: date | None = Query(None, alias="from"),
                p: Principal = Depends(require("tools:read")),
                db: AsyncSession = Depends(get_db)):
    """Per-tool call volume, outcome breakdown, success rate and latency (avg/p95)."""
    since = _since(date_from)
    return {"since": since, "tools": await queries.tool_stats(db, p.tenant_id, tool_name, since)}


@router.get("/errors")
async def errors(tool_name: str | None = None,
                 date_from: date | None = Query(None, alias="from"),
                 p: Principal = Depends(require("tools:read")),
                 db: AsyncSession = Depends(get_db)):
    """Every non-success outcome, grouped by tool, status and the specific check that
    failed."""
    since = _since(date_from)
    return {"since": since, "errors": await queries.error_breakdown(db, p.tenant_id, tool_name, since)}
