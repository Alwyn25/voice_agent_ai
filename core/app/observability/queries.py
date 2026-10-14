"""Read side of tool performance tracking — aggregates over tool_metrics.

Kept separate from tool_metrics.py (the write side, hooked into the executor) so that
hook stays a single small import. Nothing here is on the hot path of a turn; it is only
called from the /tools/metrics API.
"""
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.tool_metrics import ToolMetric

STATUSES = ("success", "failed", "blocked", "skipped_duplicate")


async def tool_stats(db: AsyncSession, tenant_id: UUID, tool_name: str | None = None,
                     since: datetime | None = None) -> list[dict]:
    """Per-tool call volume, outcome breakdown, success rate, and latency."""
    since = since or (datetime.now(timezone.utc) - timedelta(days=7))
    stmt = (select(
                ToolMetric.tool_name,
                func.count().label("calls"),
                *(func.sum(case((ToolMetric.status == s, 1), else_=0)).label(s) for s in STATUSES),
                func.avg(ToolMetric.duration_ms).label("avg_duration_ms"),
                func.percentile_cont(0.95).within_group(ToolMetric.duration_ms).label("p95_duration_ms"),
            )
            .where(ToolMetric.tenant_id == tenant_id, ToolMetric.created_at >= since)
            .group_by(ToolMetric.tool_name)
            .order_by(func.count().desc()))
    if tool_name:
        stmt = stmt.where(ToolMetric.tool_name == tool_name)
    rows = (await db.execute(stmt)).all()

    return [{
        "tool_name": r.tool_name,
        "calls": int(r.calls),
        "outcomes": {s: int(getattr(r, s) or 0) for s in STATUSES},
        "success_rate": round(r.success / r.calls, 4) if r.calls else None,
        "avg_duration_ms": round(r.avg_duration_ms, 1) if r.avg_duration_ms is not None else None,
        "p95_duration_ms": round(r.p95_duration_ms, 1) if r.p95_duration_ms is not None else None,
    } for r in rows]


async def error_breakdown(db: AsyncSession, tenant_id: UUID, tool_name: str | None = None,
                          since: datetime | None = None) -> list[dict]:
    """Every non-success outcome grouped by tool, status and the specific check that
    failed — the "why" behind a low success rate."""
    since = since or (datetime.now(timezone.utc) - timedelta(days=7))
    stmt = (select(ToolMetric.tool_name, ToolMetric.status, ToolMetric.failed_check,
                   func.count().label("count"))
            .where(ToolMetric.tenant_id == tenant_id, ToolMetric.created_at >= since,
                  ToolMetric.status != "success")
            .group_by(ToolMetric.tool_name, ToolMetric.status, ToolMetric.failed_check)
            .order_by(func.count().desc()))
    if tool_name:
        stmt = stmt.where(ToolMetric.tool_name == tool_name)
    rows = (await db.execute(stmt)).all()
    return [{"tool_name": r.tool_name, "status": r.status, "failed_check": r.failed_check,
             "count": int(r.count)} for r in rows]
