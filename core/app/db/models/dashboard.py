"""Vocal Backend Tasksheet P7 (Overview): the org-wide daily rollup the Overview
dashboard's summary tiles and call-activity chart read historical days from — the
org-level counterpart to `agent_metrics_daily` (app/db/models/agents.py), written by
the same Aggregator Worker (app/services/aggregator.py). Today's own numbers are
computed live from `calls`/`tool_executions` directly (see app/api/dashboard.py) —
this table only ever holds already-finished days, so a dashboard loading before the
Aggregator Worker's first tick of the day still sees an accurate "today".
"""
import uuid
from datetime import date as date_

from sqlalchemy import Date, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin


class DailyOrgMetrics(IdMixin, Base):
    __tablename__ = "daily_org_metrics"
    __table_args__ = (UniqueConstraint("tenant_id", "date", name="uq_daily_org_metrics_day"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    date: Mapped[date_] = mapped_column(Date)
    # A point-in-time gauge (how many agents were status='live'), captured AT ROLLUP TIME
    # for that day — not a true historical reconstruction (Agent has no status-history
    # table), but the closest honest approximation without one. Documented limitation:
    # a day rolled up late (the Aggregator Worker catching up after an outage) reflects
    # whenever the rollup actually ran, not necessarily that calendar day's real peak.
    active_agents: Mapped[int] = mapped_column(Integer, default=0)
    calls_placed: Mapped[int] = mapped_column(Integer, default=0)
    calls_completed: Mapped[int] = mapped_column(Integer, default=0)
    # A completed call that did NOT hand off to a human — the same "resolved" concept
    # agent_metrics_daily already uses, rolled up org-wide.
    resolved: Mapped[int] = mapped_column(Integer, default=0)
    actions_taken: Mapped[int] = mapped_column(Integer, default=0)
    avg_duration_seconds: Mapped[int] = mapped_column(Integer, default=0)
    # Sum (not average) of completed calls' duration_seconds for the day — backs the
    # Overview "Total time" tile, which is org-wide talk-time, distinct from the
    # per-call avg_duration_seconds above.
    total_duration_seconds: Mapped[int] = mapped_column(Integer, default=0)
