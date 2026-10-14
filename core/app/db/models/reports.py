"""Vocal Backend Tasksheet P9 (Analytics + Reports) — explicitly marked "deferred,
not part of design pass" in the tasksheet (the lowest-priority track in the whole
build): a bare stub table + CRUD for a future Reports page, not a reporting engine.
`config` is an opaque JSONB blob (whatever the eventual Reports UI decides a saved
report needs — filters, chart type, date range) that nothing server-side interprets
yet; see app/api/analytics.py's GET /analytics/capabilities for what a report could
actually be built FROM today.
"""
import uuid

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin, TimestampMixin


class SavedReport(IdMixin, TimestampMixin, Base):
    __tablename__ = "saved_reports"
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    config: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
