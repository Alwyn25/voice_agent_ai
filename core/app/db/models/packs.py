"""Vocal Backend Tasksheet P0, Generic Sprint 1 (Pack Architecture).

industry_packs is a small reference catalog — one row per registered
app.packs.IndustryPack subclass (see app/packs/__init__.py's PackRegistry).
tenant_pack_config is the one-row-per-tenant pointer to which pack is active, plus
tenant-specific overrides layered on top of the pack's own defaults (e.g. a
tenant-specific vocabulary tweak), so a per-tenant customisation never needs a pack
subclass of its own.

Note: `tenants.industry_pack` (app/db/models/platform.py) is a pre-existing, never-
wired-up plain string column left over from before this schema was decided —
harmless, but not what PackRegistry reads; it reads these two tables instead.
"""
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin


class IndustryPackDef(IdMixin, Base):
    __tablename__ = "industry_packs"
    __table_args__ = (UniqueConstraint("slug", name="uq_industry_pack_slug"),)
    slug: Mapped[str] = mapped_column(String(100))
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TenantPackConfig(IdMixin, Base):
    __tablename__ = "tenant_pack_config"
    __table_args__ = (UniqueConstraint("tenant_id", name="uq_tenant_pack_config_tenant"),)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    pack_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("industry_packs.id"))
    overrides: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
