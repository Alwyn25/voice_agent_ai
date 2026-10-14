"""Vocal Backend Tasksheet P0, Generic Sprint 1 (Pack Architecture): industry_packs
+ tenant_pack_config, seeded with the two built-in packs (real_estate_india,
generic) and backfilled so every existing tenant keeps its current (real-estate)
behaviour — see app/packs/__init__.py's PackRegistry for how these are read.

Additive and idempotent (guarded), so a re-run is safe.

Revision ID: 0019
Revises: 0018
"""
import uuid

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None

_PACKS = [
    ("real_estate_india", "Real Estate (India)"),
    ("generic", "Generic"),
]


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def upgrade() -> None:
    if not _has_table("industry_packs"):
        op.create_table(
            "industry_packs",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("slug", sa.String(length=100), nullable=False),
            sa.Column("name", sa.String(length=200), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("slug", name="uq_industry_pack_slug"),
        )

    if not _has_table("tenant_pack_config"):
        op.create_table(
            "tenant_pack_config",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("pack_id", UUID(as_uuid=True),
                      sa.ForeignKey("industry_packs.id"), nullable=False),
            sa.Column("overrides", JSONB(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("tenant_id", name="uq_tenant_pack_config_tenant"),
        )

    bind = op.get_bind()

    # Seed the catalog (safe to re-run: skip a slug that's already there).
    pack_ids: dict[str, uuid.UUID] = {}
    for slug, name in _PACKS:
        row = bind.execute(sa.text("SELECT id FROM industry_packs WHERE slug = :slug"),
                           {"slug": slug}).first()
        if row is not None:
            pack_ids[slug] = row[0]
            continue
        new_id = uuid.uuid4()
        bind.execute(sa.text("INSERT INTO industry_packs (id, slug, name) VALUES (:id, :slug, :name)"),
                     {"id": new_id, "slug": slug, "name": name})
        pack_ids[slug] = new_id

    # Backfill: every tenant with no tenant_pack_config row yet gets real_estate_india
    # — this backend's only tenants today are real-estate ones, so this is a
    # zero-behaviour-change default, not a guess. UUIDs generated in Python (like
    # every other row in this codebase) rather than relying on a Postgres-version-
    # specific gen_random_uuid().
    re_pack_id = pack_ids["real_estate_india"]
    unconfigured = bind.execute(sa.text("""
        SELECT t.id FROM tenants t
        LEFT JOIN tenant_pack_config tpc ON tpc.tenant_id = t.id
        WHERE tpc.id IS NULL
    """)).fetchall()
    for (tenant_id,) in unconfigured:
        bind.execute(sa.text("""
            INSERT INTO tenant_pack_config (id, tenant_id, pack_id, overrides)
            VALUES (:id, :tenant_id, :pack_id, '{}'::jsonb)
        """), {"id": uuid.uuid4(), "tenant_id": tenant_id, "pack_id": re_pack_id})


def downgrade() -> None:
    bind = op.get_bind()
    # Copilot review #13: only drop these tables if no later migration added
    # dependent FKs. Check for dependent tenant_pack_config rows first.
    if _has_table("tenant_pack_config"):
        remaining = bind.execute(sa.text("SELECT count(*) FROM tenant_pack_config")).scalar()
        if remaining == 0:
            op.drop_table("tenant_pack_config")
        else:
            # Dependent rows exist — only remove the backfill rows we created,
            # don't drop the table.
            for slug, _ in _PACKS:
                bind.execute(sa.text("""
                    DELETE FROM tenant_pack_config
                    WHERE pack_id = (SELECT id FROM industry_packs WHERE slug = :slug)
                """), {"slug": slug})
    if _has_table("industry_packs"):
        remaining = bind.execute(sa.text(
            "SELECT count(*) FROM industry_packs WHERE slug NOT IN ('real_estate_india', 'generic')"
        )).scalar()
        if remaining == 0:
            # Only our two packs exist — safe to remove them
            for slug, _ in _PACKS:
                bind.execute(sa.text("DELETE FROM industry_packs WHERE slug = :slug"), {"slug": slug})
            # Only drop the table if it's now empty
            count = bind.execute(sa.text("SELECT count(*) FROM industry_packs")).scalar()
            if count == 0:
                op.drop_table("industry_packs")
