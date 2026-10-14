"""Vocal Backend Tasksheet P9 Generic Sprint 5: two new industry_packs catalog rows
(healthcare, saas_sales) — the two new seed-stub packs registered in
app/packs/__init__.py's PackRegistry, alongside migration 0019's real_estate_india/
generic. No new tables; industry_packs already exists.

Additive and idempotent (guarded — skips a slug already present), so a re-run is
safe. No tenant is ever assigned onto either pack automatically; they exist only to
be explicitly chosen (e.g. via a future admin endpoint, or directly in
tenant_pack_config.overrides), enabling a multi-industry demo without disturbing any
existing tenant's current pack.

Revision ID: 0028
Revises: 0027
"""
import uuid

import sqlalchemy as sa
from alembic import op

revision = "0028"
down_revision = "0027"
branch_labels = None
depends_on = None

_PACKS = [
    ("healthcare", "Healthcare"),
    ("saas_sales", "SaaS Sales"),
]


def upgrade() -> None:
    bind = op.get_bind()
    for slug, name in _PACKS:
        row = bind.execute(sa.text("SELECT id FROM industry_packs WHERE slug = :slug"),
                           {"slug": slug}).first()
        if row is not None:
            continue
        bind.execute(sa.text("INSERT INTO industry_packs (id, slug, name) VALUES (:id, :slug, :name)"),
                     {"id": uuid.uuid4(), "slug": slug, "name": name})


def downgrade() -> None:
    bind = op.get_bind()
    for slug, _ in _PACKS:
        # Copilot review #14/#15: reassign any tenants on this pack to 'generic'
        # before deleting the row, so we don't leave orphaned FK references.
        generic_id = bind.execute(sa.text(
            "SELECT id FROM industry_packs WHERE slug = 'generic'")).scalar()
        pack_id = bind.execute(sa.text(
            "SELECT id FROM industry_packs WHERE slug = :slug"), {"slug": slug}).scalar()
        if pack_id and generic_id:
            bind.execute(sa.text("""
                UPDATE tenant_pack_config SET pack_id = :generic_id
                WHERE pack_id = :pack_id
            """), {"generic_id": generic_id, "pack_id": pack_id})
        # Only delete the row this migration created (safe: slug is unique).
        if pack_id:
            bind.execute(sa.text("DELETE FROM industry_packs WHERE slug = :slug"), {"slug": slug})
