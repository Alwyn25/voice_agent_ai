"""Housekeeping: rename a handful of indexes whose live name drifted from what
their current SQLAlchemy model declaration actually implies, found by running
`alembic check` at the end of the Wave 1-10 build as a final correctness pass.
Purely cosmetic — every index below already exists and already works, this only
renames it so `alembic check` (and anyone reading the schema) sees no drift:

  - ix_centres_tenant_id / ix_executives_centre_id / ix_executives_tenant_id:
    migration 0022 (Wave 5's Centre->Location/Executive->StaffMember rename)
    renamed the TABLES but Postgres does not rename an index just because its
    table was renamed, so these kept their pre-rename names.
  - ix_knowledge_documents_kb: migration 0020 (Wave 3) created this index with
    an explicit short name instead of the same auto-generated name
    `knowledge_base_id: Mapped[...] = mapped_column(..., index=True)` implies.

Additive and idempotent (guarded — renames only if the old name exists and the
new one doesn't yet), so a re-run is safe.

Revision ID: 0030
Revises: 0029
"""
import sqlalchemy as sa
from alembic import op

revision = "0030"
down_revision = "0029"
branch_labels = None
depends_on = None

_RENAMES = [
    ("ix_centres_tenant_id", "ix_locations_tenant_id"),
    ("ix_executives_centre_id", "ix_staff_members_location_id"),
    ("ix_executives_tenant_id", "ix_staff_members_tenant_id"),
    ("ix_knowledge_documents_kb", "ix_knowledge_documents_knowledge_base_id"),
]


def _index_names() -> set[str]:
    bind = op.get_bind()
    rows = bind.execute(sa.text(
        "SELECT indexname FROM pg_indexes WHERE schemaname = 'public'")).all()
    return {r[0] for r in rows}


def upgrade() -> None:
    existing = _index_names()
    for old, new in _RENAMES:
        if old in existing and new not in existing:
            op.execute(sa.text(f'ALTER INDEX "{old}" RENAME TO "{new}"'))


def downgrade() -> None:
    existing = _index_names()
    for old, new in _RENAMES:
        if new in existing and old not in existing:
            op.execute(sa.text(f'ALTER INDEX "{new}" RENAME TO "{old}"'))
