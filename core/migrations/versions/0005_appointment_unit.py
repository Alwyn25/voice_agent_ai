"""Bind the chosen unit configuration to an appointment.

Adds appointments.unit_type (canonical, indexed — "3 BHK") and appointments.unit_context
(JSONB snapshot). This is the durable answer to "the flat I selected": before it, the unit
lived only in the transient retrieval-focus hint, which drifts across a multi-turn booking,
so a caller who chose the 3 BHK could be quoted the 2.5 BHK cost sheet.

Revision ID: 0005
Revises: 0004

Defensive like 0003/0006: on a FRESH database 0001's create_all builds appointments from the
live models, which already carry unit_type/unit_context — so a plain add_column raised
DuplicateColumn and a new environment could not be migrated at all. On a database that
migrated before these columns existed they are missing and must be added. Each object is
checked before it is created, so both paths work and a re-run is a no-op.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

TABLE = "appointments"
INDEX = "ix_appt_unit_type"


def _columns() -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(TABLE)}


def _indexes() -> set[str]:
    return {i["name"] for i in sa.inspect(op.get_bind()).get_indexes(TABLE)}


def upgrade() -> None:
    cols = _columns()
    if "unit_type" not in cols:
        op.add_column(TABLE, sa.Column("unit_type", sa.String(length=60), nullable=True))
    if "unit_context" not in cols:
        # server_default so the column backfills on existing rows without a table rewrite;
        # dropped afterwards so the application default (an empty dict from the ORM) governs
        # new inserts rather than a database-side literal.
        op.add_column(TABLE,
                      sa.Column("unit_context", JSONB(), nullable=False, server_default="{}"))
        op.alter_column(TABLE, "unit_context", server_default=None)
    # On a fresh DB the model's own single-column index (ix_appointments_unit_type) exists;
    # this compound one is still added, exactly as on a migrated DB.
    if INDEX not in _indexes():
        op.create_index(INDEX, TABLE, ["tenant_id", "unit_type"])


def downgrade() -> None:
    if INDEX in _indexes():
        op.drop_index(INDEX, table_name=TABLE)
    cols = _columns()
    for col in ("unit_context", "unit_type"):
        if col in cols:
            op.drop_column(TABLE, col)
