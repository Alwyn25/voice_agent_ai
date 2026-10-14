"""Vocal Backend Tasksheet P4, Generic Sprint 2 (Model Renames) — Centre ->
Location, Executive -> StaffMember, Customer.assigned_executive_id ->
assigned_staff_id.

Scoped to the model/admin-CRUD layer only (see app/db/models/scheduling.py's
Location docstring): appointments.centre_id/executive_id columns and their
values are UNCHANGED — those are the live booking tool's wire contract with
the LLM, deferred to P6's Generic Sprint 3 (Engine Vocabulary). Postgres
follows a table rename through existing FK constraints automatically (they
reference the table by OID, not name), so appointments' FKs keep working
without being touched here.

Additive/rename-only and idempotent (guarded), so a re-run is safe. No data
loss: RENAME TABLE/COLUMN preserve every row.

Revision ID: 0022
Revises: 0021
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def _columns(table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    if _has_table("centres") and not _has_table("locations"):
        op.rename_table("centres", "locations")
    if "location_type" not in _columns("locations"):
        op.add_column("locations", sa.Column(
            "location_type", JSONB(), nullable=False, server_default="{}"))

    if _has_table("executives") and not _has_table("staff_members"):
        op.rename_table("executives", "staff_members")
    staff_cols = _columns("staff_members")
    if "centre_id" in staff_cols and "location_id" not in staff_cols:
        op.alter_column("staff_members", "centre_id", new_column_name="location_id")

    policy_cols = _columns("slot_policies")
    if "centre_id" in policy_cols and "location_id" not in policy_cols:
        op.alter_column("slot_policies", "centre_id", new_column_name="location_id")

    customer_cols = _columns("customers")
    if "assigned_executive_id" in customer_cols and "assigned_staff_id" not in customer_cols:
        op.alter_column("customers", "assigned_executive_id", new_column_name="assigned_staff_id")

    # Data migration: existing owner_type discriminator values on working_hours/
    # holidays follow the entity rename too (an internal discriminator, not a
    # tool-facing value — see Location's docstring for the wire-contract line).
    op.execute("UPDATE working_hours SET owner_type = 'location' WHERE owner_type = 'centre'")
    op.execute("UPDATE working_hours SET owner_type = 'staff_member' WHERE owner_type = 'executive'")
    op.execute("UPDATE holidays SET owner_type = 'location' WHERE owner_type = 'centre'")
    op.execute("UPDATE holidays SET owner_type = 'staff_member' WHERE owner_type = 'executive'")


def downgrade() -> None:
    op.execute("UPDATE holidays SET owner_type = 'executive' WHERE owner_type = 'staff_member'")
    op.execute("UPDATE holidays SET owner_type = 'centre' WHERE owner_type = 'location'")
    op.execute("UPDATE working_hours SET owner_type = 'executive' WHERE owner_type = 'staff_member'")
    op.execute("UPDATE working_hours SET owner_type = 'centre' WHERE owner_type = 'location'")

    if "assigned_staff_id" in _columns("customers"):
        op.alter_column("customers", "assigned_staff_id", new_column_name="assigned_executive_id")
    if "location_id" in _columns("slot_policies"):
        op.alter_column("slot_policies", "location_id", new_column_name="centre_id")
    if "location_id" in _columns("staff_members"):
        op.alter_column("staff_members", "location_id", new_column_name="centre_id")
    if _has_table("staff_members") and not _has_table("executives"):
        op.rename_table("staff_members", "executives")
    if "location_type" in _columns("locations"):
        op.drop_column("locations", "location_type")
    if _has_table("locations") and not _has_table("centres"):
        op.rename_table("locations", "centres")
