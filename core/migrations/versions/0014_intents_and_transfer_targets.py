"""Custom Instructions Phase 2 — intent vocabulary, transfer targets, escalation target.

  * ``intent_vocabulary`` — per-tenant EXTENSIONS to the canonical intent set. The routing
    compiler grounds ``turn.intent`` conditions against this so a rule only keys on a label
    the planner can emit (spec §4.3).
  * ``transfer_targets`` — the named humans/queues a routing instruction may escalate to
    (spec §4.4 / §5).
  * ``escalations.target`` — the resolved transfer-target key stamped on a handoff.

Additive and idempotent (guarded), so a re-run is safe.

Revision ID: 0014
Revises: 0013
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def _columns(table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    if not _has_table("intent_vocabulary"):
        op.create_table(
            "intent_vocabulary",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("name", sa.String(length=60), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("aliases", JSONB(), nullable=False, server_default="[]"),
            sa.Column("examples", JSONB(), nullable=False, server_default="[]"),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("created_by", UUID(as_uuid=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True),
                      server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True),
                      server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("tenant_id", "name", name="uq_intent_vocab_tenant_name"),
        )

    if not _has_table("transfer_targets"):
        op.create_table(
            "transfer_targets",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("key", sa.String(length=60), nullable=False),
            sa.Column("display_name", sa.String(length=120), nullable=False),
            sa.Column("destination", JSONB(), nullable=False, server_default="{}"),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("created_by", UUID(as_uuid=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True),
                      server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True),
                      server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("tenant_id", "key", name="uq_transfer_target_tenant_key"),
        )

    if "target" not in _columns("escalations"):
        op.add_column("escalations", sa.Column("target", sa.String(length=60), nullable=True))


def downgrade() -> None:
    if "target" in _columns("escalations"):
        op.drop_column("escalations", "target")
    if _has_table("transfer_targets"):
        op.drop_table("transfer_targets")
    if _has_table("intent_vocabulary"):
        op.drop_table("intent_vocabulary")
