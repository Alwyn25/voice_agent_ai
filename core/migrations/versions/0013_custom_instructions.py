"""Custom Instructions & Dynamic Knowledge (Phase 1).

Two new tables (see docs/custom-instructions-implementation-plan.md):

  * ``custom_instructions`` — admin-authored behavioural changes. Phase 1 stores/serves
    the ``advisory`` kind (compiled_prompt rendered into the responder prompt); the
    ``rule_set_id`` column is present now so the Phase-2 ``routing`` kind can own a
    generated RuleSet without a further migration.
  * ``project_facts_dynamic`` — time-bounded facts merged into the PROJECT FACTS block,
    self-expiring by date filter.

Both are additive and idempotent (guarded on table existence) so a re-run is safe.

Revision ID: 0013
Revises: 0012
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return sa.inspect(op.get_bind()).has_table(name)


def _ensure_index(name: str, table: str, cols: list[str]) -> None:
    """Create the index unless it exists. Guarded SEPARATELY from the table: on a fresh DB,
    0001's create_all has already built the table from the models (so the create_table branch
    is skipped), and these compound indexes — which the models do not declare — must still be
    added, or a new environment silently lacks the hot-path loader index."""
    if name not in {i["name"] for i in sa.inspect(op.get_bind()).get_indexes(table)}:
        op.create_index(name, table, cols)


def upgrade() -> None:
    if not _has_table("custom_instructions"):
        op.create_table(
            "custom_instructions",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("project_id", UUID(as_uuid=True),
                      sa.ForeignKey("projects.id"), nullable=True),
            sa.Column("raw_text", sa.Text(), nullable=False),
            sa.Column("kind", sa.String(length=20), nullable=False),
            sa.Column("compiled_prompt", sa.Text(), nullable=True),
            sa.Column("rule_set_id", UUID(as_uuid=True),
                      sa.ForeignKey("rule_sets.id"), nullable=True),
            sa.Column("scope", JSONB(), nullable=False, server_default="{}"),
            sa.Column("priority", sa.Integer(), nullable=False, server_default="100"),
            sa.Column("preview", JSONB(), nullable=True),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("root_id", UUID(as_uuid=True), nullable=True),
            sa.Column("created_by", UUID(as_uuid=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True),
                      server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True),
                      server_default=sa.func.now(), nullable=False),
        )
    # Hot-path loader filter: active rows for a tenant (project scoping done in Python).
    _ensure_index("ix_custom_instructions_active", "custom_instructions",
                  ["tenant_id", "is_active"])
    # Version-history lookups by logical instruction.
    _ensure_index("ix_custom_instructions_tenant_project", "custom_instructions",
                  ["tenant_id", "project_id"])
    _ensure_index("ix_custom_instructions_root", "custom_instructions", ["root_id"])

    if not _has_table("project_facts_dynamic"):
        op.create_table(
            "project_facts_dynamic",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("project_id", UUID(as_uuid=True),
                      sa.ForeignKey("projects.id"), nullable=True),
            sa.Column("label", sa.String(length=120), nullable=True),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("valid_from", sa.Date(), nullable=True),
            sa.Column("valid_until", sa.Date(), nullable=True),
            sa.Column("priority", sa.Integer(), nullable=False, server_default="100"),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
            sa.Column("created_by", UUID(as_uuid=True), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True),
                      server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True),
                      server_default=sa.func.now(), nullable=False),
        )
    _ensure_index("ix_project_facts_dynamic_active", "project_facts_dynamic",
                  ["tenant_id", "project_id", "is_active"])


def downgrade() -> None:
    if _has_table("project_facts_dynamic"):
        op.drop_table("project_facts_dynamic")
    if _has_table("custom_instructions"):
        op.drop_table("custom_instructions")
