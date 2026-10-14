"""Multi-project-per-tenant: a projects table + a project_id on the tables that need scoping.

Adds:
  * projects — one row per development a tenant sells (facts, synonyms, dialled number,
    aliases, default centre).
  * knowledge_documents.project_id, knowledge_chunks.project_id — so retrieval can isolate
    one project's chunks from another's.
  * appointments.project_id — so a visit records which project it is for.
  * conversations.project_id — the project a call resolved to.

Every project_id is NULLABLE. A tenant with no projects row keeps working exactly as before
(everything reads as tenant-wide); the columns only start to matter once projects exist and
the backfill (scripts/seed_projects.py) tags the existing rows.

Revision ID: 0008
Revises: 0007
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def _has_column(table: str, column: str) -> bool:
    insp = sa.inspect(op.get_bind())
    if table not in insp.get_table_names():
        return False
    return any(col["name"] == column for col in insp.get_columns(table))


def _add_project_id(table: str, *, index: bool = True) -> None:
    """Add a nullable project_id FK to an existing table, idempotently."""
    if _has_column(table, "project_id"):
        return
    op.add_column(table, sa.Column("project_id", UUID(as_uuid=True),
                                   sa.ForeignKey("projects.id"), nullable=True))
    if index:
        op.create_index(f"ix_{table}_project_id", table, ["project_id"])


def upgrade() -> None:
    if not _has_table("projects"):
        op.create_table(
            "projects",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column("tenant_id", UUID(as_uuid=True),
                      sa.ForeignKey("tenants.id"), nullable=False, index=True),
            sa.Column("slug", sa.String(length=100), nullable=False),
            sa.Column("name", sa.String(length=200), nullable=False),
            sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
            sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("phone_did", sa.String(length=20), nullable=True),
            sa.Column("aliases", ARRAY(sa.String()), nullable=False, server_default="{}"),
            sa.Column("facts", JSONB(), nullable=False, server_default="[]"),
            sa.Column("synonyms", JSONB(), nullable=False, server_default="[]"),
            sa.Column("default_centre_id", UUID(as_uuid=True),
                      sa.ForeignKey("centres.id"), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True),
                      server_default=sa.func.now(), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True),
                      server_default=sa.func.now(), nullable=False),
            sa.UniqueConstraint("tenant_id", "slug", name="uq_project_tenant_slug"),
        )
        op.create_index("ix_project_did", "projects", ["tenant_id", "phone_did"])

    _add_project_id("knowledge_documents")
    # knowledge_chunks is filtered by the (tenant, project) pair on the retrieval hot path,
    # so it gets a compound index rather than the plain one _add_project_id makes.
    if not _has_column("knowledge_chunks", "project_id"):
        op.add_column("knowledge_chunks",
                      sa.Column("project_id", UUID(as_uuid=True),
                                sa.ForeignKey("projects.id"), nullable=True))
        op.create_index("ix_chunks_project", "knowledge_chunks",
                        ["tenant_id", "project_id"])
    _add_project_id("appointments")
    _add_project_id("conversations")
    _add_project_id("customer_product_interest")


def downgrade() -> None:
    for table in ("conversations", "appointments", "customer_product_interest"):
        if _has_column(table, "project_id"):
            op.drop_index(f"ix_{table}_project_id", table_name=table)
            op.drop_column(table, "project_id")
    if _has_column("knowledge_chunks", "project_id"):
        op.drop_index("ix_chunks_project", table_name="knowledge_chunks")
        op.drop_column("knowledge_chunks", "project_id")
    if _has_column("knowledge_documents", "project_id"):
        op.drop_index("ix_knowledge_documents_project_id", table_name="knowledge_documents")
        op.drop_column("knowledge_documents", "project_id")
    if _has_table("projects"):
        op.drop_index("ix_project_did", table_name="projects")
        op.drop_table("projects")
