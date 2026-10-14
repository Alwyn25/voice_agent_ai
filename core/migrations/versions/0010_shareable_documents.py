"""Shareable documents — files a caller can be sent (floor plan, brochure, price sheet,
…), the source of truth the send_document tool resolves a request against.

Separate from knowledge_documents on purpose: those are chunked/embedded TEXT the agent reads
from; these are binaries with a delivery URL, keyed by what a caller asks for rather than by
semantic similarity. See app/db/models/documents.py.

tenant_id is required; project_id is NULLABLE, mirroring 0008/0009 — a tenant that seeds no
documents is unaffected, and documents fall back to tenant-wide when a tenant runs one project.

Revision ID: 0010
Revises: 0009
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def _has_table(name: str) -> bool:
    return name in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    if _has_table("shareable_documents"):
        return
    op.create_table(
        "shareable_documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", UUID(as_uuid=True),
                  sa.ForeignKey("tenants.id"), nullable=False, index=True),
        sa.Column("project_id", UUID(as_uuid=True),
                  sa.ForeignKey("projects.id"), nullable=True),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("doc_type", sa.String(length=40), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("file_url", sa.Text(), nullable=False),
        sa.Column("file_name", sa.String(length=200), nullable=True),
        sa.Column("mime_type", sa.String(length=100), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("metadata", JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_shareable_docs_tenant_project", "shareable_documents",
                    ["tenant_id", "project_id"])


def downgrade() -> None:
    if _has_table("shareable_documents"):
        op.drop_index("ix_shareable_docs_tenant_project", table_name="shareable_documents")
        op.drop_table("shareable_documents")
