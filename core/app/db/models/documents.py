"""A SHAREABLE DOCUMENT — a file a caller can be sent (floor plan, brochure, price
sheet, payment plan, location map, …), as opposed to knowledge the agent reads.

WHY THIS EXISTS, AND WHY IT IS NOT ``knowledge_documents``. The knowledge base holds text
that is chunked, embedded and retrieved so the agent can ANSWER from it. A floor plan is not
that: nobody wants its geometry described down a phone line, they want the PDF on their
WhatsApp. These are files with a URL to deliver, keyed by what a caller asks for ("send the
3 BHK floor plan", "can you WhatsApp the brochure") rather than by semantic similarity. Mixing
them into ``knowledge_documents`` would either pollute retrieval with un-embeddable binaries or
force the send path to reverse-engineer a file URL out of a RAG chunk. A first-class table for
sendable assets keeps each concern clean.

The ``send_document`` tool resolves a caller's request against these rows (by type and by a
token match on title/description), then hands the winner's ``file_url`` to the configured
WhatsApp provider. The delivery itself is audited on the ordinary ``tool_executions`` row.

Multi-project, exactly like the rest of the engine: ``project_id`` is nullable and falls back
to tenant-wide, so a single-project tenant is unchanged and a caller pinned to DSR can never be
sent Lemon Tree's floor plan. ``doc_type`` is a small controlled vocabulary the tool filters on;
``is_active`` retires a superseded version without deleting its history.
"""
import uuid

from sqlalchemy import Boolean, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin, TimestampMixin

# The controlled vocabulary the tool filters and disambiguates on. Deliberately broad — the
# feature is "send ANY document", not "send floor plans" — with "other" as the catch-all so a
# tenant is never blocked from adding a category the engine did not anticipate.
DOC_TYPES = (
    "floor_plan", "brochure", "price_sheet", "payment_plan", "cost_sheet",
    "location_map", "amenities", "master_plan", "specification", "legal",
    "agreement", "offer", "other",
)


class ShareableDocument(IdMixin, TimestampMixin, Base):
    __tablename__ = "shareable_documents"
    __table_args__ = (
        # The send path filters by (tenant, project) on every request, mirroring units/chunks.
        Index("ix_shareable_docs_tenant_project", "tenant_id", "project_id"),
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    # Nullable, like the other multi-project columns — tenant-wide when unset.
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True)

    # What a person calls it — "DSR Highland Greenz — 3 BHK Floor Plan". The primary thing a
    # caller's request ("the 3 BHK floor plan") is token-matched against.
    title: Mapped[str] = mapped_column(String(300))
    # One of DOC_TYPES. The tool's coarse filter: "send the brochure" narrows to doc_type in a
    # single step before any title matching.
    doc_type: Mapped[str] = mapped_column(String(40))
    # A one-line human description, also searched — lets a caller's wording ("the layout for
    # the three-bed", "the payment schedule") reach a document whose title phrases it otherwise.
    description: Mapped[str | None] = mapped_column(Text)

    # The URL the provider actually delivers. A publicly reachable https link for the WhatsApp
    # Cloud API (it fetches the file itself), or a signed URL — the engine does not host files.
    file_url: Mapped[str] = mapped_column(Text)
    # The name the recipient sees on the file in WhatsApp. Defaults from the title when unset.
    file_name: Mapped[str | None] = mapped_column(String(200))
    # e.g. "application/pdf". Advisory — passed to the provider where it accepts one.
    mime_type: Mapped[str | None] = mapped_column(String(100))

    # A retired document stays for history but is never resolved as a send target.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    meta: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
