"""A PROJECT — a single development a tenant sells (e.g. "DSR Highland Greenz").

WHY THIS EXISTS. The engine was built one-project-per-tenant: retrieval is scoped by
``tenant_id`` alone, ``tenant.settings["project_facts"]`` names a single project, and a
booking records only where/when, never WHICH project. That is correct for a tenant with one
development, and wrong the moment a developer runs several — a caller asking a Lemon Tree
question could be answered from a DSR cost sheet, because both projects' chunks share one
retrieval pool, and "what projects do you have?" always returned the one hardcoded set of
facts.

A ``projects`` row makes the project a first-class dimension WITHIN a tenant. Documents,
chunks, appointments and conversations carry an optional ``project_id``; retrieval filters on
it; the prompt's PROJECT FACTS block is taken from the resolved project. Everything is
NULLABLE and falls back to tenant-level behaviour, so a tenant with zero or one project is
byte-for-byte unchanged — the multi-project machinery costs nothing until a second project
exists.

RESOLUTION DATA lives here too, so the engine carries no telephony or vocabulary specifics:
  * ``phone_did`` — the inbound number callers dial for THIS project (``Call.to_number``).
    The strongest deterministic resolution signal; null when projects share one hotline.
  * ``aliases`` — surface names a caller might say ("Highland", "the Sarjapur project"),
    matched when they name a project out loud.
  * ``facts`` / ``synonyms`` — per-project copies of what ``tenant.settings`` held globally,
    so each project answers from its own curated facts and vocabulary bridges.
  * ``default_centre_id`` — the site office a visit to this project is booked at, so booking
    knows the centre from the project without asking.
  * ``is_default`` — the project a tenant falls back to when resolution is otherwise
    ambiguous AND only one project is marked default; the single-project migration sets this.
"""
import uuid

from sqlalchemy import Boolean, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin, TimestampMixin


class Project(IdMixin, TimestampMixin, Base):
    __tablename__ = "projects"
    __table_args__ = (
        UniqueConstraint("tenant_id", "slug", name="uq_project_tenant_slug"),
        # Resolution by dialled number is a hot lookup on every inbound call.
        Index("ix_project_did", "tenant_id", "phone_did"),
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    slug: Mapped[str] = mapped_column(String(100))
    name: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20), default="active")  # active|inactive
    # The fallback project when resolution is ambiguous and exactly one is marked default.
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)

    # --- resolution signals ---
    # The inbound number for THIS project (Call.to_number). Null when projects share a line.
    phone_did: Mapped[str | None] = mapped_column(String(20))
    # Surface names a caller might say for this project, lowercased. Matched when they name it.
    aliases: Mapped[list] = mapped_column(ARRAY(String), default=list)

    # --- per-project content (mirrors what tenant.settings held globally) ---
    # [{"label", "value"}] — the always-answerable facts for this project's prompt block.
    facts: Mapped[list] = mapped_column(JSONB, default=list)
    # [[surface phrases]] — this project's retrieval synonym groups.
    synonyms: Mapped[list] = mapped_column(JSONB, default=list)

    # The site office a visit to this project is booked at. Column name kept as
    # default_centre_id (not renamed to default_location_id) even though the FK
    # target table is now `locations` — see app/db/models/scheduling.py's
    # Location docstring for why this rename pass stops at the wire-contract
    # boundary with the live call engine.
    default_centre_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("locations.id"))
