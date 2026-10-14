"""A UNIT — one buyable flat configuration/size in a project's inventory.

WHY THIS EXISTS. Pricing and inventory questions ("anything around 1.1 cr?", "cheapest
3 BHK?", "what does a 2.5 BHK cost?") were answered by a language model free-associating
over retrieved PROSE from the cost-sheet documents. That works for a feature yes/no but not
for a numeric comparison: asked for units "around 1.1 cr" the model latched onto a scripted
"nothing under one crore" line and never did the arithmetic of which units actually sit near
that figure. The grounding guards police feature claims and completion claims — they never
checked that a numeric answer compared against the right rows.

The engine already routes the two other question-classes whose answer is NOT in a document
to a structured source: scheduling goes to the scheduler, "is my visit booked?" goes to the
customer's record. Pricing/inventory is the third such class, and this table is its source of
truth. The ``inventory_query`` tool filters and ranks these rows deterministically; its result
lands in the responder's VERIFIED RESULTS, which already outrank retrieved prose.

Everything is tenant- and project-scoped, mirroring the multi-project machinery: ``project_id``
is nullable and falls back to tenant-wide, so a tenant that seeds no units simply keeps the
old prose-only behaviour and this table costs nothing.

HONESTY IS A FIRST-CLASS FIELD. ``price_published`` records that a configuration EXISTS but
has no cost sheet (DSR's 1 BHK / 1.5 BHK; every Lemon Tree config, whose brochure publishes
no price). The tool surfaces that plainly — "the 1 BHK cost sheet isn't published" — instead
of reaching for another configuration's rate, which is the exact mistake the prose guidance
kept having to warn against.
"""
import uuid

from sqlalchemy import Boolean, ForeignKey, Index, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, IdMixin, TimestampMixin


class Unit(IdMixin, TimestampMixin, Base):
    __tablename__ = "units"
    __table_args__ = (
        # The inventory query filters by (tenant, project) on every pricing turn.
        Index("ix_units_tenant_project", "tenant_id", "project_id"),
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), index=True)
    # Nullable, like the rest of the multi-project columns — tenant-wide when unset.
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True)

    # The configuration exactly as the project names it: "2 BHK", "2.5 BHK", "3 BHK". Stored
    # canonicalised (see app/packs/real_estate_india/seeds/seed_units.py) so a query
    # for "3bhk" matches "3 BHK".
    bhk_type: Mapped[str] = mapped_column(String(30))
    # Super built-up area, the basis the cost sheets quote a rate against. Nullable because a
    # config-only row (a Lemon Tree configuration with no published sheet) has no single size.
    super_builtup_sqft: Mapped[int | None] = mapped_column(Integer, nullable=True)
    carpet_sqft: Mapped[int | None] = mapped_column(Integer, nullable=True)
    facing: Mapped[str | None] = mapped_column(String(40), nullable=True)
    tower: Mapped[str | None] = mapped_column(String(40), nullable=True)

    # Rate on super built-up area. Null when no cost sheet exists for this configuration.
    rate_psf: Mapped[int | None] = mapped_column(Integer, nullable=True)
    flat_cost: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # The ALL-IN published total (flat + infrastructure + clubhouse + parking) — the figure a
    # caller is actually quoted, and the one budget queries compare against. Null when no price
    # is published. Whole rupees.
    all_in_total: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # EXISTS-but-unpriced is a real, common state (1 BHK here, all of Lemon Tree). False means
    # the tool must say the price is not published, never estimate one.
    price_published: Mapped[bool] = mapped_column(Boolean, default=True)
    # available | sold | hold — so a query can exclude what a caller cannot actually buy.
    availability: Mapped[str] = mapped_column(String(20), default="available")

    # Where the figure came from (which cost sheet), for traceability — never spoken.
    source_note: Mapped[str | None] = mapped_column(String(200), nullable=True)
