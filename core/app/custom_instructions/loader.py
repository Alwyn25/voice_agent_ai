"""Turn-time loaders. Deterministic, cheap, no LLM.

Both queries are per-turn and uncached — like ``active_rule_sets`` and the project-facts
query they sit beside — so an admin activate/deactivate/delete takes effect on the very
next turn of any call, including one already in progress, with no cache invalidation.
"""
from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.custom_instructions.scope import TurnScope, matches
from app.db.models import CustomInstruction, ProjectFactDynamic


def _project_filter(model, project_id: uuid.UUID | None):
    """Match tenant-wide rows (project_id IS NULL) plus, when a project is resolved, that
    project's own rows."""
    if project_id is None:
        return model.project_id.is_(None)
    return or_(model.project_id.is_(None), model.project_id == project_id)


async def load_active_advisory(
    db: AsyncSession, tenant_id: uuid.UUID, *,
    project_id: uuid.UUID | None,
    turn: TurnScope,
) -> list[dict]:
    """Active advisory rows whose scope matches this turn, ordered by priority then age.

    Returns ``[{"id", "priority", "text"}]``. The count is tiny (per-tenant caps), so the
    scope match runs in Python for clarity rather than as jsonb SQL.
    """
    rows = (await db.execute(
        select(CustomInstruction).where(
            CustomInstruction.tenant_id == tenant_id,
            CustomInstruction.is_active.is_(True),
            CustomInstruction.kind == "advisory",
            _project_filter(CustomInstruction, project_id),
        ).order_by(CustomInstruction.priority.asc(), CustomInstruction.created_at.asc())
    )).scalars().all()

    out: list[dict] = []
    for r in rows:
        if not r.compiled_prompt:
            continue
        if matches(r.scope, turn):
            out.append({"id": str(r.id), "priority": r.priority,
                        "text": r.compiled_prompt.strip()})
    return out


async def load_active_dynamic_facts(
    db: AsyncSession, tenant_id: uuid.UUID, *,
    project_id: uuid.UUID | None,
    today: date,
) -> list[dict]:
    """Active, in-window dynamic facts for the tenant/project, highest priority first.

    Returns ``[{"label", "value"}]`` — the exact shape the existing PROJECT FACTS block
    consumes — so the caller merges them straight into ``project_facts``.
    """
    rows = (await db.execute(
        select(ProjectFactDynamic).where(
            ProjectFactDynamic.tenant_id == tenant_id,
            ProjectFactDynamic.is_active.is_(True),
            _project_filter(ProjectFactDynamic, project_id),
            or_(ProjectFactDynamic.valid_from.is_(None),
                ProjectFactDynamic.valid_from <= today),
            or_(ProjectFactDynamic.valid_until.is_(None),
                ProjectFactDynamic.valid_until >= today),
        ).order_by(ProjectFactDynamic.priority.desc(),
                   ProjectFactDynamic.created_at.asc())
    )).scalars().all()

    out: list[dict] = []
    for r in rows:
        content = (r.content or "").strip()
        if not content:
            continue
        out.append({"label": (r.label or "Offer"), "value": content})
    return out
