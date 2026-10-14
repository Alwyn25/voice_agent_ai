"""Orchestration used by the admin API. Keeps the router thin and the feature logic here.

  * load_grounding    — assemble the intent vocab + transfer targets a routing rule grounds on.
  * compile_preview   — run the one classify+compile call, grounded, return the review payload.
  * assert_can_activate (advisory) — per-tenant caps at activation.
  * activate_owned_ruleset / deactivate_owned_ruleset (routing) — a routing instruction OWNS
    a generated RuleSet (versioned by a stable name derived from its root_id) and reuses the
    existing rule engine end to end.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.custom_instructions import caps, intents, routing
from app.custom_instructions.compiler import (CompileResult, GroundingContext,
                                              compile_instruction)
from app.db.models import (CustomInstruction, IntentVocabularyEntry, RuleSet,
                           TransferTarget)


async def load_grounding(db: AsyncSession, tenant_id: uuid.UUID) -> GroundingContext:
    """The intent vocabulary + active transfer targets + tenant tools the compiler grounds
    routing on. Tools are what a ``require`` rule gates ("collect X before Y")."""
    from app.engine.tools import registry
    vocab = await intents.load_vocabulary(db, tenant_id)
    targets = (await db.execute(select(TransferTarget).where(
        TransferTarget.tenant_id == tenant_id,
        TransferTarget.is_active.is_(True)))).scalars().all()
    targets_block = "\n".join(f"- {t.key} — {t.display_name}" for t in targets)
    tools = await registry.list_tools(db, tenant_id, enabled_only=True)
    tools_block = "\n".join(f"- {t.name} — {t.description or ''}".rstrip(" —") for t in tools)
    return GroundingContext(
        vocab_names=set(intents.names(vocab)),
        target_keys={t.key for t in targets},
        tool_names={t.name for t in tools},
        vocab_block=intents.render_for_prompt(vocab),
        targets_block=targets_block,
        tools_block=tools_block)


async def compile_preview(db: AsyncSession, tenant_id: uuid.UUID,
                          raw_text: str) -> CompileResult:
    grounding = await load_grounding(db, tenant_id)
    return await compile_instruction(raw_text, grounding=grounding)


async def _active_advisory_snippets(
    db: AsyncSession, tenant_id: uuid.UUID, *, exclude_id: uuid.UUID | None,
) -> list[str]:
    q = select(CustomInstruction.compiled_prompt).where(
        CustomInstruction.tenant_id == tenant_id,
        CustomInstruction.is_active.is_(True),
        CustomInstruction.kind == "advisory",
    )
    if exclude_id is not None:
        q = q.where(CustomInstruction.id != exclude_id)
    rows = (await db.execute(q)).scalars().all()
    return [r for r in rows if r]


async def assert_can_activate_advisory(
    db: AsyncSession, tenant_id: uuid.UUID, row: CustomInstruction,
) -> None:
    """Raise ``caps.CapExceeded`` / ``ValueError`` if an advisory row can't be activated."""
    if not row.compiled_prompt:
        raise ValueError("Instruction has no compiled advisory text; recompile it first.")
    others = await _active_advisory_snippets(db, tenant_id, exclude_id=row.id)
    caps.check_activation(active_snippets=others, new_snippet=row.compiled_prompt)


async def provision_and_activate(
    db: AsyncSession, tenant_id: uuid.UUID, row: CustomInstruction, *,
    activated_by: uuid.UUID | None,
    target_destinations: dict[str, dict] | None = None,
) -> RuleSet:
    """Create the intents/targets a routing instruction needs, then activate its RuleSet — all
    in ONE transaction (the caller commits). This is what lets a non-technical admin add a
    routing rule end to end: the plain-English instruction carries a provisioning plan the
    compiler built, and approving it here upserts the tenant intent vocabulary + transfer
    target and only then compiles the rule against the now-complete grounding.

    ``target_destinations`` maps a proposed target key -> its destination
    ({"kind": "phone"|"sip"|..., "value": "+91..."}), collected inline at the review gate. A
    target the plan marks ``needs_destination`` MUST have one, or activation is refused (the
    rule would escalate to a target that goes nowhere).
    """
    plan = (row.preview or {}).get("provisioning_plan") or {}
    destinations = target_destinations or {}

    # 1) Tenant intents. Idempotent by (tenant, name): enrich an existing row, else create.
    for it in plan.get("intents", []):
        name = it.get("name")
        if not name:
            continue
        existing = (await db.execute(select(IntentVocabularyEntry).where(
            IntentVocabularyEntry.tenant_id == tenant_id,
            IntentVocabularyEntry.name == name))).scalar_one_or_none()
        if existing is None:
            existing = IntentVocabularyEntry(tenant_id=tenant_id, name=name,
                                             created_by=activated_by)
            db.add(existing)
        existing.description = (it.get("description") or "").strip() or None
        existing.aliases = list(it.get("aliases") or [])
        existing.examples = list(it.get("examples") or [])
        existing.is_active = True

    # 2) Transfer targets. The destination is the one thing no system can invent.
    for tg in plan.get("targets", []):
        key = tg.get("key")
        if not key:
            continue
        dest = destinations.get(key) or {}
        if tg.get("needs_destination") and not (dest.get("value") or "").strip():
            raise ValueError(
                f"transfer target '{tg.get('display_name') or key}' needs a destination "
                f"(the number or SIP address the agent should transfer to)")
        existing = (await db.execute(select(TransferTarget).where(
            TransferTarget.tenant_id == tenant_id,
            TransferTarget.key == key))).scalar_one_or_none()
        if existing is None:
            existing = TransferTarget(tenant_id=tenant_id, key=key, created_by=activated_by)
            db.add(existing)
        existing.display_name = (tg.get("display_name") or key).strip()
        if dest:
            existing.destination = {"kind": dest.get("kind") or tg.get("destination_kind")
                                    or "phone", "value": (dest.get("value") or "").strip()}
        existing.is_active = True

    # Make the new rows visible to load_grounding's SELECT below, in this same transaction.
    await db.flush()

    # 3) Build + activate the rule. activate_owned_ruleset re-validates against LIVE grounding,
    # which now includes everything we just provisioned — so it grounds, or we raise.
    return await activate_owned_ruleset(db, tenant_id, row, activated_by=activated_by)


async def activate_owned_ruleset(
    db: AsyncSession, tenant_id: uuid.UUID, row: CustomInstruction, *,
    activated_by: uuid.UUID | None,
) -> RuleSet:
    """Build/activate the RuleSet a routing instruction owns, and link it back on the row.

    Re-validates against the CURRENT grounding (vocab/targets may have changed since
    compile), writes a new RuleSet version under the stable per-instruction name, and
    deactivates older versions of that name. Raises ValueError if it no longer grounds.
    """
    preview = row.preview or {}
    when, then = preview.get("rule_when"), preview.get("rule_then")
    priority = int(preview.get("rule_priority") or 100)
    gate_tools = preview.get("gate_tools") or []
    root = row.root_id or row.id

    grounding = await load_grounding(db, tenant_id)
    compiled, yaml_source, problems = routing.build_and_validate(
        instruction_id=root, when=when, then=then, priority=priority,
        vocab_names=grounding.vocab_names, target_keys=grounding.target_keys,
        gate_tools=gate_tools, tool_names=grounding.tool_names)
    if problems or compiled is None:
        raise ValueError("routing rule no longer grounds: " + "; ".join(problems)
                         if problems else "routing rule could not be compiled")

    name = routing.rule_set_name(root)
    existing = (await db.execute(select(RuleSet).where(
        RuleSet.tenant_id == tenant_id, RuleSet.name == name)
        .order_by(RuleSet.version.desc()))).scalars().all()
    next_version = (existing[0].version + 1) if existing else 1
    for r in existing:
        r.is_active = False
    rs = RuleSet(tenant_id=tenant_id, name=name, category="custom_instruction",
                 yaml_source=yaml_source, compiled=compiled, version=next_version,
                 is_active=True, activated_by=activated_by,
                 activated_at=datetime.now(timezone.utc))
    db.add(rs)
    await db.flush()
    row.rule_set_id = rs.id
    return rs


async def deactivate_owned_ruleset(
    db: AsyncSession, tenant_id: uuid.UUID, row: CustomInstruction,
) -> None:
    """Deactivate every RuleSet version owned by this instruction (by its stable name), so
    the rule stops firing on the next turn."""
    root = row.root_id or row.id
    name = routing.rule_set_name(root)
    rows = (await db.execute(select(RuleSet).where(
        RuleSet.tenant_id == tenant_id, RuleSet.name == name))).scalars().all()
    for r in rows:
        r.is_active = False
