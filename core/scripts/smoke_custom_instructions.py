#!/usr/bin/env python3
"""Smoke test for Custom Instructions & Dynamic Knowledge (Phase 1 + Phase 2).

Runs against the live DB. Needs NO LLM credits — it exercises the runtime plumbing
directly (loaders, scope match, render, intent-vocabulary normalisation, routing rule
firing, targeted-escalation resolution) rather than the author-time compile call. It
inserts rows tagged "SMOKE:" under the first seeded tenant, asserts each effect, and
cleans them up at the end so re-runs stay idempotent.

    docker compose exec -T -e LLM_PROVIDER=mock core python scripts/smoke_custom_instructions.py

For the author-time COMPILE step (which does call the LLM), use the HTTP checks in the
handover notes with a real LLM key instead.
"""
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.custom_instructions import intents, routing, service  # noqa: E402
from app.custom_instructions.loader import (load_active_advisory,  # noqa: E402
                                            load_active_dynamic_facts)
from app.custom_instructions.render import render_advisory_block  # noqa: E402
from app.custom_instructions.scope import TurnScope  # noqa: E402
from app.db.base import get_session_factory  # noqa: E402
from app.db.models import (Customer, CustomInstruction, IntentVocabularyEntry,  # noqa: E402
                           ProjectFactDynamic, RuleSet, Tenant, TransferTarget)
from app.engine.planner.confidence import normalize_intent  # noqa: E402
from app.engine.rules import engine as rule_engine  # noqa: E402
from app.engine.rules.context import build_context  # noqa: E402

_MARK = "SMOKE:"
_fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        _fail += 1


async def _cleanup(db, tenant_id) -> None:
    for r in (await db.execute(select(CustomInstruction).where(
            CustomInstruction.tenant_id == tenant_id,
            CustomInstruction.raw_text.like(f"{_MARK}%")))).scalars().all():
        await db.delete(r)
    for r in (await db.execute(select(ProjectFactDynamic).where(
            ProjectFactDynamic.tenant_id == tenant_id,
            ProjectFactDynamic.content.like(f"{_MARK}%")))).scalars().all():
        await db.delete(r)
    for r in (await db.execute(select(IntentVocabularyEntry).where(
            IntentVocabularyEntry.tenant_id == tenant_id,
            IntentVocabularyEntry.name.in_(["smoke_loan", "smoke_prov_loan"])))).scalars().all():
        await db.delete(r)
    for r in (await db.execute(select(TransferTarget).where(
            TransferTarget.tenant_id == tenant_id,
            TransferTarget.key.in_(["smoke_finance_desk", "smoke_prov_desk"])))).scalars().all():
        await db.delete(r)
    for r in (await db.execute(select(RuleSet).where(
            RuleSet.tenant_id == tenant_id,
            RuleSet.name.like("ci_%"),
            RuleSet.category == "custom_instruction"))).scalars().all():
        await db.delete(r)
    await db.commit()


async def main() -> int:
    async with get_session_factory()() as db:
        tenant = (await db.execute(
            select(Tenant).order_by(Tenant.created_at.asc()).limit(1))).scalar_one_or_none()
        if tenant is None:
            print("No tenant found — create your tenant/user first (scripts/seed.py).")
            return 1
        tenant_id = tenant.id
        # A customer is optional here — only build_context uses it, and it accepts None.
        cust = (await db.execute(select(Customer).where(
            Customer.tenant_id == tenant_id).limit(1))).scalar_one_or_none()
        today = date.today()
        print(f"tenant: {tenant.name}  ({tenant_id})")
        print(f"customer: {'yes' if cust else 'none (routing test will use an anonymous turn)'}\n")
        await _cleanup(db, tenant_id)  # start clean

        # ---------- Phase 1: advisory instruction ----------
        print("Phase 1 — advisory instruction")
        adv = CustomInstruction(
            tenant_id=tenant_id, project_id=None,
            raw_text=f"{_MARK} be more formal with corporate callers",
            kind="advisory",
            compiled_prompt="Be more formal when the caller identifies as a corporate buyer.",
            scope={}, priority=100, is_active=True, version=1)
        adv.root_id = None
        db.add(adv)
        await db.commit()
        rows = await load_active_advisory(
            db, tenant_id, project_id=None,
            turn=TurnScope(direction="inbound", state=None, intent=None, today=today))
        check("advisory row loaded", any("corporate buyer" in r["text"] for r in rows),
              f"got {rows}")
        block = render_advisory_block(rows)
        check("advisory block renders subordinate", bool(block) and "TENANT PREFERENCES" in block
              and "NEVER" in block)

        # scope filter: an outbound-only advisory must NOT match an inbound turn
        adv_out = CustomInstruction(
            tenant_id=tenant_id, project_id=None, raw_text=f"{_MARK} outbound only",
            kind="advisory", compiled_prompt="Lead with the location on outbound calls.",
            scope={"direction": "outbound"}, priority=100, is_active=True, version=1)
        db.add(adv_out)
        await db.commit()
        rows_in = await load_active_advisory(
            db, tenant_id, project_id=None,
            turn=TurnScope(direction="inbound", state=None, intent=None, today=today))
        check("scope filter excludes outbound-only on inbound turn",
              all("location on outbound" not in r["text"] for r in rows_in))

        # ---------- Phase 1: dynamic knowledge ----------
        print("\nPhase 1 — dynamic knowledge")
        fact = ProjectFactDynamic(
            tenant_id=tenant_id, project_id=None, label="Festive offer",
            content=f"{_MARK} Festive offer: 10% discount on bookings before 31 Oct.",
            valid_from=today - timedelta(days=1), valid_until=today + timedelta(days=30),
            priority=10, is_active=True)
        expired = ProjectFactDynamic(
            tenant_id=tenant_id, project_id=None, label="Old offer",
            content=f"{_MARK} Expired offer — should never load.",
            valid_from=today - timedelta(days=60), valid_until=today - timedelta(days=1),
            priority=10, is_active=True)
        db.add_all([fact, expired])
        await db.commit()
        facts = await load_active_dynamic_facts(db, tenant_id, project_id=None, today=today)
        vals = [f["value"] for f in facts]
        check("active dynamic fact loaded", any("Festive offer" in v for v in vals))
        check("expired dynamic fact excluded", all("Expired offer" not in v for v in vals))

        # ---------- Phase 2: intent vocabulary ----------
        print("\nPhase 2 — intent vocabulary")
        db.add(IntentVocabularyEntry(
            tenant_id=tenant_id, name="smoke_loan",
            description="Loan / finance questions",
            aliases=["loan", "finance", "emi", "home loan"], examples=[], is_active=True))
        await db.commit()
        vocab = await intents.load_vocabulary(db, tenant_id)
        extra = intents.extra_aliases(vocab)
        check("tenant intent in vocabulary", "smoke_loan" in intents.names(vocab))
        check("free-form label normalises to tenant intent",
              normalize_intent("asking about a home loan", extra) == "smoke_loan",
              normalize_intent("asking about a home loan", extra))
        check("core safety intent still wins",
              normalize_intent("talk_to_a_human", {"smoke_loan": ["human"]}) == "request_human")

        # ---------- Phase 2: transfer target ----------
        print("\nPhase 2 — transfer target + routing rule")
        db.add(TransferTarget(
            tenant_id=tenant_id, key="smoke_finance_desk",
            display_name="Finance Desk", destination={"kind": "queue", "value": "finance"},
            is_active=True))
        await db.commit()

        # Build + validate a routing rule against the LIVE vocab/targets, then activate it.
        vnames = set(intents.names(await intents.load_vocabulary(db, tenant_id)))
        tkeys = {t.key for t in (await db.execute(select(TransferTarget).where(
            TransferTarget.tenant_id == tenant_id,
            TransferTarget.is_active.is_(True)))).scalars().all()}
        compiled, yaml_src, probs = routing.build_and_validate(
            instruction_id="smoke-root",
            when={"any": [{"field": "turn.intent", "op": "eq", "value": "smoke_loan"}]},
            then=[{"escalate": {"reason": "Loan enquiry", "target": "smoke_finance_desk",
                                "type": "immediate"}}],
            priority=10, vocab_names=vnames, target_keys=tkeys)
        check("routing rule grounds + compiles", not probs and compiled is not None,
              str(probs))
        if compiled:
            db.add(RuleSet(tenant_id=tenant_id, name=routing.rule_set_name("smoke-root"),
                           category="custom_instruction", yaml_source=yaml_src,
                           compiled=compiled, version=1, is_active=True))
            await db.commit()

            # A loan-intent turn must fire the routing escalation with the resolved target.
            ctx = await build_context(db, tenant_id, evaluation_point="turn", customer=cust,
                                      turn={"intent": "smoke_loan"})
            decision = await rule_engine.evaluate_point(
                db, tenant_id, evaluation_point="turn", ctx=ctx)
            esc = [e for e in decision.escalations if e.body.get("target") == "smoke_finance_desk"]
            check("routing rule fires on loan intent", bool(esc),
                  f"escalations={[e.rule_id for e in decision.escalations]}")

            # Targeted-escalation resolution: the named target resolves against the registry.
            tt = (await db.execute(select(TransferTarget).where(
                TransferTarget.tenant_id == tenant_id,
                TransferTarget.key == "smoke_finance_desk",
                TransferTarget.is_active.is_(True)))).scalar_one_or_none()
            check("transfer target resolves in registry", tt is not None and tt.key == "smoke_finance_desk")

        # ---------- Phase 2b: self-provisioning activation ----------
        # The production path a non-technical admin takes: a plain-English routing instruction
        # whose intent AND target do not exist yet is saved INACTIVE with a provisioning plan
        # (what the compiler emits), then approved at the review gate — provision_and_activate
        # creates the intent + target and activates the rule in one transaction.
        print("\nPhase 2b — self-provisioning activation (new intent + new target)")
        plan = {"intents": [{"name": "smoke_prov_loan", "description": "Loan questions",
                             "aliases": ["loan", "home loan", "emi", "financing"],
                             "examples": ["I need a home loan"]}],
                "targets": [{"key": "smoke_prov_desk", "display_name": "Provisioned Desk",
                             "destination_kind": "phone", "needs_destination": True}]}
        prov = CustomInstruction(
            tenant_id=tenant_id, project_id=None,
            raw_text=f"{_MARK} connect loan enquiries to the provisioned desk",
            kind="routing", scope={}, priority=10, is_active=False, version=1,
            preview={"rule_when": {"any": [{"field": "turn.intent", "op": "eq",
                                            "value": "smoke_prov_loan"}]},
                     "rule_then": [{"escalate": {"reason": "Loan enquiry",
                                                 "target": "smoke_prov_desk",
                                                 "type": "immediate"}}],
                     "rule_priority": 10, "provisioning_plan": plan})
        db.add(prov)
        await db.flush()
        prov.root_id = prov.id
        await db.commit()

        # Refusing without a destination protects against an escalation that goes nowhere.
        refused = False
        try:
            await service.provision_and_activate(db, tenant_id, prov, activated_by=None,
                                                 target_destinations=None)
        except ValueError:
            refused = True
        await db.rollback()
        check("activation refused when transfer destination missing", refused)

        prov = (await db.execute(select(CustomInstruction).where(
            CustomInstruction.id == prov.id))).scalar_one()
        rs = await service.provision_and_activate(
            db, tenant_id, prov, activated_by=None,
            target_destinations={"smoke_prov_desk": {"kind": "phone", "value": "+91 98000 00000"}})
        await db.commit()
        check("owned RuleSet built + active", rs is not None and rs.is_active)

        vocab2 = intents.names(await intents.load_vocabulary(db, tenant_id))
        check("intent auto-provisioned", "smoke_prov_loan" in vocab2)
        tgt = (await db.execute(select(TransferTarget).where(
            TransferTarget.tenant_id == tenant_id,
            TransferTarget.key == "smoke_prov_desk"))).scalar_one_or_none()
        check("target auto-provisioned with destination",
              tgt is not None and tgt.destination.get("value") == "+91 98000 00000")

        # And the end-to-end effect: a loan-intent turn now fires the provisioned escalation.
        ctx2 = await build_context(db, tenant_id, evaluation_point="turn", customer=cust,
                                   turn={"intent": "smoke_prov_loan"})
        dec2 = await rule_engine.evaluate_point(db, tenant_id, evaluation_point="turn", ctx=ctx2)
        check("provisioned rule fires end to end",
              any(e.body.get("target") == "smoke_prov_desk" for e in dec2.escalations))

        # ---------- cleanup ----------
        await _cleanup(db, tenant_id)
        print(f"\n{'ALL CHECKS PASSED' if _fail == 0 else str(_fail) + ' CHECK(S) FAILED'}")
        return 1 if _fail else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
