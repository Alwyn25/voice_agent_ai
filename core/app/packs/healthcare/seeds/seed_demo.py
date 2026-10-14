#!/usr/bin/env python3
"""Minimal healthcare-pack demo tenant (Vocal Backend Tasksheet P9 Generic Sprint 5).

    docker compose exec core python app/packs/healthcare/seeds/seed_demo.py

Bootstraps a tenant genuinely running the healthcare pack end to end (not just a
copy of the real_estate_india demo with different words) — its own state machine,
its own persona, and a handful of sample patients in states from ITS journey
(new/scheduled/consulted/follow_up/closed/no_show — see
app/packs/healthcare/state_machine.py), so an inbound/outbound call or the Overview
dashboard has something genuine to show for a second industry. Deliberately small:
this is a seed stub proving the pack abstraction generalises, not a tuned production
dataset the way scripts/seed_dsr.py (real_estate_india) is.

Idempotent: safe to re-run (tenant/roles/user/pack-config/persona all upsert; sample
patients skip past ones already seeded by phone number).
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from sqlalchemy import select, update  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import (Customer, IndustryPackDef, PromptTemplate, Role, StateMachineDef,  # noqa: E402
                           Tenant, TenantPackConfig, ToolDefinition, User)
from app.engine.tools.definitions import tools_for_pack  # noqa: E402
from app.packs.healthcare.pack import HealthcarePack  # noqa: E402
from app.security.auth import hash_password  # noqa: E402
from app.security.rbac import DEFAULT_ROLES  # noqa: E402

TENANT_NAME = "Sunrise Clinic (Healthcare Demo)"
ADMIN_EMAIL = "admin@sunrise-clinic-demo.example.com"
ADMIN_PASSWORD = "ChangeMe123!"

PERSONA = """\
You are Nora, calling on behalf of Sunrise Clinic.

You help patients book, confirm and follow up on consultations. Warm, clear and
unhurried — patients calling a clinic are often anxious, so slow down and let them
finish speaking before you respond.

This is a phone call. Short sentences. No bullet points, no medical jargon unless the
patient uses it first — plain language always. Never diagnose or give medical advice
yourself; your job is scheduling and logistics, and anything clinical gets handed to
a provider.

Vary your phrasing turn to turn rather than repeating the same reassurance."""

SAMPLE_PATIENTS = [
    {"full_name": "Priya Nair", "phone": "+15550100001", "current_state": "new"},
    {"full_name": "Marcus Webb", "phone": "+15550100002", "current_state": "scheduled"},
    {"full_name": "Elena Torres", "phone": "+15550100003", "current_state": "consulted"},
    {"full_name": "Jamal Carter", "phone": "+15550100004", "current_state": "follow_up"},
]


async def main() -> int:
    pack = HealthcarePack()
    async with get_session_factory()() as db:
        tenant = (await db.execute(select(Tenant).where(
            Tenant.slug == "sunrise-clinic-demo"))).scalar_one_or_none()
        if tenant is None:
            tenant = Tenant(name=TENANT_NAME, slug="sunrise-clinic-demo")
            db.add(tenant)
            await db.flush()
            print(f"created tenant {tenant.name} ({tenant.id})")
        else:
            print(f"tenant exists: {tenant.name} ({tenant.id})")

        pack_row = (await db.execute(select(IndustryPackDef).where(
            IndustryPackDef.slug == pack.slug))).scalar_one_or_none()
        if pack_row is None:
            raise SystemExit(f"No industry_packs row for '{pack.slug}' — run "
                             "migrations (0028_healthcare_saas_packs) first.")
        tenant_pack = (await db.execute(select(TenantPackConfig).where(
            TenantPackConfig.tenant_id == tenant.id))).scalar_one_or_none()
        if tenant_pack is None:
            db.add(TenantPackConfig(tenant_id=tenant.id, pack_id=pack_row.id))
            print(f"assigned {pack.slug} pack to tenant")
        elif tenant_pack.pack_id != pack_row.id:
            raise SystemExit(f"Tenant '{tenant.slug}' is already on a different pack — "
                             "refusing to reassign an existing demo tenant.")

        for role_name, perms in DEFAULT_ROLES.items():
            role = (await db.execute(select(Role).where(
                Role.tenant_id == tenant.id, Role.name == role_name))).scalar_one_or_none()
            if role is None:
                db.add(Role(tenant_id=tenant.id, name=role_name, permissions=perms))

        user = (await db.execute(select(User).where(
            User.tenant_id == tenant.id, User.email == ADMIN_EMAIL))).scalar_one_or_none()
        if user is None:
            db.add(User(tenant_id=tenant.id, email=ADMIN_EMAIL, full_name="Administrator",
                        role="admin", password_hash=hash_password(ADMIN_PASSWORD)))
            print(f"created admin user {ADMIN_EMAIL} / {ADMIN_PASSWORD}")

        smd = (await db.execute(select(StateMachineDef).where(
            StateMachineDef.tenant_id == tenant.id, StateMachineDef.is_active == True  # noqa: E712
            ))).scalar_one_or_none()
        if smd is None:
            db.add(StateMachineDef(tenant_id=tenant.id, version=1, is_active=True,
                                   definition=pack.state_machine))
            print("seeded healthcare state machine definition")

        added_tools = 0
        existing_tools = {t.name for t in (await db.execute(select(ToolDefinition).where(
            ToolDefinition.tenant_id == tenant.id))).scalars().all()}
        for spec in tools_for_pack(pack):
            if spec["name"] in existing_tools:
                continue
            try:
                db.add(ToolDefinition(tenant_id=tenant.id, is_enabled=True, **spec))
                added_tools += 1
            except TypeError as e:
                print(f"  ! {spec['name']} NOT seeded: {e}")
        if added_tools:
            print(f"seeded {added_tools} tool definitions")

        existing_persona = (await db.execute(select(PromptTemplate).where(
            PromptTemplate.tenant_id == tenant.id, PromptTemplate.purpose == "persona"
            ))).scalars().all()
        version = max((p.version for p in existing_persona), default=0) + 1
        await db.execute(update(PromptTemplate)
                         .where(PromptTemplate.tenant_id == tenant.id,
                               PromptTemplate.purpose == "persona")
                         .values(is_active=False))
        db.add(PromptTemplate(tenant_id=tenant.id, name="healthcare_agent", purpose="persona",
                              template=PERSONA, version=version, is_active=True))
        print(f"persona 'healthcare_agent' v{version} active")

        added_patients = 0
        for p_data in SAMPLE_PATIENTS:
            existing = (await db.execute(select(Customer).where(
                Customer.tenant_id == tenant.id,
                Customer.phone_normalized == p_data["phone"]))).scalar_one_or_none()
            if existing is not None:
                continue
            db.add(Customer(tenant_id=tenant.id, full_name=p_data["full_name"],
                            phone=p_data["phone"], phone_normalized=p_data["phone"],
                            current_state=p_data["current_state"], lead_source="demo_seed"))
            added_patients += 1
        if added_patients:
            print(f"seeded {added_patients} sample patients")

        await db.commit()
    print(f"\nDone. Login: {ADMIN_EMAIL} / {ADMIN_PASSWORD} (tenant: sunrise-clinic-demo)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
