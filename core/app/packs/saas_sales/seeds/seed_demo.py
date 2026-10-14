#!/usr/bin/env python3
"""Minimal saas_sales-pack demo tenant (Vocal Backend Tasksheet P9 Generic Sprint 5).

    docker compose exec core python app/packs/saas_sales/seeds/seed_demo.py

Same shape as app/packs/healthcare/seeds/seed_demo.py — see that file's docstring
for what this is and is not.

Idempotent: safe to re-run.
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
from app.packs.saas_sales.pack import SaasSalesPack  # noqa: E402
from app.security.auth import hash_password  # noqa: E402
from app.security.rbac import DEFAULT_ROLES  # noqa: E402

TENANT_NAME = "Nimbus (SaaS Sales Demo)"
ADMIN_EMAIL = "admin@nimbus-sales-demo.example.com"
ADMIN_PASSWORD = "ChangeMe123!"

PERSONA = """\
You are Sam, calling on behalf of Nimbus.

You help prospects book a demo, get set up on a trial, and move toward a decision.
Confident and low-pressure — the goal is a genuinely good fit, not a close at any
cost.

This is a phone call. Short sentences. No jargon, no "synergy", no reading off a
deck. Lead with what the prospect actually said they need, not a generic pitch.

Vary your phrasing turn to turn. If a trial or contract question comes up that needs
a real answer, say so plainly rather than guessing."""

SAMPLE_LEADS = [
    {"full_name": "Devon Blake", "phone": "+15550200001", "current_state": "new"},
    {"full_name": "Aisha Rahman", "phone": "+15550200002", "current_state": "demo_scheduled"},
    {"full_name": "Tomas Lindgren", "phone": "+15550200003", "current_state": "trial_active"},
    {"full_name": "Yuki Sato", "phone": "+15550200004", "current_state": "contract_sent"},
]


async def main() -> int:
    pack = SaasSalesPack()
    async with get_session_factory()() as db:
        tenant = (await db.execute(select(Tenant).where(
            Tenant.slug == "nimbus-sales-demo"))).scalar_one_or_none()
        if tenant is None:
            tenant = Tenant(name=TENANT_NAME, slug="nimbus-sales-demo")
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
            print("seeded saas_sales state machine definition")

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
        db.add(PromptTemplate(tenant_id=tenant.id, name="saas_sales_agent", purpose="persona",
                              template=PERSONA, version=version, is_active=True))
        print(f"persona 'saas_sales_agent' v{version} active")

        added_leads = 0
        for l_data in SAMPLE_LEADS:
            existing = (await db.execute(select(Customer).where(
                Customer.tenant_id == tenant.id,
                Customer.phone_normalized == l_data["phone"]))).scalar_one_or_none()
            if existing is not None:
                continue
            db.add(Customer(tenant_id=tenant.id, full_name=l_data["full_name"],
                            phone=l_data["phone"], phone_normalized=l_data["phone"],
                            current_state=l_data["current_state"], lead_source="demo_seed"))
            added_leads += 1
        if added_leads:
            print(f"seeded {added_leads} sample leads")

        await db.commit()
    print(f"\nDone. Login: {ADMIN_EMAIL} / {ADMIN_PASSWORD} (tenant: nimbus-sales-demo)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
