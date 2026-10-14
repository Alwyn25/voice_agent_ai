"""Seed the first tenant, default roles, and an admin user.

Usage (inside core/ with deps installed and DB up):
    python scripts/seed.py --tenant "Advora" --email admin@advora.ai --password <pw>
Idempotent: safe to run twice.
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import (IndustryPackDef, Role, RuleSet, StateMachineDef, Tenant,  # noqa: E402
                           TenantPackConfig, ToolCatalogEntry, ToolDefinition, User)
from app.engine.rules.defaults import DEFAULT_RULE_SETS  # noqa: E402
from app.engine.rules.schema import compile_rules  # noqa: E402
from app.engine.tools.definitions import tools_for_pack  # noqa: E402
from app.packs.real_estate_india.pack import RealEstateIndiaPack  # noqa: E402
from app.security.auth import hash_password  # noqa: E402
from app.security.rbac import DEFAULT_ROLES  # noqa: E402


BUILTIN_TOOL_CATALOG = [
    {"key": "shopify_order_manager", "category": "orders", "provider_name": "shopify",
     "actions": ["lookup_order", "track_order", "reschedule_delivery", "cancel_order"]},
    {"key": "stripe_refund_manager", "category": "payments", "provider_name": "stripe",
     "actions": ["quote_refund", "issue_refund", "check_status"]},
    {"key": "google_calendar_appointments", "category": "scheduling", "provider_name": "google_calendar",
     "actions": ["check_availability", "create", "reschedule", "cancel"]},
    {"key": "zendesk_ticket_manager", "category": "support", "provider_name": "zendesk",
     "actions": ["create", "update", "escalate"]},
    {"key": "hubspot_crm_manager", "category": "crm", "provider_name": "hubspot",
     "actions": ["find_contact", "update_contact", "log_activity"]},
    {"key": "shippo_delivery_manager", "category": "logistics", "provider_name": "shippo",
     "actions": ["track", "re_route", "hold"]},
    {"key": "twilio_human_handoff", "category": "communication", "provider_name": "twilio",
     "actions": ["transfer", "warm_transfer", "callback"]},
    {"key": "twilio_notification_manager", "category": "communication", "provider_name": "twilio",
     "actions": ["send_sms", "send_email"]},
    {"key": "internal_customer_profile_manager", "category": "crm", "provider_name": "internal_api",
     "actions": ["verify", "fetch_history"]},
    {"key": "internal_feedback_manager", "category": "support", "provider_name": "internal_api",
     "actions": ["ask_rating", "save_comment"]},
    {"key": "recharge_subscription_manager", "category": "orders", "provider_name": "recharge",
     "actions": ["pause", "skip", "cancel", "resume"]},
]


async def main(tenant_name: str, email: str, password: str, reseed_sm: bool = False,
               reseed_tools: bool = False, reseed_rules: bool = False) -> None:
    async with get_session_factory()() as db:
        slug = tenant_name.lower().replace(" ", "-")
        tenant = (await db.execute(select(Tenant).where(Tenant.slug == slug))).scalar_one_or_none()
        if tenant is None:
            tenant = Tenant(name=tenant_name, slug=slug)
            db.add(tenant)
            await db.flush()
            print(f"created tenant {tenant.name} ({tenant.id})")
        else:
            print(f"tenant exists: {tenant.name} ({tenant.id})")

        pack_row = (await db.execute(select(IndustryPackDef).where(
            IndustryPackDef.slug == RealEstateIndiaPack.slug))).scalar_one_or_none()
        if pack_row is None:
            pack_row = IndustryPackDef(slug=RealEstateIndiaPack.slug, name=RealEstateIndiaPack.name)
            db.add(pack_row)
            await db.flush()
        tenant_pack = (await db.execute(select(TenantPackConfig).where(
            TenantPackConfig.tenant_id == tenant.id))).scalar_one_or_none()
        if tenant_pack is None:
            db.add(TenantPackConfig(tenant_id=tenant.id, pack_id=pack_row.id))
            print(f"assigned {RealEstateIndiaPack.slug} pack to tenant")

        role_updated = 0
        for role_name, perms in DEFAULT_ROLES.items():
            role = (await db.execute(select(Role).where(Role.tenant_id == tenant.id,
                                                        Role.name == role_name))).scalar_one_or_none()
            if role is None:
                db.add(Role(tenant_id=tenant.id, name=role_name, permissions=perms))
                print(f"created role {role_name}")
            elif role.permissions != perms:
                # Unlike tools/rules/state-machine, roles carry no version history to
                # preserve — always keep an existing role's permissions in sync with
                # DEFAULT_ROLES, or a code change here silently never reaches tenants
                # seeded before it (that's exactly how a viewer stayed unable to read
                # /users after rbac.py granted viewer users:read).
                role.permissions = perms
                role_updated += 1
        if role_updated:
            print(f"updated permissions on {role_updated} existing role(s)")

        user = (await db.execute(select(User).where(User.tenant_id == tenant.id,
                                                    User.email == email))).scalar_one_or_none()
        if user is None:
            db.add(User(tenant_id=tenant.id, email=email, full_name="Administrator",
                        role="admin", password_hash=hash_password(password)))
            print(f"created admin user {email}")
        else:
            print(f"user exists: {email}")

        smd = (await db.execute(select(StateMachineDef).where(
            StateMachineDef.tenant_id == tenant.id, StateMachineDef.is_active == True))).scalars().first()  # noqa: E712
        if smd is None:
            db.add(StateMachineDef(tenant_id=tenant.id, version=1, is_active=True,
                                   definition=RealEstateIndiaPack().state_machine))
            print("seeded default state machine definition")
        elif reseed_sm:
            smd.is_active = False
            db.add(StateMachineDef(tenant_id=tenant.id, version=smd.version + 1, is_active=True,
                                   definition=RealEstateIndiaPack().state_machine))
            print(f"state machine reseeded as version {smd.version + 1} (previous deactivated)")
        else:
            print("state machine definition exists (use --reseed-state-machine to update)")

        existing_tools = {t.name for t in (await db.execute(select(ToolDefinition).where(
            ToolDefinition.tenant_id == tenant.id))).scalars().all()}
        added = 0
        # Vocal Backend Tasksheet P7 Generic Sprint 4: pack-aware (CORE_TOOLS + this
        # tenant's pack's own extras) instead of the old unconditional BUILTIN_TOOLS —
        # this bootstrap tenant is real_estate_india (see the state-machine seed just
        # above), so it still gets exactly the same 17 tools as before.
        for spec in tools_for_pack(RealEstateIndiaPack()):
            if spec["name"] in existing_tools and not reseed_tools:
                continue
            if spec["name"] in existing_tools:
                row = (await db.execute(select(ToolDefinition).where(
                    ToolDefinition.tenant_id == tenant.id,
                    ToolDefinition.name == spec["name"]))).scalar_one()
                for k, v in spec.items():
                    setattr(row, k, v)
            else:
                # Per-tool, so one bad spec cannot take the rest down with it.
                #
                # A spec with a key the model does not have raises on construction. Inside
                # a bare loop that aborts the whole pass, and the only visible symptom was
                # a tool count one lower than expected — the tool existed in code, the
                # planner could not see it, and nothing said why.
                try:
                    db.add(ToolDefinition(tenant_id=tenant.id, is_enabled=True, **spec))
                except TypeError as e:
                    print(f"  ! {spec['name']} NOT seeded: {e}")
                    continue
            added += 1
        if added:
            print(f"seeded/updated {added} tool definitions")
        else:
            print("tool definitions exist (use --reseed-tools to update)")

        rule_added = 0
        for rname, category, source in DEFAULT_RULE_SETS:
            existing = (await db.execute(select(RuleSet).where(
                RuleSet.tenant_id == tenant.id, RuleSet.name == rname)
                .order_by(RuleSet.version.desc()))).scalars().first()
            if existing and not reseed_rules:
                continue
            compiled = compile_rules(source)
            version = (existing.version + 1) if existing else 1
            if existing:
                for r in (await db.execute(select(RuleSet).where(
                        RuleSet.tenant_id == tenant.id, RuleSet.name == rname))).scalars().all():
                    r.is_active = False
            db.add(RuleSet(tenant_id=tenant.id, name=rname, category=category,
                           yaml_source=source, compiled=compiled, version=version,
                           is_active=True))
            rule_added += 1
        if rule_added:
            print(f"seeded/updated {rule_added} rule sets")
        else:
            print("rule sets exist (use --reseed-rules to update)")

        existing_catalog = {t.key for t in (await db.execute(select(ToolCatalogEntry).where(
            ToolCatalogEntry.tenant_id == tenant.id))).scalars().all()}
        catalog_added = 0
        for spec in BUILTIN_TOOL_CATALOG:
            if spec["key"] in existing_catalog:
                continue
            db.add(ToolCatalogEntry(tenant_id=tenant.id, **spec))
            catalog_added += 1
        if catalog_added:
            print(f"seeded {catalog_added} tool marketplace catalog entries")
        else:
            print("tool marketplace catalog exists")

        await db.commit()
        print("seed complete")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tenant", required=True)
    p.add_argument("--email", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--reseed-rules", action="store_true",
                   help="Deactivate current rule sets and install the built-in ones as a new version")
    p.add_argument("--reseed-tools", action="store_true",
                   help="Overwrite existing tool definitions with the built-in ones")
    p.add_argument("--reseed-state-machine", action="store_true",
                   help="Deactivate the current journey definition and insert the built-in one as a new version")
    args = p.parse_args()
    asyncio.run(main(args.tenant, args.email, args.password, args.reseed_state_machine,
                     args.reseed_tools, args.reseed_rules))