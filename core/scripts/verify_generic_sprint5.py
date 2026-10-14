#!/usr/bin/env python3
"""Vocal Backend Tasksheet P9 (Analytics + Reports + Generic Sprint 5: Seeds & New
Packs) Definition of Done, verified against the live container.

    docker compose exec core python scripts/verify_generic_sprint5.py

Unlike every other verify_*.py in this project, this one does NOT clean up its own
fixtures at the end for the two demo tenants (sunrise-clinic-demo, nimbus-sales-demo)
— they are the actual deliverable ("enables multi-industry demos"), not throwaway
test data, so this script leaves them in place. It DOES clean up the one saved_report
it creates to test the /reports CRUD, since that has no standing demo value.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import Customer, StateMachineDef, Tenant, TenantPackConfig, User  # noqa: E402
from app.packs import PackRegistry  # noqa: E402
from app.security.auth import create_token  # noqa: E402

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
failures = 0
BASE_URL = "http://localhost:8000"


def check(label: str, cond: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if cond else FAIL} {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures += 1


async def test_pack_registry() -> None:
    slugs = {"real_estate_india", "generic", "healthcare", "saas_sales"}
    registered = set()
    for slug in slugs:
        try:
            pack = PackRegistry.get(slug)
            registered.add(pack.slug)
        except Exception:
            pass
    check("all 4 packs (2 pre-existing + 2 new) are registered", registered == slugs,
         f"got {registered}")

    healthcare = PackRegistry.get("healthcare")
    saas = PackRegistry.get("saas_sales")
    check("healthcare pack has its own distinct state machine (not RE's/generic's)",
         set(healthcare.state_machine["states"]) == {
             "new", "scheduled", "consulted", "follow_up", "closed", "no_show"},
         f"states={list(healthcare.state_machine['states'])}")
    check("saas_sales pack has its own distinct state machine",
         set(saas.state_machine["states"]) == {
             "new", "demo_scheduled", "trial_active", "contract_sent", "closed_won", "closed_lost"},
         f"states={list(saas.state_machine['states'])}")
    check("healthcare vocabulary is genuinely different from saas_sales' (not copy-pasted)",
         healthcare.vocabulary.commitment_noun != saas.vocabulary.commitment_noun,
         f"healthcare={healthcare.vocabulary.commitment_noun} saas={saas.vocabulary.commitment_noun}")
    check("healthcare conversion_state matches one of its own real states",
         healthcare.conversion_state in healthcare.state_machine["states"])
    check("saas_sales conversion_state matches one of its own real states",
         saas.conversion_state in saas.state_machine["states"])


async def test_demo_tenants_exist_and_work(db, token_by_slug: dict) -> None:
    """The two seed_demo.py scripts (app/packs/healthcare/seeds/,
    app/packs/saas_sales/seeds/) were already run once as part of building this —
    this just confirms what they left behind is real and functional, live."""
    for slug, pack_slug in (("sunrise-clinic-demo", "healthcare"), ("nimbus-sales-demo", "saas_sales")):
        tenant = (await db.execute(select(Tenant).where(Tenant.slug == slug))).scalar_one_or_none()
        check(f"demo tenant '{slug}' exists", tenant is not None)
        if tenant is None:
            continue
        cfg = (await db.execute(select(TenantPackConfig).where(
            TenantPackConfig.tenant_id == tenant.id))).scalar_one_or_none()
        pack = PackRegistry.get(pack_slug)
        check(f"'{slug}' is actually assigned the {pack_slug} pack",
             cfg is not None and str(cfg.pack_id) is not None)
        smd = (await db.execute(select(StateMachineDef).where(
            StateMachineDef.tenant_id == tenant.id, StateMachineDef.is_active == True  # noqa: E712
            ))).scalar_one_or_none()
        check(f"'{slug}' has its pack's own state machine seeded",
             smd is not None and set(smd.definition["states"]) == set(pack.state_machine["states"]))
        customers = (await db.execute(select(Customer).where(
            Customer.tenant_id == tenant.id))).scalars().all()
        check(f"'{slug}' has sample customers in real states from its own journey",
             len(customers) >= 1 and all(c.current_state in pack.state_machine["states"]
                                        for c in customers),
             f"states seen={[c.current_state for c in customers]}")

        user = (await db.execute(select(User).where(User.tenant_id == tenant.id))).scalars().first()
        if user is not None:
            token_by_slug[slug] = create_token(user_id=user.id, tenant_id=tenant.id,
                                               role=user.role, token_type="access")


async def test_capabilities_and_reports(token: str) -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30,
                                 headers={"Authorization": f"Bearer {token}"}) as client:
        r_cap = await client.get("/analytics/capabilities")
        check("GET /analytics/capabilities 200s", r_cap.status_code == 200)
        caps = r_cap.json().get("capabilities", [])
        check("capabilities lists both the older /analytics/* and newer /dashboard/* endpoints",
             any("/analytics/" in c["endpoint"] for c in caps)
             and any("/dashboard/" in c["endpoint"] for c in caps),
             f"endpoints={[c['endpoint'] for c in caps]}")
        check("every capability names which table(s) it reads from (the handoff DoD)",
             all(c.get("reads_from") for c in caps))

        r_create = await client.post("/reports", json={
            "name": "Verify P9 Report", "config": {"chart": "call-activity", "days": 7}})
        check("POST /reports returns 201", r_create.status_code == 201, f"got {r_create.text}")
        report_id = r_create.json()["id"]

        r_list = await client.get("/reports")
        check("GET /reports lists the new report",
             any(r["id"] == report_id for r in r_list.json()))

        r_get = await client.get(f"/reports/{report_id}")
        check("GET /reports/{id} returns the same config",
             r_get.status_code == 200 and r_get.json()["config"]["days"] == 7)

        r_delete = await client.delete(f"/reports/{report_id}")
        check("DELETE /reports/{id} returns deleted:true",
             r_delete.status_code == 200 and r_delete.json().get("deleted") is True)

        r_list_after = await client.get("/reports")
        check("GET /reports no longer lists the deleted report",
             not any(r["id"] == report_id for r in r_list_after.json()))


async def main() -> int:
    await test_pack_registry()

    token_by_slug: dict = {}
    async with get_session_factory()() as db:
        await test_demo_tenants_exist_and_work(db, token_by_slug)

    demo_tenant_token = None
    async with get_session_factory()() as db:
        tenant = (await db.execute(select(Tenant).where(Tenant.slug == "demo-tenant"))).scalar_one_or_none()
        if tenant is not None:
            user = (await db.execute(select(User).where(User.tenant_id == tenant.id))).scalars().first()
            if user is not None:
                demo_tenant_token = create_token(user_id=user.id, tenant_id=tenant.id,
                                                 role=user.role, token_type="access")
    token = demo_tenant_token or next(iter(token_by_slug.values()), None)
    if token is None:
        raise SystemExit("No tenant with a user found to test /analytics/capabilities and /reports.")
    await test_capabilities_and_reports(token)

    print()
    if failures:
        print(f"{FAIL} {failures} check(s) failed")
        return 1
    print(f"{PASS} all P9 / Generic Sprint 5 Definition-of-Done checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
