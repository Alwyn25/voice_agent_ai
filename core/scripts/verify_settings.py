#!/usr/bin/env python3
"""Vocal Backend Tasksheet P8 (Settings tier — Integrations / Team / Billing)
Definition of Done, verified against the live container over real HTTP with a
minted JWT.

    docker compose exec -T core python scripts/verify_settings.py

Creates and cleans up its own tenant-scoped rows (a test user, a pending invite, an
integration connection) — restores the tenant's plan/seats/subscription/invoices to
their pre-run state afterward, since billing changes are tenant-wide, not scoped to
a prefix this script could just delete by name. Safe to re-run.
"""
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from sqlalchemy import delete, select  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import (Integration, Invoice, Subscription, Tenant, ToolCatalogEntry,  # noqa: E402
                           ToolConnection, User)
from app.security.auth import create_token  # noqa: E402

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
failures = 0
BASE_URL = "http://localhost:8000"
TEST_EMAIL = "verify-p8-user@example.com"
INVITE_EMAIL = "verify-p8-invite@example.com"


def check(label: str, cond: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if cond else FAIL} {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures += 1


async def _demo_tenant_and_token(db):
    tenant = (await db.execute(select(Tenant).where(Tenant.slug == "demo-tenant"))).scalars().first()
    if tenant is None:
        raise SystemExit("No demo tenant found — run scripts/seed.py first.")
    user = (await db.execute(select(User).where(User.tenant_id == tenant.id))).scalars().first()
    token = create_token(user_id=user.id, tenant_id=tenant.id, role=user.role, token_type="access")
    return tenant, token


async def _snapshot_billing(db, tenant_id: uuid.UUID) -> dict:
    tenant = (await db.execute(select(Tenant).where(Tenant.id == tenant_id))).scalar_one()
    sub = (await db.execute(select(Subscription).where(
        Subscription.tenant_id == tenant_id))).scalar_one_or_none()
    return {"plan": tenant.plan, "seats_purchased": tenant.seats_purchased,
           "had_subscription": sub is not None}


async def _restore_billing(db, tenant_id: uuid.UUID, snapshot: dict) -> None:
    tenant = (await db.execute(select(Tenant).where(Tenant.id == tenant_id))).scalar_one()
    tenant.plan = snapshot["plan"]
    tenant.seats_purchased = snapshot["seats_purchased"]
    await db.execute(delete(Invoice).where(Invoice.tenant_id == tenant_id))
    if not snapshot["had_subscription"]:
        await db.execute(delete(Subscription).where(Subscription.tenant_id == tenant_id))
    await db.flush()


async def _cleanup_fixtures(db, tenant_id: uuid.UUID) -> None:
    await db.execute(delete(User).where(User.tenant_id == tenant_id, User.email == TEST_EMAIL))
    from app.db.models import TeamInvite
    await db.execute(delete(TeamInvite).where(
        TeamInvite.tenant_id == tenant_id, TeamInvite.email == INVITE_EMAIL))
    integ = (await db.execute(select(Integration).where(
        Integration.tenant_id == tenant_id, Integration.provider_name == "shopify"
        ))).scalar_one_or_none()
    if integ is not None:
        linked = (await db.execute(select(ToolConnection).where(
            ToolConnection.integration_id == integ.id))).scalars().all()
        for conn in linked:
            conn.integration_id = None
        await db.delete(integ)
    await db.flush()


async def main() -> int:
    async with get_session_factory()() as db:
        tenant, token = await _demo_tenant_and_token(db)
        tenant_id = tenant.id
        await _cleanup_fixtures(db, tenant_id)
        await db.commit()
        billing_snapshot = await _snapshot_billing(db, tenant_id)

    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30, headers=headers) as client:
        # ---------------- Team ----------------
        r_team_before = await client.get("/team")
        check("GET /team 200s", r_team_before.status_code == 200)
        active_before = r_team_before.json()["active_count"]

        r_user = await client.post("/users", json={
            "email": TEST_EMAIL, "password": "TestPassword123!",
            "full_name": "Verify P8 User", "role": "agent"})
        check("POST /users (fixture) returns 201", r_user.status_code == 201, f"got {r_user.text}")
        new_user_id = r_user.json()["id"]

        r_invite = await client.post("/team/invites", json={"email": INVITE_EMAIL, "role": "agent"})
        check("POST /team/invites (fixture) returns 201", r_invite.status_code == 201)

        r_team_after = await client.get("/team")
        team = r_team_after.json()
        check("GET /team active_count grew by 1 after adding a user",
             team["active_count"] == active_before + 1,
             f"before={active_before} after={team['active_count']}")
        check("GET /team includes the pending invite with status='invited'",
             any(m["email"] == INVITE_EMAIL and m["status"] == "invited" for m in team["members"]),
             f"members={team['members']}")
        check("GET /team includes the new active user with status='active'",
             any(m["email"] == TEST_EMAIL and m["status"] == "active" for m in team["members"]))

        r_role = await client.patch(f"/team/{new_user_id}/role", json={"role": "supervisor"})
        check("PATCH /team/{id}/role returns 200 with the new role",
             r_role.status_code == 200 and r_role.json()["role"] == "supervisor",
             f"got {r_role.status_code}: {r_role.text}")

        # ---------------- Integrations ----------------
        r_integ_list_before = await client.get("/integrations")
        check("GET /integrations 200s", r_integ_list_before.status_code == 200)

        r_connect = await client.post("/integrations", json={
            "provider_name": "shopify", "credentials": {"api_key": "verify-test-key"}})
        check("POST /integrations returns 201 with status='connected'",
             r_connect.status_code == 201 and r_connect.json()["status"] == "connected",
             f"got {r_connect.status_code}: {r_connect.text}")
        integration_id = r_connect.json()["id"]

        async with get_session_factory()() as db:
            catalog = (await db.execute(select(ToolCatalogEntry).where(
                ToolCatalogEntry.tenant_id == tenant_id,
                ToolCatalogEntry.provider_name == "shopify"))).scalars().first()
            conn = (await db.execute(select(ToolConnection).where(
                ToolConnection.tenant_id == tenant_id,
                ToolConnection.tool_catalog_id == catalog.id))).scalars().first() if catalog else None
        check("connecting the org-level integration links the existing shopify tool_connection",
             conn is not None and str(conn.integration_id) == integration_id,
             f"conn.integration_id={conn.integration_id if conn else None}")

        r_integ_list_after = await client.get("/integrations")
        check("GET /integrations lists the newly-connected provider",
             any(i["provider_name"] == "shopify" and i["status"] == "connected"
                 for i in r_integ_list_after.json()))

        r_disconnect = await client.delete(f"/integrations/{integration_id}")
        check("DELETE /integrations/{id} returns 200 with status='disconnected'",
             r_disconnect.status_code == 200 and r_disconnect.json()["status"] == "disconnected",
             f"got {r_disconnect.status_code}: {r_disconnect.text}")

        # ---------------- Billing ----------------
        r_sub_before = await client.get("/billing/subscription")
        check("GET /billing/subscription 200s", r_sub_before.status_code == 200)
        sub_before = r_sub_before.json()
        check("seats_used reflects real active users (DoD: seat count matches active users)",
             sub_before["seats_used"] == team["active_count"],
             f"seats_used={sub_before['seats_used']} active_count={team['active_count']}")

        r_checkout = await client.post("/billing/checkout-session", json={
            "plan": "pro", "seats": sub_before["seats_used"] + 3, "billing_cycle": "annual"})
        check("POST /billing/checkout-session returns 201, activated=True (stub provider)",
             r_checkout.status_code == 201 and r_checkout.json().get("activated") is True,
             f"got {r_checkout.status_code}: {r_checkout.text}")
        checkout_url = r_checkout.json().get("checkout_url", "")
        check("checkout_url has exactly one '?' (query string not double-appended)",
             checkout_url.count("?") == 1, f"checkout_url={checkout_url}")

        r_sub_after = await client.get("/billing/subscription")
        sub_after = r_sub_after.json()
        check("GET /billing/subscription — DoD: nav chip reads live, reflects the checkout",
             sub_after["plan"] == "pro" and sub_after["seats_purchased"] == sub_before["seats_used"] + 3
             and sub_after["status"] == "active" and sub_after["billing_cycle"] == "annual",
             f"sub_after={sub_after}")

        r_invoices = await client.get("/billing/invoices")
        check("GET /billing/invoices shows a real invoice from the checkout",
             r_invoices.json()["total"] >= 1 and r_invoices.json()["items"][0]["status"] == "paid",
             f"invoices={r_invoices.json()}")

    async with get_session_factory()() as db:
        await _cleanup_fixtures(db, tenant_id)
        await _restore_billing(db, tenant_id, billing_snapshot)
        await db.commit()

    print()
    if failures:
        print(f"{FAIL} {failures} check(s) failed")
        return 1
    print(f"{PASS} all P8 settings Definition-of-Done checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
