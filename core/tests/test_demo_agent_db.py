"""Demo agent (app/demo/agent.py, app/demo/scheduling.py) against a scratch database:
provisioning from facts + dummy data, idempotency, isolation from every non-demo tenant,
dummy-data slots through the REAL scheduling service (compute_slots / book), and call start.

No LLM here; the engine turn itself is exercised live (see docs §16.5).

    TEST_ADMIN_DATABASE_URL=postgresql://aivoiceagent:aivoiceagent@127.0.0.1:5437/postgres \
        python -m pytest tests/test_demo_agent_db.py -q
"""
import asyncio
import os
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ADMIN_URL = os.environ.get("TEST_ADMIN_DATABASE_URL")
pytestmark = pytest.mark.skipif(not ADMIN_URL, reason="TEST_ADMIN_DATABASE_URL not set")

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))
IST = ZoneInfo("Asia/Kolkata")
CUSTOMER_TENANT = uuid.UUID("33333333-3333-4333-8333-333333333333")


@pytest.fixture(scope="module")
def db_url():
    import psycopg2
    name = f"demo_agent_{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(ADMIN_URL)
    admin.autocommit = True
    admin.cursor().execute(f"CREATE DATABASE {name}")
    pg = f"{ADMIN_URL.rsplit('/', 1)[0]}/{name}"
    r = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=CORE,
                       env=dict(os.environ, DATABASE_URL_SYNC=pg.replace(
                           "postgresql://", "postgresql+psycopg2://", 1)), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with psycopg2.connect(pg) as conn, conn.cursor() as cur:
        # A real customer tenant with its own engine config, to prove demos never touch it.
        cur.execute("INSERT INTO tenants (id, name, slug, timezone, settings, status, plan, "
                    "seats_purchased) VALUES (%s, 'Real Customer', 'real-customer', 'Asia/Kolkata', "
                    "'{}', 'active', 'starter', 1)", (str(CUSTOMER_TENANT),))
        cur.execute("INSERT INTO locations (id, tenant_id, name, timezone, is_active, location_type, "
                    "created_at, updated_at) VALUES (gen_random_uuid(), %s, 'HQ', 'Asia/Kolkata', "
                    "true, '{}', now(), now())", (str(CUSTOMER_TENANT),))
    try:
        yield pg
    finally:
        cur = admin.cursor()
        cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s", (name,))
        cur.execute(f"DROP DATABASE IF EXISTS {name}")
        admin.close()


def run(db_url, fn):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    async def go():
        engine = create_async_engine(db_url.replace("postgresql://", "postgresql+asyncpg://", 1),
                                     poolclass=NullPool)
        try:
            async with async_sessionmaker(engine, expire_on_commit=False)() as db:
                out = await fn(db)
                await db.commit()
                return out
        finally:
            await engine.dispose()
    return asyncio.run(go())


def sql(db_url, q, *a):
    import psycopg2
    with psycopg2.connect(db_url) as conn, conn.cursor() as cur:
        cur.execute(q, a)
        return cur.fetchall() if cur.description else None


@pytest.fixture(scope="module")
def record(db_url):
    """One real-estate record imported through the normal service path."""
    from app.demo import records as svc
    from app.demo.industries.registry import sync_seed_industries
    from tests.test_demo_importer import READY, workbook
    svc._DEF_CACHE.clear()

    async def go(db):
        await sync_seed_industries(db)
        await db.flush()
        imp = await svc.preview_import(db, data=workbook([READY]), filename="t.xlsx",
                                       industry_key="real_estate", actor="test")
        result, _ = await svc.apply_import(db, imp.id, actor="test")
        assert result["agent_errors"] == []
        return uuid.UUID(result["records"][0]["record_id"])
    rid = run(db_url, go)
    return rid


def test_provisioned_inside_the_demo_tenant(db_url, record):
    tid = sql(db_url, "SELECT tenant_id FROM demo_prospects WHERE id=%s", str(record))[0][0]
    assert sql(db_url, "SELECT kind, name FROM tenants WHERE id=%s", tid) == [("demo", "Acme Homes")]
    assert sql(db_url, "SELECT p.slug FROM tenant_pack_config c JOIN industry_packs p ON p.id=c.pack_id "
                       "WHERE c.tenant_id=%s", tid) == [("real_estate_india",)]
    tools = {r[0] for r in sql(db_url, "SELECT name FROM tool_definitions WHERE tenant_id=%s "
                                       "AND is_enabled", tid)}
    assert {"search_slots", "book_appointment", "knowledge_lookup"} <= tools
    # Allowlist: no unit inventory (it made the agent deny listed configurations) and no
    # WhatsApp document sending from a demo.
    assert not tools & {"inventory_query", "send_document", "create_customer"}
    assert sql(db_url, "SELECT location_type->>'booking_reference' FROM locations WHERE tenant_id=%s",
               tid) == [("VIS-1187",)]
    assert sql(db_url, "SELECT name, address FROM locations WHERE tenant_id=%s", tid) == [
        ("Acme Green Acres", "MG Road, Bengaluru")]                   # sales office wins
    assert sql(db_url, "SELECT count(*) FROM staff_members WHERE tenant_id=%s", tid) == [(0,)]
    # Visits are never tied to a staff member, so the model can't pass the sample sales
    # contact's NAME as executive_id (a live run did; the booking failed validation).
    schemas = sql(db_url, "SELECT name, input_schema FROM tool_definitions WHERE tenant_id=%s "
                          "AND name IN ('search_slots','book_appointment','reschedule_appointment')", tid)
    assert len(schemas) == 3
    assert all("executive_id" not in s["properties"] for _, s in schemas)
    persona = sql(db_url, "SELECT template FROM prompt_templates WHERE tenant_id=%s AND purpose='persona' "
                          "AND is_active", tid)[0][0]
    assert "Acme Green Acres" in persona and "Acme Homes" in persona
    assert "This is a demo running on sample data" in persona
    assert "booking reference VIS-1187" in persona
    assert "The caller is Rahul, a sample lead." in persona
    assert "Facebook lead form for Acme Green Acres" in persona
    facts = sql(db_url, "SELECT settings->'project_facts' FROM tenants WHERE id=%s", tid)[0][0]
    by_label = {f["label"]: f["value"] for f in facts}
    assert by_label["RERA number"] == "not published; the sales team will confirm"
    assert by_label["Configurations"] == "2 & 3 BHK (unit types: 2 BHK, 3 BHK)"
    assert sql(db_url, "SELECT count(*) FROM agents WHERE tenant_id=%s AND status='live'", tid) == [(1,)]


def test_provisioning_is_idempotent(db_url, record):
    from app.demo import records as svc
    from app.demo.agent import provision_demo_agent

    async def go(db):
        return await provision_demo_agent(db, await svc.get_record(db, record))
    assert run(db_url, go)["changed"] == []


def test_dummy_data_change_rewrites_the_persona(db_url, record):
    from app.demo import records as svc
    from app.demo.agent import provision_demo_agent

    async def go(db):
        p = await svc.get_record(db, record)
        _, data, _, _ = await svc.effective_dataset(
            db, p, await svc.load_definition(db, p.industry_key, p.industry_version))
        errors, _ = await svc.save_record_dataset(db, p, {**data, "booking_reference": "VIS-4242"},
                                                  dry_run=False)
        assert errors == []
        return await provision_demo_agent(db, p)
    assert "persona" in run(db_url, go)["changed"]
    tid = sql(db_url, "SELECT tenant_id FROM demo_prospects WHERE id=%s", str(record))[0][0]
    rows = sql(db_url, "SELECT version, is_active, template LIKE '%%VIS-4242%%' FROM prompt_templates "
                       "WHERE tenant_id=%s AND purpose='persona' ORDER BY version", tid)
    assert rows == [(1, False, False), (2, True, True)]


def test_customer_tenant_and_shared_rows_untouched(db_url, record):
    assert sql(db_url, "SELECT count(*) FROM tool_definitions WHERE tenant_id=%s", str(CUSTOMER_TENANT)) == [(0,)]
    assert sql(db_url, "SELECT count(*) FROM prompt_templates WHERE tenant_id=%s", str(CUSTOMER_TENANT)) == [(0,)]
    assert sql(db_url, "SELECT count(*) FROM tenant_pack_config WHERE tenant_id=%s", str(CUSTOMER_TENANT)) == [(0,)]
    # Every demo-written engine row belongs to a demo tenant.
    for table in ("tool_definitions", "prompt_templates", "locations",
                  "tenant_pack_config", "agents"):
        stray = sql(db_url, f"SELECT count(*) FROM {table} x JOIN tenants t ON t.id=x.tenant_id "
                            f"WHERE t.kind='customer' AND t.id<>%s", str(CUSTOMER_TENANT))
        assert stray == [(0,)], table


def test_refuses_to_provision_a_customer_tenant(db_url, record):
    from app.demo import records as svc
    from app.demo.agent import provision_demo_agent

    async def go(db):
        p = await svc.get_record(db, record)
        real = p.tenant_id
        p.tenant_id = CUSTOMER_TENANT              # pretend a bug pointed it elsewhere
        try:
            with pytest.raises(svc.Conflict, match="not a demo tenant"):
                await provision_demo_agent(db, p)
        finally:
            p.tenant_id = real
    run(db_url, go)


def test_demo_slots_through_the_real_scheduling_service(db_url, record):
    from sqlalchemy import select

    from app.db.models import Customer, Location
    from app.demo import records as svc
    from app.demo.scheduling import demo_slot_windows
    from app.services import scheduling as sched

    tid = uuid.UUID(str(sql(db_url, "SELECT tenant_id FROM demo_prospects WHERE id=%s", str(record))[0][0]))

    async def fixed(db):
        now = datetime(2026, 9, 30, 15, 0, tzinfo=IST)
        return await demo_slot_windows(db, tid, now.date(), now.date() + timedelta(days=7), now=now)
    windows = run(db_url, fixed)
    assert [(s.isoformat(), (e - s).seconds // 60) for s, e in windows] == [
        ("2026-10-01T11:00:00+05:30", 60), ("2026-10-01T16:00:00+05:30", 60),
        ("2026-10-02T11:00:00+05:30", 60)]

    async def live(db):
        loc = (await db.execute(select(Location).where(Location.tenant_id == tid))).scalar_one()
        today = datetime.now(IST).date()
        slots = await sched.compute_slots(db, tid, loc.id, today, today + timedelta(days=7))
        # Two different sample leads can both book the same first slot: each demo call
        # starts from the same fresh calendar.
        booked = []
        for n in range(2):
            c = Customer(tenant_id=tid, full_name=f"Rahul {n}", phone=f"+9100000000{n}",
                         phone_normalized=f"+9100000000{n}")
            db.add(c)
            await db.flush()
            booked.append(await sched.book(db, tenant_id=tid, customer=c, centre_id=loc.id,
                                           starts_at=slots[0].starts_at, booked_via="agent"))
        return slots, booked, today
    slots, booked, today = run(db_url, live)
    assert [s.starts_at.date() for s in slots] == [today + timedelta(days=1)] * 2 + [today + timedelta(days=2)]
    assert [b.starts_at for b in booked] == [slots[0].starts_at] * 2


def test_customer_tenant_takes_the_normal_scheduling_path(db_url):
    from sqlalchemy import select

    from app.db.models import Location
    from app.demo.scheduling import demo_slot_windows, tenant_kind
    from app.services import scheduling as sched

    async def go(db):
        assert await tenant_kind(db, CUSTOMER_TENANT) == "customer"
        assert await demo_slot_windows(db, CUSTOMER_TENANT, datetime.now().date(),
                                       datetime.now().date()) is None
        loc = (await db.execute(select(Location).where(Location.tenant_id == CUSTOMER_TENANT))).scalar_one()
        # No working hours configured → the normal computation finds nothing (it would find
        # demo slots if the demo branch leaked).
        return await sched.compute_slots(db, CUSTOMER_TENANT, loc.id, datetime.now().date(),
                                         datetime.now().date() + timedelta(days=7))
    assert run(db_url, go) == []


def test_each_call_gets_a_fresh_sample_lead_and_the_demo_greeting(db_url, record):
    from app.demo import records as svc
    from app.demo.agent import start_demo_conversation

    async def go(db):
        p = await svc.get_record(db, record)
        a = await start_demo_conversation(db, p, channel="chat")
        b = await start_demo_conversation(db, p, channel="voice")
        return a, b
    (conv_a, cust_a, greet), (conv_b, cust_b, _) = run(db_url, go)
    assert cust_a.id != cust_b.id and cust_a.phone_normalized != cust_b.phone_normalized
    assert cust_a.full_name == "Rahul" and "demo_sample" in cust_a.tags
    assert cust_a.phone_normalized.startswith("+910")
    assert greet.startswith("Hello, this is the Acme Green Acres enquiry desk. I see you enquired")
    assert "sample data" not in greet
    assert "Acme Green Acres" in greet and "{{" not in greet
    msgs = sql(db_url, "SELECT role, content FROM messages WHERE conversation_id=%s", str(conv_a.id))
    assert msgs == [("agent", greet)]
