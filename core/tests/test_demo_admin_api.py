"""Demo admin API (app/api/demo_admin.py) end to end against a scratch database:
access control, industries, workbook import (preview → apply → re-import), facts,
contacts, knowledge documents, dummy data, and per-record tenant isolation.

Embedding is replaced by a stub (no OpenAI calls): queued documents are marked ready.

    TEST_ADMIN_DATABASE_URL=postgresql://aivoiceagent:aivoiceagent@127.0.0.1:5437/postgres \
        python -m pytest tests/test_demo_admin_api.py -q
"""
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

ADMIN_URL = os.environ.get("TEST_ADMIN_DATABASE_URL")
pytestmark = pytest.mark.skipif(not ADMIN_URL, reason="TEST_ADMIN_DATABASE_URL not set")

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))

STAFF_TENANT = uuid.UUID("11111111-1111-4111-8111-111111111111")
CUSTOMER_TENANT = uuid.UUID("22222222-2222-4222-8222-222222222222")


@pytest.fixture(scope="module")
def env():
    """Scratch DB at head, staff + customer tenants, seeds synced, app wired to it."""
    import psycopg2
    name = f"demo_api_{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(ADMIN_URL)
    admin.autocommit = True
    admin.cursor().execute(f"CREATE DATABASE {name}")
    base = ADMIN_URL.rsplit("/", 1)[0]
    pg_url = f"{base}/{name}"
    r = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=CORE,
                       env=dict(os.environ, DATABASE_URL_SYNC=pg_url.replace(
                           "postgresql://", "postgresql+psycopg2://", 1)),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with psycopg2.connect(pg_url) as conn, conn.cursor() as cur:
        for tid, slug in ((STAFF_TENANT, "phonn"), (CUSTOMER_TENANT, "some-customer")):
            cur.execute("INSERT INTO tenants (id, name, slug, timezone, settings, status, plan, "
                        "seats_purchased) VALUES (%s, %s, %s, 'Asia/Kolkata', '{}', 'active', "
                        "'starter', 1)", (str(tid), slug, slug))

    os.environ.update({"DEMO_ENABLED": "true", "DEMO_ADMIN_TENANT_IDS": str(STAFF_TENANT),
                       "DATABASE_URL": pg_url.replace("postgresql://", "postgresql+asyncpg://", 1)})
    from app.config import get_settings
    get_settings.cache_clear()
    try:
        yield pg_url
    finally:
        for k in ("DEMO_ENABLED", "DEMO_ADMIN_TENANT_IDS", "DATABASE_URL"):
            os.environ.pop(k, None)
        get_settings.cache_clear()
        cur = admin.cursor()
        cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s", (name,))
        cur.execute(f"DROP DATABASE IF EXISTS {name}")
        admin.close()


@pytest.fixture(scope="module")
def client(env):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from app.api.demo_admin import router
    from app.db.base import get_db
    from app.demo import records as svc
    from app.demo.industries.registry import sync_seed_industries
    from app.security.deps import Principal, get_principal

    engine = create_async_engine(env.replace("postgresql://", "postgresql+asyncpg://", 1),
                                 poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def db_override():
        async with factory() as s:
            yield s

    state = {"principal": Principal(tenant_id=STAFF_TENANT, kind="user", user_id=None,
                                    role="admin", permissions=["*"], label="ops@phonn.ai"),
             "jobs": []}

    async def fake_ingest(jobs):
        """Stand-in for embedding: mark each queued document ready."""
        from app.db.models import KnowledgeDocument
        async with factory() as s:
            for j in jobs:
                state["jobs"].append(j)
                doc = await s.get(KnowledgeDocument, j.document_id)
                doc.status = "ready"
            await s.commit()

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_db] = db_override
    app.dependency_overrides[get_principal] = lambda: state["principal"]
    svc.run_ingest_jobs = fake_ingest
    svc._DEF_CACHE.clear()

    with TestClient(app) as c:
        c.portal.call(_sync, factory, sync_seed_industries)
        c.state = state
        c.factory = factory
        yield c


async def _sync(factory, sync):
    async with factory() as s:
        await sync(s)
        await s.commit()


def as_principal(client, **kw):
    from app.security.deps import Principal
    base = dict(tenant_id=STAFF_TENANT, kind="user", user_id=None, role="admin",
                permissions=["*"], label="ops@phonn.ai")
    client.state["principal"] = Principal(**{**base, **kw})


def sql(env, query, *args):
    import psycopg2
    with psycopg2.connect(env) as conn, conn.cursor() as cur:
        cur.execute(query, args)
        return cur.fetchall() if cur.description else None


def workbook_bytes(rows):
    from tests.test_demo_importer import workbook
    return workbook(rows)


def re_rows():
    from tests.test_demo_importer import READY, row
    second = row(**{"#": 2, "Developer": "Bay View Realty", "City": "Chennai",
                    "Buyer name": "Meena Das, Karan Shah", "Buyer role": "Head sales, Manager",
                    "Email": "meena@bayview.in, karan.shah@bayview.in",
                    "Project name": "Bay View Towers", "Location": "OMR, Chennai",
                    "Configurations": "3 BHK", "Price": "₹1.2 Cr onwards"})
    return [READY, second]


def preview(client, rows=None):
    r = client.post("/admin/demo/imports", data={"industry_key": "real_estate"},
                    files={"file": ("prospects.xlsx", workbook_bytes(rows or re_rows()),
                                     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert r.status_code == 201, r.text
    return r.json()


# ── access ───────────────────────────────────────────────────────────────────────────────

def test_customer_admin_is_refused(client):
    as_principal(client, tenant_id=CUSTOMER_TENANT)          # an admin, but of a customer
    assert client.get("/admin/demo/industries").status_code == 403
    as_principal(client, permissions=["knowledge:read"])     # our staff, but not an admin
    assert client.get("/admin/demo/industries").status_code == 403
    as_principal(client)
    assert client.get("/admin/demo/industries").status_code == 200


def test_everything_is_404_while_demo_disabled(client, env):
    from app.config import get_settings
    os.environ["DEMO_ENABLED"] = "false"
    get_settings.cache_clear()
    try:
        assert client.get("/admin/demo/industries").status_code == 404
    finally:
        os.environ["DEMO_ENABLED"] = "true"
        get_settings.cache_clear()


# ── industries ───────────────────────────────────────────────────────────────────────────

def test_industry_listed_and_ready(client):
    items = {i["key"]: i for i in client.get("/admin/demo/industries").json()}
    assert items["real_estate"]["can_go_live"] is True
    assert items["real_estate"]["current_version"] == 1


def test_validate_reports_every_problem(client):
    r = client.post("/admin/demo/industries/validate", data={"text": "key: Bad Key\nlabel: x\n"})
    body = r.json()
    assert body["valid"] is False
    assert {"(root): 'facts_fields' is a required property", "(root): 'version' is a required property",
            "key: 'Bad Key' does not match '^[a-z][a-z0-9_]{1,39}$'"} <= set(body["errors"])


def test_publish_new_version_and_refuse_identical(client):
    import yaml
    raw = client.get("/admin/demo/industries/real_estate").json()["definition"]
    raw["label"] = "Real estate (India)"
    r = client.post("/admin/demo/industries", data={"text": yaml.safe_dump(raw), "activate": "false"})
    assert r.status_code == 201 and r.json()["version"] == 2
    r = client.post("/admin/demo/industries", data={"text": yaml.safe_dump(raw), "activate": "false"})
    assert r.status_code == 409
    # Not activated: new records keep using version 1.
    assert client.get("/admin/demo/industries/real_estate").json()["current_version"] == 1


def test_facts_template_download(client):
    import io

    import openpyxl
    r = client.get("/admin/demo/industries/real_estate/facts-template.xlsx")
    assert r.status_code == 200
    ws = openpyxl.load_workbook(io.BytesIO(r.content)).active
    headers = [c.value for c in ws[1]]
    assert ws.title == "Real estate" and "Project name" in headers and "Buyer name" in headers


# ── imports → records ────────────────────────────────────────────────────────────────────

def test_import_preview_apply_and_reimport(client, env):
    pv = preview(client)
    assert pv["status"] == "previewed"
    assert pv["report"]["summary"]["creates"] == 2
    assert sql(env, "SELECT count(*) FROM demo_prospects")[0][0] == 0      # preview writes nothing

    r = client.post(f"/admin/demo/imports/{pv['id']}/apply", json={})
    assert r.status_code == 200, r.text
    res = r.json()
    assert (res["created"], res["updated"], res["contacts_added"]) == (2, 0, 4)
    assert res["documents_queued"] == 2
    assert client.post(f"/admin/demo/imports/{pv['id']}/apply", json={}).status_code == 409

    # Each record has its own demo tenant and knowledge base.
    tenants = sql(env, "SELECT t.kind, p.knowledge_base_id IS NOT NULL FROM demo_prospects p "
                       "JOIN tenants t ON t.id = p.tenant_id")
    assert tenants == [("demo", True), ("demo", True)]

    # The same workbook again: updates, nothing changed, nothing re-embedded.
    pv2 = preview(client)
    assert pv2["report"]["summary"]["updates"] == 2
    assert all(rec["effect"]["changed_facts"] == [] for rec in pv2["report"]["records"])
    jobs_before = len(client.state["jobs"])
    res2 = client.post(f"/admin/demo/imports/{pv2['id']}/apply", json={}).json()
    assert (res2["created"], res2["updated"], res2["documents_queued"]) == (0, 2, 0)
    assert len(client.state["jobs"]) == jobs_before


def _record_id(client, company):
    items = client.get("/admin/demo/records", params={"q": company}).json()["items"]
    return next(i["id"] for i in items if i["company_name"] == company)


def test_record_detail_minimum_and_generated_document(client):
    acme = client.get(f"/admin/demo/records/{_record_id(client, 'Acme Homes')}").json()
    assert acme["minimum_met"] is True and acme["record_name"] == "Acme Green Acres"
    assert acme["facts"]["configurations"]["items"] == ["2 BHK", "3 BHK"]
    assert [d["generated"] for d in acme["documents"]] == [True]
    assert acme["documents"][0]["status"] == "ready"
    assert acme["dummy_data"]["source"] == "template"
    bay = client.get(f"/admin/demo/records/{_record_id(client, 'Bay View Realty')}").json()
    assert bay["minimum_met"] is False
    assert "Sales office is missing" in bay["minimum_missing"]


def test_minimum_filter(client):
    below = client.get("/admin/demo/records", params={"minimum_met": "false"}).json()["items"]
    assert [i["company_name"] for i in below] == ["Bay View Realty"]


def test_fact_edit_rerenders_document(client, env):
    rid = _record_id(client, "Bay View Realty")
    r = client.patch(f"/admin/demo/records/{rid}/facts", json={"facts": {
        "sales_office": {"value": "Anna Salai, Chennai"},
        "configurations": {"value": "3 & 4 BHK"},
        "price": {"value": None}}})
    assert r.status_code == 200, r.text
    assert r.json()["minimum_met"] is True
    assert r.json()["facts"]["price"]["value"] is None
    assert client.patch(f"/admin/demo/records/{rid}/facts",
                        json={"facts": {"nope": {"value": "x"}}}).status_code == 422
    text = client.state["jobs"][-1].text
    assert "- Price: not published; the sales team will confirm." in text
    assert "- Sales office: Anna Salai, Chennai" in text


def test_admin_contact_review_survives_reimport(client):
    rid = _record_id(client, "Bay View Realty")
    contacts = {c["full_name"]: c for c in client.get(f"/admin/demo/records/{rid}/contacts").json()}
    karan = contacts["Karan Shah"]
    r = client.patch(f"/admin/demo/contacts/{karan['id']}",
                     json={"email": "karan@bayview.co.in", "reviewed": True})
    assert r.status_code == 200 and r.json()["needs_review"] is False
    pv = preview(client)
    client.post(f"/admin/demo/imports/{pv['id']}/apply", json={})
    again = {c["full_name"]: c for c in client.get(f"/admin/demo/records/{rid}/contacts").json()}
    assert again["Karan Shah"]["email"] == "karan@bayview.co.in"
    assert again["Karan Shah"]["needs_review"] is False


def test_contact_rules(client):
    rid = _record_id(client, "Acme Homes")
    assert client.post(f"/admin/demo/records/{rid}/contacts",
                       json={"full_name": "asha rao"}).status_code == 409          # exists
    r = client.post(f"/admin/demo/records/{rid}/contacts",
                    json={"full_name": "neha joshi", "linkedin_url": "linkedin.com/in/neha-j"})
    assert r.status_code == 201
    c = r.json()
    assert (c["full_name"], c["needs_review"]) == ("Neha Joshi", True)
    assert c["linkedin_url"] == "https://www.linkedin.com/in/neha-j/"
    assert client.patch(f"/admin/demo/contacts/{c['id']}",
                        json={"reviewed": True}).status_code == 422                # no email
    assert client.patch(f"/admin/demo/contacts/{c['id']}",
                        json={"email": "asha.rao@acmehomes.in"}).status_code == 409  # taken
    assert client.patch(f"/admin/demo/contacts/{c['id']}",
                        json={"status": "excluded"}).json()["status"] == "excluded"


# ── knowledge documents ──────────────────────────────────────────────────────────────────

def test_documents(client):
    rid = _record_id(client, "Acme Homes")
    other = _record_id(client, "Bay View Realty")
    html = b"<html><head><script>x()</script></head><body><h1>Acme Green Acres</h1><p>Clubhouse and pool.</p></body></html>"
    r = client.post(f"/admin/demo/records/{rid}/documents", data={"doc_type": "brochure"},
                    files={"file": ("brochure.html", html, "text/html")})
    assert r.status_code == 201, r.text
    doc_id = r.json()["id"]
    assert "x()" not in client.state["jobs"][-1].text
    assert "Clubhouse and pool." in client.state["jobs"][-1].text
    dup = client.post(f"/admin/demo/records/{rid}/documents", data={"doc_type": "brochure"},
                      files={"file": ("copy.html", html, "text/html")})
    assert dup.status_code == 409
    zipped = client.post(f"/admin/demo/records/{rid}/documents",
                         files={"file": ("doc.docx", b"PK\x03\x04rest", "application/octet-stream")})
    assert zipped.status_code == 422
    docs = client.get(f"/admin/demo/records/{rid}/documents").json()
    generated = next(d for d in docs if d["generated"])
    assert client.delete(f"/admin/demo/records/{rid}/documents/{generated['id']}").status_code == 409
    # Another record can't reach this record's document.
    assert client.delete(f"/admin/demo/records/{other}/documents/{doc_id}").status_code == 404
    assert client.delete(f"/admin/demo/records/{rid}/documents/{doc_id}").status_code == 200


def test_records_have_separate_tenants_and_kbs(env):
    rows = sql(env, "SELECT tenant_id, knowledge_base_id FROM demo_prospects")
    assert len({t for t, _ in rows}) == len(rows) and len({k for _, k in rows}) == len(rows)
    leaked = sql(env, "SELECT count(*) FROM knowledge_documents d JOIN demo_prospects p "
                      "ON p.knowledge_base_id = d.knowledge_base_id WHERE d.tenant_id <> p.tenant_id")
    assert leaked[0][0] == 0


# ── dummy data ───────────────────────────────────────────────────────────────────────────

def test_dummy_data_lifecycle(client):
    rid = _record_id(client, "Acme Homes")
    base = f"/admin/demo/records/{rid}/dummy-data"
    current = client.get(base).json()
    assert current["source"] == "template" and current["errors"] == []
    data = current["data"]

    bad = {**data, "free_visit_slots": [{"day_offset": 1, "time": "2026-10-01"}]}
    r = client.put(base, json=bad)
    assert r.status_code == 422 and any("free_visit_slots" in e for e in r.json()["detail"]["errors"])

    mine = {**data, "booking_reference": "VIS-5001"}
    assert client.put(base, params={"dry_run": "true"}, json=mine).json() == {"valid": True, "saved": False}
    assert client.put(base, json=mine).json()["version"] == 1
    assert client.get(base).json()["source"] == "record"

    prev = client.get(f"{base}/preview", params={"at": "2026-09-30T15:00:00+05:30"}).json()
    assert [s["date"] for s in prev["data"]["free_visit_slots"]] == ["2026-10-01", "2026-10-01", "2026-10-02"]
    assert prev["data"]["lead_source"] == "Facebook lead form for Acme Green Acres"
    assert prev["data"]["booking_reference"] == "VIS-5001"

    assert client.delete(base).json() == {"source": "template", "cleared": True}
    assert client.get(base).json()["source"] == "template"
    assert client.post(f"{base}/rollback/1").json() == {"active_version": 1}
    assert client.get(base).json()["source"] == "record"
    assert client.post(f"{base}/rollback/9").status_code == 404


# ── demo agent ───────────────────────────────────────────────────────────────────────────

def test_agent_provisioned_and_chat_starts(client, env):
    rid = _record_id(client, "Acme Homes")
    # Imports and edits already provisioned it; asking again changes nothing.
    r = client.post(f"/admin/demo/records/{rid}/agent")
    assert r.status_code == 200 and r.json()["changed"] == []
    chat = client.post(f"/admin/demo/records/{rid}/chat")
    assert chat.status_code == 201
    body = chat.json()
    assert "enquiry desk. I see you enquired" in body["opening_line"]
    assert "sample data" not in body["opening_line"]
    assert body["caller"]["name"] == "Rahul"
    t = client.get(f"/admin/demo/records/{rid}/chat/{body['session_id']}").json()
    assert t["messages"] == [{"role": "agent", "content": body["opening_line"], "at": t["messages"][0]["at"]}]
    assert t["bookings"] == []
    # A chat of one record is not reachable through another record.
    other = _record_id(client, "Bay View Realty")
    assert client.get(f"/admin/demo/records/{other}/chat/{body['session_id']}").status_code == 404


# ── choosing rows on apply, erasing records ──────────────────────────────────────────────

def _two_new_rows():
    from tests.test_demo_importer import READY, row
    base = dict(zip(__import__("tests.test_demo_importer", fromlist=["HEADER"]).HEADER, READY))
    cedar = row(**{**base, "#": 3, "Developer": "Cedar Homes", "Buyer name": "Ira Sen",
                   "Email": "ira@cedarhomes.in", "Project name": "Cedar Park"})
    dune = row(**{**base, "#": 4, "Developer": "Dune Estates", "Buyer name": "Om Rao",
                  "Email": "om@dune.in", "Project name": "Dune Bay"})
    return [cedar, dune]


def test_apply_only_the_selected_rows(client):
    pv = preview(client, _two_new_rows())
    by_company = {r["company_name"]: r["row"] for r in pv["report"]["records"]}
    url = f"/admin/demo/imports/{pv['id']}/apply"
    assert client.post(url, json={"rows": []}).status_code == 422             # nothing selected
    assert client.post(url, json={"rows": [999]}).status_code == 422          # not in the preview
    r = client.post(url, json={"rows": [by_company["Cedar Homes"]]})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["created"] == 1 and res["selected"] == 1 and res["skipped_rows"] == [by_company["Dune Estates"]]
    names = {i["company_name"] for i in client.get("/admin/demo/records", params={"limit": 200}).json()["items"]}
    assert "Cedar Homes" in names and "Dune Estates" not in names


def test_erase_a_record_removes_its_whole_tenant(client, env):
    rid = _record_id(client, "Cedar Homes")
    tid = sql(env, "SELECT tenant_id FROM demo_prospects WHERE id=%s", rid)[0][0]
    assert sql(env, "SELECT count(*) FROM knowledge_documents WHERE tenant_id=%s", tid)[0][0] >= 1
    r = client.delete(f"/admin/demo/records/{rid}")
    assert r.status_code == 200 and r.json()["records"] == 1 and r.json()["tenants"] == 1
    assert client.get(f"/admin/demo/records/{rid}").status_code == 404
    assert sql(env, "SELECT count(*) FROM tenants WHERE id=%s", tid) == [(0,)]
    # Nothing of the tenant is left in any tenant-scoped table.
    tables = [t for (t,) in sql(env, "SELECT table_name FROM information_schema.columns WHERE "
                                     "column_name='tenant_id' AND table_schema='public'")]
    for t in tables:
        assert sql(env, f'SELECT count(*) FROM "{t}" WHERE tenant_id=%s', tid) == [(0,)], t
    assert sql(env, "SELECT count(*) FROM demo_contacts WHERE full_name='Ira Sen'") == [(0,)]
    # Other records are untouched.
    assert _record_id(client, "Acme Homes")


def test_erase_refuses_a_customer_tenant(client, env):
    from app.db.models import DemoProspect
    from app.demo.erase import erase_records
    from app.demo.records import Conflict
    customer = sql(env, "SELECT id FROM tenants WHERE kind='customer' LIMIT 1")[0][0]

    async def attempt():
        async with client.factory() as db:
            try:
                await erase_records(db, [DemoProspect(id=uuid.uuid4(), tenant_id=customer)])
            except Conflict as e:
                return e.message
            finally:
                await db.rollback()
    assert "not a demo tenant" in client.portal.call(attempt)
    assert sql(env, "SELECT count(*) FROM tenants WHERE id=%s", customer) == [(1,)]
