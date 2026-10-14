"""Company links, the public demo API and the demo voice handshake, end to end against a
scratch database (embedding stubbed, no LLM):

  links      one per company record, one row per person (own recipient code + UTM tags);
             token hashed for lookup and kept only encrypted; re-export gives the same URLs;
             regenerate replaces them; a company's link can be sent to another company's
             person
  tracking   who opened: recipient / shared (second browser on a code) / unknown / self
             identified; codes of other links ignored; UTM + email step captured; GET writes
             nothing
  public     page data in the exact shape the demo page renders; 404 unknown/revoked,
             410 expired; private headers; events
  limits     one call at a time per link, calls per day, daily spend budget
  voice      demo token: wrong origin, single use, tampering; call close-out
  admin      per-link activity (people, other visitors, calls, questions), questions search,
             sessions with who, the funnel per company and per person

    TEST_ADMIN_DATABASE_URL=postgresql://aivoiceagent:aivoiceagent@127.0.0.1:5437/postgres \\
    REDIS_URL=redis://127.0.0.1:6382/0 python -m pytest tests/test_demo_links_api.py -q
"""
import asyncio
import csv
import hashlib
import io
import os
import subprocess
import sys
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

ADMIN_URL = os.environ.get("TEST_ADMIN_DATABASE_URL")
pytestmark = pytest.mark.skipif(not ADMIN_URL, reason="TEST_ADMIN_DATABASE_URL not set")

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))
STAFF = uuid.UUID("44444444-4444-4444-8444-444444444444")
DEMO_ORIGIN = "http://localhost:5173"
TOKEN_KEY = "uR5bYcG2Hk0yq0Qm3m4aQ1dO8JtV3bq7wP9xXo2nZzQ="     # test-only Fernet key


@pytest.fixture(scope="module")
def env():
    import psycopg2
    name = f"demo_links_{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(ADMIN_URL)
    admin.autocommit = True
    admin.cursor().execute(f"CREATE DATABASE {name}")
    pg = f"{ADMIN_URL.rsplit('/', 1)[0]}/{name}"
    r = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=CORE,
                       env=dict(os.environ, DATABASE_URL_SYNC=pg.replace("postgresql://", "postgresql+psycopg2://", 1)),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    with psycopg2.connect(pg) as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO tenants (id, name, slug, timezone, settings, status, plan, seats_purchased) "
                    "VALUES (%s, 'Phonn', 'phonn', 'Asia/Kolkata', '{}', 'active', 'starter', 1)", (str(STAFF),))
    overrides = {"DEMO_ENABLED": "true", "DEMO_ADMIN_TENANT_IDS": str(STAFF),
                 "DATABASE_URL": pg.replace("postgresql://", "postgresql+asyncpg://", 1),
                 "DEMO_PUBLIC_BASE_URL": DEMO_ORIGIN, "DEMO_ALLOWED_ORIGINS": DEMO_ORIGIN,
                 "REDIS_URL": os.environ.get("REDIS_URL", "redis://127.0.0.1:6382/0"),
                 "DEMO_DAILY_BUDGET_USD": "0", "DEMO_LINK_TOKEN_KEYS": TOKEN_KEY,
                 "DEMO_UTM_ENABLED": "true", "DEMO_UTM_SOURCE": "email", "DEMO_UTM_MEDIUM": "cold_email",
                 "DEMO_UTM_CAMPAIGN": "", "DEMO_EXPORT_STEPS": "e1,e2,e3,e4,li",
                 "DEMO_IDENTIFY_PROMPT": "unknown_only",
                 # Every request here comes from one test IP; per-IP limits get their
                 # own test below.
                 "DEMO_SESSIONS_PER_IP_PER_HOUR": "100", "DEMO_RATE_LIMIT_PER_MIN": "1000"}
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    _flush_redis()
    from app.config import get_settings
    get_settings.cache_clear()
    try:
        yield pg
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        get_settings.cache_clear()
        cur = admin.cursor()
        cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s", (name,))
        cur.execute(f"DROP DATABASE IF EXISTS {name}")
        admin.close()


@pytest.fixture(scope="module")
def client(env):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from app.api.demo_admin import router as admin_router
    from app.api.demo_public import router as public_router
    from app.db.base import get_db
    from app.demo import limits
    from app.demo import records as svc
    from app.demo.industries.registry import sync_seed_industries
    from app.security.deps import Principal, get_principal
    from tests.test_demo_importer import READY, row, workbook

    engine = create_async_engine(env.replace("postgresql://", "postgresql+asyncpg://", 1), poolclass=NullPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    async def db_override():
        async with factory() as s:
            yield s

    async def no_ingest(jobs):
        return None

    limits._redis = None                     # pick up REDIS_URL for this module
    limits._BUDGET_CACHE.clear()
    svc.run_ingest_jobs = no_ingest
    svc._DEF_CACHE.clear()
    app = FastAPI()
    app.include_router(admin_router)
    app.include_router(public_router)
    app.dependency_overrides[get_db] = db_override
    app.dependency_overrides[get_principal] = lambda: Principal(
        tenant_id=STAFF, kind="user", user_id=None, role="admin", permissions=["*"], label="ops")

    bay = row(**{"#": 2, "Developer": "Bay View Realty", "City": "Chennai",
                 "Buyer name": "Meena Das, Karan Shah", "Buyer role": "Head sales, Manager",
                 "Email": "meena@bayview.in, karan.shah@bayview.in", "Project name": "Bay View Towers",
                 "Location": "OMR, Chennai", "Configurations": "2 & 3 BHK", "Sales office": "Anna Salai"})
    with TestClient(app) as c:
        async def seed():
            async with factory() as s:
                await sync_seed_industries(s)
                await s.commit()
        c.portal.call(seed)
        pv = c.post("/admin/demo/imports", data={"industry_key": "real_estate"},
                    files={"file": ("p.xlsx", workbook([READY, bay]), "application/octet-stream")}).json()
        assert c.post(f"/admin/demo/imports/{pv['id']}/apply", json={}).status_code == 200
        c.factory = factory
        yield c
    asyncio.run(engine.dispose())


def _flush_redis():
    """Start from clean per-IP counters and single-use markers."""
    import redis
    r = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://127.0.0.1:6382/0"))
    for pattern in ("demo:rl:*", "demo:once:*"):
        for k in r.scan_iter(pattern):
            r.delete(k)


def sql(env, q, *a):
    import psycopg2
    with psycopg2.connect(env) as conn, conn.cursor() as cur:
        cur.execute(q, a)
        return cur.fetchall() if cur.description else None


def token_of(url: str) -> str:
    return urlsplit(url).path.rsplit("/", 1)[-1]


def query_of(url: str) -> dict:
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


def record_id(client, q: str) -> str:
    return client.get("/admin/demo/records", params={"q": q}).json()["items"][0]["id"]


@pytest.fixture(scope="module")
def generated(client):
    r = client.post("/admin/demo/links/generate", json={})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def rows(generated):
    return {row["full_name"]: row for row in generated["rows"]}


# ── links ────────────────────────────────────────────────────────────────────────────────

def test_one_link_per_company_one_row_per_person(client, env, generated, rows):
    assert {x["company"] for x in generated["links"]} == {"Acme Homes", "Bay View Realty"}
    assert all(x["created"] for x in generated["links"])
    assert set(rows) == {"Asha Rao", "Vikram Iyer", "Meena Das", "Karan Shah"}
    # Colleagues share the company's token; each has their own code.
    assert token_of(rows["Asha Rao"]["link"]) == token_of(rows["Vikram Iyer"]["link"])
    assert token_of(rows["Asha Rao"]["link"]) != token_of(rows["Meena Das"]["link"])
    refs = [query_of(r["link"])["r"] for r in rows.values()]
    assert len(set(refs)) == 4 and all(len(x) >= 8 for x in refs)
    q = query_of(rows["Asha Rao"]["link"])
    assert q == {"r": rows["Asha Rao"]["ref_code"], "utm_source": "email", "utm_medium": "cold_email",
                 "utm_campaign": "demo_real_estate", "utm_content": "e1"}
    assert query_of(rows["Asha Rao"]["link_li"])["utm_source"] == "linkedin"
    assert "r" not in query_of(rows["Asha Rao"]["company_link"])
    # No email address anywhere in a URL.
    assert all("@" not in u for r in rows.values() for k, u in r.items() if k.startswith("link"))


def test_token_hashed_and_only_encrypted_at_rest(env, rows):
    raw = token_of(rows["Asha Rao"]["link"])
    stored = sql(env, "SELECT token_hash, token_prefix, token_ciphertext FROM demo_links WHERE id=%s",
                 rows["Asha Rao"]["link_id"])[0]
    assert stored[:2] == (hashlib.sha256(raw.encode()).hexdigest(), raw[:6])
    assert stored[2] and raw not in stored[2]
    assert sql(env, "SELECT count(*) FROM demo_links WHERE token_ciphertext LIKE %s", f"%{raw}%") == [(0,)]


def test_people_are_marked_emailed(env, rows):
    assert sql(env, "SELECT count(*) FROM demo_link_recipients WHERE last_exported_at IS NOT NULL") == [(4,)]


def test_second_run_gives_the_same_urls(client, rows):
    again = client.post("/admin/demo/links/generate", json={}).json()
    assert not any(x["created"] for x in again["links"])
    assert {r["full_name"]: r["link"] for r in again["rows"]} == {k: v["link"] for k, v in rows.items()}


def test_listing_never_shows_tokens(client):
    listed = client.get("/admin/demo/links").json()
    assert len(listed) == 2
    assert all("url" not in x and "token_hash" not in x and "token_ciphertext" not in x for x in listed)
    acme = next(x for x in listed if x["record"]["company_name"] == "Acme Homes")
    assert acme["exportable"] and acme["stats"]["people"] == 2 and acme["stats"]["people_emailed"] == 2


def test_urls_endpoint_rebuilds_the_same_links(client, env, rows):
    u = client.get(f"/admin/demo/links/{rows['Asha Rao']['link_id']}/urls").json()
    by_name = {x["contact"]["name"]: x for x in u["recipients"]}
    assert by_name["Asha Rao"]["urls"]["e1"] == rows["Asha Rao"]["link"]
    assert u["company_url"] == rows["Asha Rao"]["company_link"]
    assert sql(env, "SELECT count(*) FROM audit_events WHERE action='demo.link_reveal'")[0][0] >= 1


def test_send_a_companys_link_to_another_companys_person(client, rows):
    acme_link = rows["Asha Rao"]["link_id"]
    r = client.post(f"/admin/demo/links/{acme_link}/recipients", json={"contact_id": rows["Karan Shah"]["contact_id"]})
    assert r.status_code == 201, r.text
    body = r.json()
    assert token_of(body["urls"]["e1"]) == token_of(rows["Asha Rao"]["link"])
    assert body["ref_code"] != rows["Karan Shah"]["ref_code"]       # his own code on Acme's link
    again = client.post(f"/admin/demo/links/{acme_link}/recipients", json={"contact_id": rows["Karan Shah"]["contact_id"]})
    assert again.json()["ref_code"] == body["ref_code"]              # idempotent
    csv_rows = client.post("/admin/demo/links/generate", json={"record_ids": [rows["Asha Rao"]["record_id"]]}).json()["rows"]
    karan = next(x for x in csv_rows if x["full_name"] == "Karan Shah")
    assert karan["other_company"] == "yes" and karan["company"] == "Acme Homes"


def test_record_link_endpoint_reuses_unless_replaced(client, rows):
    rid = rows["Meena Das"]["record_id"]
    same = client.post(f"/admin/demo/records/{rid}/link", json={}).json()
    assert same["created"] is False and same["id"] == rows["Meena Das"]["link_id"]
    assert {x["contact"]["name"] for x in same["recipients"]} == {"Meena Das", "Karan Shah"}


def test_regenerate_csv_replaces_links(client, rows):
    r = client.post("/admin/demo/links/generate", params={"format": "csv"},
                    json={"regenerate": True, "industry_key": "real_estate"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    out = list(csv.DictReader(io.StringIO(r.content.decode("utf-8-sig"))))
    assert len(out) == 5                    # 4 people + Karan on Acme's link too
    assert {"link", "link_e1", "link_e2", "link_li", "company_link", "ref_code", "prompt_1"} <= set(out[0])
    assert all(x["link"] == x["link_e1"] for x in out)
    assert out[0]["prompt_1"] == "3BHK ka price kya hai?"
    asha = next(x for x in out if x["full_name"] == "Asha Rao")
    assert token_of(asha["link"]) != token_of(rows["Asha Rao"]["link"])
    assert client.get(f"/public/demo/{token_of(rows['Asha Rao']['link'])}").status_code == 404   # old URL dead


# ── public page and tracking ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def acme(client):
    """Acme's current link (after the regenerate above) and its people's codes."""
    u = client.post(f"/admin/demo/records/{record_id(client, 'Acme')}/link", json={}).json()
    refs = {x["contact"]["name"]: x["ref_code"] for x in u["recipients"]}
    bay = client.post(f"/admin/demo/records/{record_id(client, 'Bay View')}/link", json={}).json()
    return {"token": token_of(u["company_url"]), "link_id": u["id"], "refs": refs,
            "bay_ref": bay["recipients"][0]["ref_code"]}


def test_page_data_shape(client, env, acme):
    events_before = sql(env, "SELECT count(*) FROM demo_events")[0][0]
    r = client.get(f"/public/demo/{acme['token']}")
    assert r.status_code == 200
    assert r.headers["x-robots-tag"].startswith("noindex") and r.headers["cache-control"] == "no-store"
    p = r.json()
    assert p["header"] == "Acme Homes × Phonn: a demo agent"
    assert p["record_name"] == "Acme Green Acres" and p["industry_label"] == "Real estate"
    assert len(p["prompts"]) == 3 and p["max_call_seconds"] == 300 and p["max_calls_per_day"] == 10
    titles = [s["title"] for s in p["sample"]["sections"]]
    assert "Free site-visit slots" in titles and "The caller (you)" in titles
    slots = next(s for s in p["sample"]["sections"] if s["title"] == "Free site-visit slots")["items"]
    assert len(slots) == 3 and " at 11 am" in slots[0]["value"]
    assert {"label": "You are", "value": "Rahul"} in p["sample"]["summary"]
    # Each section says how to show it; the visitor's own person is marked.
    kinds = {s["title"]: (s["kind"], s.get("you")) for s in p["sample"]["sections"]}
    assert kinds["The caller (you)"] == ("person", True) and kinds["Sales contact"] == ("person", False)
    assert kinds["Free site-visit slots"][0] == "slots" and kinds["What the agent will ask"][0] == "list"
    details = {i["label"]: i["value"] for i in next(
        s for s in p["sample"]["sections"] if s["title"] == "Details")["items"]}
    assert details["Enquired"] == "10 minutes before the call"
    assert len(next(i for i in p["sample"]["summary"] if i["label"] == "Free site-visit slots")["values"]) == 3
    assert p["scenario"] == {"you_play": "A home buyer who saw the project ad",
                             "goal": "A site visit booked in one of the sample slots, reference read back"}
    facts = {f["label"]: f for f in p["knowledge"]["facts"]}
    assert facts["RERA number"]["value"] is None
    assert facts["Price"]["source_url"] == "https://acmehomes.in/price"
    assert facts["Price"]["checked_on"] == "2026-10-01"
    # Nothing internal or personal leaks.
    flat = r.text
    assert "tenant" not in flat and "prospect_id" not in flat and "token_hash" not in flat
    assert "Asha" not in flat and "asha@" not in flat
    # GET writes nothing (mail scanners fetch links).
    assert sql(env, "SELECT count(*) FROM demo_events")[0][0] == events_before
    assert sql(env, "SELECT count(*) FROM demo_visitors")[0][0] == 0


def test_unknown_revoked_and_expired(client, env, acme, rows):
    assert client.get("/public/demo/" + "x" * 22).status_code == 404
    sql(env, "UPDATE demo_links SET activated_at = now() - interval '8 days', "
             "expires_at = now() - interval '1 minute' WHERE id=%s", acme["link_id"])
    try:
        assert client.get(f"/public/demo/{acme['token']}").status_code == 410
        assert client.post(f"/public/demo/{acme['token']}/visit",
                           json={"visitor_id": str(uuid.uuid4())}).status_code == 410
    finally:
        sql(env, "UPDATE demo_links SET expires_at = now() + interval '7 days' WHERE id=%s", acme["link_id"])


BROWSER = {k: str(uuid.uuid4()) for k in ("asha_laptop", "asha_phone", "copied", "wrong_code", "stranger")}


def visit(client, acme, browser, **body):
    r = client.post(f"/public/demo/{acme['token']}/visit", json={"visitor_id": BROWSER[browser], **body})
    assert r.status_code == 200, r.text
    return r.json()


def visitor_row(env, browser):
    return sql(env, "SELECT v.attribution, c.full_name, v.source_step, v.utm, v.first_viewed_at IS NOT NULL "
                    "FROM demo_visitors v LEFT JOIN demo_link_recipients r ON r.id = v.recipient_id "
                    "LEFT JOIN demo_contacts c ON c.id = r.contact_id WHERE v.visitor_key = %s", BROWSER[browser])[0]


def test_visit_on_a_persons_code_is_that_person(client, env, acme):
    utm = {"utm_source": "email", "utm_medium": "cold_email", "utm_campaign": "oct_b1", "utm_content": "e2",
           "utm_evil": "x", "utm_term": "=HYPERLINK(1)" * 20}
    r = visit(client, acme, "asha_laptop", ref=acme["refs"]["Asha Rao"], utm=utm)
    assert r == {"ask_identity": False}
    attribution, name, step, saved_utm, viewed = visitor_row(env, "asha_laptop")
    assert (attribution, name, step, viewed) == ("recipient", "Asha Rao", "e2", True)
    assert saved_utm == {k: utm[k] for k in ("utm_source", "utm_medium", "utm_campaign", "utm_content")}
    ev = sql(env, "SELECT e.type, e.source_step, c.full_name FROM demo_events e "
                  "JOIN demo_link_recipients r ON r.id = e.recipient_id JOIN demo_contacts c ON c.id = r.contact_id "
                  "WHERE e.visitor_id = %s", BROWSER["asha_laptop"])
    assert ev == [("page_viewed", "e2", "Asha Rao")]
    # Coming back later without the code (bookmark) is still Asha; first touch is kept.
    visit(client, acme, "asha_laptop", source_step="e3")
    assert visitor_row(env, "asha_laptop")[:3] == ("recipient", "Asha Rao", "e2")


def test_same_code_on_another_browser_is_shared(client, env, acme):
    r = visit(client, acme, "asha_phone", ref=acme["refs"]["Asha Rao"], source_step="e1")
    assert r == {"ask_identity": True}                  # her phone, or someone she forwarded to
    assert visitor_row(env, "asha_phone")[:3] == ("shared", "Asha Rao", "e1")


def test_no_code_or_another_links_code_is_unknown(client, env, acme):
    assert visit(client, acme, "copied") == {"ask_identity": True}
    assert visitor_row(env, "copied")[:2] == ("unknown", None)
    visit(client, acme, "wrong_code", ref=acme["bay_ref"])          # Bay View's code on Acme's link
    assert visitor_row(env, "wrong_code")[:2] == ("unknown", None)
    visit(client, acme, "stranger", ref="not a code!!")              # mangled: ignored, page still works
    assert visitor_row(env, "stranger")[:2] == ("unknown", None)


def test_identify(client, env, acme):
    url = f"/public/demo/{acme['token']}/identify"
    # A colleague who is on our list becomes that person.
    r = client.post(url, json={"visitor_id": BROWSER["copied"], "name": "Vikram", "email": "VIKRAM@acmehomes.in"})
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert visitor_row(env, "copied")[:2] == ("self_identified", "Vikram Iyer")
    # Someone we don't know keeps their own identity.
    client.post(url, json={"visitor_id": BROWSER["stranger"], "name": "Rohan Mehta",
                           "email": "rohan@acmehomes.in", "role": "CFO"})
    assert sql(env, "SELECT attribution, recipient_id, identified_name, identified_email, identified_role "
                    "FROM demo_visitors WHERE visitor_key=%s", BROWSER["stranger"]) == [
        ("self_identified", None, "Rohan Mehta", "rohan@acmehomes.in", "CFO")]
    assert client.post(url, json={"visitor_id": BROWSER["stranger"], "name": "x", "email": "nope"}).status_code == 422
    assert visit(client, acme, "stranger") == {"ask_identity": False}     # asked once is enough
    assert sql(env, "SELECT count(*) FROM demo_events WHERE type='visitor_identified'") == [(2,)]


def test_events(client, env, acme):
    r = client.post(f"/public/demo/{acme['token']}/events",
                    json={"type": "kb_tab_viewed", "visitor_id": BROWSER["asha_laptop"]})
    assert r.status_code == 204
    assert sql(env, "SELECT c.full_name FROM demo_events e JOIN demo_link_recipients r ON r.id = e.recipient_id "
                    "JOIN demo_contacts c ON c.id = r.contact_id WHERE e.type='kb_tab_viewed'") == [("Asha Rao",)]
    # Opens are only ever recorded by /visit.
    assert client.post(f"/public/demo/{acme['token']}/events",
                       json={"type": "page_viewed", "visitor_id": BROWSER["asha_laptop"]}).status_code == 422
    assert client.post(f"/public/demo/{acme['token']}/events", json={"type": "drop_tables"}).status_code == 422


# ── sessions, limits, voice ──────────────────────────────────────────────────────────────

def test_session_voice_handshake_and_close_out(client, env, acme):
    from app.demo.voice import DemoVoiceRefused, authenticate_demo_voice, finish_demo_call

    r = client.post(f"/public/demo/{acme['token']}/session", json={"visitor_id": BROWSER["asha_laptop"]})
    assert r.status_code == 201, r.text
    s = r.json()
    assert s["max_seconds"] == 300 and "/ws/voice?token=" in s["ws_url"]
    vtok = parse_qs(urlsplit(s["ws_url"]).query)["token"][0]
    row = sql(env, "SELECT d.conversation_id, c.tenant_id, t.kind, ct.full_name FROM demo_sessions d "
                   "JOIN conversations c ON c.id = d.conversation_id JOIN tenants t ON t.id = c.tenant_id "
                   "JOIN demo_link_recipients r ON r.id = d.recipient_id JOIN demo_contacts ct ON ct.id = r.contact_id "
                   "WHERE d.id=%s", s["session_id"])[0]
    assert row[2] == "demo" and row[3] == "Asha Rao"

    # One call at a time on a link.
    busy = client.post(f"/public/demo/{acme['token']}/session", json={})
    assert busy.status_code == 429 and busy.json()["detail"]["code"] == "link_busy"

    async def handshake(token, origin):
        async with client.factory() as db:
            try:
                call = await authenticate_demo_voice(db, token, origin)
                await db.commit()
                return call
            except DemoVoiceRefused as e:
                return e.code

    assert client.portal.call(handshake, vtok, "https://evil.example") == 4403
    assert client.portal.call(handshake, vtok + "x", DEMO_ORIGIN) == 4401
    call = client.portal.call(handshake, vtok, DEMO_ORIGIN)
    assert not isinstance(call, int) and call.max_seconds == 300
    assert str(call.conversation_id) == str(row[0])
    assert client.portal.call(handshake, vtok, DEMO_ORIGIN) == 4401       # single use

    # What the caller said (the engine writes these during a real call), with the engine's
    # own tool result and planner action in between, as a real call stores them.
    for i, (role, text) in enumerate([("customer", "3BHK ka price kya hai?"),
                                      ("tool", 'search_slots succeeded: {"slots": []}'),
                                      ("agent", "Price is not published; the sales team will confirm."),
                                      ("system", "action:respond"),
                                      ("customer", "Is there a discount?")]):
        sql(env, "INSERT INTO messages (id, tenant_id, conversation_id, role, content, metadata, created_at) "
                 "VALUES (gen_random_uuid(), %s, %s, %s, %s, '{}', now() + make_interval(secs => %s))",
            str(row[1]), str(row[0]), role, text, i + 1)

    # The sample lead books during the call → outcome goal_completed.
    sql(env, "INSERT INTO locations (id, tenant_id, name, timezone, is_active, location_type, created_at, updated_at) "
             "SELECT gen_random_uuid(), %s, 'x', 'Asia/Kolkata', true, '{}', now(), now() "
             "WHERE NOT EXISTS (SELECT 1 FROM locations WHERE tenant_id=%s)", str(row[1]), str(row[1]))
    sql(env, "INSERT INTO appointments (id, tenant_id, customer_id, centre_id, starts_at, ends_at, status, booked_via, "
             "reminder_state, unit_context, created_at, updated_at) SELECT gen_random_uuid(), c.tenant_id, c.customer_id, "
             "(SELECT id FROM locations WHERE tenant_id=c.tenant_id LIMIT 1), now() + interval '1 day', "
             "now() + interval '1 day 1 hour', 'booked', 'agent', '{}', '{}', now(), now() FROM conversations c WHERE c.id=%s",
        str(row[0]))

    async def close():
        async with client.factory() as db:
            await finish_demo_call(db, call, end_reason="user_hangup")
            await db.commit()
    client.portal.call(close)
    done = sql(env, "SELECT end_reason, outcome, duration_seconds IS NOT NULL, ended_at IS NOT NULL "
                    "FROM demo_sessions WHERE id=%s", s["session_id"])[0]
    assert done == ("user_hangup", "goal_completed", True, True)


def test_calls_per_day_limit(client, env, acme):
    sql(env, "UPDATE demo_links SET max_calls_per_day = 2 WHERE id=%s", acme["link_id"])
    sql(env, "UPDATE demo_sessions SET ended_at = now() WHERE demo_link_id=%s AND ended_at IS NULL", acme["link_id"])
    assert client.post(f"/public/demo/{acme['token']}/session", json={}).status_code == 201   # 2nd today
    sql(env, "UPDATE demo_sessions SET ended_at = now() WHERE demo_link_id=%s AND ended_at IS NULL", acme["link_id"])
    r = client.post(f"/public/demo/{acme['token']}/session", json={})
    assert r.status_code == 429 and r.json()["detail"]["code"] == "daily_limit"
    sql(env, "UPDATE demo_links SET max_calls_per_day = NULL WHERE id=%s", acme["link_id"])


def test_each_call_gets_full_time_counted_from_first_audio(client, env, acme):
    """Every call gets the full max_call_seconds (calls are limited per day, not by a time
    pool), and a call's length runs from the agent's first audio, not from connecting."""
    from datetime import datetime, timedelta, timezone

    from app.demo.voice import DemoCall, finish_demo_call
    url = f"/public/demo/{acme['token']}/session"
    sql(env, "UPDATE demo_links SET max_calls_per_day = 50 WHERE id=%s", acme["link_id"])
    sql(env, "UPDATE demo_sessions SET ended_at = now() WHERE demo_link_id=%s AND ended_at IS NULL", acme["link_id"])
    try:
        first = client.post(url, json={"visitor_id": BROWSER["copied"]}).json()
        assert first["max_seconds"] == 300

        # 200 s of talk after the greeting started playing.
        s = sql(env, "SELECT conversation_id FROM demo_sessions WHERE id=%s", first["session_id"])[0]
        call = DemoCall(session_id=uuid.UUID(first["session_id"]), link_id=uuid.UUID(acme["link_id"]),
                        tenant_id=uuid.uuid4(), conversation_id=s[0], max_seconds=300)

        async def close(audio_started_at):
            async with client.factory() as db:
                await finish_demo_call(db, call, end_reason="user_hangup", audio_started_at=audio_started_at)
                await db.commit()
        client.portal.call(close, datetime.now(timezone.utc) - timedelta(seconds=200))
        assert sql(env, "SELECT duration_seconds FROM demo_sessions WHERE id=%s", first["session_id"]) == [(200,)]

        # Talking again starts a fresh call with the full time.
        again = client.post(url, json={"visitor_id": BROWSER["copied"]})
        assert again.status_code == 201 and again.json()["max_seconds"] == 300
    finally:
        sql(env, "UPDATE demo_links SET max_calls_per_day = NULL WHERE id=%s", acme["link_id"])
        sql(env, "UPDATE demo_sessions SET ended_at = now() WHERE demo_link_id=%s AND ended_at IS NULL", acme["link_id"])


def test_daily_budget(client, env, acme):
    from app.config import get_settings
    from app.demo import limits
    tid = sql(env, "SELECT p.tenant_id FROM demo_links l JOIN demo_prospects p ON p.id=l.prospect_id "
                   "WHERE l.id=%s", acme["link_id"])[0][0]
    # One turn that cost ~$12.50 on gpt-4o.
    sql(env, "INSERT INTO decision_traces (id, tenant_id, conversation_id, turn_index, retrieved_memory, "
             "retrieved_chunks, rules_evaluated, prompt_versions, tokens, model_used, created_at) VALUES "
             "(gen_random_uuid(), %s, gen_random_uuid(), 0, '{}', '[]', '[]', '{}', "
             "'{\"planner\": {\"prompt\": 1000000, \"completion\": 1000000}}', 'gpt-4o', now())", str(tid))
    os.environ["DEMO_DAILY_BUDGET_USD"] = "10"
    get_settings.cache_clear()
    limits._BUDGET_CACHE.clear()
    try:
        r = client.post(f"/public/demo/{acme['token']}/session", json={})
        assert r.status_code == 429 and r.json()["detail"]["code"] == "budget"
    finally:
        os.environ["DEMO_DAILY_BUDGET_USD"] = "0"
        get_settings.cache_clear()


def test_session_that_never_connects_frees_the_link(client, env, acme):
    """Talk pressed, then the microphone denied or the tab closed: the issued session must not
    block the link once its voice token has expired."""
    url = f"/public/demo/{acme['token']}/session"
    sql(env, "UPDATE demo_sessions SET ended_at = now() WHERE demo_link_id=%s AND ended_at IS NULL", acme["link_id"])
    assert client.post(url, json={}).status_code == 201                 # issued, never connected
    assert client.post(url, json={}).json()["detail"]["code"] == "link_busy"
    sql(env, "UPDATE demo_sessions SET created_at = now() - interval '3 minutes' "
             "WHERE demo_link_id=%s AND started_at IS NULL AND ended_at IS NULL", acme["link_id"])
    assert client.post(url, json={}).status_code == 201
    sql(env, "UPDATE demo_sessions SET ended_at = now() WHERE demo_link_id=%s AND ended_at IS NULL", acme["link_id"])


def test_simultaneous_starts_get_one_slot(client, env, acme):
    """Four browsers press Talk at the same moment on a one-call-at-a-time link: the limits
    are counted and the session inserted under one lock, so exactly one gets the call."""
    import httpx
    url = f"/public/demo/{acme['token']}/session"
    sql(env, "UPDATE demo_links SET max_calls_per_day = 50 WHERE id=%s", acme["link_id"])
    sql(env, "UPDATE demo_sessions SET ended_at = now() WHERE demo_link_id=%s AND ended_at IS NULL", acme["link_id"])

    async def race():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app),
                                     base_url="http://test") as ac:
            return await asyncio.gather(*[ac.post(url, json={}) for _ in range(4)])
    try:
        results = client.portal.call(race)
        assert sorted(r.status_code for r in results) == [201, 429, 429, 429], [r.text for r in results]
        assert {r.json()["detail"]["code"] for r in results if r.status_code == 429} == {"link_busy"}
        assert sql(env, "SELECT count(*) FROM demo_sessions WHERE demo_link_id=%s AND ended_at IS NULL",
                   acme["link_id"]) == [(1,)]
    finally:
        sql(env, "UPDATE demo_links SET max_calls_per_day = NULL WHERE id=%s", acme["link_id"])
        sql(env, "UPDATE demo_sessions SET ended_at = now() WHERE demo_link_id=%s AND ended_at IS NULL", acme["link_id"])


# ── leads: the contact form, the sheet, the read-back ────────────────────────────────────

def test_contact_form_saves_the_lead_once(client, env, acme):
    url = f"/public/demo/{acme['token']}/contact"
    asha = {"visitor_id": BROWSER["asha_laptop"], "kind": "schedule_call", "name": "Asha Rao",
            "email": "Asha.Rao@acmehomes.in"}
    r = client.post(url, json=asha)
    assert r.status_code == 422 and "phone" in r.json()["detail"]["message"]       # we call them back
    assert client.post(url, json={**asha, "email": "not-an-email", "phone": "9876543210"}).status_code == 422
    body = {**asha, "phone": "98765 43210", "preferred_time": "after 4 pm", "message": "=HYPERLINK(1) <b>hi</b>"}
    first = client.post(url, json=body)
    assert first.status_code == 201 and first.json()["created"] is True
    again = client.post(url, json=body)                                             # double click
    assert again.json() == {"lead_id": first.json()["lead_id"], "created": False}
    row = sql(env, "SELECT l.kind, l.phone, l.email, c.full_name, l.company, l.last_session_id IS NOT NULL, "
                   "l.call_status FROM demo_leads l LEFT JOIN demo_contacts c ON c.id = l.contact_id "
                   "WHERE l.sheet_row_key=%s", first.json()["lead_id"])
    assert row == [("schedule_call", "+919876543210", "asha.rao@acmehomes.in", "Asha Rao", "Acme Homes", True, "New")]
    assert sql(env, "SELECT count(*) FROM demo_events WHERE type='contact_submitted'") == [(1,)]
    assert client.post("/public/demo/" + "x" * 22 + "/contact", json={**body}).status_code == 404


def test_contact_from_an_unknown_visitor_says_who_they_are(client, env, acme):
    r = client.post(f"/public/demo/{acme['token']}/contact",
                    json={"visitor_id": BROWSER["wrong_code"], "kind": "contact", "name": "Neel Shah",
                          "email": "neel@acmehomes.in", "role": "CTO"})
    assert r.status_code == 201
    assert sql(env, "SELECT attribution, identified_name, identified_role FROM demo_visitors WHERE visitor_key=%s",
               BROWSER["wrong_code"]) == [("self_identified", "Neel Shah", "CTO")]


@pytest.fixture
def sheet_on(monkeypatch):
    from app.config import get_settings
    from app.demo.leads import sheets
    from tests.test_demo_leads_sheet import FakeSheet
    fake = FakeSheet({"Sheet1": []})
    for k, v in {"DEMO_LEADS_GSHEET_ID": "test-sheet", "DEMO_EMAIL_PROVIDER": "none",
                 "DEMO_SHEET_LEADS_TAB": "Leads", "DEMO_SHEET_ACTIVITY_TAB": "Demo activity"}.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    monkeypatch.setattr(sheets, "open_backend", lambda: fake)
    yield fake
    get_settings.cache_clear()


def _sheet_rows(fake, tab):
    header, *rows = fake.data[tab]
    return [dict(zip(header, r)) for r in rows]


def test_leads_go_to_the_sheet_and_the_team_status_comes_back(client, env, sheet_on):
    r = client.post("/admin/demo/leads/sync").json()
    assert r["pushed"]["appended"] == 2 and r["synced"]["activity_rows"] == 2
    leads = {x["Name"]: x for x in _sheet_rows(sheet_on, "Leads")}
    asha = leads["Asha Rao"]
    assert asha["Kind"] == "Schedule a call" and asha["Phone"] == "+919876543210"
    assert asha["Company"] == "Acme Homes" and asha["Industry"] == "Real estate"
    assert asha["Demo calls"] == "1" and asha["Goal completed"] == "yes"
    assert "3BHK ka price kya hai?" in asha["Questions asked"]
    assert asha["Message"].startswith("'=HYPERLINK")                   # text, not a formula
    assert asha["Status"] == "New"
    activity = {x["Company"]: x for x in _sheet_rows(sheet_on, "Demo activity")}
    assert "Asha Rao" in activity["Acme Homes"]["Opened by"]
    assert activity["Acme Homes"]["Lead submitted"].startswith(("Schedule a call", "Contact us"))

    # The team works the lead in the sheet; the next sync copies it back.
    header = sheet_on.data["Leads"][0]
    row = next(r for r in sheet_on.data["Leads"][1:] if r[header.index("Name")] == "Asha Rao")
    row[header.index("Status")], row[header.index("Called by")] = "Contacted", "Karthik"
    row[header.index("Notes")] = "Wants a pilot in November"
    assert client.post("/admin/demo/leads/sync").json()["synced"]["status_updates"] == 1
    listed = {x["name"]: x for x in client.get("/admin/demo/leads").json()}
    assert listed["Asha Rao"]["call_status"] == "Contacted" and listed["Asha Rao"]["called_by"] == "Karthik"
    assert listed["Asha Rao"]["delivery"] == "delivered" and listed["Asha Rao"]["on_our_list"] == "Asha Rao"
    assert client.post("/admin/demo/leads/sync").json()["pushed"]["appended"] == 0    # nothing twice


def test_a_lead_survives_a_sheet_outage(client, env, acme, sheet_on, monkeypatch):
    from app.demo.leads import sheets
    r = client.post(f"/public/demo/{acme['token']}/contact",
                    json={"visitor_id": BROWSER["stranger"], "kind": "real_calls_pilot", "name": "Rohan Mehta",
                          "email": "rohan@acmehomes.in", "phone": "+91 99887 76655"})
    assert r.status_code == 201
    lid = r.json()["lead_id"]

    def down():
        raise sheets.SheetError("Google is down")
    monkeypatch.setattr(sheets, "open_backend", down)
    assert client.post("/admin/demo/leads/sync").json()["pushed"]["failed"] == 1
    assert sql(env, "SELECT notify_attempts, sheet_synced_at IS NULL, last_notify_error LIKE '%%Google is down%%' "
                    "FROM demo_leads WHERE sheet_row_key=%s", lid) == [(1, True, True)]
    monkeypatch.setattr(sheets, "open_backend", lambda: sheet_on)
    lead_uuid = sql(env, "SELECT id FROM demo_leads WHERE sheet_row_key=%s", lid)[0][0]
    done = client.post(f"/admin/demo/leads/{lead_uuid}/resend").json()
    assert done["lead"]["delivery"] == "delivered" and done["lead"]["notify_attempts"] == 0
    assert lid in [x["Lead ID"] for x in _sheet_rows(sheet_on, "Leads")]

    # Rows deleted by mistake come back with a resync; rows still there are left alone.
    sheet_on.data["Leads"] = sheet_on.data["Leads"][:1]
    assert client.post("/admin/demo/leads/resync-sheet").json()["appended"] == 3
    assert len(_sheet_rows(sheet_on, "Leads")) == 3


def test_leads_xlsx_download(client):
    r = client.get("/admin/demo/leads.xlsx")
    assert r.status_code == 200 and r.content[:2] == b"PK"


# ── admin: who did what ──────────────────────────────────────────────────────────────────

def test_link_activity(client, acme):
    a = client.get(f"/admin/demo/links/{acme['link_id']}/activity").json()
    people = {p["contact"]["name"]: p for p in a["people"]}
    asha = people["Asha Rao"]
    assert asha["opened"] and asha["browsers"] == 2 and asha["forwarded_browsers"] == 1
    assert asha["views"] >= 2 and asha["calls"] == 1 and asha["goal_completed"] and asha["questions"] == 2
    assert people["Vikram Iyer"]["opened"]                 # identified himself on a copied link
    assert people["Karan Shah"]["contact"]["other_company"] and not people["Karan Shah"]["opened"]
    others = {o["name"]: o for o in a["other_visitors"]}
    assert others["Rohan Mehta"]["email"] == "rohan@acmehomes.in" and others["Rohan Mehta"]["via"] is None
    assert others["Neel Shah"]["email"] == "neel@acmehomes.in"   # wrong code, then the contact form
    qs = [q for q in a["questions"] if q["who"]["name"] == "Asha Rao"]
    assert [q["text"] for q in qs] == ["Is there a discount?", "3BHK ka price kya hai?"]
    assert qs[1]["reply"].startswith("Price is not published") and qs[0]["reply"] is None
    assert any(t["type"] == "visitor_identified" for t in a["timeline"])


def test_questions_search(client):
    found = client.get("/admin/demo/questions", params={"q": "discount"}).json()
    assert [(q["text"], q["company"], q["who"]["name"]) for q in found] == [
        ("Is there a discount?", "Acme Homes", "Asha Rao")]
    assert client.get("/admin/demo/questions", params={"q": "100%_"}).json() == []   # LIKE wildcards are literal


def test_sessions_transcript_and_metrics(client, env):
    sessions = client.get("/admin/demo/sessions").json()
    booked = [s for s in sessions if s["outcome"] == "goal_completed"]
    assert booked and booked[0]["company"] == "Acme Homes" and booked[0]["who"]["name"] == "Asha Rao"
    t = client.get(f"/admin/demo/sessions/{booked[0]['id']}/transcript").json()
    assert t["who"]["email"] == "asha.rao@acmehomes.in"
    assert t["messages"][0]["role"] == "agent"
    assert "enquiry desk. I see you enquired" in t["messages"][0]["content"]
    # Only what was said: no tool results or planner actions.
    assert [(m["role"], m["content"]) for m in t["messages"][1:]] == [
        ("customer", "3BHK ka price kya hai?"),
        ("agent", "Price is not published; the sales team will confirm."),
        ("customer", "Is there a discount?")]
    assert len(t["bookings"]) == 1
    assert client.get(f"/admin/demo/sessions/{uuid.uuid4()}/transcript").status_code == 404

    m = client.get("/admin/demo/metrics").json()
    assert m["records"] == 2 and m["contacts"] == 4 and m["links_active"] == 2
    assert m["companies_opened"] == 1 and m["companies_talked"] == 1 and m["visits_booked"] >= 1
    # Karan is on two companies' links but is one person.
    assert m["people_emailed"] == 4 and m["people_opened"] == 2 and m["people_talked"] == 1
    assert m["unknown_visitors"] == 2 and m["open_rate"] == 0.5
    assert m["spend_today_usd"] > 0
    assert client.get("/admin/demo/metrics", params={"industry_key": "nope"}).json()["records"] == 0


def test_revoke_and_extend(client, acme):
    ext = client.post(f"/admin/demo/links/{acme['link_id']}/extend", json={"days": 5}).json()
    assert ext["status"] == "active"
    rv = client.post(f"/admin/demo/links/{acme['link_id']}/revoke", json={"reason": "test"}).json()
    assert rv["status"] == "revoked"
    assert client.get(f"/public/demo/{acme['token']}").status_code == 404
    assert client.post(f"/admin/demo/links/{acme['link_id']}/extend", json={"days": 5}).status_code == 409
    r = client.post(f"/admin/demo/links/{acme['link_id']}/recipients", json={"contact_id": str(uuid.uuid4())})
    assert r.status_code == 409


def test_public_api_is_404_while_disabled(client, acme):
    from app.config import get_settings
    bay = client.post(f"/admin/demo/records/{record_id(client, 'Bay View')}/link", json={}).json()
    os.environ["DEMO_ENABLED"] = "false"
    get_settings.cache_clear()
    try:
        assert client.get(f"/public/demo/{token_of(bay['company_url'])}").status_code == 404
    finally:
        os.environ["DEMO_ENABLED"] = "true"
        get_settings.cache_clear()


def test_per_ip_rate_limit(client):
    from app.config import get_settings
    bay = client.post(f"/admin/demo/records/{record_id(client, 'Bay View')}/link", json={}).json()
    _flush_redis()
    os.environ["DEMO_RATE_LIMIT_PER_MIN"] = "2"
    get_settings.cache_clear()
    try:
        token = token_of(bay["company_url"])
        codes = [client.get(f"/public/demo/{token}").status_code for _ in range(3)]
        assert codes[:2] != [429, 429] and codes[2] == 429
    finally:
        os.environ["DEMO_RATE_LIMIT_PER_MIN"] = "1000"
        get_settings.cache_clear()
        _flush_redis()


def test_missing_token_key_refuses_new_links(client):
    from app.config import get_settings
    os.environ["DEMO_LINK_TOKEN_KEYS"] = ""
    get_settings.cache_clear()
    try:
        r = client.post(f"/admin/demo/records/{record_id(client, 'Acme')}/link", json={"replace": True})
        assert r.status_code == 409 and "DEMO_LINK_TOKEN_KEYS" in r.json()["detail"]["message"]
    finally:
        os.environ["DEMO_LINK_TOKEN_KEYS"] = TOKEN_KEY
        get_settings.cache_clear()
