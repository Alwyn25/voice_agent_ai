"""The demo schema must be identical whichever way a database reaches revision 0038.

Migration 0001 builds a fresh database with Base.metadata.create_all(), so on a new
environment the demo tables come from app/db/models/demo.py and 0038's guarded
create_table is skipped. On an existing database (already past 0001) the tables come from
0038's own DDL. If the two drift, fresh environments silently lose CHECK constraints,
partial unique indexes or server defaults. This test builds both and compares the
catalogue, then checks the constraints actually bite.

Needs a Postgres it may create and drop a scratch database on (the docker-compose one):

    TEST_ADMIN_DATABASE_URL=postgresql://aivoiceagent:aivoiceagent@127.0.0.1:5437/postgres \
        python -m pytest tests/test_demo_schema_parity.py -q

Skipped when TEST_ADMIN_DATABASE_URL is not set.
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

_CATALOG = """
SELECT 'col', table_name, column_name,
       data_type || coalesce('(' || character_maximum_length || ')', ''),
       is_nullable, coalesce(column_default, '')
  FROM information_schema.columns
 WHERE table_name LIKE 'demo_%' OR (table_name = 'tenants' AND column_name = 'kind')
UNION ALL
SELECT 'con', cl.relname, c.conname, c.contype::text, pg_get_constraintdef(c.oid), ''
  FROM pg_constraint c JOIN pg_class cl ON cl.oid = c.conrelid
 WHERE cl.relname LIKE 'demo_%' OR c.conname = 'ck_tenants_kind'
UNION ALL
SELECT 'idx', tablename, indexname, '', indexdef, ''
  FROM pg_indexes WHERE tablename LIKE 'demo_%'
ORDER BY 1, 2, 3
"""


@pytest.fixture(scope="module")
def scratch_db():
    import psycopg2
    name = f"demo_parity_{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(ADMIN_URL)
    admin.autocommit = True
    admin.cursor().execute(f"CREATE DATABASE {name}")
    base, _ = ADMIN_URL.rsplit("/", 1)
    url = f"{base}/{name}"
    try:
        yield url
    finally:
        cur = admin.cursor()
        cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s",
                    (name,))
        cur.execute(f"DROP DATABASE IF EXISTS {name}")
        admin.close()


def _alembic(url: str, *args: str) -> None:
    sync_url = url.replace("postgresql://", "postgresql+psycopg2://", 1)
    r = subprocess.run([sys.executable, "-m", "alembic", *args], cwd=CORE,
                       env=dict(os.environ, DATABASE_URL_SYNC=sync_url),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"alembic {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}"


def _catalog(url: str) -> list[tuple]:
    import psycopg2
    with psycopg2.connect(url) as conn, conn.cursor() as cur:
        cur.execute(_CATALOG)
        return cur.fetchall()


def test_fresh_and_upgraded_schemas_match(scratch_db):
    # Path A: a brand-new database (0001's create_all builds the demo tables from the models).
    _alembic(scratch_db, "upgrade", "head")
    fresh = _catalog(scratch_db)
    assert any(r[1] == "demo_leads" for r in fresh), "demo tables missing after fresh upgrade"

    # Path B: drop back below 0038, then let 0038's and 0039's own DDL build them.
    _alembic(scratch_db, "downgrade", "0036")
    assert not [r for r in _catalog(scratch_db) if r[1].startswith("demo_")]
    _alembic(scratch_db, "upgrade", "head")
    upgraded = _catalog(scratch_db)
    _assert_same(fresh, upgraded, "0038 + 0039")

    # Path C: a database that ran 0038 before 0039 existed: 0039 down, then up again.
    _alembic(scratch_db, "downgrade", "0038")
    assert any(r[2] == "uq_demo_links_one_active" for r in _catalog(scratch_db))
    _alembic(scratch_db, "upgrade", "head")
    _assert_same(fresh, _catalog(scratch_db), "0039 over an existing 0038")


def _assert_same(fresh: list[tuple], other: list[tuple], label: str) -> None:
    only_fresh = sorted(set(fresh) - set(other))
    only_other = sorted(set(other) - set(fresh))
    assert not only_fresh and not only_other, (
        f"models (fresh DB) and {label} (existing DB) disagree:\n"
        f"  only from models: {only_fresh}\n  only from {label}: {only_other}")


def test_constraints_bite(scratch_db):
    import psycopg2
    from psycopg2 import errors

    conn = psycopg2.connect(scratch_db)
    conn.autocommit = True
    cur = conn.cursor()

    def rejected(sql, params=(), exc=errors.IntegrityError):
        with pytest.raises(exc):
            cur.execute(sql, params)

    cur.execute("INSERT INTO tenants (id, name, slug, timezone, settings, status, plan, "
                "seats_purchased) VALUES (gen_random_uuid(), 'c', %s, 'Asia/Kolkata', '{}', "
                "'active', 'starter', 1) RETURNING kind", (f"c-{uuid.uuid4().hex[:6]}",))
    assert cur.fetchone()[0] == "customer"
    rejected("UPDATE tenants SET kind = 'other'")

    rejected("INSERT INTO demo_industries (key, label) VALUES ('Bad Key', 'x')")
    cur.execute("INSERT INTO demo_industries (key, label) VALUES ('real_estate', 'Real estate') "
                "RETURNING status, current_version")
    assert cur.fetchone() == ("draft", 1)
    cur.execute("INSERT INTO demo_industry_versions (industry_key, version, definition) "
                "VALUES ('real_estate', 1, '{}')")
    # A prospect can only name an industry version that exists.
    prospect = ("INSERT INTO demo_prospects (id, industry_key, industry_version, record_key, "
                "company_name) VALUES (gen_random_uuid(), 'real_estate', %s, %s, %s) "
                "RETURNING id, status")
    rejected(prospect, (9, "x--x", "X"))
    cur.execute(prospect, (1, "dsr--highland-greenz", "DSR"))
    pid, status = cur.fetchone()
    assert status == "draft"
    rejected(prospect, (1, "dsr--highland-greenz", "DSR again"))   # same record imported twice

    contact = ("INSERT INTO demo_contacts (id, prospect_id, full_name, email) "
               "VALUES (gen_random_uuid(), %s, %s, %s) RETURNING id")
    cur.execute(contact, (pid, "Rahul Rao", "rahul@x.com"))
    cid = cur.fetchone()[0]
    rejected(contact, (pid, "rahul rao", None))                 # same person, different case
    rejected(contact, (pid, "Someone Else", "RAHUL@x.com"))     # same email at the same company

    link = ("INSERT INTO demo_links (id, prospect_id, token_hash, token_prefix, "
            "expires_at) VALUES (gen_random_uuid(), %s, %s, 'abc', now() + interval '7 days') RETURNING id")
    cur.execute(link, (pid, "a" * 64))
    lid = cur.fetchone()[0]
    rejected(link, (pid, "b" * 64))            # a second active link for the same company
    rejected(link, (pid, "NOT-A-HASH"))        # token_hash must be a sha256 hex digest

    recipient = ("INSERT INTO demo_link_recipients (id, demo_link_id, contact_id, ref_code) "
                 "VALUES (gen_random_uuid(), %s, %s, %s) RETURNING id")
    cur.execute(recipient, (lid, cid, "Ab3_x-9QzK"))
    rid = cur.fetchone()[0]
    rejected(recipient, (lid, cid, "Zz9_y-1QaB"))     # the same person twice on one link
    rejected(recipient, (lid, cid, "bad code!"))       # outside the code alphabet

    visitor = ("INSERT INTO demo_visitors (id, demo_link_id, visitor_key, recipient_id, attribution) "
               "VALUES (gen_random_uuid(), %s, %s, %s, %s) RETURNING attribution, utm")
    key = str(uuid.uuid4())
    cur.execute(visitor, (lid, key, rid, "recipient"))
    assert cur.fetchone() == ("recipient", {})
    rejected(visitor, (lid, key, None, "unknown"))     # one row per browser per link
    rejected(visitor, (lid, str(uuid.uuid4()), None, "guessed"))
    cur.execute("INSERT INTO demo_events (id, demo_link_id, type) "
                "VALUES (gen_random_uuid(), %s, 'visitor_identified')", (lid,))

    lead = ("INSERT INTO demo_leads (id, kind, name, email, sheet_row_key) "
            "VALUES (gen_random_uuid(), 'contact', 'A', %s, %s)")
    cur.execute(lead, ("a@x.com", "L1"))
    rejected(lead, ("A@X.com", "L2"))          # same lead again the same day
    rejected("INSERT INTO demo_leads (id, kind, name, email, sheet_row_key) "
             "VALUES (gen_random_uuid(), 'spam', 'A', 'b@x.com', 'L3')")
    conn.close()


def test_downgrade_refuses_while_demo_tenants_exist(scratch_db):
    import psycopg2
    with psycopg2.connect(scratch_db) as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO tenants (id, name, slug, timezone, settings, status, plan, "
                    "seats_purchased, kind) VALUES (gen_random_uuid(), 'd', %s, 'Asia/Kolkata', "
                    "'{}', 'active', 'starter', 1, 'demo')", (f"demo-{uuid.uuid4().hex[:6]}",))
    sync_url = scratch_db.replace("postgresql://", "postgresql+psycopg2://", 1)
    r = subprocess.run([sys.executable, "-m", "alembic", "downgrade", "0036"], cwd=CORE,
                       env=dict(os.environ, DATABASE_URL_SYNC=sync_url),
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "demo tenant(s) still exist" in r.stderr + r.stdout


def test_0039_carries_per_person_links_over():
    """A database with per-person links (0038): each link's person becomes its recipient,
    their calls keep that attribution, and the links are revoked (their tokens were never
    stored, so they can't be exported again)."""
    import psycopg2
    name = f"demo_carry_{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(ADMIN_URL)
    admin.autocommit = True
    admin.cursor().execute(f"CREATE DATABASE {name}")
    url = f"{ADMIN_URL.rsplit('/', 1)[0]}/{name}"
    try:
        _alembic(url, "upgrade", "head")
        _alembic(url, "downgrade", "0038")
        with psycopg2.connect(url) as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO demo_industries (key, label) VALUES ('real_estate', 'RE')")
            cur.execute("INSERT INTO demo_industry_versions (industry_key, version, definition) "
                        "VALUES ('real_estate', 1, '{}')")
            cur.execute("INSERT INTO demo_prospects (id, industry_key, industry_version, record_key, "
                        "company_name) VALUES (gen_random_uuid(), 'real_estate', 1, 'a--b', 'A') RETURNING id")
            pid = cur.fetchone()[0]
            people = []
            for who in ("Asha", "Ravi"):
                cur.execute("INSERT INTO demo_contacts (id, prospect_id, full_name) "
                            "VALUES (gen_random_uuid(), %s, %s) RETURNING id", (pid, who))
                cid = cur.fetchone()[0]
                cur.execute("INSERT INTO demo_links (id, contact_id, prospect_id, token_hash, token_prefix, "
                            "expires_at) VALUES (gen_random_uuid(), %s, %s, %s, 'abc', now() + interval '7 days') "
                            "RETURNING id", (cid, pid, uuid.uuid4().hex * 2))
                lid = cur.fetchone()[0]
                cur.execute("INSERT INTO demo_sessions (id, demo_link_id) VALUES (gen_random_uuid(), %s)", (lid,))
                people.append((cid, lid))
        _alembic(url, "upgrade", "head")
        with psycopg2.connect(url) as conn, conn.cursor() as cur:
            for cid, lid in people:
                cur.execute("SELECT r.contact_id, l.status, s.recipient_id = r.id FROM demo_links l "
                            "JOIN demo_link_recipients r ON r.demo_link_id = l.id "
                            "JOIN demo_sessions s ON s.demo_link_id = l.id WHERE l.id = %s", (lid,))
                assert cur.fetchall() == [(cid, "revoked", True)]
    finally:
        cur = admin.cursor()
        cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s", (name,))
        cur.execute(f"DROP DATABASE IF EXISTS {name}")
        admin.close()
