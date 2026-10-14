"""Seed sync (app/demo/industries/registry.py) against a real database: creates missing
industries with their version and dummy-data template, is idempotent, and never
overwrites an industry an admin already owns.

Needs a Postgres it may create and drop a scratch database on (the docker-compose one):

    TEST_ADMIN_DATABASE_URL=postgresql://aivoiceagent:aivoiceagent@127.0.0.1:5437/postgres \
        python -m pytest tests/test_demo_registry_db.py -q
"""
import asyncio
import copy
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


@pytest.fixture(scope="module")
def db_url():
    import psycopg2
    name = f"demo_registry_{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(ADMIN_URL)
    admin.autocommit = True
    admin.cursor().execute(f"CREATE DATABASE {name}")
    base = ADMIN_URL.rsplit("/", 1)[0]
    url = f"{base}/{name}"
    r = subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=CORE,
                       env=dict(os.environ, DATABASE_URL_SYNC=url.replace(
                           "postgresql://", "postgresql+psycopg2://", 1)),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    try:
        yield url.replace("postgresql://", "postgresql+asyncpg://", 1)
    finally:
        cur = admin.cursor()
        cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s",
                    (name,))
        cur.execute(f"DROP DATABASE IF EXISTS {name}")
        admin.close()


def run(db_url, fn):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    async def go():
        engine = create_async_engine(db_url)
        try:
            async with async_sessionmaker(engine, expire_on_commit=False)() as db:
                out = await fn(db)
                await db.commit()
                return out
        finally:
            await engine.dispose()
    return asyncio.run(go())


def test_sync_creates_then_is_idempotent_and_never_overwrites(db_url):
    from sqlalchemy import select, update

    from app.db.models import DemoIndustry, DemoIndustryVersion, DemoSampleDataset
    from app.demo.industries.definition import compile_definition
    from app.demo.industries.registry import load_seed_definitions, sync_seed_industries

    seeds = load_seed_definitions()
    assert "real_estate" in run(db_url, lambda db: sync_seed_industries(db, seeds))
    assert run(db_url, lambda db: sync_seed_industries(db, seeds)) == []    # idempotent

    async def state(db):
        ind = await db.get(DemoIndustry, "real_estate")
        versions = (await db.execute(select(DemoIndustryVersion.version).where(
            DemoIndustryVersion.industry_key == "real_estate"))).scalars().all()
        templates = (await db.execute(select(DemoSampleDataset).where(
            DemoSampleDataset.industry_key == "real_estate",
            DemoSampleDataset.prospect_id.is_(None)))).scalars().all()
        return ind, versions, templates

    ind, versions, templates = run(db_url, state)
    assert (ind.status, ind.current_version, versions) == ("active", 1, [1])
    assert len(templates) == 1 and templates[0].is_active
    assert templates[0].data["booking_reference"] == "VIS-1187"

    # An admin renames the industry; a newer seed version arrives with a deploy.
    run(db_url, lambda db: db.execute(update(DemoIndustry).where(
        DemoIndustry.key == "real_estate").values(label="Real estate (admin edit)")))
    re_seed = next(d for d in seeds if d.key == "real_estate")
    raw2 = copy.deepcopy(re_seed.raw)
    raw2["version"] = 2
    raw2["label"] = "Real estate v2 from seed"
    assert run(db_url, lambda db: sync_seed_industries(db, [compile_definition(raw2)])) == []

    ind, versions, _ = run(db_url, state)
    assert ind.label == "Real estate (admin edit)"      # admin's edit survives
    assert versions == [1]                              # seed v2 not applied automatically
