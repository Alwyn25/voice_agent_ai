"""migrations/env.py: what `alembic upgrade head` does when the database is stamped with a
revision this code has no script for (ALEMBIC_UNKNOWN_REVISION = error | skip | rebase).

Each test gets a scratch database migrated to 0036 and then stamped with a revision the
code doesn't have: 0037 (a migration from another branch, below this code's head) or 0099
(the database is ahead of the code: a rollback).

    TEST_ADMIN_DATABASE_URL=postgresql://aivoiceagent:aivoiceagent@127.0.0.1:5437/postgres \
        python -m pytest tests/test_migration_unknown_revision.py -q
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


def _alembic(url: str, *args: str, mode: str | None = None):
    env = dict(os.environ, DATABASE_URL_SYNC=url.replace("postgresql://", "postgresql+psycopg2://", 1))
    env.pop("ALEMBIC_UNKNOWN_REVISION", None)
    if mode:
        env["ALEMBIC_UNKNOWN_REVISION"] = mode
    return subprocess.run([sys.executable, "-m", "alembic", *args], cwd=CORE, env=env,
                          capture_output=True, text=True)


def _sql(url: str, q: str, *a):
    import psycopg2
    with psycopg2.connect(url) as conn, conn.cursor() as cur:
        cur.execute(q, a)
        return cur.fetchall() if cur.description else None


@pytest.fixture
def db_at():
    """Factory: a scratch DB migrated to 0036, then stamped with `fake`."""
    import psycopg2
    admin = psycopg2.connect(ADMIN_URL)
    admin.autocommit = True
    made = []

    def make(fake: str) -> str:
        name = f"mig_unknown_{uuid.uuid4().hex[:8]}"
        admin.cursor().execute(f"CREATE DATABASE {name}")
        made.append(name)
        url = f"{ADMIN_URL.rsplit('/', 1)[0]}/{name}"
        r = _alembic(url, "upgrade", "0036")
        assert r.returncode == 0, r.stderr
        _sql(url, "UPDATE alembic_version SET version_num = %s", fake)
        return url
    yield make
    cur = admin.cursor()
    for name in made:
        cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s", (name,))
        cur.execute(f"DROP DATABASE IF EXISTS {name}")
    admin.close()


def _version(url):
    return [r[0] for r in _sql(url, "SELECT version_num FROM alembic_version")]


HEAD = "0039"     # this code's newest migration


def _has_0038(url):
    # ck_tenants_kind is created only by migration 0038 (the model has no CHECK).
    return bool(_sql(url, "SELECT 1 FROM pg_constraint WHERE conname = 'ck_tenants_kind'"))


def test_default_error_fails_as_before_with_a_hint(db_at):
    url = db_at("0037")
    r = _alembic(url, "upgrade", "head")
    assert r.returncode != 0
    assert "ALEMBIC_UNKNOWN_REVISION=rebase" in r.stderr
    assert _version(url) == ["0037"] and not _has_0038(url)


def test_skip_leaves_the_database_alone(db_at):
    url = db_at("0037")
    r = _alembic(url, "upgrade", "head", mode="skip")
    assert r.returncode == 0, r.stderr
    assert "Skipping migrations" in r.stderr
    assert _version(url) == ["0037"] and not _has_0038(url)


def test_rebase_applies_what_this_code_has_after_the_known_ancestor(db_at):
    url = db_at("0037")
    r = _alembic(url, "upgrade", "head", mode="rebase")
    assert r.returncode == 0, r.stderr
    assert "re-stamped to 0036" in r.stderr
    assert _version(url) == [HEAD] and _has_0038(url)
    # Running again is a no-op.
    assert _alembic(url, "upgrade", "head", mode="rebase").returncode == 0
    assert _version(url) == [HEAD]


def test_rebase_never_rewinds_a_database_that_is_ahead(db_at):
    url = db_at("0099")
    r = _alembic(url, "upgrade", "head", mode="rebase")
    assert r.returncode == 0, r.stderr
    assert "ahead of this code" in r.stderr
    assert _version(url) == ["0099"] and not _has_0038(url)


def test_manual_stamp_still_works_whatever_the_mode(db_at):
    url = db_at("0037")
    r = _alembic(url, "stamp", "0036", "--purge", mode="skip")
    assert r.returncode == 0, r.stderr
    assert _version(url) == ["0036"]


def test_bad_mode_is_refused(db_at):
    url = db_at("0037")
    r = _alembic(url, "upgrade", "head", mode="yolo")
    assert r.returncode != 0 and "ALEMBIC_UNKNOWN_REVISION must be one of" in r.stderr
