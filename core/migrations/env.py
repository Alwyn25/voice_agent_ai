import logging
import os
import re
import sys
from logging.config import fileConfig

from alembic import context
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings  # noqa: E402
from app.db.base import Base  # noqa: E402
import app.db.models  # noqa: F401,E402  (registers all tables)

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
log = logging.getLogger("alembic.env")

# What to do when the database is stamped with a revision this code has no script for —
# e.g. a migration applied from another branch (R5.04's 0037 on a database whose code
# doesn't have it yet). ALEMBIC_UNKNOWN_REVISION:
#
#   error   (default) stop, as alembic always has, with a message saying what to do.
#   skip    leave the database alone and run no migrations, so the app can still start.
#           Also the right choice for a rollback (older code against a newer database).
#   rebase  for a migration from another branch: treat the unknown revision as the nearest
#           known revision below it (0037 → 0036), re-stamp, and apply what this code has
#           after that (0038). Requires those migrations to be re-runnable (this project's
#           are guarded) and numeric revision ids. Never used when the database is AHEAD of
#           this code — that is a rollback, and it is skipped instead, because re-stamping
#           backwards would make the next deploy re-run migrations already applied.
UNKNOWN_REVISION_MODES = ("error", "skip", "rebase")


def get_url() -> str:
    return os.environ.get("DATABASE_URL_SYNC") or get_settings().database_url_sync


def _num(rev: str) -> int | None:
    return int(rev) if re.fullmatch(r"\d+", rev or "") else None


def _reconcile_unknown_revisions(engine, script: ScriptDirectory) -> bool:
    """Handle a database stamped with revisions this code doesn't know. Returns False when
    migrations should be skipped this run, True to go ahead."""
    mode = os.environ.get("ALEMBIC_UNKNOWN_REVISION", "error").strip().lower()
    if mode not in UNKNOWN_REVISION_MODES:
        raise RuntimeError(f"ALEMBIC_UNKNOWN_REVISION must be one of {UNKNOWN_REVISION_MODES}, "
                           f"got {mode!r}")
    with engine.connect() as conn:
        if not inspect(conn).has_table("alembic_version"):
            return True                                   # fresh database
        stamped = [r[0] for r in conn.execute(text("SELECT version_num FROM alembic_version"))]
    known = {s.revision for s in script.walk_revisions()}
    unknown = [r for r in stamped if r not in known]
    if not unknown:
        return True

    heads = script.get_heads()
    hint = (f"The database is at {', '.join(unknown)}, which this code has no migration script "
            f"for (code head: {', '.join(heads)}). Set ALEMBIC_UNKNOWN_REVISION=rebase if it came "
            f"from another branch, or =skip to start without migrating.")
    if mode == "error":
        log.error(hint)
        return True          # alembic then fails exactly as it always has
    if mode == "skip":
        log.warning("%s Skipping migrations (ALEMBIC_UNKNOWN_REVISION=skip).", hint)
        return False

    # rebase
    known_nums = sorted(n for n in (_num(r) for r in known) if n is not None)
    plan = {}
    for rev in unknown:
        n = _num(rev)
        if n is None:
            log.warning("%s Revision %r is not numeric, so it can't be placed; skipping "
                        "migrations.", hint, rev)
            return False
        if not known_nums or n > known_nums[-1]:
            log.warning("The database is at %s, ahead of this code's newest migration (%s): "
                        "this is a rollback, not a missing branch migration. Skipping migrations "
                        "rather than re-stamping backwards.", rev, ", ".join(heads))
            return False
        below = [k for k in known_nums if k < n]
        if not below:
            log.warning("No known revision below %s to rebase onto; skipping migrations.", rev)
            return False
        target = next(r for r in known if _num(r) == below[-1])
        plan[rev] = target
    with engine.begin() as conn:
        for rev, target in plan.items():
            conn.execute(text("UPDATE alembic_version SET version_num = :t WHERE version_num = :r"),
                         {"t": target, "r": rev})
            log.warning("ALEMBIC_UNKNOWN_REVISION=rebase: database was at %s (no script in this "
                        "code); re-stamped to %s so the migrations after it can run. They are "
                        "guarded, so anything %s already created is left as is.", rev, target, rev)
    return True


def run_migrations_offline() -> None:
    context.configure(url=get_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def _is_upgrade() -> bool:
    """Only `upgrade` reconciles. stamp/downgrade/current must keep working unchanged, so a
    person can always fix a version table by hand (e.g. `alembic stamp 0036 --purge`)."""
    cmd = getattr(getattr(config, "cmd_opts", None), "cmd", None)
    fn = cmd[0] if isinstance(cmd, tuple) and cmd else None
    return fn is None or getattr(fn, "__name__", "") == "upgrade"


def run_migrations_online() -> None:
    engine = create_engine(get_url())
    if _is_upgrade() and not _reconcile_unknown_revisions(engine, ScriptDirectory.from_config(config)):
        return
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
