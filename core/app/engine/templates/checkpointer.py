"""Postgres-backed LangGraph checkpointer (Vocal Backend Tasksheet P3) — keyed by
call_id (the thread_id in LangGraph's terms), so a call can resume exactly where
it left off after a dropped media stream.

`AsyncPostgresSaver` owns and versions its own schema (checkpoints/
checkpoint_writes/checkpoint_blobs/checkpoint_migrations), created by
`.setup()` — idempotent, safe to call every time a checkpointer is opened,
rather than forced through this app's own Alembic history. It manages its own
psycopg3 connection, entirely separate from the app's asyncpg SQLAlchemy engine.

Usage (an async context manager, so its connection pool has a clear lifetime —
see app/engine/templates/compiler.py's run_graph() for the actual call site):

    async with open_checkpointer() as saver:
        compiled = graph.compile(checkpointer=saver)
        await compiled.ainvoke(state, config={"configurable": {"thread_id": call_id}})

P6's Call Runtime Service (its own long-lived process) is expected to open one
checkpointer for its whole lifetime via this same helper, rather than per call —
short-lived-per-call is the right shape only for this wave's dry-run/verify use.
"""
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.config import get_settings


def checkpointer_conn_string() -> str:
    """A plain psycopg3 DSN — AsyncPostgresSaver doesn't understand SQLAlchemy's
    driver-qualified `postgresql+asyncpg://` URL, just `postgresql://`."""
    return get_settings().database_url.replace("postgresql+asyncpg://", "postgresql://")


def open_checkpointer():
    """Returns the async context manager; call sites do
    `async with open_checkpointer() as saver: ...` and call `await saver.setup()`
    once inside before first use."""
    return AsyncPostgresSaver.from_conn_string(checkpointer_conn_string())
