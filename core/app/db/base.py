import asyncio
import uuid
from datetime import datetime

from sqlalchemy import DateTime, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import get_settings


class Base(DeclarativeBase):
    # Every model's TimestampMixin.updated_at (onupdate=func.now()) is a server-computed
    # column, so after a flush SQLAlchemy marks it expired rather than trusting the
    # in-memory value — normally invisible, since the next access just reloads it. But an
    # AsyncSession's ORM-attribute lazy-reload machinery needs an active greenlet bridge,
    # which isn't there for a plain (non-awaited) attribute access — e.g. a response
    # builder called right after `await db.commit()`, a pattern all over this codebase's
    # API layer (see integrations_tools.py's _catalog_out, agents.py's _agent_out).
    # That raised sqlalchemy.exc.MissingGreenlet, surfaced to the frontend as "Could not
    # reach the backend". eager_defaults fetches server defaults via RETURNING in the
    # same flush statement instead, so updated_at/created_at are already populated and
    # never need that later reload at all — inherited by every subclass since
    # __mapper_args__ is a normal (overridable) class attribute in SQLAlchemy declarative.
    __mapper_args__ = {"eager_defaults": True}


class IdMixin:
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        # Explicit pool sizing (P2.05): parallel preconditions fan a turn out to several
        # side-sessions at once, so the default size-5 pool would force cold overflow
        # connections under any concurrency. pool_size holds the warm connections primed at
        # startup; max_overflow is burst headroom. pool_pre_ping keeps a connection the server
        # dropped while idle from surfacing as a mid-call error -- it costs one lightweight
        # round-trip on checkout, which is why a warm pool (not just a big one) matters.
        _engine = create_async_engine(
            settings.database_url, pool_pre_ping=True,
            pool_size=settings.db_pool_size, max_overflow=settings.db_max_overflow)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


async def get_db():
    async with get_session_factory()() as session:
        yield session


async def warm_pool(connections: int) -> int:
    """Open `connections` pooled DB connections CONCURRENTLY so they return to the pool warm
    (P2.05 support). Returns how many were actually primed.

    Parallel preconditions fan a booking turn out to several side-sessions at once; if the pool
    holds only one warm connection, the rest are built cold mid-call -- a fresh TLS handshake to
    the remote Postgres that surfaced live as a ~145ms precondition spike. Priming must be
    CONCURRENT: opening connections one after another just hands back the same single connection
    each time and warms nothing. A barrier holds every connection open at the same instant, so
    the pool is forced to create that many distinct physical connections, which then go back
    into it warm.

    Best-effort by contract: the caller runs this in the background at startup and logs the
    count; it never raises for a routine connectivity hiccup beyond what the driver itself
    raises, which the caller already guards.
    """
    factory = get_session_factory()
    # Warming past pool_size is wasted -- a connection beyond pool_size is an overflow one the
    # pool discards on return rather than keeping warm.
    warm = max(1, min(connections, get_settings().db_pool_size))

    if warm == 1 or not hasattr(asyncio, "Barrier"):     # Barrier is Python 3.11+
        async with factory() as db:
            await db.execute(text("SELECT 1"))            # connect + pool + TLS
        return 1

    barrier = asyncio.Barrier(warm)

    async def _prime() -> None:
        async with factory() as db:
            await db.execute(text("SELECT 1"))
            try:
                # Rendezvous so all `warm` connections are checked out at once; never hang boot
                # on a straggler -- a connection released early is still returned to the pool warm.
                await asyncio.wait_for(barrier.wait(), timeout=10)
            except (asyncio.TimeoutError, asyncio.BrokenBarrierError):
                pass

    await asyncio.gather(*[_prime() for _ in range(warm)])
    return warm
