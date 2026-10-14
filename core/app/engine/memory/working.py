"""Working memory — the live scratchpad for an in-progress conversation.

Redis-backed, one hash per conversation. Holds what must survive between turns but
not beyond the call: the pending confirmation, identity-verification state, turn
counter, and recent actions. Everything durable is written to Postgres separately,
so losing Redis degrades a call rather than losing it.
"""
import json
import logging
import uuid
from typing import Any

import redis.asyncio as aioredis

from app.config import get_settings

logger = logging.getLogger(__name__)

TTL_SECONDS = 60 * 60  # call duration + 1h
_LOCK_TTL_SECONDS = 5   # per-conversation write lock timeout

_client: aioredis.Redis | None = None


def _redis() -> aioredis.Redis:
    global _client
    if _client is None:
        _client = aioredis.from_url(get_settings().redis_url, decode_responses=True)
    return _client


def _key(conversation_id: uuid.UUID) -> str:
    return f"session:{conversation_id}"


def _lock_key(conversation_id: uuid.UUID) -> str:
    return f"session:{conversation_id}:write_lock"


async def load(conversation_id: uuid.UUID) -> dict[str, Any]:
    try:
        raw = await _redis().get(_key(conversation_id))
    except Exception:
        return _empty()
    if not raw:
        return _empty()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return _empty()


async def save(conversation_id: uuid.UUID, session: dict[str, Any]) -> None:
    """Write session to Redis under a per-conversation lock to prevent concurrent overwrites.

    The lock is a best-effort guard: if acquiring it fails (Redis down, timeout) we write
    anyway so a degraded Redis never silently swallows a session update.
    """
    r = _redis()
    lock_acquired = False
    try:
        # NX = only set if not exists (atomic test-and-set).
        lock_acquired = await r.set(_lock_key(conversation_id), "1",
                                    nx=True, ex=_LOCK_TTL_SECONDS)
        if not lock_acquired:
            # Another coroutine is writing; log and proceed anyway — best effort.
            logger.warning("working.save: lock contention on %s, writing without lock",
                           conversation_id)
        await r.set(_key(conversation_id), json.dumps(session, default=str), ex=TTL_SECONDS)
    except Exception:
        pass  # best effort — the conversation still works from Postgres
    finally:
        if lock_acquired:
            try:
                await r.delete(_lock_key(conversation_id))
            except Exception:
                pass


async def clear(conversation_id: uuid.UUID) -> None:
    try:
        await _redis().delete(_key(conversation_id))
    except Exception:
        pass


def _empty() -> dict[str, Any]:
    return {"pending_action": None, "identity_verified": False,
            "turn_index": 0, "recent_actions": [], "escalated": False,
            # What the call is currently ABOUT — carried across turns so a follow-up
            # retrieves the right variant. Sessions written before this key existed read
            # back without it, and every use is `.get(...) or {}`, so in-flight calls are
            # unaffected.
            "active_entities": {},
            # The BookingTask projection (see engine.planner.booking_task). One per
            # conversation, rebuilt each turn. None until a booking is in flight; sessions
            # written before this key existed read back without it, and every use is
            # `.get("booking_task")`, so in-flight calls are unaffected.
            "booking_task": None,
            # Tracks how many times the engine has proactively re-engaged a silent caller.
            # Used by conversational_signal.classify() to tighten the BACKCHANNEL threshold
            # after repeated re-engagements. Missing on sessions written before this key
            # existed; all reads use `.get("reengage_count", 0)`.
            "reengage_count": 0}