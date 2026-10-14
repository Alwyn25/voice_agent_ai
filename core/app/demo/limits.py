"""Abuse and cost limits for public demo calls (docs/gtm-demo-links.md §8.4). Every value
comes from the DEMO_* settings; a link may override the per-link ones.

  per IP          requests per minute on /public/demo, new calls per hour     (Redis; in-process
                                                                               if Redis is down)
  per link        calls today, calls at once, lifetime minutes                 (database)
  per IP per link call time across every browser on one network                (database)
  global          calls at once across every demo                               (database)
  spend           today's demo spend vs DEMO_DAILY_BUDGET_USD                   (database)
  bot check       Cloudflare Turnstile (required outside local development)

The caller's IP is the one our own proxies recorded (DEMO_CLIENT_IP_HEADER /
DEMO_TRUSTED_PROXY_HOPS), never the part of X-Forwarded-For the client can write.

A call counts as "live" from the moment its session is issued until it ends, but never for
longer than its maximum length plus a margin, so a crashed call can't hold a slot forever.
The database limits are checked and the session inserted under one lock, so two requests at
the same moment can't both take the last slot.
"""
import hashlib
import hmac
import ipaddress
import logging
import time
from urllib.parse import urlsplit
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import httpx
from sqlalchemy import func, or_, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import DecisionTrace, DemoLink, DemoSession, Tenant
from app.db.models.platform import TENANT_KIND_DEMO

log = logging.getLogger(__name__)
_LIVE_MARGIN_S = 120
VOICE_TOKEN_TTL_S = 120          # a demo voice token must be used within this (app/demo/voice.py)
_BUDGET_CACHE: dict[str, tuple[float, Decimal]] = {}      # day → (computed_at, usd)
_BUDGET_TTL_S = 60


class LimitHit(Exception):
    """A limit refused the request. `code` is stable for the UI; `message` is shown."""

    def __init__(self, code: str, message: str, *, status: int = 429):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


BUSY = "The demo is busy right now. Please try again in a few minutes."


# ── IP ───────────────────────────────────────────────────────────────────────────────────

_last_warned: dict[str, float] = {}


def _warn_every(key: str, seconds: float, msg: str, *args, level=logging.WARNING, **kw) -> None:
    """Log at most once per `seconds` per key: a broken dependency must not flood the logs
    with one line per request."""
    now = time.monotonic()
    if now - _last_warned.get(key, -seconds) >= seconds:
        _last_warned[key] = now
        log.log(level, msg, *args, **kw)


def _valid_ip(value: str) -> str | None:
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        return None


def client_ip(request) -> str:
    """The caller's IP, as our own proxies saw it (DEMO_CLIENT_IP_HEADER /
    DEMO_TRUSTED_PROXY_HOPS in app/config.py).

    Each proxy appends the address it received the request from to X-Forwarded-For, so with
    N proxies the client is the N-th entry from the right. Everything to its left was sent by
    the client and is ignored: taking the first entry would let anyone pick a new IP per
    request and walk past every per-IP limit. A header that is missing, too short or not an
    IP falls back to the socket peer (logged: it means the hop count is wrong)."""
    s = get_settings()
    peer = request.client.host if request.client else "0.0.0.0"
    header = s.demo_client_ip_header.strip().lower()
    hops = s.demo_trusted_proxy_hops
    if hops <= 0 or not header:
        return peer
    raw = request.headers.get(header)
    if raw:
        if header == "x-forwarded-for":
            chain = [p.strip() for p in raw.split(",") if p.strip()]
            ip = _valid_ip(chain[-hops]) if len(chain) >= hops else None
        else:
            ip = _valid_ip(raw.split(",")[-1])
        if ip:
            return ip
    _warn_every("client_ip", 300, "demo client IP: %s is missing or has fewer than %d entries; "
                "using the socket peer. Check DEMO_CLIENT_IP_HEADER / DEMO_TRUSTED_PROXY_HOPS.",
                header, hops)
    return peer


def ip_hash(ip: str) -> str:
    """Raw IPs are never stored; a keyed hash still lets limits and abuse checks group by IP."""
    return hmac.new(get_settings().demo_ip_hash_secret.encode(), ip.encode(), hashlib.sha256).hexdigest()


# ── Redis rate limits ────────────────────────────────────────────────────────────────────

_redis = None


def _client():
    global _redis
    if _redis is None:
        import redis.asyncio as aioredis
        _redis = aioredis.from_url(get_settings().redis_url, decode_responses=True)
    return _redis


_LOCAL_COUNTS: dict[str, tuple[int, float]] = {}     # key → (count, monotonic expiry)
_LOCAL_MAX_KEYS = 50_000


def _local_incr(k: str, window_s: int) -> int:
    """The same fixed-window counter in this process's memory, used while Redis is down.
    Per process rather than shared, so with several replicas the effective limit is a few
    times higher, but it still stops one client from hammering the API."""
    now = time.monotonic()
    if len(_LOCAL_COUNTS) >= _LOCAL_MAX_KEYS:
        for old in [x for x, (_, exp) in _LOCAL_COUNTS.items() if exp <= now]:
            del _LOCAL_COUNTS[old]
        if len(_LOCAL_COUNTS) >= _LOCAL_MAX_KEYS:     # a flood of distinct IPs: start over
            _LOCAL_COUNTS.clear()
    n, exp = _LOCAL_COUNTS.get(k, (0, now + window_s + 5))
    if exp <= now:
        n, exp = 0, now + window_s + 5
    _LOCAL_COUNTS[k] = (n + 1, exp)
    return n + 1


async def rate_limit(bucket: str, key: str, *, limit: int, window_s: int) -> None:
    """Fixed-window counter in Redis. If Redis is unreachable the count moves to this
    process's memory (logged): a Redis outage neither takes every demo down nor removes
    the per-IP limits."""
    k = f"demo:rl:{bucket}:{key}:{int(time.time() // window_s)}"
    try:
        r = _client()
        n = await r.incr(k)
        if n == 1:
            await r.expire(k, window_s + 5)
    except Exception:  # noqa: BLE001
        _warn_every("redis_rl", 60, "demo rate limit: redis unavailable, counting in process",
                    exc_info=True)
        n = _local_incr(k, window_s)
    if n > limit:
        raise LimitHit("rate_limited", "Too many requests. Please wait a moment and try again.")


async def burn_once(key: str, ttl_s: int) -> bool:
    """True the first time `key` is seen within ttl_s, False after (single-use tokens).
    Fails CLOSED: without Redis a token can't be proven unused, so it's refused."""
    try:
        return bool(await _client().set(f"demo:once:{key}", "1", nx=True, ex=ttl_s))
    except Exception:  # noqa: BLE001
        log.error("demo single-use check failed: redis unavailable", exc_info=True)
        return False


# ── Turnstile ────────────────────────────────────────────────────────────────────────────

TURNSTILE_ACTIONS = {"session": "demo_session", "contact": "demo_contact"}   # the page sets these


def _demo_hostnames() -> set[str]:
    from app.demo.settings import csv_list
    return {(urlsplit(o).hostname or "").lower()
            for o in csv_list(get_settings().demo_allowed_origins)} - {""}


async def verify_turnstile(token: str | None, ip: str, *, action: str) -> None:
    """Cloudflare's verdict on the page's Turnstile token. Fails CLOSED: if Cloudflare can't
    be reached the call or form is refused, since letting everything through is exactly what
    a bot would wait for.

    Outside local development the token must also have been issued on one of our demo
    origins (so a token solved on someone else's page can't be replayed here) and for this
    action (so a token from the contact form can't start a call). Cloudflare's test keys
    report a placeholder hostname, which is why local skips those two checks."""
    s = get_settings()
    if not s.demo_turnstile_enabled:
        return
    if not token:
        raise LimitHit("bot_check", "Please complete the check and try again.", status=403)
    refused = LimitHit("bot_check", "We couldn't verify you're a person. Please try again.", status=403)
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.post("https://challenges.cloudflare.com/turnstile/v0/siteverify",
                                  data={"secret": s.demo_turnstile_secret, "response": token,
                                        "remoteip": ip})
            result = r.json()
    except Exception:  # noqa: BLE001
        _warn_every("turnstile", 60, "turnstile verification unavailable", exc_info=True)
        raise refused from None
    if not result.get("success"):
        log.info("turnstile refused: %s", result.get("error-codes"))
        raise refused
    if s.env != "local":
        hostname = (result.get("hostname") or "").lower()
        if hostname not in _demo_hostnames():
            log.warning("turnstile token from another site refused (hostname %r)", hostname)
            raise refused
        if result.get("action") != TURNSTILE_ACTIONS[action]:
            log.warning("turnstile token for another action refused (%r, wanted %r)",
                        result.get("action"), TURNSTILE_ACTIONS[action])
            raise refused


# ── call limits ──────────────────────────────────────────────────────────────────────────

def link_limit(link: DemoLink, name: str) -> int:
    s = get_settings()
    default = {"max_calls_per_day": s.demo_max_calls_per_link_per_day,
               "max_call_seconds": s.demo_max_call_seconds,
               "max_total_minutes": s.demo_max_minutes_per_link,
               "max_concurrent_calls": s.demo_max_concurrent_calls_per_link}[name]
    return getattr(link, name) or default


def _live_clause(now: datetime):
    """Sessions holding a call slot. A connected call counts until it ends (at most its
    maximum length plus a margin, in case the process died mid-call). A session that never
    connected (microphone denied, tab closed) stops counting once its voice token has
    expired, so it can't block the link for minutes."""
    s = get_settings()
    oldest = now - timedelta(seconds=s.demo_max_call_seconds + _LIVE_MARGIN_S)
    unconnected_oldest = now - timedelta(seconds=VOICE_TOKEN_TTL_S + 30)
    return (DemoSession.ended_at.is_(None), DemoSession.created_at > oldest,
            or_(DemoSession.started_at.is_not(None), DemoSession.created_at > unconnected_oldest))


def _day_start(now: datetime) -> datetime:
    tz = ZoneInfo(get_settings().demo_budget_timezone)
    local = now.astimezone(tz)
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


async def spend_today_usd(db: AsyncSession, now: datetime) -> Decimal:
    """Today's demo spend, priced from the usage every engine turn records (tokens per
    stage, TTS characters, STT seconds) on decision_traces in demo tenants. Cached for a
    minute: a budget check per call start must not scan the day's traces every time."""
    from app.observability.pricing import turn_cost
    start = _day_start(now)
    key = start.date().isoformat()
    cached = _BUDGET_CACHE.get(key)
    if cached and time.monotonic() - cached[0] < _BUDGET_TTL_S:
        return cached[1]
    rows = (await db.execute(
        select(DecisionTrace.tokens, DecisionTrace.model_used, DecisionTrace.tts_characters,
               DecisionTrace.stt_seconds)
        .join(Tenant, Tenant.id == DecisionTrace.tenant_id)
        .where(Tenant.kind == TENANT_KIND_DEMO, DecisionTrace.created_at >= start))).all()
    total = Decimal("0")
    for tokens, model, tts, stt in rows:
        c = turn_cost(tokens_by_stage=tokens or {}, model_used=model, tts_characters=tts,
                      stt_seconds=float(stt) if stt is not None else None)
        total += c.llm_usd + c.tts_usd + c.stt_usd
    _BUDGET_CACHE.clear()
    _BUDGET_CACHE[key] = (time.monotonic(), total)
    return total


# pg_advisory_xact_lock key for issuing demo sessions (any fixed bigint; "demo" in ASCII).
_SESSION_LOCK_KEY = 0x64656D6F
_SESSION_LOCK_WAIT = "5s"


async def lock_session_issue(db: AsyncSession) -> None:
    """Serialise issuing demo sessions until this transaction ends (commit or rollback).

    The call limits count live sessions and the caller then inserts one; without this, two
    requests at the same moment both see "0 live" and both get the link's single slot (or
    the last global one). One lock for all links, because the global count spans them; it
    is held only for the checks and a few inserts, not for the call. Waiting is bounded: a
    stuck holder turns into "busy", not a hung request."""
    try:
        await db.execute(text(f"SET LOCAL lock_timeout = '{_SESSION_LOCK_WAIT}'"))
        await db.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": _SESSION_LOCK_KEY})
    except DBAPIError:
        log.warning("demo session lock not acquired within %s", _SESSION_LOCK_WAIT)
        raise LimitHit("busy", BUSY) from None


async def check_can_start_call(db: AsyncSession, link: DemoLink, *, now: datetime) -> None:
    """Raise LimitHit when this link may not start another call right now. Call under
    lock_session_issue, and insert the session in the same transaction."""
    s = get_settings()
    live = _live_clause(now)

    at_once = (await db.execute(select(func.count()).select_from(DemoSession).where(
        DemoSession.demo_link_id == link.id, *live))).scalar_one()
    if at_once >= link_limit(link, "max_concurrent_calls"):
        raise LimitHit("link_busy", "A call is already running on this link. End it before "
                                    "starting another.")

    global_live = (await db.execute(select(func.count()).select_from(DemoSession).where(*live))).scalar_one()
    if global_live >= s.demo_max_concurrent_calls_global:
        raise LimitHit("busy", BUSY)

    today = (await db.execute(select(func.count()).select_from(DemoSession).where(
        DemoSession.demo_link_id == link.id, DemoSession.created_at >= _day_start(now)))).scalar_one()
    if today >= link_limit(link, "max_calls_per_day"):
        raise LimitHit("daily_limit", "This demo link has reached today's call limit. "
                                      "Please try again tomorrow.")

    used_s = (await db.execute(select(func.coalesce(func.sum(DemoSession.duration_seconds), 0)).where(
        DemoSession.demo_link_id == link.id))).scalar_one()
    if used_s >= link_limit(link, "max_total_minutes") * 60:
        raise LimitHit("used_up", "This demo link has used all its call time. Reply to our "
                                  "email and we'll extend it.")

    if s.demo_daily_budget_usd > 0:
        spent = await spend_today_usd(db, now)
        if spent >= Decimal(str(s.demo_daily_budget_usd)):
            log.error("demo daily budget reached: $%s of $%s", spent, s.demo_daily_budget_usd)
            raise LimitHit("budget", BUSY)
        if spent >= Decimal(str(s.demo_daily_budget_usd)) * s.demo_budget_alert_pct / 100:
            log.warning("demo spend at %s%% of the daily budget ($%s of $%s)",
                        s.demo_budget_alert_pct, spent, s.demo_daily_budget_usd)
