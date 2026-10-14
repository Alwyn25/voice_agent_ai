"""Unit tests for the demo abuse limits that need no database (app/demo/limits.py):

  client IP     taken from our own proxy's entry, never the client-written part of
                X-Forwarded-For; edge headers; fallbacks
  rate limit    keeps counting in process while Redis is down
  Turnstile     fails closed; outside local the token must come from a demo origin and
                carry this action

    cd core && python -m pytest tests/test_demo_limits.py -q
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.demo import limits  # noqa: E402


@pytest.fixture
def settings(monkeypatch):
    """Point limits at a Settings built here; tests tweak it via the returned setter."""
    current = {"s": Settings(_env_file=None)}

    def use(**overrides):
        current["s"] = Settings(_env_file=None, **overrides)
        return current["s"]

    monkeypatch.setattr(limits, "get_settings", lambda: current["s"])
    import app.demo.settings  # noqa: F401  (csv_list, used by _demo_hostnames)
    return use


def _req(headers: dict | None = None, peer: str = "10.0.0.7"):
    return SimpleNamespace(headers={k.lower(): v for k, v in (headers or {}).items()},
                           client=SimpleNamespace(host=peer))


# ── client IP ────────────────────────────────────────────────────────────────────────────

def test_ip_is_our_proxys_entry_not_the_clients(settings):
    settings(demo_trusted_proxy_hops=1)
    # A client sends its own X-Forwarded-For; Railway appends the real address.
    req = _req({"X-Forwarded-For": "6.6.6.6, 1.2.3.4, 203.0.113.9"})
    assert limits.client_ip(req) == "203.0.113.9"


def test_spoofed_entries_all_land_in_one_bucket(settings):
    settings(demo_trusted_proxy_hops=1)
    ips = {limits.client_ip(_req({"X-Forwarded-For": f"{n}.{n}.{n}.{n}, 203.0.113.9"}))
           for n in range(1, 50)}
    assert ips == {"203.0.113.9"}


def test_two_proxies(settings):
    settings(demo_trusted_proxy_hops=2)      # Cloudflare → Railway
    req = _req({"X-Forwarded-For": "6.6.6.6, 198.51.100.4, 172.68.1.1"})
    assert limits.client_ip(req) == "198.51.100.4"


def test_edge_header(settings):
    settings(demo_client_ip_header="cf-connecting-ip")
    assert limits.client_ip(_req({"CF-Connecting-IP": "2001:db8::1"})) == "2001:db8::1"


def test_hops_zero_is_the_socket_peer(settings):
    settings(demo_trusted_proxy_hops=0)
    assert limits.client_ip(_req({"X-Forwarded-For": "6.6.6.6"}, peer="192.0.2.5")) == "192.0.2.5"


@pytest.mark.parametrize("xff", [None, "", "not-an-ip", " , "])
def test_bad_or_missing_header_falls_back_to_peer(settings, xff):
    settings(demo_trusted_proxy_hops=1)
    headers = {} if xff is None else {"X-Forwarded-For": xff}
    assert limits.client_ip(_req(headers, peer="192.0.2.5")) == "192.0.2.5"


def test_chain_shorter_than_hops_falls_back_to_peer(settings):
    settings(demo_trusted_proxy_hops=2)
    assert limits.client_ip(_req({"X-Forwarded-For": "203.0.113.9"}, peer="192.0.2.5")) == "192.0.2.5"


# ── rate limit without Redis ─────────────────────────────────────────────────────────────

class _DownRedis:
    async def incr(self, k):
        raise ConnectionError("redis down")


def test_rate_limit_counts_in_process_when_redis_is_down(settings, monkeypatch):
    settings()
    monkeypatch.setattr(limits, "_client", lambda: _DownRedis())
    limits._LOCAL_COUNTS.clear()

    async def run():
        for _ in range(3):
            await limits.rate_limit("t", "ip-a", limit=3, window_s=60)
        with pytest.raises(limits.LimitHit) as e:
            await limits.rate_limit("t", "ip-a", limit=3, window_s=60)
        assert e.value.code == "rate_limited"
        await limits.rate_limit("t", "ip-b", limit=3, window_s=60)     # another IP: own count
    asyncio.run(run())
    limits._LOCAL_COUNTS.clear()


def test_local_counter_window_expires(monkeypatch):
    limits._LOCAL_COUNTS.clear()
    clock = {"t": 1000.0}
    monkeypatch.setattr(limits.time, "monotonic", lambda: clock["t"])
    assert [limits._local_incr("k", 60) for _ in range(3)] == [1, 2, 3]
    clock["t"] += 70
    assert limits._local_incr("k", 60) == 1
    limits._LOCAL_COUNTS.clear()


def test_local_counter_memory_is_bounded(monkeypatch):
    limits._LOCAL_COUNTS.clear()
    monkeypatch.setattr(limits, "_LOCAL_MAX_KEYS", 100)
    for i in range(250):
        limits._local_incr(f"k{i}", 60)
    assert len(limits._LOCAL_COUNTS) <= 100
    limits._LOCAL_COUNTS.clear()


# ── Turnstile ────────────────────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def _cloudflare(monkeypatch, body=None, fail=False):
    class Client:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, data):
            if fail:
                raise TimeoutError("cloudflare unreachable")
            return _Resp(body)
    monkeypatch.setattr(limits.httpx, "AsyncClient", Client)


PROD = dict(env="production", demo_turnstile_enabled=True, demo_turnstile_secret="s",
            demo_allowed_origins="https://demo.phonn.ai")


def _verify(token="tok", action="session"):
    asyncio.run(limits.verify_turnstile(token, "203.0.113.9", action=action))


def test_turnstile_off(settings):
    settings(demo_turnstile_enabled=False)
    _verify(token=None)


def test_turnstile_missing_token(settings):
    settings(**PROD)
    with pytest.raises(limits.LimitHit) as e:
        _verify(token=None)
    assert e.value.code == "bot_check" and e.value.status == 403


def test_turnstile_passes_for_our_origin_and_action(settings, monkeypatch):
    settings(**PROD)
    _cloudflare(monkeypatch, {"success": True, "hostname": "demo.phonn.ai", "action": "demo_session"})
    _verify()


@pytest.mark.parametrize("body", [
    {"success": False, "error-codes": ["invalid-input-response"]},
    {"success": True, "hostname": "evil.example", "action": "demo_session"},     # solved elsewhere
    {"success": True, "hostname": "demo.phonn.ai", "action": "demo_contact"},    # other widget
    {"success": True, "hostname": "demo.phonn.ai"},                              # no action
])
def test_turnstile_refusals(settings, monkeypatch, body):
    settings(**PROD)
    _cloudflare(monkeypatch, body)
    with pytest.raises(limits.LimitHit) as e:
        _verify()
    assert e.value.code == "bot_check"


def test_turnstile_fails_closed_when_cloudflare_is_down(settings, monkeypatch):
    settings(**PROD)
    _cloudflare(monkeypatch, fail=True)
    with pytest.raises(limits.LimitHit):
        _verify()


def test_turnstile_local_accepts_test_key_responses(settings, monkeypatch):
    """Cloudflare's always-pass test keys report hostname example.com and no action."""
    settings(**{**PROD, "env": "local"})
    _cloudflare(monkeypatch, {"success": True, "hostname": "example.com"})
    _verify()
