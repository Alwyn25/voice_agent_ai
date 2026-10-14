"""Password hashing, JWT issuing/verification, API key hashing."""
import asyncio
import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.config import get_settings


# ---- passwords ----
#
# bcrypt is INTENTIONALLY slow — it burns ~250-400 ms of CPU at the default cost factor,
# which is what makes a stolen hash expensive to brute-force. That same cost is a hazard on
# an async server: called directly inside a coroutine it blocks the whole event loop for the
# duration, freezing every other in-flight request — including live voice turns sharing the
# loop. The P1.08 loop monitor caught exactly this: a ~400 ms stall on the /auth/login path
# (see docs/bridge-latency.md).
#
# The two flavours below exist for that reason:
#   - hash_password / verify_password       : synchronous, for sync callers (seed scripts,
#                                             tests) where blocking is fine because there is
#                                             no event loop to starve.
#   - hash_password_async / verify_password_async : run the SAME work in a worker thread via
#                                             asyncio.to_thread, so the loop stays free. Async
#                                             request handlers MUST use these. bcrypt's C
#                                             routine releases the GIL while hashing, so the
#                                             thread runs truly in parallel and the loop keeps
#                                             servicing other tasks.

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except ValueError:
        return False


async def hash_password_async(plain: str) -> str:
    """Hash off the event loop — use this from any async handler (see note above)."""
    return await asyncio.to_thread(hash_password, plain)


async def verify_password_async(plain: str, hashed: str) -> bool:
    """Verify off the event loop — use this from any async handler (see note above)."""
    return await asyncio.to_thread(verify_password, plain, hashed)


# ---- JWT ----

def _now() -> datetime:
    return datetime.now(timezone.utc)


def create_token(*, user_id: uuid.UUID, tenant_id: uuid.UUID, role: str, token_type: str) -> str:
    s = get_settings()
    if token_type == "access":
        exp = _now() + timedelta(minutes=s.access_token_minutes)
    else:
        exp = _now() + timedelta(days=s.refresh_token_days)
    payload = {
        "sub": str(user_id),
        "tenant_id": str(tenant_id),
        "role": role,
        "type": token_type,
        "iat": _now(),
        "exp": exp,
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_token(token: str, *, expected_type: str) -> dict:
    """Raises jwt.PyJWTError on any problem; caller maps to 401."""
    s = get_settings()
    payload = jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
    if payload.get("type") != expected_type:
        raise jwt.InvalidTokenError(f"expected {expected_type} token")
    return payload


# ---- API keys ----

def generate_api_key() -> tuple[str, str]:
    """Returns (plaintext_key, hash). Plaintext is shown once at creation."""
    plain = "vak_" + secrets.token_urlsafe(32)
    return plain, hash_api_key(plain)


def hash_api_key(plain: str) -> str:
    return hashlib.sha256(plain.encode()).hexdigest()
