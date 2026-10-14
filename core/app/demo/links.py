"""Private demo links (docs/gtm-demo-links.md §8.1, §9).

One link per COMPANY RECORD. Everyone emailed about that company gets the same link, each
with their own recipient code, so a visit is attributed to a person without their email
address ever appearing in a URL (URLs end up in proxy and mail-scanner logs, and travel
with a forwarded email):

    {DEMO_PUBLIC_BASE_URL}/t/{token}?r={ref_code}&utm_source=…&utm_content=e1

  * The token is 128 bits from secrets.token_urlsafe(16). Lookup is by its SHA-256. It is
    also stored encrypted (DEMO_LINK_TOKEN_KEYS) so the link can be exported again for a
    person added later; a database dump alone does not reveal it.
  * One active link per record (partial unique index): making a new one revokes the old,
    and its recipients move to the new link with fresh codes.
  * A recipient code is attribution, not a credential: without the token it opens nothing,
    and a code of another link is ignored.
  * A link past expires_at is expired whatever its stored status says; the hourly job
    (phase 7) only tidies the column.
"""
import hashlib
import hmac
import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

from cryptography.fernet import Fernet, InvalidToken, MultiFernet
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import DemoBatch, DemoContact, DemoLink, DemoLinkRecipient, DemoProspect
from app.demo.records import Conflict, Invalid, NotFound, load_definition
from app.demo.settings import csv_list

TOKEN_BYTES = 16
REF_BYTES = 7                       # → 10 url-safe characters, 56 bits
LIMITS = ("max_calls_per_day", "max_call_seconds", "max_total_minutes", "max_concurrent_calls")


def new_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def new_ref_code() -> str:
    return secrets.token_urlsafe(REF_BYTES)


def hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ── token encryption ─────────────────────────────────────────────────────────────────────

def _fernet() -> MultiFernet:
    keys = csv_list(get_settings().demo_link_token_keys)
    if not keys:
        raise Conflict("DEMO_LINK_TOKEN_KEYS is not set, so links can't be created or exported")
    return MultiFernet([Fernet(k.encode()) for k in keys])


def encrypt_token(raw: str) -> str:
    return _fernet().encrypt(raw.encode()).decode()


def reveal_token(link: DemoLink) -> str:
    """The raw token of a link, for exporting its URLs. Refused for links made before
    tokens were kept (0039) or whose key has been removed from DEMO_LINK_TOKEN_KEYS."""
    if not link.token_ciphertext:
        raise Conflict("this link was made before links could be exported again; make a new one")
    try:
        raw = _fernet().decrypt(link.token_ciphertext.encode()).decode()
    except InvalidToken:
        raise Conflict("this link's token can't be decrypted with DEMO_LINK_TOKEN_KEYS "
                       "(was its key removed?); make a new link") from None
    if not hmac.compare_digest(hash_token(raw), link.token_hash):
        raise Conflict("this link's stored token doesn't match its hash; make a new link")
    return raw


# ── URLs ─────────────────────────────────────────────────────────────────────────────────

def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:100] or "demo"


def campaign_for(prospect: DemoProspect, batch: DemoBatch | None) -> str:
    """utm_campaign: the batch name, else DEMO_UTM_CAMPAIGN, else demo_<industry>."""
    if batch is not None:
        return _slug(batch.name)
    return get_settings().demo_utm_campaign or f"demo_{prospect.industry_key}"


def link_url(raw: str, *, ref: str | None = None, step: str | None = None,
             campaign: str | None = None) -> str:
    """The URL a person is sent. Without `ref` it is the plain company link (visits on it
    are "unknown" until the visitor says who they are)."""
    s = get_settings()
    params: dict[str, str] = {}
    if ref:
        params["r"] = ref
    if s.demo_utm_enabled:
        linkedin = step == "li"
        params["utm_source"] = "linkedin" if linkedin else s.demo_utm_source
        params["utm_medium"] = "social" if linkedin else s.demo_utm_medium
        if campaign:
            params["utm_campaign"] = campaign
        if step:
            params["utm_content"] = step
    elif step:
        params["s"] = step
    base = f"{s.demo_public_base_url.rstrip('/')}/t/{raw}"
    return f"{base}?{urlencode(params)}" if params else base


def effective_status(link: DemoLink, now: datetime | None = None) -> str:
    """active | expired | revoked — expiry is judged by the clock, not only the column."""
    if link.status == "revoked":
        return "revoked"
    if link.status == "expired" or link.expires_at <= (now or now_utc()):
        return "expired"
    return "active"


# ── lookups ──────────────────────────────────────────────────────────────────────────────

async def find_by_token(db: AsyncSession, raw: str) -> DemoLink | None:
    """The link for a raw token, or None. Lookup is by the hash's unique index, so the
    comparison is on an indexed equality of digests, never on the secret itself."""
    if not raw or len(raw) > 256:
        return None
    digest = hash_token(raw)
    link = (await db.execute(select(DemoLink).where(DemoLink.token_hash == digest))).scalar_one_or_none()
    if link is not None and not hmac.compare_digest(link.token_hash, digest):
        return None
    return link


async def get_link(db: AsyncSession, link_id: uuid.UUID) -> DemoLink:
    link = await db.get(DemoLink, link_id)
    if link is None:
        raise NotFound("link not found")
    return link


async def active_link(db: AsyncSession, prospect_id: uuid.UUID) -> DemoLink | None:
    return (await db.execute(select(DemoLink).where(
        DemoLink.prospect_id == prospect_id, DemoLink.status == "active"))).scalar_one_or_none()


# ── issuing ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IssuedLink:
    link: DemoLink
    prospect: DemoProspect
    raw: str                         # the token: never logged, never stored in clear
    created: bool                    # False = the record's existing active link


def _check_limits(limits: dict | None) -> dict:
    limits = {k: v for k, v in (limits or {}).items() if v is not None}
    for k, v in limits.items():
        if k not in LIMITS:
            raise Invalid(f"unknown limit {k!r}")
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise Invalid(f"{k} must be a positive whole number")
    return limits


async def issue_link(db: AsyncSession, prospect: DemoProspect, *, replace: bool = False,
                     ttl_days: int | None = None, limits: dict | None = None,
                     created_by: uuid.UUID | None = None) -> IssuedLink:
    """The record's company link. Returns its current active link if it has a usable one
    (unless replace=True); otherwise makes a new one, revoking the old and moving its
    recipients over with fresh codes (the old URLs stop working)."""
    if prospect.tenant_id is None:
        raise Conflict(f"{prospect.company_name} has no demo yet")
    definition = await load_definition(db, prospect.industry_key, prospect.industry_version)
    if not definition.has_agent:
        raise Conflict(f"{prospect.company_name}: {definition.label} has no demo agent yet "
                       "(data-only industry), so it can't have a demo link")
    s = get_settings()
    ttl = ttl_days or s.demo_link_ttl_days
    if not 1 <= ttl <= 90:
        raise Invalid("ttl_days must be between 1 and 90")
    limits = _check_limits(limits)
    now = now_utc()

    current = await active_link(db, prospect.id)
    if current is not None and not replace and effective_status(current, now) == "active":
        try:
            return IssuedLink(link=current, prospect=prospect, raw=reveal_token(current), created=False)
        except Conflict:
            pass                     # not exportable: replace it below

    carried: list[uuid.UUID] = []
    if current is not None:
        carried = list((await db.execute(select(DemoLinkRecipient.contact_id).where(
            DemoLinkRecipient.demo_link_id == current.id)
            .order_by(DemoLinkRecipient.created_at))).scalars())
        current.status, current.revoked_at = "revoked", now
        current.revoked_reason = "replaced by a new link"
        await db.flush()             # the partial unique index allows one active link per record

    raw = new_token()
    link = DemoLink(prospect_id=prospect.id, token_hash=hash_token(raw), token_prefix=raw[:6],
                    token_ciphertext=encrypt_token(raw), status="active", activated_at=now,
                    expires_at=now + timedelta(days=ttl), created_by=created_by, **limits)
    db.add(link)
    await db.flush()
    for contact_id in carried:
        db.add(DemoLinkRecipient(demo_link_id=link.id, contact_id=contact_id,
                                 ref_code=new_ref_code(), created_by=created_by))
    await db.flush()
    return IssuedLink(link=link, prospect=prospect, raw=raw, created=True)


async def ensure_recipient(db: AsyncSession, link: DemoLink, contact: DemoContact, *,
                           created_by: uuid.UUID | None = None) -> DemoLinkRecipient:
    """This person's recipient code on this link (made on first use)."""
    if contact.status != "active":
        raise Conflict(f"{contact.full_name} is excluded; re-activate them first")
    existing = (await db.execute(select(DemoLinkRecipient).where(
        DemoLinkRecipient.demo_link_id == link.id,
        DemoLinkRecipient.contact_id == contact.id))).scalar_one_or_none()
    if existing is not None:
        return existing
    r = DemoLinkRecipient(demo_link_id=link.id, contact_id=contact.id, ref_code=new_ref_code(),
                          created_by=created_by)
    db.add(r)
    await db.flush()
    return r


async def recipients_of(db: AsyncSession, link_id: uuid.UUID) -> list[tuple[DemoLinkRecipient, DemoContact]]:
    return list((await db.execute(
        select(DemoLinkRecipient, DemoContact).join(DemoContact, DemoContact.id == DemoLinkRecipient.contact_id)
        .where(DemoLinkRecipient.demo_link_id == link_id)
        .order_by(DemoContact.full_name))).all())


def export_steps() -> list[str]:
    return csv_list(get_settings().demo_export_steps)


def recipient_urls(raw: str, ref: str, campaign: str) -> dict[str, str]:
    """{step: URL} for one person, one per DEMO_EXPORT_STEPS entry."""
    return {step: link_url(raw, ref=ref, step=step, campaign=campaign) for step in export_steps()}


def link_out(link: DemoLink, *, prospect: DemoProspect | None = None) -> dict:
    out = {"id": str(link.id), "record_id": str(link.prospect_id),
           "token_prefix": link.token_prefix, "status": effective_status(link),
           "exportable": bool(link.token_ciphertext),
           "activated_at": link.activated_at, "expires_at": link.expires_at,
           "revoked_at": link.revoked_at, "revoked_reason": link.revoked_reason,
           "limits": {k: getattr(link, k) for k in LIMITS}}
    if prospect is not None:
        out["record"] = {"company_name": prospect.company_name}
    return out
