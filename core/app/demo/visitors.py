"""Who opened a company link (docs/gtm-demo-links.md §9.1).

Every browser that opens a link becomes one DemoVisitor, keyed by the page's first-party
visitor id. Its attribution is decided from what it arrived with:

  * a recipient code (?r=) of THIS link, first browser to use it   → recipient
  * the same code, but another browser already used it             → shared (that person on
    a second device, or a colleague they forwarded the email to)
  * no code, or a code of another link                             → unknown
  * the visitor typed a name and email                             → self_identified; when
    the email is one of the company's contacts, the visitor becomes that person

First touch wins for how they arrived (code, email step, UTM tags); last_seen_at moves on
every touch. Nothing here runs on a plain GET of the page: mail scanners fetch links but
never run the page's script, so only real visits create visitors.
"""
import re
import uuid
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DemoContact, DemoEvent, DemoLink, DemoLinkRecipient, DemoVisitor
from app.demo.links import ensure_recipient
from app.demo.settings import STEP_RE

REF_RE = re.compile(r"^[A-Za-z0-9_-]{6,16}$")
UTM_KEYS = ("utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term")
_UTM_VALUE = re.compile(r"^[\w .:/+@-]{1,100}$")
EMAIL_RE = re.compile(r"^[^@\s<>\"',;]{1,64}@[A-Za-z0-9.-]{1,253}\.[A-Za-z]{2,63}$")


def clean_ref(ref: str | None) -> str | None:
    return ref if ref and REF_RE.match(ref) else None


def clean_step(step: str | None) -> str | None:
    return step if step and STEP_RE.match(step) else None


def clean_utm(utm: dict | None) -> dict:
    """Only the five standard UTM keys, short plain values; anything else is dropped."""
    out = {}
    for k in UTM_KEYS:
        v = (utm or {}).get(k)
        if isinstance(v, str) and _UTM_VALUE.match(v.strip()):
            out[k] = v.strip()
    return out


def step_from(source_step: str | None, utm: dict) -> str | None:
    """The email step: ?s= when UTM is off, utm_content when it's on."""
    return clean_step(source_step) or clean_step(utm.get("utm_content"))


async def _recipient_by_ref(db: AsyncSession, link: DemoLink, ref: str | None,
                            *, lock: bool = False) -> DemoLinkRecipient | None:
    if not ref:
        return None
    q = select(DemoLinkRecipient).where(DemoLinkRecipient.demo_link_id == link.id,
                                        DemoLinkRecipient.ref_code == ref)
    if lock:
        q = q.with_for_update()
    return (await db.execute(q)).scalar_one_or_none()


async def touch_visitor(db: AsyncSession, link: DemoLink, *, visitor_key: uuid.UUID,
                        now: datetime, ref: str | None = None, utm: dict | None = None,
                        source_step: str | None = None, viewed: bool = False,
                        ip_hash: str | None = None, user_agent: str | None = None) -> DemoVisitor:
    """The visitor for this browser on this link, created on first sight and attributed."""
    ref = clean_ref(ref)
    utm = clean_utm(utm)
    step = step_from(source_step, utm)
    await db.execute(
        pg_insert(DemoVisitor)
        .values(id=uuid.uuid4(), demo_link_id=link.id, visitor_key=visitor_key, attribution="unknown",
                first_seen_at=now, last_seen_at=now, landing_ref=ref, source_step=step, utm=utm,
                ip_hash=ip_hash, user_agent=(user_agent or "")[:400] or None)
        .on_conflict_do_nothing(constraint="uq_demo_visitors_link_key"))
    v = (await db.execute(select(DemoVisitor).where(
        DemoVisitor.demo_link_id == link.id, DemoVisitor.visitor_key == visitor_key)
        .with_for_update())).scalar_one()
    v.last_seen_at = max(v.last_seen_at, now)
    if viewed and v.first_viewed_at is None:
        v.first_viewed_at = now
    if v.landing_ref is None and ref:
        v.landing_ref = ref
    if v.source_step is None and step:
        v.source_step = step
    if not v.utm and utm:
        v.utm = utm
    if v.recipient_id is None and v.attribution == "unknown":
        # Locking the recipient makes "first browser on this code" exact under concurrency.
        r = await _recipient_by_ref(db, link, ref, lock=True)
        if r is not None:
            taken = (await db.execute(select(func.count()).select_from(DemoVisitor).where(
                DemoVisitor.recipient_id == r.id, DemoVisitor.attribution == "recipient",
                DemoVisitor.id != v.id))).scalar_one()
            v.recipient_id = r.id
            v.attribution = "shared" if taken else "recipient"
    await db.flush()
    return v


def should_ask_identity(v: DemoVisitor, mode: str) -> bool:
    if mode == "off" or v.identified_at is not None:
        return False
    return mode == "always" or v.attribution in ("unknown", "shared")


async def identify(db: AsyncSession, link: DemoLink, v: DemoVisitor, *, name: str, email: str,
                   role: str | None, now: datetime) -> DemoVisitor:
    """Record who a visitor says they are. If the email belongs to one of the company's
    contacts (or anyone this link was sent to), the visitor becomes that person; otherwise
    the visit stays credited to whoever's email brought them, with the stated identity."""
    email = email.strip().lower()
    v.identified_name = name.strip()[:120]
    v.identified_email = email[:254]
    v.identified_role = (role or "").strip()[:120] or None
    v.identified_at = now
    v.attribution = "self_identified"

    match = (await db.execute(
        select(DemoLinkRecipient).join(DemoContact, DemoContact.id == DemoLinkRecipient.contact_id)
        .where(DemoLinkRecipient.demo_link_id == link.id, func.lower(DemoContact.email) == email)
    )).scalars().first()
    if match is None:
        contact = (await db.execute(select(DemoContact).where(
            DemoContact.prospect_id == link.prospect_id, DemoContact.status == "active",
            func.lower(DemoContact.email) == email))).scalars().first()
        if contact is not None:
            match = await ensure_recipient(db, link, contact)
    if match is not None:
        v.recipient_id = match.id
    db.add(DemoEvent(demo_link_id=link.id, visitor_id=v.visitor_key, demo_visitor_id=v.id,
                     recipient_id=v.recipient_id, type="visitor_identified", occurred_at=now,
                     meta={"matched_contact": match is not None}))
    await db.flush()
    return v
