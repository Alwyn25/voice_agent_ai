"""Public demo API (docs/gtm-demo-links.md §8.2): what a prospect's demo page calls. No
login — the private link's token is the only credential, and it scopes everything.

  GET  /public/demo/{token}            page data (404 unknown or revoked, 410 expired)
  POST /public/demo/{token}/visit      "a person is looking at the page" beacon: records the
                                       open and who it is (recipient code, email step, UTM)
  POST /public/demo/{token}/identify   a visitor says who they are (optional form)
  POST /public/demo/{token}/session    start a voice call → short-lived voice token + ws_url
  POST /public/demo/{token}/events     other funnel beacons from the page
  POST /public/demo/{token}/contact    Contact us / Schedule a call / Run on real calls /
                                       Rebuild my demo (also on an expired link)

Rules for this surface:
  * GET never writes. Mail security scanners fetch every link in an email; an "open" is
    only the page's own /visit beacon after a person has seen it.
  * The link is per company; who a visit belongs to comes from the recipient code in that
    person's copy of the URL (app/demo/visitors.py). A code of another link is ignored.
  * Unknown and revoked links get the same 404, so a guessed token reveals nothing.
  * Responses are never indexed or cached, and carry no referrer.
  * Every limit is enforced here and again at the voice handshake (app/demo/limits.py).
"""
import re
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.base import get_db
from app.db.models import DemoEvent, DemoLink, DemoProspect, DemoSession, DemoVisitor
from app.demo import limits
from app.demo.agent import start_demo_conversation
from app.demo.links import effective_status, find_by_token
from app.demo.page import page_data
from app.demo.records import DemoError
from app.demo.leads import service as leads
from app.demo.visitors import EMAIL_RE, identify, should_ask_identity, touch_visitor
from app.services.phone import normalize_phone
from app.demo.voice import issue_voice_token

router = APIRouter(prefix="/public/demo", tags=["demo-public"])

PRIVATE_HEADERS = {"X-Robots-Tag": "noindex, nofollow, noarchive",
                   "Referrer-Policy": "no-referrer",
                   "Cache-Control": "no-store"}
# page_viewed is not here: an open is only ever recorded by /visit.
CLIENT_EVENTS = {"kb_tab_viewed", "sample_tab_viewed", "call_started",
                 "call_ended", "transcript_viewed", "cta_real_calls_clicked"}


class Arrival(BaseModel):
    """How this browser arrived, read from the URL by the page (then removed from the
    address bar, so a link copied from it doesn't carry someone else's code). Bad values
    are ignored rather than refused: a mangled URL must still open the demo."""
    visitor_id: uuid.UUID | None = None
    ref: str | None = Field(None, max_length=32)
    source_step: str | None = Field(None, max_length=10)
    utm: dict[str, str] | None = Field(None, max_length=10)


def _private(response: Response) -> None:
    response.headers.update(PRIVATE_HEADERS)


def _gone(detail: str = "This demo has expired") -> HTTPException:
    return HTTPException(status.HTTP_410_GONE, detail, headers=PRIVATE_HEADERS)


def _not_found() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, "This demo link isn't valid", headers=PRIVATE_HEADERS)


def _limit(e: limits.LimitHit) -> HTTPException:
    return HTTPException(e.status, {"code": e.code, "message": e.message}, headers=PRIVATE_HEADERS)


async def _enabled():
    if not get_settings().demo_enabled:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")


async def _active_link(db: AsyncSession, token: str) -> DemoLink:
    link = await find_by_token(db, token)
    if link is None or effective_status(link) == "revoked":
        raise _not_found()
    if effective_status(link) == "expired":
        raise _gone()
    return link


async def _throttle(request: Request, bucket: str, limit: int, window_s: int) -> str:
    ip = limits.client_ip(request)
    try:
        await limits.rate_limit(bucket, limits.ip_hash(ip), limit=limit, window_s=window_s)
    except limits.LimitHit as e:
        raise _limit(e) from None
    return ip


async def _visitor(db: AsyncSession, link: DemoLink, body: Arrival, request: Request, ip: str,
                   *, viewed: bool = False) -> DemoVisitor | None:
    if body.visitor_id is None:
        return None
    return await touch_visitor(db, link, visitor_key=body.visitor_id, now=datetime.now(timezone.utc),
                               ref=body.ref, utm=body.utm, source_step=body.source_step,
                               viewed=viewed, ip_hash=limits.ip_hash(ip),
                               user_agent=request.headers.get("user-agent"))


@router.get("/{token}", dependencies=[Depends(_enabled)])
async def get_page(token: str, request: Request, response: Response,
                   db: AsyncSession = Depends(get_db)):
    await _throttle(request, "page", get_settings().demo_rate_limit_per_min, 60)
    link = await _active_link(db, token)
    _private(response)
    return await page_data(db, link)


class VisitIn(Arrival):
    visitor_id: uuid.UUID


@router.post("/{token}/visit", dependencies=[Depends(_enabled)])
async def record_visit(token: str, body: VisitIn, request: Request, response: Response,
                       db: AsyncSession = Depends(get_db)):
    """The page's beacon once a person has seen it (visible for 2 s): the "opened" signal.
    Returns whether to show the optional "who's trying this?" card."""
    ip = await _throttle(request, "events", get_settings().demo_rate_limit_per_min, 60)
    link = await _active_link(db, token)
    v = await _visitor(db, link, body, request, ip, viewed=True)
    db.add(DemoEvent(demo_link_id=link.id, visitor_id=v.visitor_key, demo_visitor_id=v.id,
                     recipient_id=v.recipient_id, type="page_viewed", source_step=v.source_step,
                     meta={"utm": v.utm} if v.utm else {}))
    ask = should_ask_identity(v, get_settings().demo_identify_prompt)
    await db.commit()
    _private(response)
    return {"ask_identity": ask}


class IdentifyIn(BaseModel):
    visitor_id: uuid.UUID
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(max_length=254)
    role: str | None = Field(None, max_length=120)


@router.post("/{token}/identify", dependencies=[Depends(_enabled)])
async def identify_visitor(token: str, body: IdentifyIn, request: Request, response: Response,
                           db: AsyncSession = Depends(get_db)):
    """A visitor tells us who they are (optional). Nothing is returned about other people:
    the response is the same whether or not the email matched a contact."""
    ip = await _throttle(request, "events", get_settings().demo_rate_limit_per_min, 60)
    if not body.name.strip() or not EMAIL_RE.match(body.email.strip()):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            {"code": "invalid", "message": "Please enter your name and a valid email."},
                            headers=PRIVATE_HEADERS)
    link = await _active_link(db, token)
    v = await _visitor(db, link, Arrival(visitor_id=body.visitor_id), request, ip)
    await identify(db, link, v, name=body.name, email=body.email, role=body.role,
                   now=datetime.now(timezone.utc))
    await db.commit()
    _private(response)
    return {"ok": True}


class SessionIn(Arrival):
    turnstile_token: str | None = Field(None, max_length=2048)


def _ws_url(request: Request, voice_token: str) -> str:
    """wss://<this API's host>/ws/voice?token=… — the scheme follows the proxy's
    X-Forwarded-Proto so it is wss behind Railway's TLS."""
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
    ws = "wss" if proto == "https" else "ws"
    return f"{ws}://{host}/ws/voice?token={voice_token}"


@router.post("/{token}/session", status_code=status.HTTP_201_CREATED, dependencies=[Depends(_enabled)])
async def start_session(token: str, body: SessionIn, request: Request, response: Response,
                        db: AsyncSession = Depends(get_db)):
    s = get_settings()
    ip = await _throttle(request, "session", s.demo_sessions_per_ip_per_hour, 3600)
    link = await _active_link(db, token)
    ip_hash = limits.ip_hash(ip)
    try:
        # Bot check first: it's a network call, and must not be made while holding the lock.
        await limits.verify_turnstile(body.turnstile_token, ip, action="session")
        # From here to the commit below, no other session can be issued: the limits are
        # counted and this session inserted as one step (app/demo/limits.py).
        await limits.lock_session_issue(db)
        await limits.check_can_start_call(db, link, now=datetime.now(timezone.utc))
    except limits.LimitHit as e:
        raise _limit(e) from None
    prospect = await db.get(DemoProspect, link.prospect_id)
    try:
        conv, _customer, _greeting = await start_demo_conversation(db, prospect, channel="voice")
    except DemoError:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            {"code": "unavailable", "message": "This demo isn't available right now."},
                            headers=PRIVATE_HEADERS) from None
    max_seconds = limits.link_limit(link, "max_call_seconds")
    # Who is calling: usually known from /visit already; touched here too in case the
    # visitor pressed Talk before the page beacon fired.
    v = await _visitor(db, link, body, request, ip, viewed=True)
    session = DemoSession(demo_link_id=link.id, visitor_id=body.visitor_id,
                          demo_visitor_id=v.id if v else None,
                          recipient_id=v.recipient_id if v else None,
                          conversation_id=conv.id, ip_hash=ip_hash,
                          user_agent=(request.headers.get("user-agent") or "")[:400] or None)
    db.add(session)
    await db.flush()
    voice_token = issue_voice_token(session=session, link=link, prospect=prospect,
                                    max_seconds=max_seconds)
    await db.commit()
    _private(response)
    return {"session_id": str(session.id), "ws_url": _ws_url(request, voice_token),
            "max_seconds": max_seconds}


class EventIn(Arrival):
    type: str = Field(max_length=40)
    session_id: uuid.UUID | None = None


@router.post("/{token}/events", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(_enabled)])
async def record_event(token: str, body: EventIn, request: Request,
                       db: AsyncSession = Depends(get_db)):
    ip = await _throttle(request, "events", get_settings().demo_rate_limit_per_min, 60)
    if body.type not in CLIENT_EVENTS:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "unknown event type")
    link = await _active_link(db, token)
    session_id = None
    if body.session_id is not None:
        owned = (await db.execute(select(DemoSession.id).where(
            DemoSession.id == body.session_id, DemoSession.demo_link_id == link.id))).scalar_one_or_none()
        session_id = owned          # a session of another link is silently dropped
    v = await _visitor(db, link, body, request, ip, viewed=True)     # a click: a person
    db.add(DemoEvent(demo_link_id=link.id, visitor_id=body.visitor_id, session_id=session_id,
                     demo_visitor_id=v.id if v else None, recipient_id=v.recipient_id if v else None,
                     type=body.type, source_step=v.source_step if v else None, meta={}))
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT, headers=PRIVATE_HEADERS)


class ContactIn(Arrival):
    kind: Literal["contact", "schedule_call", "real_calls_pilot", "rebuild_request"]
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(max_length=254)
    phone: str | None = Field(None, max_length=30)
    company: str | None = Field(None, max_length=200)
    role: str | None = Field(None, max_length=120)
    preferred_time: str | None = Field(None, max_length=120)
    message: str | None = Field(None, max_length=2000)
    turnstile_token: str | None = Field(None, max_length=2048)


def _invalid(message: str) -> HTTPException:
    return HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, {"code": "invalid", "message": message},
                         headers=PRIVATE_HEADERS)


@router.post("/{token}/contact", status_code=status.HTTP_201_CREATED, dependencies=[Depends(_enabled)])
async def submit_contact(token: str, body: ContactIn, request: Request, response: Response,
                         db: AsyncSession = Depends(get_db)):
    """A lead from the demo page. Saved before we answer, so it survives a mail or Google
    outage; the email and the sheet row follow from the background worker. Works on an
    expired link too (that page offers "Rebuild my demo"), never on a revoked or unknown one.
    Sending the same email and kind twice on one day returns the first lead."""
    s = get_settings()
    ip = await _throttle(request, "contact", s.demo_sessions_per_ip_per_hour, 3600)
    link = await find_by_token(db, token)
    if link is None or effective_status(link) == "revoked":
        raise _not_found()
    if not body.name.strip():
        raise _invalid("Please enter your name.")
    if not EMAIL_RE.match(body.email.strip()):
        raise _invalid("Please enter a valid email address.")
    phone = None
    if body.phone and body.phone.strip():
        digits = re.sub(r"\D", "", body.phone)
        if not 8 <= len(digits) <= 15:
            raise _invalid("Please enter a valid phone number.")
        phone = normalize_phone(body.phone)[:20]
    if body.kind in leads.PHONE_REQUIRED and phone is None:
        raise _invalid("Please add a phone number so we can call you.")
    try:
        await limits.verify_turnstile(body.turnstile_token, ip, action="contact")
    except limits.LimitHit as e:
        raise _limit(e) from None

    now = datetime.now(timezone.utc)
    v = await _visitor(db, link, body, request, ip, viewed=True)
    lead, created = await leads.create_lead(
        db, link=link, visitor=v, kind=body.kind, name=body.name, email=body.email, phone=phone,
        company=body.company, role=body.role, preferred_time=body.preferred_time,
        message=body.message, now=now)
    # The form tells us who an unknown (or forwarded-to) visitor is.
    if created and v is not None and v.identified_at is None and v.attribution in ("unknown", "shared"):
        await identify(db, link, v, name=body.name, email=body.email, role=body.role, now=now)
    await db.commit()
    leads.wake()
    _private(response)
    return {"lead_id": lead.sheet_row_key, "created": created}
