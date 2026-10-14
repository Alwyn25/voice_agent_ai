"""Vocal Backend Tasksheet P0 (Foundation) — the endpoints that were genuinely
missing: POST /orgs, GET /me (alias), GET /activity, and the team-invite flow.

organizations/users/api_keys/activity_events already exist under other names
(tenants/users/api_keys/audit_events) with working CRUD+auth+RBAC — see
app/db/models/platform.py and app/services/activity.py for the mapping notes. This
module only adds what those didn't already cover.
"""
import logging
import secrets
import uuid
from datetime import date, datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import me as auth_me
from app.db.base import get_db
from app.db.models import (AuditEvent, Conversation, IndustryPackDef, Message, Notification,
                           Role, Tenant, TeamInvite, TenantPackConfig, ToolCatalogEntry,
                           ToolDefinition, User)
from app.engine.tools.definitions import tools_for_pack
from app.packs.real_estate_india.pack import RealEstateIndiaPack
from app.security.auth import hash_api_key, hash_password_async
from app.security.deps import Principal, get_principal, require
from app.security.rbac import DEFAULT_ROLES
from app.services.activity import append_activity_event

router = APIRouter(tags=["foundation"])
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- GET /me
# Literal alias for the tasksheet's exact endpoint name; /auth/me (pre-existing) does
# the same thing and stays for backward compatibility.
router.add_api_route("/me", auth_me, methods=["GET"])


# --------------------------------------------------------------------------- POST /orgs
class OrgCreate(BaseModel):
    name: str
    slug: str
    timezone: str = "Asia/Kolkata"
    plan: str = "starter"
    seats_purchased: int = 1
    admin_email: EmailStr
    admin_password: str
    admin_name: str = "Administrator"


@router.post("/orgs", status_code=status.HTTP_201_CREATED)
async def create_org(body: OrgCreate, db: AsyncSession = Depends(get_db)):
    """Bootstrap a brand-new tenant + its admin user + default roles. Deliberately
    takes no auth dependency (there is no tenant/principal yet to authenticate
    against) — this is the one endpoint meant for internal/admin use only, same as
    the tasksheet's note says; front it with network-level access control, not RBAC."""
    existing = (await db.execute(select(Tenant).where(Tenant.slug == body.slug))).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "An organization with that slug already exists")

    tenant = Tenant(name=body.name, slug=body.slug, timezone=body.timezone,
                    plan=body.plan, seats_purchased=body.seats_purchased)
    db.add(tenant)
    await db.flush()

    # Default every newly-bootstrapped org onto real_estate_india rather than
    # leaving it with no tenant_pack_config row. The generic pack (PackRegistry's
    # fallback for an unconfigured tenant) is infrastructure-only right now — its
    # state names aren't yet wired into the tool/analytics/booking layers that
    # still hardcode the real-estate vocabulary (that wiring is P6/P7's Generic
    # Sprints 3-4) — so a tenant actually left on it today would have booking
    # silently rejected and analytics silently wrong, not just different wording.
    # real_estate_india is the only pack with full engine support until then.
    pack_row = (await db.execute(select(IndustryPackDef).where(
        IndustryPackDef.slug == RealEstateIndiaPack.slug))).scalar_one_or_none()
    pack = RealEstateIndiaPack()
    if pack_row is not None:
        db.add(TenantPackConfig(tenant_id=tenant.id, pack_id=pack_row.id))
    else:
        # Should only happen if migration 0019's seed insert hasn't run yet.
        # Logged rather than silent: this tenant falls through to
        # PackRegistry's generic-pack fallback — exactly the state the comment
        # above says this block exists to prevent.
        logger.warning("industry_packs has no '%s' row — tenant %s created with no "
                       "tenant_pack_config; it will fall back to the generic pack",
                       RealEstateIndiaPack.slug, tenant.id)

    # Vocal Backend Tasksheet P7 Generic Sprint 4: this endpoint previously seeded NO
    # tool_definitions at all for a new tenant (only the CLI scripts/seed.py bootstrap
    # script did) — a tenant created through the real product flow had zero usable
    # engine tools (no search_slots/book_appointment/…) until someone ran that script
    # by hand. tools_for_pack() gives this tenant exactly the same pack-aware set
    # scripts/seed.py now seeds too.
    for spec in tools_for_pack(pack):
        try:
            db.add(ToolDefinition(tenant_id=tenant.id, is_enabled=True, **spec))
        except TypeError:
            logger.exception("tool spec %r not seedable for tenant %s", spec.get("name"), tenant.id)

    for role_name, perms in DEFAULT_ROLES.items():
        db.add(Role(tenant_id=tenant.id, name=role_name, permissions=perms))

    admin = User(tenant_id=tenant.id, email=body.admin_email, full_name=body.admin_name,
                role="admin", password_hash=await hash_password_async(body.admin_password))
    db.add(admin)
    await db.flush()

    db.add(AuditEvent(tenant_id=tenant.id, actor=f"user:{admin.email}", action="orgs.create",
                      resource_type="tenant", resource_id=str(tenant.id),
                      after={"name": tenant.name, "slug": tenant.slug, "plan": tenant.plan}))
    await db.commit()
    return {"id": str(tenant.id), "name": tenant.name, "slug": tenant.slug,
           "plan": tenant.plan, "seats_purchased": tenant.seats_purchased,
           "admin_user_id": str(admin.id)}


# --------------------------------------------------------------------------- GET /activity
@router.get("/activity")
async def activity_feed(
        category: str | None = Query(None, description="Matches the action prefix before the first '.', e.g. 'tools'"),
        date_from: date | None = Query(None, alias="from"),
        date_to: date | None = Query(None, alias="to"),
        page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200),
        p: Principal = Depends(require("audit:read")),
        db: AsyncSession = Depends(get_db)):
    """Paginated activity feed, filterable by category — powers the Overview feed
    (P7). Backed by the same audit_events table GET /audit/events already reads."""
    q = select(AuditEvent).where(AuditEvent.tenant_id == p.tenant_id)
    if category:
        q = q.where(AuditEvent.action.startswith(f"{category}."))
    if date_from:
        q = q.where(AuditEvent.created_at >= datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc))
    if date_to:
        q = q.where(AuditEvent.created_at <= datetime.combine(date_to, datetime.max.time(), tzinfo=timezone.utc))

    total = int((await db.execute(select(func.count()).select_from(q.subquery()))).scalar_one())
    rows = (await db.execute(q.order_by(AuditEvent.created_at.desc())
                             .offset((page - 1) * page_size).limit(page_size))).scalars().all()
    return {"items": [{"id": str(e.id), "actor": e.actor, "action": e.action,
                       "category": e.action.split(".", 1)[0] if e.action else None,
                       "resource_type": e.resource_type, "resource_id": e.resource_id,
                       "before": e.before, "after": e.after, "at": e.created_at}
                      for e in rows],
           "page": page, "page_size": page_size, "total": total}


# --------------------------------------------------------------------------- Team invites
class InviteCreate(BaseModel):
    email: EmailStr
    role: str


def _invite_out(row: TeamInvite, token: str | None = None) -> dict:
    out = {"id": str(row.id), "email": row.email, "role": row.role,
          "invited_by": str(row.invited_by) if row.invited_by else None,
          "expires_at": row.expires_at, "accepted_at": row.accepted_at,
          "created_at": row.created_at}
    if token:
        out["token"] = token  # shown once, at creation, like an API key
    return out


@router.get("/team/invites")
async def list_invites(p: Principal = Depends(require("users:read")),
                       db: AsyncSession = Depends(get_db)):
    """Pending invites only — accepted ones become real users, visible via GET /users."""
    rows = (await db.execute(select(TeamInvite).where(
        TeamInvite.tenant_id == p.tenant_id, TeamInvite.accepted_at.is_(None))
        .order_by(TeamInvite.created_at.desc()))).scalars().all()
    return [_invite_out(r) for r in rows]


@router.post("/team/invites", status_code=status.HTTP_201_CREATED)
async def create_invite(body: InviteCreate, p: Principal = Depends(require("users:write")),
                        db: AsyncSession = Depends(get_db)):
    existing = (await db.execute(select(User).where(
        User.tenant_id == p.tenant_id, User.email == body.email))).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "That email is already a member")
    pending = (await db.execute(select(TeamInvite).where(
        TeamInvite.tenant_id == p.tenant_id, TeamInvite.email == body.email,
        TeamInvite.accepted_at.is_(None)))).scalar_one_or_none()
    if pending is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "That email already has a pending invite")

    plain_token = "invite_" + secrets.token_urlsafe(24)
    row = TeamInvite(tenant_id=p.tenant_id, email=body.email, role=body.role,
                     invited_by=p.user_id, token_hash=hash_api_key(plain_token),
                     expires_at=datetime.now(timezone.utc) + timedelta(days=7))
    db.add(row)
    await db.flush()
    append_activity_event(db, p, "team.invite", resource_type="team_invite",
                          resource_id=str(row.id), after={"email": row.email, "role": row.role})
    await db.commit()
    return _invite_out(row, token=plain_token)


@router.delete("/team/invites/{invite_id}")
async def revoke_invite(invite_id: uuid.UUID, p: Principal = Depends(require("users:write")),
                        db: AsyncSession = Depends(get_db)):
    row = (await db.execute(select(TeamInvite).where(
        TeamInvite.tenant_id == p.tenant_id, TeamInvite.id == invite_id))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Invite not found")
    if row.accepted_at is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Invite already accepted")
    await db.delete(row)
    append_activity_event(db, p, "team.invite_revoke", resource_type="team_invite", resource_id=str(invite_id))
    await db.commit()
    return {"id": str(invite_id), "revoked": True}


class InviteAccept(BaseModel):
    token: str
    full_name: str
    password: str


@router.post("/team/invites/{invite_id}/accept", status_code=status.HTTP_201_CREATED)
async def accept_invite(invite_id: uuid.UUID, body: InviteAccept, db: AsyncSession = Depends(get_db)):
    """No auth dependency — the caller doesn't have an account yet; the invite token
    itself is the credential."""
    row = (await db.execute(select(TeamInvite).where(TeamInvite.id == invite_id))).scalar_one_or_none()
    if row is None or row.token_hash != hash_api_key(body.token):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Invite not found")
    if row.accepted_at is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Invite already accepted")
    if row.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status.HTTP_410_GONE, "Invite has expired")

    user = User(tenant_id=row.tenant_id, email=row.email, full_name=body.full_name,
               role=row.role, password_hash=await hash_password_async(body.password))
    db.add(user)
    row.accepted_at = datetime.now(timezone.utc)
    await db.flush()
    event = AuditEvent(id=uuid.uuid4(), tenant_id=row.tenant_id, actor=f"user:{user.email}",
                       action="team.invite_accepted", resource_type="user", resource_id=str(user.id))
    db.add(event)
    if row.invited_by is not None:
        # The inviter has no live Principal here (this route has no auth
        # dependency — see the docstring), so build the Notification row directly
        # rather than routing through append_activity_event's Principal-shaped API.
        db.add(Notification(tenant_id=row.tenant_id, user_id=row.invited_by, event_id=event.id))
    await db.commit()
    return {"user_id": str(user.id), "email": user.email, "role": user.role}


# --------------------------------------------------------------------------- Notifications
def _require_user_principal(p: Principal) -> None:
    """Notifications and search are read against a specific user's inbox / session
    context — an api_key principal (no user_id) has neither."""
    if p.user_id is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Requires a user principal, not an API key")


@router.get("/notifications")
async def list_notifications(limit: int = Query(20, ge=1, le=100),
                             p: Principal = Depends(get_principal),
                             db: AsyncSession = Depends(get_db)):
    """Unread count + latest N for the bell icon in the global header (every screen)."""
    _require_user_principal(p)
    unread_count = int((await db.execute(
        select(func.count()).select_from(Notification).where(
            Notification.tenant_id == p.tenant_id, Notification.user_id == p.user_id,
            Notification.read_at.is_(None)))).scalar_one())
    rows = (await db.execute(
        select(Notification, AuditEvent)
        .join(AuditEvent, Notification.event_id == AuditEvent.id)
        .where(Notification.tenant_id == p.tenant_id, Notification.user_id == p.user_id)
        .order_by(Notification.created_at.desc()).limit(limit))).all()
    return {
        "unread_count": unread_count,
        "items": [{"id": str(n.id), "event_id": str(n.event_id), "read_at": n.read_at,
                  "created_at": n.created_at, "actor": e.actor, "action": e.action,
                  "resource_type": e.resource_type, "resource_id": e.resource_id}
                 for n, e in rows],
    }


@router.patch("/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: uuid.UUID, p: Principal = Depends(get_principal),
                                 db: AsyncSession = Depends(get_db)):
    _require_user_principal(p)
    row = (await db.execute(select(Notification).where(
        Notification.tenant_id == p.tenant_id, Notification.user_id == p.user_id,
        Notification.id == notification_id))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Notification not found")
    if row.read_at is None:
        row.read_at = datetime.now(timezone.utc)
        await db.commit()
    return {"id": str(row.id), "read_at": row.read_at}


@router.post("/notifications/read-all")
async def mark_all_notifications_read(p: Principal = Depends(get_principal), db: AsyncSession = Depends(get_db)):
    _require_user_principal(p)
    now = datetime.now(timezone.utc)
    result = await db.execute(
        update(Notification)
        .where(Notification.tenant_id == p.tenant_id, Notification.user_id == p.user_id,
              Notification.read_at.is_(None))
        .values(read_at=now))
    await db.commit()
    return {"marked_read": result.rowcount}


# --------------------------------------------------------------------------- Search
@router.get("/search")
async def search(
        q: str = Query(..., min_length=1),
        types: list[str] | None = Query(
            None, description="calls|tools — 'agents' is accepted but returns nothing until P4"),
        limit: int = Query(10, ge=1, le=50),
        p: Principal = Depends(get_principal),
        db: AsyncSession = Depends(get_db)):
    """Cross-entity search for the 'Search agents, calls, tools' header bar.

    Scoped to what actually exists today: calls (Conversation.summary + Message
    transcript content) and tools (both the built-in engine catalog and the P1
    connectable tool marketplace). 'agents' is a valid `types` value that returns an
    empty list until the P4 agents table lands — kept in the contract now so the
    frontend doesn't need a later breaking change, per the tasksheet's stated intent."""
    _require_user_principal(p)
    # `is not None`, not a truthiness check — types=[] (every filter checkbox
    # unchecked) must return nothing, not silently fall back to "every type".
    wanted = set(types) if types is not None else {"calls", "tools", "agents"}
    # Escape SQL LIKE metacharacters in the caller's own query text — otherwise a
    # literal "%"/"_" in the search term (a discount code "50%OFF", a reference like
    # "ORDER_123") is read as a wildcard instead of a literal character.
    escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    like = f"%{escaped}%"
    results: list[dict] = []

    if "calls" in wanted:
        conv_rows = (await db.execute(
            select(Conversation).where(Conversation.tenant_id == p.tenant_id,
                                       Conversation.summary.ilike(like, escape="\\"))
            .order_by(Conversation.started_at.desc()).limit(limit))).scalars().all()
        results += [{"type": "call", "id": str(c.id),
                    "title": (c.summary or "")[:120] or None,
                    "snippet": c.summary, "at": c.started_at} for c in conv_rows]

        msg_rows = (await db.execute(
            select(Message, Conversation)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(Message.tenant_id == p.tenant_id, Message.content.ilike(like, escape="\\"))
            .order_by(Message.created_at.desc()).limit(limit))).all()
        results += [{"type": "call", "id": str(conv.id),
                    "title": (conv.summary or "")[:120] or None,
                    "snippet": msg.content[:200], "at": msg.created_at} for msg, conv in msg_rows]

    if "tools" in wanted:
        engine_tools = (await db.execute(
            select(ToolDefinition).where(ToolDefinition.tenant_id == p.tenant_id,
                                         ToolDefinition.name.ilike(like, escape="\\"))
            .limit(limit))).scalars().all()
        results += [{"type": "tool", "id": str(t.id), "title": t.name,
                    "snippet": t.category, "at": None} for t in engine_tools]

        catalog_tools = (await db.execute(
            select(ToolCatalogEntry).where(ToolCatalogEntry.tenant_id == p.tenant_id,
                                           ToolCatalogEntry.key.ilike(like, escape="\\"))
            .limit(limit))).scalars().all()
        results += [{"type": "tool", "id": str(t.id), "title": t.key,
                    "snippet": t.category, "at": None} for t in catalog_tools]

    # No blanket results[:limit] here — each branch above already caps itself at
    # `limit`, and slicing the combined list post-hoc silently drops an entire
    # requested type when an earlier type alone fills the slice (e.g. calls
    # returning 2*limit rows would previously crowd out every tool match).
    return {"query": q, "items": results}
