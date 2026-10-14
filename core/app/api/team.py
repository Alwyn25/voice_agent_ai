"""Vocal Backend Tasksheet P8 (Settings tier): GET /team, PATCH /team/{id}/role.

Reconciles onto the pre-existing users/team_invites (P0) rather than inventing a new
member concept: a "team member" here is either an active User row or a pending
TeamInvite (P0's own team-invite flow, already fully built — POST /team/invites,
.../accept, revoke — this just needed a single combined LIST view with the exact
member/status contract the tasksheet's Settings > Team screen expects, which
GET /users alone never had, and PATCH /team/{id}/role, an alias over PATCH /users/{id}
scoped to just the one field this screen actually changes).
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import AuditEvent, TeamInvite, User
from app.security.deps import Principal, require
from app.security.rbac import DEFAULT_ROLES

router = APIRouter(prefix="/team", tags=["settings"])

# Copilot review #20: the set of valid role names, derived from DEFAULT_ROLES.
VALID_ROLES = set(DEFAULT_ROLES.keys())


@router.get("")
async def list_team(p: Principal = Depends(require("team:read")), db: AsyncSession = Depends(get_db)):
    users = (await db.execute(select(User).where(
        User.tenant_id == p.tenant_id).order_by(User.created_at))).scalars().all()
    # A revoked invite is deleted outright (see DELETE /team/invites/{id} in
    # foundation.py), so accepted_at IS NULL alone means "still pending".
    pending = (await db.execute(select(TeamInvite).where(
        TeamInvite.tenant_id == p.tenant_id,
        TeamInvite.accepted_at.is_(None)).order_by(TeamInvite.created_at))).scalars().all()

    members = [{"id": str(u.id), "type": "user", "email": u.email, "full_name": u.full_name,
               "role": u.role, "status": "active" if u.is_active else "disabled",
               "joined_at": u.created_at} for u in users]
    members += [{"id": str(inv.id), "type": "invite", "email": inv.email, "full_name": None,
                "role": inv.role, "status": "invited", "joined_at": None,
                "expires_at": inv.expires_at} for inv in pending]
    # Copilot review #19: count only active users, not disabled ones.
    active_count = sum(1 for u in users if u.is_active)
    return {"members": members, "active_count": active_count, "pending_count": len(pending)}


class RoleChange(BaseModel):
    role: str


@router.patch("/{user_id}/role")
async def change_role(user_id: uuid.UUID, body: RoleChange,
                      p: Principal = Depends(require("team:write")),
                      db: AsyncSession = Depends(get_db)):
    # Copilot review #20: validate role name against known roles.
    if body.role not in VALID_ROLES:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"role must be one of {sorted(VALID_ROLES)}")
    user = (await db.execute(select(User).where(
        User.tenant_id == p.tenant_id, User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Team member not found")
    before = user.role
    user.role = body.role
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="team.role_change",
                      resource_type="user", resource_id=str(user_id),
                      before={"role": before}, after={"role": body.role}))
    await db.commit()
    return {"id": str(user_id), "role": user.role}
