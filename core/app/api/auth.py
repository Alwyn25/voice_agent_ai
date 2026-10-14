import uuid
from datetime import datetime, timezone

import jwt as pyjwt
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import AuditEvent, User
from app.security.auth import create_token, decode_token, verify_password_async
from app.security.deps import Principal, get_principal

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.email == body.email))).scalar_one_or_none()
    # Hash off the event loop (bcrypt is ~400 ms of CPU) so a login cannot stall in-flight
    # voice turns — see app/security/auth.py and docs/bridge-latency.md (P1.08).
    if user is None or not await verify_password_async(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    if not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Account disabled")

    user.last_login_at = datetime.now(timezone.utc)
    db.add(AuditEvent(tenant_id=user.tenant_id, actor=f"user:{user.email}", action="auth.login"))
    await db.commit()

    from app.config import get_settings
    return TokenResponse(
        access_token=create_token(user_id=user.id, tenant_id=user.tenant_id, role=user.role, token_type="access"),
        refresh_token=create_token(user_id=user.id, tenant_id=user.tenant_id, role=user.role, token_type="refresh"),
        expires_in=get_settings().access_token_minutes * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    try:
        payload = decode_token(body.refresh_token, expected_type="refresh")
    except pyjwt.PyJWTError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid refresh token: {e}")
    user = (await db.execute(select(User).where(User.id == uuid.UUID(payload["sub"])))).scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found or inactive")

    from app.config import get_settings
    return TokenResponse(
        access_token=create_token(user_id=user.id, tenant_id=user.tenant_id, role=user.role, token_type="access"),
        refresh_token=create_token(user_id=user.id, tenant_id=user.tenant_id, role=user.role, token_type="refresh"),
        expires_in=get_settings().access_token_minutes * 60,
    )


@router.get("/me")
async def me(principal: Principal = Depends(get_principal),
             db: AsyncSession = Depends(get_db)):
    """Who is holding this token.

    The identity fields matter as much as the permissions: a console that knows your role
    but not your name has to invent a label for the signed-in user, and every dashboard
    that did so ended up displaying a hardcoded demo account. An API key has no user
    behind it, so those stay null rather than being filled with something plausible.
    """
    email = name = None
    if principal.user_id:
        user = (await db.execute(select(User).where(
            User.id == principal.user_id))).scalar_one_or_none()
        if user is not None:
            email = user.email
            name = getattr(user, "full_name", None) or user.email.split("@")[0]

    return {
        "kind": principal.kind,
        "tenant_id": str(principal.tenant_id),
        "user_id": str(principal.user_id) if principal.user_id else None,
        "email": email,
        "name": name,
        "role": principal.role,
        "permissions": principal.permissions,
    }