"""FastAPI dependencies: current principal (user or API key) + permission guard."""
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import jwt as pyjwt
from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import ApiKey, Role, User
from app.security.auth import decode_token, hash_api_key
from app.security.rbac import DEFAULT_ROLES, permissions_for


@dataclass
class Principal:
    tenant_id: uuid.UUID
    kind: str  # "user" | "api_key"
    user_id: uuid.UUID | None = None
    role: str | None = None
    permissions: list[str] = field(default_factory=list)
    label: str = ""

    @property
    def actor(self) -> str:
        return f"{self.kind}:{self.label or self.user_id}"


async def get_principal(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
) -> Principal:
    if x_api_key:
        row = (await db.execute(select(ApiKey).where(ApiKey.key_hash == hash_api_key(x_api_key)))).scalar_one_or_none()
        if row is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")
        if row.expires_at and row.expires_at < datetime.now(timezone.utc):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "API key expired")
        return Principal(tenant_id=row.tenant_id, kind="api_key", permissions=list(row.scopes or []), label=row.name)

    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1]
        try:
            payload = decode_token(token, expected_type="access")
        except pyjwt.PyJWTError as e:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid token: {e}")
        user_id = uuid.UUID(payload["sub"])
        user = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
        if user is None or not user.is_active:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "User not found or inactive")
        role = (await db.execute(
            select(Role).where(Role.tenant_id == user.tenant_id, Role.name == user.role)
        )).scalar_one_or_none()
        # Copilot review #10: fall back to DEFAULT_ROLES when the DB role row is
        # missing or stale — ensures new permissions (team:*, billing:*, integrations:*)
        # are available to existing tenants without a backfill migration.
        perms = list(role.permissions) if role and role.permissions else DEFAULT_ROLES.get(user.role, [])
        return Principal(tenant_id=user.tenant_id, kind="user", user_id=user.id,
                         role=user.role, permissions=perms, label=user.email)

    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing credentials")


def require(permission: str):
    async def guard(principal: Principal = Depends(get_principal)) -> Principal:
        if not permissions_for(principal.permissions, permission):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Missing permission: {permission}")
        return principal
    return guard
