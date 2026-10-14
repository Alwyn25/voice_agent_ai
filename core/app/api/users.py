import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import ApiKey, AuditEvent, Role, User
from app.security.auth import generate_api_key, hash_password_async
from app.security.deps import Principal, require

router = APIRouter(tags=["users & api keys"])


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    role: str = "agent"


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    full_name: str
    role: str
    is_active: bool

    class Config:
        from_attributes = True


@router.post("/users", response_model=UserOut, status_code=201)
async def create_user(body: UserCreate, principal: Principal = Depends(require("users:write")),
                      db: AsyncSession = Depends(get_db)):
    role = (await db.execute(select(Role).where(Role.tenant_id == principal.tenant_id,
                                                Role.name == body.role))).scalar_one_or_none()
    if role is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Unknown role: {body.role}")
    exists = (await db.execute(select(User).where(User.tenant_id == principal.tenant_id,
                                                  User.email == body.email))).scalar_one_or_none()
    if exists:
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already exists")

    user = User(tenant_id=principal.tenant_id, email=body.email, full_name=body.full_name,
                role=body.role, password_hash=await hash_password_async(body.password))
    db.add(user)
    db.add(AuditEvent(tenant_id=principal.tenant_id, actor=principal.actor, action="users.create",
                      resource_type="user", after={"email": body.email, "role": body.role}))
    await db.commit()
    await db.refresh(user)
    return user


@router.get("/users", response_model=list[UserOut])
async def list_users(principal: Principal = Depends(require("users:read")),
                     db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(User).where(User.tenant_id == principal.tenant_id))).scalars().all()
    return rows


@router.patch("/users/{user_id}", response_model=UserOut)
async def update_user(user_id: uuid.UUID, body: dict,
                      principal: Principal = Depends(require("users:write")),
                      db: AsyncSession = Depends(get_db)):
    user = (await db.execute(select(User).where(User.tenant_id == principal.tenant_id,
                                                User.id == user_id))).scalar_one_or_none()
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    allowed = {"full_name", "role", "is_active"}
    before = {k: getattr(user, k) for k in allowed}
    for k, v in body.items():
        if k in allowed:
            setattr(user, k, v)
    if "password" in body:
        user.password_hash = await hash_password_async(body["password"])
    db.add(AuditEvent(tenant_id=principal.tenant_id, actor=principal.actor, action="users.update",
                      resource_type="user", resource_id=str(user_id),
                      before=before, after={k: getattr(user, k) for k in allowed}))
    await db.commit()
    await db.refresh(user)
    return user


class ApiKeyCreate(BaseModel):
    name: str
    scopes: list[str] = []
    expires_at: datetime | None = None


@router.post("/api-keys", status_code=201)
async def create_api_key(body: ApiKeyCreate, principal: Principal = Depends(require("api_keys:write")),
                         db: AsyncSession = Depends(get_db)):
    plain, key_hash = generate_api_key()
    row = ApiKey(tenant_id=principal.tenant_id, name=body.name, key_hash=key_hash,
                 scopes=body.scopes, expires_at=body.expires_at)
    db.add(row)
    db.add(AuditEvent(tenant_id=principal.tenant_id, actor=principal.actor, action="api_keys.create",
                      resource_type="api_key", after={"name": body.name, "scopes": body.scopes}))
    await db.commit()
    return {"id": str(row.id), "name": row.name, "key": plain,
            "note": "Store this key now — it is shown only once."}


@router.get("/api-keys")
async def list_api_keys(principal: Principal = Depends(require("api_keys:read")),
                        db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(ApiKey).where(ApiKey.tenant_id == principal.tenant_id))).scalars().all()
    return [{"id": str(r.id), "name": r.name, "scopes": r.scopes,
             "expires_at": r.expires_at, "last_used_at": r.last_used_at} for r in rows]
