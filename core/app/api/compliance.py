"""Do-Not-Contact list and consent ledger (minimal surface; extended in a later stage).

DNC is the platform's hardest guarantee: the calling_policy rule set reads this list
as a priority-0 terminal rule, so an entry here blocks outbound contact everywhere.
"""
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import AuditEvent, ConsentRecord, Customer, DncEntry
from app.security.deps import Principal, require
from app.services.phone import normalize_phone

router = APIRouter(tags=["compliance"])


class DncIn(BaseModel):
    phone: str
    scope: str = "all"            # all | marketing | voice
    source: str = "customer_request"
    customer_id: uuid.UUID | None = None
    expires_at: datetime | None = None


@router.post("/dnc", status_code=201)
async def add_dnc(body: DncIn, p: Principal = Depends(require("customers:write")),
                  db: AsyncSession = Depends(get_db)):
    phone = normalize_phone(body.phone)
    existing = (await db.execute(select(DncEntry).where(
        DncEntry.tenant_id == p.tenant_id, DncEntry.phone_normalized == phone,
        DncEntry.scope == body.scope))).scalar_one_or_none()
    if existing:
        return {"id": str(existing.id), "phone_normalized": phone, "scope": existing.scope,
                "note": "Already on the list"}
    row = DncEntry(tenant_id=p.tenant_id, phone_normalized=phone, scope=body.scope,
                   source=body.source, customer_id=body.customer_id,
                   added_at=datetime.now(timezone.utc), expires_at=body.expires_at)
    db.add(row)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="dnc.add",
                      resource_type="dnc_entry",
                      after={"phone": phone, "scope": body.scope, "source": body.source}))
    await db.commit()
    return {"id": str(row.id), "phone_normalized": phone, "scope": row.scope, "active": True}


@router.get("/dnc")
async def list_dnc(scope: str | None = None,
                   p: Principal = Depends(require("customers:read")),
                   db: AsyncSession = Depends(get_db)):
    q = select(DncEntry).where(DncEntry.tenant_id == p.tenant_id)
    if scope:
        q = q.where(DncEntry.scope == scope)
    rows = (await db.execute(q.order_by(DncEntry.created_at.desc()))).scalars().all()
    return [{"id": str(r.id), "phone_normalized": r.phone_normalized, "scope": r.scope,
             "source": r.source, "added_at": r.added_at, "expires_at": r.expires_at}
            for r in rows]


class DncRemove(BaseModel):
    reason: str


@router.delete("/dnc/{entry_id}")
async def remove_dnc(entry_id: uuid.UUID, body: DncRemove,
                     p: Principal = Depends(require("dnc:remove")),
                     db: AsyncSession = Depends(get_db)):
    """Admin only, reason required, and the removal itself is audited."""
    row = (await db.execute(select(DncEntry).where(
        DncEntry.tenant_id == p.tenant_id, DncEntry.id == entry_id))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "DNC entry not found")
    before = {"phone": row.phone_normalized, "scope": row.scope, "source": row.source}
    await db.delete(row)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="dnc.remove",
                      resource_type="dnc_entry", resource_id=str(entry_id),
                      before=before, after={"reason": body.reason}))
    await db.commit()
    return {"removed": True, "reason": body.reason}


class ConsentIn(BaseModel):
    consent_type: str            # calls | recording | marketing | data_processing
    granted: bool
    channel: str | None = None
    evidence_uri: str | None = None


@router.post("/customers/{customer_id}/consents", status_code=201)
async def record_consent(customer_id: uuid.UUID, body: ConsentIn,
                         p: Principal = Depends(require("customers:write")),
                         db: AsyncSession = Depends(get_db)):
    c = (await db.execute(select(Customer).where(
        Customer.tenant_id == p.tenant_id, Customer.id == customer_id))).scalar_one_or_none()
    if c is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found")
    row = ConsentRecord(tenant_id=p.tenant_id, customer_id=customer_id,
                        consent_type=body.consent_type, granted=body.granted,
                        channel=body.channel, evidence_uri=body.evidence_uri)
    db.add(row)
    await db.commit()
    return {"id": str(row.id), "consent_type": row.consent_type, "granted": row.granted}


@router.get("/customers/{customer_id}/consents")
async def list_consents(customer_id: uuid.UUID,
                        p: Principal = Depends(require("customers:read")),
                        db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(ConsentRecord).where(
        ConsentRecord.tenant_id == p.tenant_id,
        ConsentRecord.customer_id == customer_id)
        .order_by(ConsentRecord.recorded_at.desc()))).scalars().all()
    return [{"id": str(r.id), "consent_type": r.consent_type, "granted": r.granted,
             "channel": r.channel, "evidence_uri": r.evidence_uri,
             "recorded_at": r.recorded_at} for r in rows]
