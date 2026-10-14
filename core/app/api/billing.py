"""Vocal Backend Tasksheet P8 (Settings tier — Billing): subscription/invoices/
checkout. `plan`/`seats_purchased` stay authoritative on Tenant (P0) — this router
merges them with `Subscription`'s own fields rather than treating either as a second
source of truth. Definition of Done: the seat-count nav chip reads live from
GET /billing/subscription, computed here from a real count of active users, never
client-side math.
"""
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import AuditEvent, Invoice, Subscription, Tenant, User
from app.security.deps import Principal, require
from app.services.billing.base import BillingError
from app.services.billing.provider import get_billing_provider

router = APIRouter(prefix="/billing", tags=["settings"])


async def _tenant(db: AsyncSession, tenant_id: uuid.UUID) -> Tenant:
    return (await db.execute(select(Tenant).where(Tenant.id == tenant_id))).scalar_one()


async def _subscription(db: AsyncSession, tenant_id: uuid.UUID) -> Subscription | None:
    return (await db.execute(select(Subscription).where(
        Subscription.tenant_id == tenant_id))).scalar_one_or_none()


@router.get("/subscription")
async def get_subscription(p: Principal = Depends(require("billing:read")),
                           db: AsyncSession = Depends(get_db)):
    tenant = await _tenant(db, p.tenant_id)
    sub = await _subscription(db, p.tenant_id)
    seats_used = int((await db.execute(select(func.count()).select_from(User).where(
        User.tenant_id == p.tenant_id, User.is_active == True))).scalar_one())  # noqa: E712
    return {
        "tenant_name": tenant.name,
        "plan": tenant.plan, "seats_purchased": tenant.seats_purchased, "seats_used": seats_used,
        "billing_cycle": sub.billing_cycle if sub else None,
        "status": sub.status if sub else "no_subscription",
        "current_period_end": sub.current_period_end if sub else None,
        "has_payment_method": bool(sub and sub.stripe_customer_id),
    }


@router.get("/invoices")
async def list_invoices(page: int = Query(1, ge=1), page_size: int = Query(25, ge=1, le=100),
                        p: Principal = Depends(require("billing:read")),
                        db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(Invoice).where(Invoice.tenant_id == p.tenant_id)
        .order_by(Invoice.period_end.desc())
        .offset((page - 1) * page_size).limit(page_size))).scalars().all()
    total = int((await db.execute(select(func.count()).select_from(Invoice).where(
        Invoice.tenant_id == p.tenant_id))).scalar_one())
    return {"items": [{"id": str(i.id), "amount_cents": i.amount_cents, "currency": i.currency,
                       "status": i.status, "period_start": i.period_start, "period_end": i.period_end,
                       "pdf_url": i.pdf_url} for i in rows],
           "page": page, "page_size": page_size, "total": total}


class CheckoutSessionIn(BaseModel):
    plan: str
    seats: int
    billing_cycle: str = "monthly"
    success_url: str = "https://app.example.com/settings/billing?success=1"
    cancel_url: str = "https://app.example.com/settings/billing?canceled=1"


# Flat per-seat pricing for the stub provider's simulated invoice — a real processor
# would price this itself; kept trivially simple since nothing downstream depends on
# the number being realistic, only on an invoice genuinely existing after checkout.
_PLAN_SEAT_PRICE_CENTS = {"starter": 2900, "pro": 4900, "enterprise": 9900}


@router.post("/checkout-session", status_code=status.HTTP_201_CREATED)
async def create_checkout_session(body: CheckoutSessionIn,
                                  p: Principal = Depends(require("billing:write")),
                                  db: AsyncSession = Depends(get_db)):
    if body.seats < 1:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "seats must be at least 1")
    # Copilot review #17: validate plan and billing_cycle against known values.
    valid_plans = set(_PLAN_SEAT_PRICE_CENTS.keys())
    if body.plan not in valid_plans:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"plan must be one of {sorted(valid_plans)}")
    if body.billing_cycle not in ("monthly", "annual"):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "billing_cycle must be 'monthly' or 'annual'")

    provider = get_billing_provider()
    try:
        session = await provider.create_checkout_session(
            tenant_id=str(p.tenant_id), plan=body.plan, seats=body.seats,
            success_url=body.success_url, cancel_url=body.cancel_url)
    except NotImplementedError as e:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, str(e))
    except BillingError as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(e))

    activated = False
    if session.immediately_active:
        tenant = await _tenant(db, p.tenant_id)
        before = {"plan": tenant.plan, "seats_purchased": tenant.seats_purchased}
        tenant.plan = body.plan
        tenant.seats_purchased = body.seats

        sub = await _subscription(db, p.tenant_id)
        now = datetime.now(timezone.utc)
        period_end = now + (timedelta(days=365) if body.billing_cycle == "annual" else timedelta(days=30))
        if sub is None:
            sub = Subscription(tenant_id=p.tenant_id, billing_cycle=body.billing_cycle,
                               status="active", current_period_end=period_end,
                               stripe_customer_id=f"stub_cus_{p.tenant_id.hex[:16]}")
            db.add(sub)
        else:
            sub.billing_cycle = body.billing_cycle
            sub.status = "active"
            sub.current_period_end = period_end

        price = _PLAN_SEAT_PRICE_CENTS.get(body.plan, _PLAN_SEAT_PRICE_CENTS["starter"])
        db.add(Invoice(tenant_id=p.tenant_id, amount_cents=price * body.seats, currency="usd",
                       status="paid", period_start=now, period_end=period_end))

        db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="billing.checkout_completed",
                          resource_type="subscription", before=before,
                          after={"plan": body.plan, "seats_purchased": body.seats,
                                "billing_cycle": body.billing_cycle}))
        activated = True

    await db.commit()
    return {"provider": session.provider, "session_id": session.session_id,
           "checkout_url": session.checkout_url, "activated": activated}
