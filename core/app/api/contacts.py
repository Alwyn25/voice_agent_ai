"""Vocal Backend Tasksheet P5 — "contacts" (first_name, last_name, phone_e164,
email, external_ref, tags) is the pre-existing `customers` table (app/db/models/
crm.py's Customer), reused rather than duplicated — the exact same reconciliation
P0 already established for organizations/tenants. Customer has no separate
first_name/last_name (only `full_name`), so bulk-import joins the two CSV columns
at import time rather than splitting Customer's schema, which is referenced far
too widely (engine/planner/*, analytics, campaigns) to safely touch for a
cosmetic column split — the same "deep engine internals" boundary Wave 5's
Centre/Executive rename already drew.

This is the only genuinely new endpoint P5 needs for contacts — everything else
(list/get/update) already exists at GET/POST/PATCH /customers.
"""
import csv
import io

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import AuditEvent, Customer
from app.security.deps import Principal, require
from app.services.phone import normalize_phone

router = APIRouter(prefix="/contacts", tags=["contacts"])

REQUIRED_COLUMNS = {"phone_e164"}
MAX_ROWS = 5000


@router.post("/bulk-import", status_code=status.HTTP_201_CREATED)
async def bulk_import(p: Principal = Depends(require("customers:write")),
                      db: AsyncSession = Depends(get_db), file: UploadFile = File(...)):
    """CSV columns: first_name, last_name, phone_e164 (required), email,
    external_ref, tags (semicolon-separated within the cell). A phone that
    already matches an existing customer updates that row instead of creating a
    duplicate — the common bulk-refresh case (re-uploading an updated list)."""
    raw = await file.read()
    # Copilot review #22: cap file size to prevent unbounded memory use.
    max_size = 10 * 1024 * 1024  # 10 MB
    if len(raw) > max_size:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"File too large ({len(raw)} bytes, max {max_size})")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "File must be UTF-8 encoded CSV")

    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None or not REQUIRED_COLUMNS.issubset(set(reader.fieldnames)):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"CSV must have a header row including: {sorted(REQUIRED_COLUMNS)}")

    created, updated, errors = 0, 0, []
    for i, row in enumerate(reader, start=2):  # row 1 is the header
        if i - 1 > MAX_ROWS:
            errors.append({"row": i, "reason": f"stopped at {MAX_ROWS} rows — split the file"})
            break
        phone_raw = (row.get("phone_e164") or "").strip()
        if not phone_raw:
            errors.append({"row": i, "reason": "phone_e164 is required"})
            continue
        normalized = normalize_phone(phone_raw)

        first = (row.get("first_name") or "").strip()
        last = (row.get("last_name") or "").strip()
        full_name = (f"{first} {last}".strip()) or None
        tags = [t.strip() for t in (row.get("tags") or "").split(";") if t.strip()]

        existing = (await db.execute(select(Customer).where(
            Customer.tenant_id == p.tenant_id, Customer.phone_normalized == normalized))).scalar_one_or_none()
        if existing is not None:
            if full_name:
                existing.full_name = full_name
            if row.get("email"):
                existing.email = row["email"].strip()
            if row.get("external_ref"):
                existing.external_ref = row["external_ref"].strip()
            if tags:
                existing.tags = sorted(set(existing.tags or []) | set(tags))
            updated += 1
        else:
            db.add(Customer(tenant_id=p.tenant_id, full_name=full_name, phone=phone_raw,
                            phone_normalized=normalized, email=row.get("email") or None,
                            external_ref=row.get("external_ref") or None, tags=tags))
            created += 1

    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="contacts.bulk_import",
                      resource_type="customer",
                      after={"created": created, "updated": updated, "errors": len(errors)}))
    await db.commit()
    return {"created": created, "updated": updated, "errors": errors}
