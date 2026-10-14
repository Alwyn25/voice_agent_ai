"""Erase demo company records completely (docs/gtm-demo-links.md §12, prospect_erasure).

A record owns a whole demo tenant (§2a): its agent, knowledge base and chunks, the sample
leads, conversations, messages, bookings, traces. Erasing it removes, in order:

  1. its leads (demo_leads only points at the record, so they would otherwise survive)
  2. the record itself; the database cascades its people, facts, dummy data, links and,
     through the links, recipients, visitors, calls (demo_sessions) and events
  3. every row of every tenant-scoped table (any table with a tenant_id column) in its tenant
  4. the tenant

Step 3 is generic so a table added later is erased too. Tables are deleted in an order
that respects foreign keys between them: each pass deletes what it can, inside a
savepoint, and retries the rest, so no hand-kept table list can fall out of date.

Refuses any tenant that is not kind='demo' or that has users: this can never touch a
customer's data.
"""
import uuid

from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import DemoLead, DemoProspect, Tenant, User
from app.db.models.platform import TENANT_KIND_DEMO
from app.demo.records import Conflict

_TENANT_TABLES = text("""
    SELECT c.table_name FROM information_schema.columns c
      JOIN information_schema.tables t USING (table_schema, table_name)
     WHERE c.table_schema = current_schema() AND c.column_name = 'tenant_id'
       AND t.table_type = 'BASE TABLE' AND c.table_name <> 'demo_prospects'
     ORDER BY c.table_name""")


async def erase_records(db: AsyncSession, prospects: list[DemoProspect]) -> dict:
    """Erase these records and their demo tenants. Returns counts. The caller commits."""
    if not prospects:
        return {"records": 0, "tenants": 0, "rows": 0}
    prospect_ids = [p.id for p in prospects]
    tenant_ids = [p.tenant_id for p in prospects if p.tenant_id is not None]
    if tenant_ids:
        kinds = (await db.execute(select(Tenant.kind).where(Tenant.id.in_(tenant_ids)))).scalars().all()
        if any(k != TENANT_KIND_DEMO for k in kinds):
            raise Conflict("refusing to erase: a record points at a tenant that is not a demo tenant")
        users = (await db.execute(select(func.count()).select_from(User).where(
            User.tenant_id.in_(tenant_ids)))).scalar_one()
        if users:
            raise Conflict("refusing to erase: a demo tenant has user accounts")

    await db.execute(delete(DemoLead).where(DemoLead.prospect_id.in_(prospect_ids)))
    await db.execute(delete(DemoProspect).where(DemoProspect.id.in_(prospect_ids)))
    rows = await _erase_tenant_rows(db, tenant_ids) if tenant_ids else 0
    if tenant_ids:
        await db.execute(delete(Tenant).where(Tenant.id.in_(tenant_ids), Tenant.kind == TENANT_KIND_DEMO))
    await db.flush()
    return {"records": len(prospect_ids), "tenants": len(tenant_ids), "rows": rows}


async def _erase_tenant_rows(db: AsyncSession, tenant_ids: list[uuid.UUID]) -> int:
    tables = list((await db.execute(_TENANT_TABLES)).scalars())
    total = 0
    remaining = tables
    while remaining:
        blocked = []
        for table in remaining:
            try:
                async with db.begin_nested():
                    r = await db.execute(text(f'DELETE FROM "{table}" WHERE tenant_id = ANY(:ids)'),
                                         {"ids": tenant_ids})
                    total += r.rowcount or 0
            except IntegrityError:
                blocked.append(table)        # another table still points at its rows
        if len(blocked) == len(remaining):
            raise Conflict("could not erase demo tenant data; blocked tables: " + ", ".join(blocked))
        remaining = blocked
    return total
