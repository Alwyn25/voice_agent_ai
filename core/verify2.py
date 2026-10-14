import asyncio, uuid
from sqlalchemy import select
from app.db.base import get_session_factory
from app.db.models import Customer
from app.engine.tools.types import ToolContext
from app.engine.tools.builtin.lookup_tools import my_appointments, my_tickets

CUST = uuid.UUID("74fff3e5-2ed9-4fb5-bbf5-e496bf62b459")

async def main():
    sf = get_session_factory()
    async with sf() as db:
        cust = (await db.execute(select(Customer).where(Customer.id == CUST))).scalar_one()
        ctx = ToolContext(db=db, tenant_id=cust.tenant_id, actor="test", customer=cust,
                          conversation_id=None, decision_trace_id=None,
                          identity_verified=False, project_id=None, extra={})
        a = await my_appointments(ctx, {})
        print("APPTS has_upcoming:", a["has_upcoming"], "| upcoming:",
              [(x["when"], x["status"]) for x in a["upcoming"]], "| note:", a["note"])
        t = await my_tickets(ctx, {})
        print("TICKETS open_count:", t["open_count"], "| note:", t["note"])
asyncio.run(main())
