import asyncio, uuid
from sqlalchemy import select
from app.db.base import get_session_factory
from app.db.models import Customer
from app.engine.tools.types import ToolContext
from app.engine.tools.builtin.lookup_tools import my_appointments

CUST = uuid.UUID("74fff3e5-2ed9-4fb5-bbf5-e496bf62b459")

async def main():
    sf = get_session_factory()
    async with sf() as db:
        cust = (await db.execute(select(Customer).where(Customer.id == CUST))).scalar_one()
        ctx = ToolContext(db=db, tenant_id=cust.tenant_id, actor="test", customer=cust,
                          conversation_id=None, decision_trace_id=None,
                          identity_verified=False, project_id=None, extra={})
        out = await my_appointments(ctx, {})
        print("has_upcoming:", out["has_upcoming"])
        print("upcoming    :", [(a["when"], a["status"]) for a in out["upcoming"]])
        print("live        :", [(a["when"], a["status"]) for a in out["live"]])
        print("earlier_today:", [(a["when"], a["status"]) for a in out["earlier_today"]])
        print("note        :", out["note"])
        out2 = await my_appointments(ctx, {"include_past": True})
        print("past sample :", len(out2["past"]), "rows")

asyncio.run(main())
