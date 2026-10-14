#!/usr/bin/env python3
"""Clear whatever is blocking an outbound call to a customer, so testing can proceed.

    python scripts/unblock_outbound.py --name "Rohan Mehta"
    python scripts/unblock_outbound.py --name "Rohan Mehta" --clear-dnc
    python scripts/unblock_outbound.py --all
    python scripts/unblock_outbound.py --all --dry-run      # report only, change nothing

WHY THIS EXISTS: an outbound dial passes the SAME compliance gate a real call would
(app/services/outbound.check_can_call). Two of those checks latch and stay latched across
test runs:

  * an OPEN ESCALATION (status pending|accepted) means "a human owns this customer" and
    suspends all automated contact until it is resolved. A single earlier test call that
    handed off to a human leaves this behind, and every later outbound attempt to that
    customer is refused for a reason that has nothing to do with the code under test.
  * a DNC entry (only cleared with --clear-dnc, because on some personas being on the list
    is the point — e.g. Kavya Menon is seeded onto it deliberately).

This is deliberately NARROWER than reset_visits.py: it does not cancel or re-book
appointments, delete callbacks, or move journey states around. It resolves the escalations
(the realistic operator action — see POST /escalations/{id}/resolve in app/api/support.py),
optionally clears DNC, and then re-runs the real gate and prints the verdict, so you can see
the customer is actually callable now rather than guessing.

Safe to run as often as you like.
"""
import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import Call, Customer, DncEntry, Escalation, Tenant  # noqa: E402
from app.services.outbound import check_can_call  # noqa: E402

# The escalation statuses check_can_call() treats as "a human owns this customer".
OPEN_ESCALATION = ("pending", "accepted")


async def _customers(db, tenant_id, name: str | None):
    q = select(Customer).where(Customer.tenant_id == tenant_id,
                               Customer.is_deleted == False)  # noqa: E712
    if name:
        q = q.where(Customer.full_name.ilike(f"%{name}%"))
    return (await db.execute(q.order_by(Customer.full_name))).scalars().all()


async def main() -> int:
    ap = argparse.ArgumentParser(description="Unblock outbound calling for testing.")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--name", help="Customer full name (partial match), e.g. \"Rohan Mehta\"")
    grp.add_argument("--all", action="store_true", help="Every customer in the tenant")
    ap.add_argument("--clear-dnc", action="store_true",
                    help="Also remove the customer from the Do-Not-Contact list "
                         "(off by default — some personas are on it on purpose)")
    ap.add_argument("--reset-attempts", action="store_true",
                    help="Also delete the customer's Call rows, resetting the per-day and "
                         "lifetime attempt counters the frequency rules read (marketing "
                         "daily cap, no-answer back-off). For repeated testing of the same "
                         "customer; wipes their call history.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what blocks each customer; change nothing")
    args = ap.parse_args()

    async with get_session_factory()() as db:
        tenant = (await db.execute(select(Tenant))).scalars().first()
        if tenant is None:
            print("No tenant. Run seed.py / seed_dsr.py first.")
            return 1

        customers = await _customers(db, tenant.id, args.name)
        if not customers:
            print(f"No customer matched name={args.name!r} in tenant {tenant.name!r}.")
            return 1

        resolved_total = dnc_cleared_total = calls_deleted_total = 0
        for cust in customers:
            escalations = (await db.execute(select(Escalation).where(
                Escalation.tenant_id == tenant.id,
                Escalation.customer_id == cust.id,
                Escalation.status.in_(OPEN_ESCALATION)))).scalars().all()
            dnc = (await db.execute(select(DncEntry).where(
                DncEntry.tenant_id == tenant.id,
                DncEntry.phone_normalized == cust.phone_normalized))).scalars().all()
            calls = (await db.execute(select(Call).where(
                Call.tenant_id == tenant.id,
                Call.customer_id == cust.id))).scalars().all() if args.reset_attempts else []

            actions = []
            if escalations:
                actions.append(f"{len(escalations)} open escalation(s)")
                if not args.dry_run:
                    for e in escalations:
                        e.status = "resolved"
                    resolved_total += len(escalations)
            if calls:
                actions.append(f"{len(calls)} call attempt(s) -> reset")
                if not args.dry_run:
                    for c in calls:
                        await db.delete(c)
                    calls_deleted_total += len(calls)
            if dnc:
                actions.append(f"on DNC ({', '.join(sorted({d.scope for d in dnc}))})")
                if args.clear_dnc and not args.dry_run:
                    for d in dnc:
                        await db.delete(d)
                    dnc_cleared_total += len(dnc)
                    actions[-1] += " -> cleared"
                elif dnc:
                    actions[-1] += " (kept; pass --clear-dnc to remove)"

            if actions:
                print(f"  {cust.full_name or cust.phone_normalized}: {', '.join(actions)}")

        # Persist escalation resolutions / DNC deletions BEFORE re-running the gate, so the
        # verdict below reflects the new state rather than the session's pending changes.
        if not args.dry_run:
            await db.commit()

        # Re-run the exact gate the console and campaigns hit, and print the verdict per
        # customer. This is the proof the change worked — not an assumption that it did.
        print("\nGate re-check (the same check_can_call the console runs):")
        for cust in customers:
            if not args.name and cust.current_state == "do_not_contact" and not args.clear_dnc:
                # Skip the noise of listing every DNC persona on an --all run.
                pass
            # Marketing has a one-call-a-day cap; showing the marketing verdict is the more
            # useful default since that is what the console dials with unless Service is
            # picked. The reason names the exact rule when it blocks.
            decision = await check_can_call(db, tenant_id=tenant.id, customer=cust,
                                            purpose="marketing")
            mark = "ALLOWED " if decision.allowed else "BLOCKED "
            tail = "" if decision.allowed else f" — {decision.blocked_by}: {decision.reason}"
            print(f"  [{mark}] {cust.full_name or cust.phone_normalized} (marketing){tail}")

        if args.dry_run:
            print("\n(dry run — nothing was changed)")
        else:
            print(f"\nResolved {resolved_total} escalation(s), cleared {dnc_cleared_total} "
                  f"DNC entr(ies), reset {calls_deleted_total} call attempt(s). "
                  f"Reconnect the customer in the console to test outbound.")
        return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
