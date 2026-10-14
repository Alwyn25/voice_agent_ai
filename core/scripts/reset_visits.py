#!/usr/bin/env python3
"""Put the demo's site visits back into a state that matches the leads' journey states.

    docker compose exec core python scripts/reset_visits.py

WHY THIS EXISTS: seed_dsr.py books Aditya and Priya a day or two ahead, then guards itself
with `if call("GET", "/appointments"): skip(...)`. So it books ONCE, ever. Days pass, those
visits fall into the past, and testing reschedules and cancels moves them around — but
Aditya's journey state still says site_visit_scheduled.

The engine is then correct and useless at the same time: "I couldn't find any current
bookings for you" is true, while the dashboard, the funnel and every rule keyed on the
journey state say he has one booked. Several confusing test calls came from this, not from
the agent.

Cancels stale visits, books fresh ones a couple of days out, and leaves journey states
agreeing with reality. Safe to run as often as you like.
"""
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import (Appointment, CallbackRequest, Location,  # noqa: E402
                           Customer, DncEntry, Escalation, Tenant, Ticket,
                           ToolExecution)
from app.services import scheduling as sched  # noqa: E402

# Every lead, back to the state seed_dsr gave them.
#
# Resetting only two of them was not enough. The suite books Sanjay a site visit, which
# moves him from qualified to site_visit_scheduled — and book_appointment is not allowed
# from site_visit_scheduled, so on the NEXT run the candidate filter drops it and the turn
# fails with "Model proposed unavailable tool 'book_appointment'". That reads like a
# regression in the candidate filter. It is yesterday's test still sitting in the database.
SEEDED_STATES = {
    "Vikram Singh": "negotiation",
    "Rohan Mehta": "lead",
    "Kavya Menon": "do_not_contact",     # seeded on the list on purpose — left alone
    "Priya Nair": "site_visit_done",
    "Sanjay Gupta": "qualified",
    "Meera Iyer": "dormant",
    "Aditya Verma": "site_visit_scheduled",
    "Nisha Pillai": "contacted",
}
WANT_VISIT = ["Aditya Verma"]          # the one lead who must have a live booking
# Testing the opt-out flow puts these two on the do-not-contact list and leaves them there,
# so the next run has nothing to opt out of and the test cannot pass twice. Kavya is seeded
# on the list deliberately and is left alone.
# Kavya is deliberately on the list and stays there; everyone else gets cleared.
CLEAR_DNC = {n: st for n, st in SEEDED_STATES.items() if st != "do_not_contact"}
ACTIVE = ("booked", "rescheduled")


async def main() -> int:
    async with get_session_factory()() as db:
        tenant = (await db.execute(select(Tenant))).scalars().first()
        if tenant is None:
            print("No tenant. Run seed.py first.")
            return 1

        centre = (await db.execute(select(Location).where(
            Location.tenant_id == tenant.id,
            Location.name.ilike("%Site Office%")))).scalars().first()
        if centre is None:
            print("No site office centre. Run seed_dsr.py first.")
            return 1

        appts = (await db.execute(select(Appointment).where(
            Appointment.tenant_id == tenant.id,
            Appointment.status.in_(ACTIVE)))).scalars().all()
        for a in appts:
            a.status = "cancelled"
        print(f"cancelled {len(appts)} stale visit(s)")

        # Cancelling above strands every caller in site_visit_scheduled with no live visit —
        # the exact stale state that deadlocks a fresh booking ("Tool not allowed in state
        # 'site_visit_scheduled'"). Reconcile them to "qualified" (the state a cancellation
        # transitions to), mirroring the engine's own self-heal at context assembly, so the
        # demo never accumulates un-bookable callers across runs. WANT_VISIT re-books below,
        # which restores site_visit_scheduled for those with a real appointment again.
        scheduled = (await db.execute(select(Customer).where(
            Customer.tenant_id == tenant.id,
            Customer.current_state == "site_visit_scheduled"))).scalars().all()
        healed = 0
        for cust in scheduled:
            live = (await db.execute(select(Appointment).where(
                Appointment.tenant_id == tenant.id, Appointment.customer_id == cust.id,
                Appointment.status.in_(ACTIVE)))).scalars().first()
            if live is None:
                cust.current_state = "qualified"
                healed += 1
        if healed:
            print(f"reconciled {healed} stale site_visit_scheduled caller(s) -> qualified")

        for name in WANT_VISIT:
            cust = (await db.execute(select(Customer).where(
                Customer.tenant_id == tenant.id,
                Customer.full_name == name))).scalars().first()
            if cust is None:
                print(f"  ! {name} not found")
                continue

            booked = None
            for offset in (2, 3, 4, 5, 6, 7):
                day = date.today() + timedelta(days=offset)
                slots = await sched.compute_slots(db, tenant.id, centre.id, day, day)
                if not slots:
                    continue
                appt = await sched.book(db, tenant_id=tenant.id, customer=cust,
                                        centre_id=centre.id, starts_at=slots[0].starts_at)
                booked = appt.starts_at
                break

            if booked:
                cust.current_state = "site_visit_scheduled"
                print(f"  {name}: booked {booked:%a %d %b %H:%M}, state site_visit_scheduled")
            else:
                print(f"  ! {name}: no free slot in the next week")

        moved = []
        for name, want in SEEDED_STATES.items():
            if name in WANT_VISIT:
                continue                      # set above, alongside a real booking
            cust = (await db.execute(select(Customer).where(
                Customer.tenant_id == tenant.id,
                Customer.full_name == name))).scalars().first()
            if cust is not None and cust.current_state != want:
                moved.append(f"{name}: {cust.current_state} -> {want}")
                cust.current_state = want
        for line in moved:
            print(f"  {line}")
        if not moved:
            print("  all other leads already in their seeded states")

        # Callbacks, and the execution records that make them idempotent.
        #
        # schedule_callback dedupes on callback:{customer}:{requested_for}, matched against
        # every past successful execution. So the second run of the suite asks for the same
        # 6am callback, the executor finds the first one and returns skipped_duplicate with
        # the ORIGINAL stored result — which was written before calling-hours clamping
        # existed, and still says six in the morning. The tool is behaving correctly; the
        # test is being answered by a fossil.
        cb = (await db.execute(select(CallbackRequest).where(
            CallbackRequest.tenant_id == tenant.id))).scalars().all()
        for c in cb:
            await db.delete(c)
        stale_exec = (await db.execute(select(ToolExecution).where(
            ToolExecution.tenant_id == tenant.id,
            ToolExecution.tool_name.in_((
                "schedule_callback", "record_do_not_contact", "record_customer_fact",
                "book_appointment", "cancel_appointment", "create_ticket",
                "update_customer_profile"))))).scalars().all()
        for e in stale_exec:
            await db.delete(e)
        if cb or stale_exec:
            print(f"  cleared {len(cb)} callback(s) and {len(stale_exec)} idempotency record(s)")

        # Tickets and escalations from earlier testing.
        #
        # These are the only demo records nothing was clearing, so they accumulated across
        # every run — a Work Queue full of a fortnight's test complaints, and an escalation
        # count on the dashboard that meant nothing. They also change behaviour: an open
        # escalation blocks outbound calls to that customer, so leftover ones quietly make
        # the next test fail for a reason that has nothing to do with the code.
        tickets = (await db.execute(select(Ticket).where(
            Ticket.tenant_id == tenant.id))).scalars().all()
        escalations = (await db.execute(select(Escalation).where(
            Escalation.tenant_id == tenant.id))).scalars().all()
        for row in tickets + escalations:
            await db.delete(row)
        if tickets or escalations:
            print(f"  cleared {len(tickets)} ticket(s) and {len(escalations)} escalation(s)")

        # Anyone left in a support state has nothing to be in it for now.
        stuck = (await db.execute(select(Customer).where(
            Customer.tenant_id == tenant.id,
            Customer.current_state.in_(("support_active", "complaint_raised"))))).scalars().all()
        for cust in stuck:
            want = SEEDED_STATES.get(cust.full_name, "lead")
            print(f"  {cust.full_name}: {cust.current_state} -> {want}")
            cust.current_state = want

        for name, back_to in CLEAR_DNC.items():
            cust = (await db.execute(select(Customer).where(
                Customer.tenant_id == tenant.id,
                Customer.full_name == name))).scalars().first()
            if cust is None:
                continue
            entries = (await db.execute(select(DncEntry).where(
                DncEntry.tenant_id == tenant.id,
                DncEntry.phone_normalized == cust.phone_normalized))).scalars().all()
            for e in entries:
                await db.delete(e)
            if cust.current_state == "do_not_contact":
                cust.current_state = back_to
            if entries or cust.current_state == back_to:
                print(f"  {name}: off the DNC list, state {back_to}")

        await db.commit()

    print("\nRe-check with: ask Aditya's line \"what's my site visit booked for\".")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))