"""Seed a handful of real, completed calls (with conversations, transcripts and tool
traces) against an existing tenant's campaign and agent, so the UI's real data paths
(Calls page, an opened campaign's results table, the customer-detail drawer) have
something genuine to render instead of the "no data yet" preview rows.

Usage (inside core/ with deps installed and DB up):
    python scripts/seed_demo_calls.py --tenant-slug demo-tenant

Idempotent: running it twice skips customers/campaign that already exist by name/phone,
but always adds a fresh batch of calls (each call is a new event, not a singleton).
"""
import argparse
import asyncio
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import (Agent, Call, Campaign, CampaignTarget, Conversation,  # noqa: E402
                           Customer, DecisionTrace, Message, Tenant)

DEMO_CUSTOMERS = [
    {
        "full_name": "Dana Whitfield", "phone": "+14155550132",
        "summary": "Parcel delayed; moved delivery to Thursday AM.",
        "intent": "Rescheduled", "sentiment": "positive", "outcome": "Rescheduled delivery",
        "transcript": [
            ("agent", "Hi, this is Ava from Northline. How can I help today?"),
            ("customer", "My order was meant to arrive today and it has not turned up."),
            ("agent", "Sorry about that. Can I take the order number?"),
            ("customer", "It is 48120."),
            ("agent", "Thanks. That parcel is with the carrier and now shows Thursday. "
                       "I can move the delivery to Thursday morning if that suits."),
            ("customer", "Thursday morning works."),
            ("agent", "Done — Thursday between 9 and 12, and I have texted the confirmation. "
                       "Anything else?"),
            ("customer", "No, that is all. Thank you."),
        ],
        "tools": [
            ("Customer Profile Manager", "verified caller by order number", "success"),
            ("Order Manager", "fetched shipment status", "success"),
            ("Order Manager", "rescheduled delivery to Thu 9-12", "success"),
            ("Notification Manager", "sent SMS confirmation", "success"),
        ],
        "duration_seconds": 161, "follow_up_required": False,
    },
    {
        "full_name": "Marcus Idowu", "phone": "+13125550188",
        "summary": "Damaged lamp; $89 refunded to card.",
        "intent": "Refund requested", "sentiment": "neutral", "outcome": "Refund issued",
        "transcript": [
            ("agent", "Hi, this is Ava from Northline. How can I help today?"),
            ("customer", "The lamp I ordered arrived with a cracked base."),
            ("agent", "I'm sorry to hear that — let me pull up your order."),
            ("customer", "It's order 48211."),
            ("agent", "Found it. I can refund the $89 to your original card right now."),
            ("customer", "That works."),
            ("agent", "Done — refund issued, 3-5 business days to appear. Anything else?"),
            ("customer", "No, thanks."),
        ],
        "tools": [
            ("Customer Profile Manager", "verified caller by order number", "success"),
            ("Order Manager", "fetched order details", "success"),
            ("Refund Manager", "issued $89 refund", "success"),
        ],
        "duration_seconds": 198, "follow_up_required": False,
    },
    {
        "full_name": "Priya Nair", "phone": "+919845550177",
        "summary": "Wants a callback after checking with her partner about the renewal.",
        "intent": "Renewal", "sentiment": "neutral", "outcome": "Follow-up requested",
        "transcript": [
            ("agent", "Hi, this is Ava from Northline, calling about your upcoming renewal."),
            ("customer", "Oh — I need to check with my partner before deciding."),
            ("agent", "No problem. When's a good time to call back?"),
            ("customer", "Try me again tomorrow evening."),
            ("agent", "Noted — I'll have someone call back tomorrow evening. Thanks for your time."),
        ],
        "tools": [
            ("Customer Profile Manager", "verified caller by phone", "success"),
            ("Scheduling Manager", "logged callback request", "success"),
        ],
        "duration_seconds": 74, "follow_up_required": True,
    },
]


async def main(tenant_slug: str, campaign_name: str) -> None:
    async with get_session_factory()() as db:
        tenant = (await db.execute(select(Tenant).where(
            Tenant.slug == tenant_slug))).scalar_one_or_none()
        if tenant is None:
            print(f"tenant '{tenant_slug}' not found — run scripts/seed.py first")
            return

        agent = (await db.execute(select(Agent).where(
            Agent.tenant_id == tenant.id, Agent.status == "live"))).scalars().first()
        if agent is None:
            agent = (await db.execute(select(Agent).where(
                Agent.tenant_id == tenant.id))).scalars().first()
        if agent is None:
            print("no agent exists for this tenant — create one in the UI first")
            return
        print(f"using agent '{agent.name}' ({agent.id})")

        campaign = (await db.execute(select(Campaign).where(
            Campaign.tenant_id == tenant.id, Campaign.name == campaign_name))).scalar_one_or_none()
        if campaign is None:
            campaign = Campaign(tenant_id=tenant.id, name=campaign_name, status="running",
                                agent_id=agent.id)
            db.add(campaign)
            await db.flush()
            print(f"created campaign '{campaign.name}' ({campaign.id})")
        else:
            print(f"campaign exists: '{campaign.name}' ({campaign.id})")

        now = datetime.now(timezone.utc)
        for i, spec in enumerate(DEMO_CUSTOMERS):
            customer = (await db.execute(select(Customer).where(
                Customer.tenant_id == tenant.id,
                Customer.phone_normalized == spec["phone"]))).scalar_one_or_none()
            if customer is None:
                customer = Customer(tenant_id=tenant.id, full_name=spec["full_name"],
                                    phone=spec["phone"], phone_normalized=spec["phone"],
                                    current_state="customer", lead_source="demo_seed")
                db.add(customer)
                await db.flush()
                print(f"  created customer {customer.full_name}")

            target = (await db.execute(select(CampaignTarget).where(
                CampaignTarget.campaign_id == campaign.id,
                CampaignTarget.customer_id == customer.id))).scalar_one_or_none()
            if target is None:
                db.add(CampaignTarget(tenant_id=tenant.id, campaign_id=campaign.id,
                                      customer_id=customer.id, status="done", attempts=1,
                                      last_attempt_at=now - timedelta(minutes=30 * (i + 1))))

            started_at = now - timedelta(minutes=30 * (i + 1))
            ended_at = started_at + timedelta(seconds=spec["duration_seconds"])
            conv = Conversation(
                tenant_id=tenant.id, customer_id=customer.id, channel="voice",
                direction="outbound", status="completed", started_at=started_at,
                ended_at=ended_at, summary=spec["summary"], outcome=spec["outcome"],
                sentiment=spec["sentiment"], intent_labels=[spec["intent"]],
                campaign_id=campaign.id,
            )
            db.add(conv)
            await db.flush()

            for turn_index, (role, text) in enumerate(spec["transcript"]):
                db.add(Message(tenant_id=tenant.id, conversation_id=conv.id, role=role,
                               content=text,
                               created_at=started_at + timedelta(seconds=turn_index * 12)))

            for turn_index, (tool, action, result) in enumerate(spec["tools"]):
                db.add(DecisionTrace(
                    tenant_id=tenant.id, conversation_id=conv.id, customer_id=customer.id,
                    turn_index=turn_index, proposed_tool=tool, action_taken="executed",
                    validation_result={"result": result},
                    planner_reasoning=action,
                    created_at=started_at + timedelta(seconds=turn_index * 14),
                ))

            call = Call(
                tenant_id=tenant.id, conversation_id=conv.id, customer_id=customer.id,
                provider="mock", provider_call_id=f"demo-{uuid.uuid4().hex[:12]}",
                from_number="+18005550100", to_number=spec["phone"], purpose="marketing",
                campaign_id=campaign.id, status="completed", attempt=1,
                answered_at=started_at, ended_at=ended_at,
                duration_seconds=spec["duration_seconds"], hangup_cause="normal_clearing",
                disclosure_played=True, cost=round(spec["duration_seconds"] * 0.008, 4),
                agent_id=agent.id, transcript=[{"speaker": r, "text": t} for r, t in spec["transcript"]],
                summary=spec["summary"], intent=spec["intent"], sentiment=spec["sentiment"],
                follow_up_required=spec["follow_up_required"],
                created_at=started_at,
            )
            db.add(call)
            print(f"  added call for {customer.full_name} ({spec['intent']})")

        await db.commit()
        print("seed complete")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tenant-slug", default="demo-tenant")
    p.add_argument("--campaign-name", default="Delivery Delay Apology — Demo")
    args = p.parse_args()
    asyncio.run(main(args.tenant_slug, args.campaign_name))
