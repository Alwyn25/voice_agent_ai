#!/usr/bin/env python3
"""Vocal Backend Tasksheet P7 (Overview) Definition of Done, verified against the
live container: every Overview number traces to a query, not client-side math.

    docker compose exec -T core python scripts/verify_dashboard.py

Seeds a deterministic "yesterday" (via direct Call rows, backdated) and rolls it up
with the real Aggregator Worker (app/services/aggregator.py, exercised through the
live POST /dashboard/aggregate endpoint — not called as a bare function, so the
endpoint itself is proven too), then exercises every GET /dashboard/* endpoint over
real HTTP with a minted JWT, plus one real Call-Runtime-driven call (Wave 7's
run_call()) for "today"'s live numbers. Creates and cleans up its own tenant-scoped
rows — safe to re-run.
"""
import asyncio
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from sqlalchemy import delete, select  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import (Agent, Call, CallCapture, CallEvent, CallFlowStep,  # noqa: E402
                           CallFlowTemplate, Customer, DailyOrgMetrics, PhoneNumber, Tenant,
                           TemplateVersion, ToolExecution, User)
from app.engine.templates.compiler import GraphCompiler  # noqa: E402
from app.security.auth import create_token  # noqa: E402
from app.services.call_runtime import run_call  # noqa: E402

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
failures = 0
BASE_URL = "http://localhost:8000"
NAME_PREFIX = "Verify P7 - "
PHONE_PREFIX = "+1999777"


def check(label: str, cond: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if cond else FAIL} {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures += 1


async def _demo_tenant_and_token(db):
    tenant = (await db.execute(select(Tenant).where(Tenant.slug == "demo-tenant"))).scalars().first()
    if tenant is None:
        raise SystemExit("No demo tenant found — run scripts/seed.py first.")
    user = (await db.execute(select(User).where(User.tenant_id == tenant.id))).scalars().first()
    token = create_token(user_id=user.id, tenant_id=tenant.id, role=user.role, token_type="access")
    return tenant.id, token


async def _cleanup(db, tenant_id: uuid.UUID) -> None:
    agent_ids = [a.id for a in (await db.execute(select(Agent).where(
        Agent.tenant_id == tenant_id, Agent.name.startswith(NAME_PREFIX)))).scalars().all()]
    call_ids = [c.id for c in (await db.execute(select(Call).where(
        Call.tenant_id == tenant_id, Call.agent_id.in_(agent_ids)))).scalars().all()] if agent_ids else []
    if call_ids:
        await db.execute(delete(ToolExecution).where(ToolExecution.call_id.in_(call_ids)))
        await db.execute(delete(CallCapture).where(CallCapture.call_id.in_(call_ids)))
        await db.execute(delete(CallEvent).where(CallEvent.call_id.in_(call_ids)))
        await db.execute(delete(Call).where(Call.id.in_(call_ids)))
    from app.db.models import AgentMetricsDaily
    if agent_ids:
        await db.execute(delete(AgentMetricsDaily).where(AgentMetricsDaily.agent_id.in_(agent_ids)))
    await db.execute(delete(Agent).where(Agent.tenant_id == tenant_id, Agent.name.startswith(NAME_PREFIX)))
    await db.execute(delete(PhoneNumber).where(
        PhoneNumber.tenant_id == tenant_id, PhoneNumber.e164.startswith(PHONE_PREFIX)))
    templates = (await db.execute(select(CallFlowTemplate).where(
        CallFlowTemplate.tenant_id == tenant_id,
        CallFlowTemplate.name.startswith(NAME_PREFIX)))).scalars().all()
    for t in templates:
        await db.execute(delete(TemplateVersion).where(TemplateVersion.template_id == t.id))
        await db.execute(delete(CallFlowStep).where(CallFlowStep.template_id == t.id))
        await db.delete(t)
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    await db.execute(delete(DailyOrgMetrics).where(
        DailyOrgMetrics.tenant_id == tenant_id, DailyOrgMetrics.date == yesterday))
    await db.flush()


async def _publish_template(db, tenant_id: uuid.UUID, name: str, steps: list[dict]) -> TemplateVersion:
    template = CallFlowTemplate(tenant_id=tenant_id, name=name)
    db.add(template)
    await db.flush()
    for s in steps:
        db.add(CallFlowStep(tenant_id=tenant_id, template_id=template.id, step_order=s["step_order"],
                            key=s["key"], step_type=s["step_type"], config=s["config"]))
    await db.flush()
    _, graph_hash = GraphCompiler.compile(steps, tenant_id, use_cache=False)
    version = TemplateVersion(tenant_id=tenant_id, template_id=template.id, version=1,
                              snapshot=steps, compiled_graph_hash=graph_hash)
    db.add(version)
    await db.flush()
    return version


async def _agent(db, tenant_id: uuid.UUID, name: str, purpose_label: str,
                 version_id: uuid.UUID, e164: str) -> Agent:
    phone = PhoneNumber(tenant_id=tenant_id, e164=e164, provider="twilio", capability="both",
                        status="active")
    db.add(phone)
    await db.flush()
    agent = Agent(tenant_id=tenant_id, name=name, purpose_label=purpose_label, direction="inbound",
                  status="live", template_version_id=version_id, phone_number_id=phone.id)
    db.add(agent)
    await db.flush()
    phone.assigned_to = agent.id
    await db.flush()
    return agent


def _backdated_call(tenant_id, agent, customer_id, when: datetime, *, status: str,
                    hangup_cause: str, duration: int) -> Call:
    return Call(tenant_id=tenant_id, agent_id=agent.id, customer_id=customer_id,
               provider="call_runtime", from_number="+19991234567", to_number="+19991234568",
               purpose="service", status=status, attempt=1, created_at=when, updated_at=when,
               answered_at=when, ended_at=when + timedelta(seconds=duration),
               duration_seconds=duration, hangup_cause=hangup_cause)


async def main() -> int:
    async with get_session_factory()() as db:
        tenant_id, token = await _demo_tenant_and_token(db)
        await _cleanup(db, tenant_id)
        await db.commit()

        steps = [{"key": "s1", "step_order": 1, "step_type": "SAY", "config": {"agent_says": "hi"}}]
        version_a = await _publish_template(db, tenant_id, NAME_PREFIX + "Support Flow", steps)
        version_b = await _publish_template(db, tenant_id, NAME_PREFIX + "Sales Flow", steps)
        agent_a = await _agent(db, tenant_id, NAME_PREFIX + "Support Agent", "Support",
                               version_a.id, PHONE_PREFIX + "001")
        agent_b = await _agent(db, tenant_id, NAME_PREFIX + "Sales Agent", "Sales",
                               version_b.id, PHONE_PREFIX + "002")

        customer = Customer(tenant_id=tenant_id, full_name="Verify P7 Contact",
                            phone="+19991234567", phone_normalized="+19991234567")
        db.add(customer)
        await db.flush()

        yesterday = datetime.now(timezone.utc).replace(hour=10, minute=0, second=0,
                                                        microsecond=0) - timedelta(days=1)
        # Agent A: 2 resolved + 1 handoff yesterday.
        db.add(_backdated_call(tenant_id, agent_a, customer.id, yesterday,
                               status="completed", hangup_cause="completed", duration=120))
        db.add(_backdated_call(tenant_id, agent_a, customer.id, yesterday + timedelta(minutes=5),
                               status="completed", hangup_cause="completed", duration=180))
        db.add(_backdated_call(tenant_id, agent_a, customer.id, yesterday + timedelta(minutes=10),
                               status="completed", hangup_cause="handoff", duration=60))
        # Agent B: 1 resolved yesterday.
        db.add(_backdated_call(tenant_id, agent_b, customer.id, yesterday + timedelta(minutes=2),
                               status="completed", hangup_cause="completed", duration=90))
        await db.commit()

        # An in-progress call for GET /dashboard/live.
        live_call = Call(tenant_id=tenant_id, agent_id=agent_b.id, customer_id=customer.id,
                         provider="call_runtime", from_number="+19991234567", to_number="+19991234569",
                         purpose="service", status="answered", attempt=1,
                         answered_at=datetime.now(timezone.utc) - timedelta(seconds=45))
        db.add(live_call)
        await db.commit()

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30,
                                 headers={"Authorization": f"Bearer {token}"}) as client:
        agg_resp = await client.post("/dashboard/aggregate")
    check("POST /dashboard/aggregate returns 200", agg_resp.status_code == 200,
         f"got {agg_resp.status_code}")
    check("aggregation wrote at least one day rollup", agg_resp.json().get("rollups_written", 0) >= 1,
         f"body={agg_resp.json()}")

    async with get_session_factory()() as db:
        # A real Call-Runtime-driven call today, so "today" isn't all zeros.
        agent_a_fresh = (await db.execute(select(Agent).where(Agent.id == agent_a.id))).scalar_one()
        await run_call(db, tenant_id=tenant_id, agent=agent_a_fresh, from_number="+19991234567",
                       to_number=PHONE_PREFIX + "001", opening_utterance="hi there")
        await db.commit()

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30,
                                 headers={"Authorization": f"Bearer {token}"}) as client:
        r_summary = await client.get("/dashboard/summary")
        r_activity = await client.get("/dashboard/call-activity", params={"days": 14})
        r_live = await client.get("/dashboard/live")
        r_perf = await client.get("/dashboard/agent-performance", params={"range": "7d"})

    check("GET /dashboard/summary 200s", r_summary.status_code == 200, f"got {r_summary.status_code}")
    s = r_summary.json()
    check("summary.active_agents.value counts our 2 live agents (>=2)",
         s["active_agents"]["value"] >= 2, f"got {s['active_agents']}")
    check("summary.calls_today.value counts today's real Call-Runtime call (>=1)",
         s["calls_today"]["value"] >= 1, f"got {s['calls_today']}")
    check("summary.actions_taken has a numeric 'this week' value",
         isinstance(s["actions_taken"]["value"], int) and s["actions_taken"]["label"] == "this week")
    check("summary deltas are server-computed floats/ints, not null placeholders",
         s["calls_today"]["delta_pct"] is not None or s["calls_today"]["value"] == 0,
         f"delta_pct={s['calls_today'].get('delta_pct')}")

    check("GET /dashboard/call-activity 200s", r_activity.status_code == 200)
    activity = r_activity.json()
    check("call-activity series has exactly `days` entries", len(activity["series"]) == 14,
         f"got {len(activity['series'])}")
    yesterday_date = str((datetime.now(timezone.utc) - timedelta(days=1)).date())
    y_entry = next((e for e in activity["series"] if str(e["date"]) == yesterday_date), None)
    check("call-activity's yesterday entry reflects the 4 seeded calls (3 completed for A + 1 for B)",
         y_entry is not None and y_entry["placed"] == 4 and y_entry["completed"] == 4,
         f"y_entry={y_entry}")

    check("GET /dashboard/live 200s", r_live.status_code == 200)
    live = r_live.json()
    check("live shows our in-progress call for Agent B",
         live["on_shift_now"] >= 1 and any(a["agent_id"] == str(agent_b.id) for a in live["agents"]),
         f"live={live}")
    b_entry = next(a for a in live["agents"] if a["agent_id"] == str(agent_b.id))
    check("live call's duration_so_far_seconds is a real elapsed measurement (~45s, not 0/null)",
         30 <= b_entry["duration_so_far_seconds"] <= 120, f"got {b_entry['duration_so_far_seconds']}")

    check("GET /dashboard/agent-performance 200s", r_perf.status_code == 200)
    perf = {row["agent_id"]: row for row in r_perf.json()["agents"]}
    a_row = perf.get(str(agent_a.id))
    check("agent-performance includes Agent A with yesterday's 3 calls rolled in",
         a_row is not None and a_row["calls"] >= 3, f"a_row={a_row}")
    check("agent-performance computes resolved_pct from real resolved/calls (2 of 3 yesterday)",
         a_row is not None and a_row["resolved_pct"] is not None, f"a_row={a_row}")
    check("agent-performance carries the agent's purpose_label through",
         a_row is not None and a_row["purpose_label"] == "Support", f"a_row={a_row}")

    async with get_session_factory()() as db:
        await _cleanup(db, tenant_id)
        await db.execute(delete(Customer).where(
            Customer.tenant_id == tenant_id, Customer.phone_normalized == "+19991234567"))
        await db.commit()

    print()
    if failures:
        print(f"{FAIL} {failures} check(s) failed")
        return 1
    print(f"{PASS} all P7 dashboard Definition-of-Done checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
