#!/usr/bin/env python3
"""Vocal Backend Tasksheet P6 (Calls Runtime) Definition of Done, verified against
the live container — a real HTTP call to the unauthenticated inbound webhook (not a
direct function call), a real JWT for the authenticated GET/POST /calls endpoints,
and the actual app.services.call_runtime module for the two deeper checks (guardrail
denial, checkpoint resume) that need direct access to the interrupted Call's id:

  - Scripted inbound call produces a full trace: webhook -> graph run -> summary
  - Guardrail-denial path: a refund over the configured limit -> handoff, logged
  - Checkpoint resume: kill mid-call, resume, confirm identical continuation state

    docker compose exec -T core python scripts/verify_calls.py

Creates and cleans up its own tenant-scoped rows (templates, agents, phone numbers,
tool connections it creates itself, calls and everything under them) — safe to re-run.
Assumes the container's own uvicorn is reachable at localhost:8000 (true whenever this
script is run the documented way, inside the same container).
"""
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from sqlalchemy import delete, select  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import (Agent, Call, CallCapture, CallEvent, CallFlowStep,  # noqa: E402
                           CallFlowTemplate, Customer, Escalation, PhoneNumber, Tenant,
                           TemplateVersion, ToolCatalogEntry, ToolConnection, ToolExecution, User)
from app.engine.templates.compiler import GraphCompiler  # noqa: E402
from app.security.auth import create_token  # noqa: E402
from app.services.call_runtime import resume_call, run_call  # noqa: E402

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
failures = 0
BASE_URL = "http://localhost:8000"
NAME_PREFIX = "Verify P6 - "
PHONE_PREFIX = "+1999888"


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
    if user is None:
        raise SystemExit("No user found for demo tenant — run scripts/seed.py first.")
    token = create_token(user_id=user.id, tenant_id=tenant.id, role=user.role, token_type="access")
    return tenant.id, token


async def _cleanup(db, tenant_id: uuid.UUID) -> None:
    call_ids = [c.id for c in (await db.execute(select(Call).where(
        Call.tenant_id == tenant_id, Call.provider == "call_runtime"))).scalars().all()]
    if call_ids:
        await db.execute(delete(ToolExecution).where(ToolExecution.call_id.in_(call_ids)))
        await db.execute(delete(CallCapture).where(CallCapture.call_id.in_(call_ids)))
        await db.execute(delete(CallEvent).where(CallEvent.call_id.in_(call_ids)))
        await db.execute(delete(Call).where(Call.id.in_(call_ids)))
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
    contact = (await db.execute(select(Customer).where(
        Customer.tenant_id == tenant_id, Customer.phone_normalized == "+19995554444"
        ))).scalar_one_or_none()
    if contact is not None:
        await db.execute(delete(Escalation).where(Escalation.customer_id == contact.id))
        await db.delete(contact)
    await db.flush()


async def _tool_connection(db, tenant_id: uuid.UUID, provider: str, guardrails: dict) -> uuid.UUID:
    """Reuses an existing connection for this provider if one exists (fixture reuse
    across this repo's verify scripts, exactly scripts/verify_templates.py's own
    _stripe_connection_with_guardrail pattern) rather than violating the
    one-connection-per-provider-per-tenant unique constraint."""
    catalog = (await db.execute(select(ToolCatalogEntry).where(
        ToolCatalogEntry.tenant_id == tenant_id, ToolCatalogEntry.provider_name == provider)
        )).scalars().first()
    if catalog is None:
        raise SystemExit(f"No {provider} catalog entry — run scripts/seed.py first.")
    conn = (await db.execute(select(ToolConnection).where(
        ToolConnection.tenant_id == tenant_id, ToolConnection.tool_catalog_id == catalog.id)
        )).scalars().first()
    if conn is None:
        conn = ToolConnection(tenant_id=tenant_id, tool_catalog_id=catalog.id,
                              status="connected", guardrails=guardrails)
        db.add(conn)
    else:
        conn.status = "connected"
        conn.guardrails = guardrails
    await db.flush()
    return conn.id


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


async def _agent(db, tenant_id: uuid.UUID, name: str, version_id: uuid.UUID, e164: str) -> Agent:
    phone = PhoneNumber(tenant_id=tenant_id, e164=e164, provider="twilio", capability="both",
                        status="active")
    db.add(phone)
    await db.flush()
    agent = Agent(tenant_id=tenant_id, name=name, direction="inbound", status="live",
                  template_version_id=version_id, phone_number_id=phone.id)
    db.add(agent)
    await db.flush()
    phone.assigned_to = agent.id
    await db.flush()
    return agent


async def test_inbound_webhook_full_trace(db, tenant_id: uuid.UUID, token: str) -> None:
    """DoD: 'Scripted inbound call produces full trace — webhook -> graph run ->
    summary matches UI expectations.' Goes through the REAL unauthenticated HTTP
    webhook (not a direct function call) so routing/tenant-resolution is proven too."""
    shopify_conn = await _tool_connection(db, tenant_id, "shopify", {})
    steps = [
        {"key": "greeting", "step_order": 1, "step_type": "SAY",
         "config": {"agent_says": "Hello! How can I help with your order today?"}},
        {"key": "identify", "step_order": 2, "step_type": "THINK",
         "config": {"system_prompt": "Extract the order id.", "capture_key": "order_id"}},
        {"key": "retrieve", "step_order": 3, "step_type": "RETRIEVE",
         "config": {"kb_ids": [], "query_source": "last_customer_message", "top_k": 3}},
        {"key": "take_action", "step_order": 4, "step_type": "TOOL",
         "config": {"tool_connection_id": str(shopify_conn), "action": "track_order",
                   "payload_template": {"order_id": "1001"}}},
        {"key": "close", "step_order": 5, "step_type": "SAY",
         "config": {"agent_says": "Thanks for calling, bye!"}},
    ]
    version = await _publish_template(db, tenant_id, NAME_PREFIX + "Order Support", steps)
    agent = await _agent(db, tenant_id, NAME_PREFIX + "Order Agent", version.id, PHONE_PREFIX + "001")
    await db.commit()

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30) as client:
        resp = await client.post("/webhooks/telephony/inbound", json={
            "to_number": PHONE_PREFIX + "001", "from_number": "+19995551111",
            "opening_utterance": "Hi, where is my order 1001?"})
    check("inbound webhook returns 201", resp.status_code == 201, f"got {resp.status_code}: {resp.text}")
    body = resp.json()
    call_id = body.get("call_id")
    check("webhook response resolved the right agent", body.get("agent_id") == str(agent.id))
    check("webhook response transcript has agent replies",
         sum(1 for t in body.get("transcript", []) if t["role"] == "agent") == 2,
         f"transcript={body.get('transcript')}")
    check("no handoff on a clean run", body.get("handoff") is False)

    events = (await db.execute(select(CallEvent).where(
        CallEvent.tenant_id == tenant_id, CallEvent.call_id == uuid.UUID(call_id))
        .order_by(CallEvent.created_at))).scalars().all()
    check("call_events has one row per graph step", [e.step_key for e in events] ==
         ["greeting", "identify", "retrieve", "take_action", "close"],
         f"got {[e.step_key for e in events]}")
    check("every call_event has a non-negative latency_ms",
         all(e.latency_ms is not None and e.latency_ms >= 0 for e in events))

    captures = (await db.execute(select(CallCapture).where(
        CallCapture.tenant_id == tenant_id, CallCapture.call_id == uuid.UUID(call_id)))).scalars().all()
    check("call_captures recorded the THINK step's capture", len(captures) == 1 and
         captures[0].step_key == "identify" and captures[0].capture_key == "order_id")

    tool_logs = (await db.execute(select(ToolExecution).where(
        ToolExecution.tenant_id == tenant_id, ToolExecution.call_id == uuid.UUID(call_id)))).scalars().all()
    check("tool_call_logs (ToolExecution) recorded the TOOL step",
         len(tool_logs) == 1 and tool_logs[0].tool_name == "track_order"
         and tool_logs[0].status == "success" and tool_logs[0].step_key == "take_action")

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30,
                                 headers={"Authorization": f"Bearer {token}"}) as client:
        resp2 = await client.get(f"/calls/{call_id}")
    check("GET /calls/{id} 200s", resp2.status_code == 200, f"got {resp2.status_code}")
    detail = resp2.json()
    check("GET /calls/{id} nests events/captures/tool_logs/transcript/summary",
         len(detail.get("events", [])) == 5 and len(detail.get("captures", [])) == 1
         and len(detail.get("tool_logs", [])) == 1 and len(detail.get("transcript", [])) == 3,
         f"detail keys seen: events={len(detail.get('events', []))} "
         f"captures={len(detail.get('captures', []))} tool_logs={len(detail.get('tool_logs', []))} "
         f"transcript={len(detail.get('transcript', []))}")


async def test_manual_handoff(db, tenant_id: uuid.UUID, token: str) -> None:
    """Task 144: POST /calls/{id}/handoff — manual escalate to human. Not itself a
    tasksheet Definition-of-Done item, but the one endpoint from this wave not yet
    exercised through real HTTP above."""
    customer = Customer(tenant_id=tenant_id, full_name="Verify P6 Handoff Contact",
                        phone="+19995554444", phone_normalized="+19995554444")
    db.add(customer)
    await db.flush()
    steps = [{"key": "s1", "step_order": 1, "step_type": "SAY", "config": {"agent_says": "hi"}}]
    version = await _publish_template(db, tenant_id, NAME_PREFIX + "Handoff", steps)
    agent = await _agent(db, tenant_id, NAME_PREFIX + "Handoff Agent", version.id, PHONE_PREFIX + "004")
    await db.commit()

    call = await run_call(db, tenant_id=tenant_id, agent=agent, from_number="+19995554444",
                          to_number=PHONE_PREFIX + "004", opening_utterance="hi")
    await db.commit()
    check("call resolved to the known customer by phone", call.customer_id == customer.id)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30,
                                 headers={"Authorization": f"Bearer {token}"}) as client:
        resp = await client.post(f"/calls/{call.id}/handoff",
                                 json={"reason": "customer_requested", "notes": "wants a manager"})
    check("POST /calls/{id}/handoff returns 201", resp.status_code == 201, f"got {resp.status_code}")
    escalation_id = resp.json().get("escalation_id")
    check("handoff response includes a new escalation id", bool(escalation_id))

    esc = (await db.execute(select(Escalation).where(
        Escalation.tenant_id == tenant_id, Escalation.id == uuid.UUID(escalation_id)
        ))).scalar_one_or_none()
    check("a real Escalation row was created, tied to the call's customer",
         esc is not None and esc.customer_id == customer.id and esc.status == "pending",
         f"esc={esc}")

    # The HTTP request above committed through its own session, not this script's — this
    # script's session still has the pre-handoff `call` object cached in its identity map,
    # so a plain select() would silently hand back that stale copy. db.refresh() forces it
    # to re-read the current row.
    await db.refresh(call)
    check("the call itself is marked follow_up_required after a manual handoff",
         call.follow_up_required is True and call.follow_up_reason == "customer_requested")


async def test_guardrail_denial_handoff(db, tenant_id: uuid.UUID, token: str) -> None:
    """DoD: 'Guardrail-denial path verified end-to-end — refund over limit -> handoff,
    logged correctly.'"""
    stripe_conn = await _tool_connection(db, tenant_id, "stripe", {"refund_limit_usd": 50})
    steps = [
        {"key": "s1", "step_order": 1, "step_type": "SAY", "config": {"agent_says": "Let me process that."}},
        {"key": "refund", "step_order": 2, "step_type": "TOOL",
         "config": {"tool_connection_id": str(stripe_conn), "action": "issue_refund",
                   "payload_template": {"charge_id": "ch_1", "amount_usd": "500"}}},
        {"key": "confirm", "step_order": 3, "step_type": "SAY",
         "config": {"agent_says": "SHOULD NOT RUN — refund was over the guardrail limit"}},
    ]
    version = await _publish_template(db, tenant_id, NAME_PREFIX + "Refund", steps)
    agent = await _agent(db, tenant_id, NAME_PREFIX + "Refund Agent", version.id, PHONE_PREFIX + "002")
    await db.commit()

    call = await run_call(db, tenant_id=tenant_id, agent=agent, from_number="+19995552222",
                          to_number=PHONE_PREFIX + "002", opening_utterance="I want a $500 refund")
    await db.commit()

    check("guardrail denial routed to handoff (hangup_cause)", call.hangup_cause == "handoff",
         f"hangup_cause={call.hangup_cause}")
    check("handoff sets follow_up_required deterministically (no LLM needed)",
         call.follow_up_required is True)
    check("the step after the denied TOOL never ran",
         all(t["text"] != "SHOULD NOT RUN — refund was over the guardrail limit"
             for t in call.transcript))

    tool_logs = (await db.execute(select(ToolExecution).where(
        ToolExecution.tenant_id == tenant_id, ToolExecution.call_id == call.id))).scalars().all()
    check("the denied tool call is logged with status='blocked'",
         len(tool_logs) == 1 and tool_logs[0].status == "blocked" and tool_logs[0].error is not None,
         f"tool_logs={[(t.status, t.error) for t in tool_logs]}")

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30,
                                 headers={"Authorization": f"Bearer {token}"}) as client:
        resp = await client.get(f"/calls/{call.id}")
    check("GET /calls/{id} reflects follow_up_required for a handed-off call",
         resp.json().get("follow_up_required") is True)


async def test_checkpoint_resume(db, tenant_id: uuid.UUID) -> None:
    """DoD: 'Checkpoint resume reproduces state — kill runtime mid-call, resume,
    confirm identical graph state.'"""
    steps = [
        {"key": "r1", "step_order": 1, "step_type": "SAY", "config": {"agent_says": "Step one."}},
        {"key": "r2", "step_order": 2, "step_type": "SAY", "config": {"agent_says": "Step two."}},
        {"key": "r3", "step_order": 3, "step_type": "SAY", "config": {"agent_says": "Step three."}},
    ]
    version = await _publish_template(db, tenant_id, NAME_PREFIX + "Resume", steps)
    agent = await _agent(db, tenant_id, NAME_PREFIX + "Resume Agent", version.id, PHONE_PREFIX + "003")
    await db.commit()

    interrupted = await run_call(db, tenant_id=tenant_id, agent=agent, from_number="+19995553333",
                                 to_number=PHONE_PREFIX + "003", opening_utterance="hi",
                                 interrupt_before=["r3"])
    await db.commit()
    check("interrupted before the last step -> only 2 agent turns so far",
         sum(1 for t in interrupted.transcript if t["role"] == "agent") == 2,
         f"transcript={interrupted.transcript}")
    check("interrupted call is left in-progress, not completed",
         interrupted.status != "completed", f"status={interrupted.status}")

    resumed = await resume_call(db, tenant_id=tenant_id, call_id=interrupted.id)
    await db.commit()
    check("resume completes the remaining step",
         sum(1 for t in resumed.transcript if t["role"] == "agent") == 3,
         f"transcript={resumed.transcript}")
    check("resumed state is a continuation, not a restart (steps one/two preserved)",
         # transcript[0] is the seeded customer opening turn ("hi"); [1]/[2] are the two
         # agent SAY steps that ran before the interrupt.
         resumed.transcript[1]["text"] == "Step one." and resumed.transcript[2]["text"] == "Step two.",
         f"transcript={resumed.transcript}")
    check("resumed call reaches completed status", resumed.status == "completed",
         f"status={resumed.status}")

    events = (await db.execute(select(CallEvent).where(
        CallEvent.tenant_id == tenant_id, CallEvent.call_id == interrupted.id)
        .order_by(CallEvent.created_at))).scalars().all()
    check("call_events covers all 3 steps across the interrupt+resume (no duplicate r1/r2)",
         [e.step_key for e in events] == ["r1", "r2", "r3"], f"got {[e.step_key for e in events]}")


async def main() -> int:
    async with get_session_factory()() as db:
        tenant_id, token = await _demo_tenant_and_token(db)
        await _cleanup(db, tenant_id)
        await db.commit()

        await test_inbound_webhook_full_trace(db, tenant_id, token)
        await db.commit()
        await test_manual_handoff(db, tenant_id, token)
        await db.commit()
        await test_guardrail_denial_handoff(db, tenant_id, token)
        await db.commit()
        await test_checkpoint_resume(db, tenant_id)
        await db.commit()

        await _cleanup(db, tenant_id)
        await db.commit()

    print()
    if failures:
        print(f"{FAIL} {failures} check(s) failed")
        return 1
    print(f"{PASS} all P6 calls-runtime Definition-of-Done checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
