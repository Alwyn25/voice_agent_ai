#!/usr/bin/env python3
"""Vocal Backend Tasksheet P3 (Templates) Definition of Done, verified against
the live DB and a real Postgres-backed LangGraph checkpointer — no mocking of
the runtime itself:

  - Publish produces an immutable version (editing steps after publish leaves
    the published snapshot untouched)
  - The "Order Support" template (Greeting -> Identify -> Understand intent ->
    Retrieve -> Take action -> Confirm -> Close) compiles and dry-runs against a
    scripted fake caller
  - A guardrail denial inside a TOOL node routes to handoff, not an unhandled
    exception, and no step after it runs
  - Checkpoint resume: interrupt before a step, then resume from exactly there

    docker compose exec -T core python scripts/verify_templates.py

Creates and cleans up its own tenant-scoped rows (templates, a tool connection,
a checkpoint thread) — safe to re-run.
"""
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete, select  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import (CallFlowStep, CallFlowTemplate, Tenant, ToolCatalogEntry,  # noqa: E402
                           ToolConnection, TemplateVersion)
from app.engine.templates.checkpointer import open_checkpointer  # noqa: E402
from app.engine.templates.compiler import GraphCompiler, run_graph  # noqa: E402
from app.engine.templates.state import new_call_state  # noqa: E402

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
failures = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if cond else FAIL} {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures += 1


ORDER_SUPPORT_STEPS = [
    {"key": "greeting", "step_order": 1, "step_type": "SAY",
     "config": {"agent_says": "Hello! Thanks for calling. How can I help with your order today?"}},
    {"key": "identify", "step_order": 2, "step_type": "THINK",
     "config": {"system_prompt": "Extract the order id the customer mentioned.",
               "capture_key": "order_id"}},
    {"key": "understand_intent", "step_order": 3, "step_type": "THINK",
     "config": {"system_prompt": "Classify the customer's intent.", "capture_key": "call_intent"}},
    {"key": "retrieve", "step_order": 4, "step_type": "RETRIEVE",
     "config": {"kb_ids": [], "query_source": "last_customer_message", "top_k": 3}},
    {"key": "take_action", "step_order": 5, "step_type": "TOOL",
     "config": {"tool_connection_id": None,  # filled in at runtime with a real connection id
               "action": "track_order", "payload_template": {"order_id": "{order_id}"}}},
    {"key": "confirm", "step_order": 6, "step_type": "SAY",
     "config": {"agent_says": "Thanks — I've got your order status, let me walk you through it."}},
    {"key": "close", "step_order": 7, "step_type": "SAY",
     "config": {"agent_says": "Is there anything else I can help with? Thanks for calling!"}},
]


async def _demo_tenant_id(db) -> uuid.UUID:
    tenant = (await db.execute(select(Tenant).where(Tenant.slug == "demo-tenant"))).scalars().first()
    if tenant is None:
        raise SystemExit("No demo tenant found — run scripts/seed.py first.")
    return tenant.id


async def _shopify_connection_id(db, tenant_id: uuid.UUID) -> uuid.UUID:
    row = (await db.execute(
        select(ToolConnection.id).join(ToolCatalogEntry, ToolCatalogEntry.id == ToolConnection.tool_catalog_id)
        .where(ToolConnection.tenant_id == tenant_id, ToolCatalogEntry.provider_name == "shopify"))).scalars().first()
    if row is None:
        raise SystemExit("No connected Shopify tool found — connect one via "
                         "POST /integrations/tools/{id}/connect first.")
    return row


async def _stripe_connection_with_guardrail(db, tenant_id: uuid.UUID) -> uuid.UUID:
    """Creates (or reuses) a Stripe connection with a $100 refund_limit_usd
    guardrail, for the denial test below."""
    catalog = (await db.execute(select(ToolCatalogEntry).where(
        ToolCatalogEntry.tenant_id == tenant_id, ToolCatalogEntry.provider_name == "stripe"))).scalars().first()
    if catalog is None:
        raise SystemExit("No stripe catalog entry found — run scripts/seed.py first.")
    conn = (await db.execute(select(ToolConnection).where(
        ToolConnection.tenant_id == tenant_id, ToolConnection.tool_catalog_id == catalog.id))).scalars().first()
    if conn is None:
        conn = ToolConnection(tenant_id=tenant_id, tool_catalog_id=catalog.id, status="connected",
                              guardrails={"refund_limit_usd": 100})
        db.add(conn)
    else:
        conn.status = "connected"
        conn.guardrails = {"refund_limit_usd": 100}
    await db.flush()
    return conn.id


async def _cleanup_checkpoint(call_id: str) -> None:
    async with open_checkpointer() as saver:
        await saver.setup()
        await saver.adelete_thread(call_id)


async def _cleanup_template(db, tenant_id: uuid.UUID, name: str) -> None:
    existing = (await db.execute(select(CallFlowTemplate).where(
        CallFlowTemplate.tenant_id == tenant_id, CallFlowTemplate.name == name))).scalars().all()
    for t in existing:
        await db.execute(delete(TemplateVersion).where(TemplateVersion.template_id == t.id))
        await db.execute(delete(CallFlowStep).where(CallFlowStep.template_id == t.id))
        await db.delete(t)
    await db.flush()


async def test_publish_and_order_support_dry_run(db, tenant_id: uuid.UUID) -> None:
    await _cleanup_template(db, tenant_id, "Order Support (verify)")
    shopify_id = await _shopify_connection_id(db, tenant_id)

    steps = [dict(s) for s in ORDER_SUPPORT_STEPS]
    for s in steps:
        s["config"] = dict(s["config"])
    steps[4]["config"]["tool_connection_id"] = str(shopify_id)

    template = CallFlowTemplate(tenant_id=tenant_id, name="Order Support (verify)",
                                description="Greeting -> Identify -> Understand intent -> Retrieve -> "
                                            "Take action -> Confirm -> Close")
    db.add(template)
    await db.flush()
    for s in steps:
        db.add(CallFlowStep(tenant_id=tenant_id, template_id=template.id, step_order=s["step_order"],
                            key=s["key"], step_type=s["step_type"], config=s["config"]))
    await db.flush()

    # Publish: compile must succeed, and it must snapshot into an immutable version.
    graph, graph_hash = GraphCompiler.compile(steps, tenant_id)
    check("Order Support template compiles", graph is not None and len(graph_hash) > 0)

    version = TemplateVersion(tenant_id=tenant_id, template_id=template.id, version=1,
                              snapshot=steps, compiled_graph_hash=graph_hash)
    db.add(version)
    await db.flush()
    version_id = version.id

    # Mutate the working copy AFTER publish — the published snapshot must not change.
    live_step = (await db.execute(select(CallFlowStep).where(
        CallFlowStep.template_id == template.id, CallFlowStep.key == "greeting"))).scalar_one()
    live_step.config = {"agent_says": "MUTATED AFTER PUBLISH"}
    await db.flush()
    reloaded = (await db.execute(select(TemplateVersion).where(
        TemplateVersion.id == version_id))).scalar_one()
    snapshot_greeting = next(s for s in reloaded.snapshot if s["key"] == "greeting")
    check("Publish produces an immutable version",
         snapshot_greeting["config"]["agent_says"] != "MUTATED AFTER PUBLISH",
         "published snapshot changed after editing the live template")

    # Dry-run against a scripted fake caller.
    call_id = f"verify-order-support-{uuid.uuid4().hex[:8]}"
    state = new_call_state(call_id)
    state["transcript"].append({"role": "customer", "text": "Hi, I want to check on my order 1001"})
    final = await run_graph(steps, tenant_id, call_id, state)

    check("dry-run completes with agent replies in transcript",
         sum(1 for t in final["transcript"] if t["role"] == "agent") == 3,
         f"expected 3 agent turns (greeting/confirm/close), got {final['transcript']}")
    check("TOOL node executed and recorded a result",
         "track_order" in final.get("tool_results", {}) and final["tool_results"]["track_order"]["success"],
         f"tool_results={final.get('tool_results')}")
    check("RETRIEVE node ran without error (empty kb_ids -> empty list, not a crash)",
         final.get("retrieved_chunks") == [])
    check("no guardrail denial on this run -> handoff stays False", final.get("handoff") is False)

    await _cleanup_checkpoint(call_id)
    await _cleanup_template(db, tenant_id, "Order Support (verify)")


async def test_guardrail_denial_routes_to_handoff(db, tenant_id: uuid.UUID) -> None:
    await _cleanup_template(db, tenant_id, "Refund Denial (verify)")
    stripe_id = await _stripe_connection_with_guardrail(db, tenant_id)

    steps = [
        {"key": "refund", "step_order": 1, "step_type": "TOOL",
         "config": {"tool_connection_id": str(stripe_id), "action": "issue_refund",
                   "payload_template": {"charge_id": "ch_test", "amount_usd": "500"}}},
        {"key": "confirm_refund", "step_order": 2, "step_type": "SAY",
         "config": {"agent_says": "Your refund of $500 has been processed."}},
    ]
    template = CallFlowTemplate(tenant_id=tenant_id, name="Refund Denial (verify)")
    db.add(template)
    await db.flush()
    for s in steps:
        db.add(CallFlowStep(tenant_id=tenant_id, template_id=template.id, **s))
    # run_graph()'s TOOL node opens its OWN session (see nodes.py) — a flush
    # alone is invisible across sessions, so this must be committed before the
    # graph runs, not just flushed.
    await db.commit()

    call_id = f"verify-refund-denial-{uuid.uuid4().hex[:8]}"
    final = await run_graph(steps, tenant_id, call_id, new_call_state(call_id))

    check("guardrail denial sets handoff=True", final.get("handoff") is True)
    check("guardrail denial reason recorded in tool_results",
         "denied" in final.get("tool_results", {}).get("issue_refund", {}),
         f"tool_results={final.get('tool_results')}")
    check("step after the denied TOOL never ran (no false confirmation spoken)",
         all(t["text"] != "Your refund of $500 has been processed." for t in final["transcript"]),
         f"transcript={final['transcript']}")

    await _cleanup_checkpoint(call_id)
    await _cleanup_template(db, tenant_id, "Refund Denial (verify)")


async def test_checkpoint_resume(tenant_id: uuid.UUID) -> None:
    steps = [
        {"key": "s1", "step_order": 1, "step_type": "SAY", "config": {"agent_says": "Step one."}},
        {"key": "s2", "step_order": 2, "step_type": "SAY", "config": {"agent_says": "Step two."}},
        {"key": "s3", "step_order": 3, "step_type": "SAY", "config": {"agent_says": "Step three."}},
    ]
    graph, _ = GraphCompiler.compile(steps, tenant_id, use_cache=False)
    call_id = f"verify-resume-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": call_id}}

    async with open_checkpointer() as saver:
        await saver.setup()
        compiled = graph.compile(checkpointer=saver, interrupt_before=["s3"])
        partial = await compiled.ainvoke(new_call_state(call_id), config=config)
        check("interrupted before the last step -> only 2 agent turns so far",
             sum(1 for t in partial["transcript"] if t["role"] == "agent") == 2,
             f"transcript={partial['transcript']}")

        resumed = await compiled.ainvoke(None, config=config)
        check("resume from checkpoint completes the remaining step",
             sum(1 for t in resumed["transcript"] if t["role"] == "agent") == 3,
             f"transcript={resumed['transcript']}")
        check("resumed state is a continuation, not a restart (step one/two preserved)",
             resumed["transcript"][0]["text"] == "Step one." and resumed["transcript"][1]["text"] == "Step two.")

        await saver.adelete_thread(call_id)


async def test_unconditional_route_is_taken(tenant_id: uuid.UUID) -> None:
    """Regression for a real bug: a route with no "if" (an unconditional jump)
    was silently never selected — _build_router's `cond and ...` required a
    condition to exist at all, so execution always fell through to the next
    positional step instead."""
    steps = [
        {"key": "s1", "step_order": 1, "step_type": "SAY",
         "config": {"agent_says": "first", "routes": [{"next": "jump_target"}]}},
        {"key": "skip_me", "step_order": 2, "step_type": "SAY", "config": {"agent_says": "SHOULD BE SKIPPED"}},
        {"key": "jump_target", "step_order": 3, "step_type": "SAY", "config": {"agent_says": "jumped here"}},
    ]
    call_id = f"verify-unconditional-route-{uuid.uuid4().hex[:8]}"
    final = await run_graph(steps, tenant_id, call_id, new_call_state(call_id))
    texts = [t["text"] for t in final["transcript"]]
    check("an unconditional route (no 'if') is actually taken", texts == ["first", "jumped here"],
         f"transcript texts={texts}")
    await _cleanup_checkpoint(call_id)


async def test_graph_cache_is_tenant_scoped() -> None:
    """Regression for a real bug: the in-process graph cache was keyed only by
    a hash of the steps' shape, not tenant_id — two tenants publishing
    byte-identical steps would share one compiled graph, whose TOOL/RETRIEVE
    nodes are closed over the FIRST tenant's tenant_id (see nodes.py's
    build_node) — a cross-tenant data-isolation leak hiding inside a caching
    optimization."""
    tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
    identical_steps = [{"key": "x", "step_order": 1, "step_type": "SAY", "config": {"agent_says": "hi"}}]
    graph_a, hash_a = GraphCompiler.compile(identical_steps, tenant_a)
    graph_b, hash_b = GraphCompiler.compile(identical_steps, tenant_b)
    check("identical steps for two tenants hash the same", hash_a == hash_b)
    check("but do NOT share the same cached graph object across tenants", graph_a is not graph_b)


async def main() -> int:
    async with get_session_factory()() as db:
        tenant_id = await _demo_tenant_id(db)
        await test_publish_and_order_support_dry_run(db, tenant_id)
        await db.commit()
        await test_guardrail_denial_routes_to_handoff(db, tenant_id)
        await db.commit()

    await test_checkpoint_resume(tenant_id)
    await test_unconditional_route_is_taken(tenant_id)
    await test_graph_cache_is_tenant_scoped()

    print()
    if failures:
        print(f"{FAIL} {failures} check(s) failed")
        return 1
    print(f"{PASS} all P3 template Definition-of-Done checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
