#!/usr/bin/env python3
"""Regression checks for the Template Builder's generic per-step fields (Uses /
Move on when / Captures) added on top of P3 — every step type now optionally
carries `agent_says`, a generic `captures` list, and a `move_on_when` self-loop
condition, replacing the old per-type-only fields the editor used to expose
(TOOL's separate Action box, THINK's System prompt box, RETRIEVE's Literal
query box). See app/engine/templates/nodes.py's `_speak`/`_primary_capture_key`/
`_apply_generic_captures` and compiler.py's `_move_on_when_route`.

    docker compose exec -T core python scripts/verify_generic_step_fields.py

Creates and cleans up its own checkpoint threads only — no template/tool rows
(reuses the demo tenant's existing Shopify connection). Safe to re-run.
"""
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.db.base import get_session_factory  # noqa: E402
from app.db.models import Tenant, ToolCatalogEntry, ToolConnection  # noqa: E402
from app.engine.templates.compiler import run_graph  # noqa: E402
from app.engine.templates.state import new_call_state  # noqa: E402

PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
failures = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global failures
    print(f"{PASS if cond else FAIL} {label}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures += 1


async def _demo_tenant_and_shopify_connection(db) -> tuple[uuid.UUID, uuid.UUID]:
    tenant = (await db.execute(select(Tenant).where(Tenant.slug == "demo-tenant"))).scalars().first()
    if tenant is None:
        raise SystemExit("No demo tenant found — run scripts/seed.py first.")
    conn_id = (await db.execute(
        select(ToolConnection.id).join(ToolCatalogEntry, ToolCatalogEntry.id == ToolConnection.tool_catalog_id)
        .where(ToolConnection.tenant_id == tenant.id, ToolCatalogEntry.provider_name == "shopify"))).scalars().first()
    if conn_id is None:
        raise SystemExit("No connected Shopify tool found — connect one via "
                         "POST /integrations/tools/{id}/connect first.")
    return tenant.id, conn_id


async def test_agent_says_and_captures_on_a_tool_step(tenant_id: uuid.UUID, shopify_conn: uuid.UUID) -> None:
    """A TOOL step with no explicit `action` (the editor's settings panel
    dropped that field in favor of a single "Uses" picker) auto-derives one
    from the connection's own catalog; `agent_says` and `captures` both work
    on a non-SAY/non-THINK step type now."""
    steps = [
        {"key": "check", "step_order": 1, "step_type": "TOOL",
         "config": {"agent_says": "Let me check that for you.",
                   "tool_connection_id": str(shopify_conn),
                   "captures": ["order_checked", "extra_marker"]}},
    ]
    call_id = f"verify-generic-fields-tool-{uuid.uuid4().hex[:8]}"
    final = await run_graph(steps, tenant_id, call_id, new_call_state(call_id))

    check("TOOL step speaks its agent_says line",
         any(t["text"] == "Let me check that for you." for t in final["transcript"]))
    check("TOOL step auto-derived an action with no explicit config['action']",
         len(final.get("tool_results", {})) == 1, f"tool_results={final.get('tool_results')}")
    check("primary capture key got the real tool result",
         final.get("captures", {}).get("order_checked") == next(iter(final["tool_results"].values()), None))
    check("secondary capture key got a boolean 'this step ran' marker",
         final.get("captures", {}).get("extra_marker") is True)


async def test_move_on_when_loops_when_never_satisfied(tenant_id: uuid.UUID) -> None:
    """A RETRIEVE step with move_on_when=until_result and no kb_ids can never
    produce a result — it must actually keep looping back to itself (not
    silently fall through), and the loop must be bounded by LangGraph's own
    recursion guard rather than hanging the process."""
    steps = [
        {"key": "search", "step_order": 1, "step_type": "RETRIEVE",
         "config": {"kb_ids": [], "move_on_when": {"mode": "until_result"}}},
    ]
    call_id = f"verify-generic-fields-loop-{uuid.uuid4().hex[:8]}"
    hit_recursion_guard = False
    try:
        await run_graph(steps, tenant_id, call_id, new_call_state(call_id))
    except Exception as e:
        hit_recursion_guard = "recursion" in str(e).lower() or "Recursion" in type(e).__name__
    check("an unsatisfiable move_on_when condition loops until the recursion guard, "
         "rather than hanging or silently passing through", hit_recursion_guard)


async def test_move_on_when_advances_once_satisfied(tenant_id: uuid.UUID) -> None:
    """A THINK step's move_on_when checks its own primary capture key
    (`captures.<key>`, a dotted CallState path) — once the LLM classification
    writes that key, the step must advance normally on the very next check,
    not loop even once more."""
    steps = [
        {"key": "classify", "step_order": 1, "step_type": "THINK",
         "config": {"captures": ["intent"], "move_on_when": {"mode": "until_result"}}},
        {"key": "done", "step_order": 2, "step_type": "SAY", "config": {"agent_says": "done"}},
    ]
    call_id = f"verify-generic-fields-resolves-{uuid.uuid4().hex[:8]}"
    final = await run_graph(steps, tenant_id, call_id, new_call_state(call_id))
    check("move_on_when advances past the step once its capture key is set",
         any(t["text"] == "done" for t in final["transcript"]), f"transcript={final['transcript']}")


async def main() -> int:
    async with get_session_factory()() as db:
        tenant_id, shopify_conn = await _demo_tenant_and_shopify_connection(db)

    await test_agent_says_and_captures_on_a_tool_step(tenant_id, shopify_conn)
    await test_move_on_when_loops_when_never_satisfied(tenant_id)
    await test_move_on_when_advances_once_satisfied(tenant_id)

    print()
    if failures:
        print(f"{FAIL} {failures} check(s) failed")
        return 1
    print(f"{PASS} all generic step-field checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
