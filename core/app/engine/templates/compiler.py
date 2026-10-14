"""GraphCompiler (Vocal Backend Tasksheet P3) — "core of the whole runtime".
Builds a LangGraph StateGraph from an ordered list of step configs, wiring
conditional edges from each step's `config["routes"]` (falling straight to
END whenever a node has set `handoff=True`, regardless of routes — the
guardrail-denial-routes-to-handoff Definition of Done). The built StateGraph is
cached in-process, keyed by a hash of the steps themselves, so publishing the
same steps twice (or two calls running the same published version concurrently)
never rebuilds the same graph structure twice.

Actually running a graph (binding a checkpointer, ainvoke) is `run_graph()`
below, kept separate from `compile()` itself since a checkpointer has its own
connection lifecycle (see checkpointer.py) that compiling a graph's shape does
not need to open.
"""
import hashlib
import json
import uuid

from langgraph.graph import END, START, StateGraph

from app.engine.templates.checkpointer import open_checkpointer
from app.engine.templates.nodes import build_node
from app.engine.templates.state import CallState

_GraphCacheKey = tuple[str, str]  # (tenant_id, graph_hash) — see compile()'s cache_key
_GRAPH_CACHE: dict[_GraphCacheKey, StateGraph] = {}
_GRAPH_CACHE_ORDER: list[_GraphCacheKey] = []
_GRAPH_CACHE_MAX = 64

HANDOFF = "__handoff__"
_DEFAULT = "__default__"


def compute_graph_hash(steps: list[dict]) -> str:
    """Deterministic over the steps' own shape (key/step_type/config), not step
    row ids or timestamps — two templates with identical steps compile to the
    identical cached graph, and republishing unchanged steps reuses the version's
    old hash rather than minting a new one."""
    payload = json.dumps(
        [{"key": s["key"], "step_type": s["step_type"], "config": s.get("config", {})} for s in steps],
        sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _resolve_field(state: CallState, field: str):
    value: object = state
    for part in field.split("."):
        if isinstance(value, dict):
            value = value.get(part)
        else:
            return None
    return value


def _condition_matches(state: CallState, cond: dict | None) -> bool:
    if cond is None:
        # A route with no "if" is unconditional (an "else"-style catch-all,
        # not a route that can never fire) — `cond and ...` previously
        # required a condition to exist at all, so an author's deliberate
        # unconditional `{"next": "x"}` route silently never matched and
        # execution always fell through to the next positional step instead.
        return True
    value = _resolve_field(state, cond.get("field", ""))
    if "empty" in cond:
        # {"field": ..., "empty": true/false} — a falsy value (None, "", [],
        # {}) counts as empty. Added for move_on_when's "wait for a result"
        # self-loop (see GraphCompiler.compile), but usable by any route.
        return (not value) == bool(cond["empty"])
    return value == cond.get("equals")


def _build_router(routes: list[dict]):
    def router(state: CallState) -> str:
        if state.get("handoff"):
            return HANDOFF
        for route in routes:
            if _condition_matches(state, route.get("if")):
                return route["next"]
        return _DEFAULT
    return router


# TOOL/RETRIEVE always have a real result field to wait on — TOOL always has
# a tool_connection_id (required by _validate_step_config) and RETRIEVE
# always has kb_ids (same), so every step of that type genuinely populates
# this field on every run.
_RESULT_FIELD_BY_STEP_TYPE = {
    "TOOL": "tool_results",
    "RETRIEVE": "retrieved_chunks",
}


def _move_on_when_route(step: dict) -> dict | None:
    """Derives a synthetic self-loop route from `config.move_on_when` (an
    editor-facing {"mode": "until_result"} toggle) — checked first, ahead of
    any author-written `routes`, since "the result isn't ready yet" should
    always win over an unconditional catch-all route. Not persisted back into
    the step's own `config.routes`; recomputed on every compile from the one
    source of truth (move_on_when) instead of two.

    THINK's result field is its own primary capture key rather than a fixed
    CallState field — `captures.<key>` still resolves via _resolve_field's
    dotted-path walk. SAY/THINK's own kb_ids-driven "Uses" (see nodes.py's
    _maybe_retrieve) is optional, unlike TOOL/RETRIEVE's required fields, so
    both fall back to retrieved_chunks ONLY when kb_ids is actually
    configured — without that guard, a SAY/THINK step with no Uses selected
    would check a CallState field it never touches, which stays empty
    forever and loops until the recursion guard on every single call. A
    THINK step with neither a capture key nor kb_ids, or a SAY step with no
    kb_ids, simply has nothing to wait on."""
    config = step.get("config") or {}
    if (config.get("move_on_when") or {}).get("mode") != "until_result":
        return None
    step_type = step["step_type"]
    if step_type == "THINK":
        capture_key = (config.get("captures") or [config.get("capture_key")])[0]
        field = f"captures.{capture_key}" if capture_key else None
        if field is None and config.get("kb_ids"):
            field = "retrieved_chunks"
    elif step_type == "SAY":
        field = "retrieved_chunks" if config.get("kb_ids") else None
    else:
        field = _RESULT_FIELD_BY_STEP_TYPE.get(step_type)
    if not field:
        return None
    return {"if": {"field": field, "empty": True}, "next": step["key"]}


class GraphCompiler:
    @staticmethod
    def compute_hash(steps: list[dict]) -> str:
        return compute_graph_hash(steps)

    @staticmethod
    def compile(steps: list[dict], tenant_id: uuid.UUID, use_cache: bool = True) -> tuple[StateGraph, str]:
        """`steps`: ordered dicts with `key`, `step_type`, `step_order`, `config`
        (the same shape as call_flow_steps rows / a template_version's snapshot).
        Returns (graph, compiled_graph_hash); the graph is NOT yet
        `.compile(checkpointer=...)`-ed — see run_graph()."""
        if not steps:
            raise ValueError("A template needs at least one step to compile")

        graph_hash = compute_graph_hash(steps)
        # The cache key includes tenant_id, not just the hash of the steps'
        # shape — TOOL/RETRIEVE nodes close over `tenant_id` (see
        # build_node()/nodes.py), so two tenants publishing byte-identical
        # steps must NOT share a cached graph: doing so would let tenant B's
        # compiled TOOL/RETRIEVE nodes execute against tenant A's tool
        # connections and knowledge bases.
        cache_key = (str(tenant_id), graph_hash)
        if use_cache and cache_key in _GRAPH_CACHE:
            _GRAPH_CACHE_ORDER.remove(cache_key)
            _GRAPH_CACHE_ORDER.append(cache_key)
            return _GRAPH_CACHE[cache_key], graph_hash

        ordered = sorted(steps, key=lambda s: s["step_order"])
        keys = [s["key"] for s in ordered]
        if len(set(keys)) != len(keys):
            raise ValueError("Step keys must be unique within a template")
        key_set = set(keys)

        graph = StateGraph(CallState)
        for step in ordered:
            graph.add_node(step["key"], build_node(step["step_type"], step.get("config") or {}, tenant_id))
        graph.add_edge(START, keys[0])

        for i, step in enumerate(ordered):
            default_next = keys[i + 1] if i + 1 < len(keys) else END
            routes = (step.get("config") or {}).get("routes") or []
            # Checked ahead of the author's own routes — "the result isn't
            # ready yet" wins over an unconditional catch-all (see its own
            # docstring for why this is derived fresh each compile, not
            # stored in config.routes).
            move_on_route = _move_on_when_route(step)
            if move_on_route is not None:
                routes = [move_on_route, *routes]
            for route in routes:
                target = route.get("next")
                if target not in key_set and target != HANDOFF:
                    raise ValueError(
                        f"Step '{step['key']}' routes to unknown step '{target}'")

            path_map = {_DEFAULT: default_next, HANDOFF: END}
            for route in routes:
                target = route["next"]
                path_map[target] = END if target == HANDOFF else target
            graph.add_conditional_edges(step["key"], _build_router(routes), path_map)

        if use_cache:
            _GRAPH_CACHE[cache_key] = graph
            _GRAPH_CACHE_ORDER.append(cache_key)
            while len(_GRAPH_CACHE_ORDER) > _GRAPH_CACHE_MAX:
                _GRAPH_CACHE.pop(_GRAPH_CACHE_ORDER.pop(0), None)

        return graph, graph_hash


async def run_graph(steps: list[dict], tenant_id: uuid.UUID, call_id: str, initial_state: CallState) -> CallState:
    """Compiles (or reuses the cached build of) `steps`, binds a short-lived
    Postgres checkpointer keyed by `call_id`, and runs the graph once to
    completion. Used by POST /templates/{id}/publish's compile-check, the
    scripted dry-run/resume verification, and — as a reference shape — by
    whatever P6's Call Runtime Service eventually invokes per real call."""
    graph, _ = GraphCompiler.compile(steps, tenant_id)
    async with open_checkpointer() as saver:
        await saver.setup()
        compiled = graph.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": call_id}}
        return await compiled.ainvoke(initial_state, config=config)
