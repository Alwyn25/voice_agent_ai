"""The four node factories (Vocal Backend Tasksheet P3) — one per step_type.
Each factory takes a step's `config` dict and returns an async LangGraph node
function `(CallState) -> dict` (a partial state update). Every node opens its
own short-lived DB session via get_session_factory() rather than sharing one
across the whole graph run, since a call's steps execute one after another, not
concurrently, and a node has no natural "request" to inherit a session from —
matching the same reasoning as app/engine/rag/kb_retrieval.py's per-task sessions.
"""
import uuid

import app.integrations  # noqa: F401 — registers every mock adapter (see app/integrations/adapters/)
from sqlalchemy import select

from app.db.base import get_session_factory
from app.db.models import ToolCatalogEntry, ToolConnection
from app.engine.rag.kb_retrieval import retrieve_for_kb_ids
from app.engine.templates.state import CallState
from app.integrations.base import get_adapter
from app.providers.llm.base import LLMError
from app.providers.llm.router import get_llm
from app.services.credentials import decrypt_credentials


class _SafeDict(dict):
    """A caller referencing a capture that isn't there yet gets the literal
    placeholder back, never a KeyError — a step config's wording is the
    template's author's problem to fix, not a reason to crash a live call."""
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _flat_context(state: CallState) -> dict:
    """captures + call_id + retrieved_text, flattened for `str.format_map` —
    the substitution surface every step type's `agent_says` gets. Nested dict
    values in captures are left as-is (format_map only substitutes top-level
    keys); a config wanting a nested field should capture it under its own key
    at THINK time instead. `retrieved_text` is the joined content of whatever
    RETRIEVE (or another step's own "Uses" knowledge bases — see
    _maybe_retrieve) most recently put in state["retrieved_chunks"]."""
    ctx = dict(state.get("captures") or {})
    ctx["call_id"] = state.get("call_id")
    ctx["retrieved_text"] = " ".join(c.get("content", "") for c in (state.get("retrieved_chunks") or []))
    return ctx


def _render(template: str, state: CallState) -> str:
    return str(template).format_map(_SafeDict(_flat_context(state)))


def _speak(config: dict, state: CallState) -> list:
    """Every step type may carry an optional `agent_says` line now (the editor
    shows one "What the agent says" box regardless of step type) — SAY's own
    factory already renders it as its entire job; TOOL/THINK/RETRIEVE call this
    to prepend the same transitional line ("Let me check that for you — one
    moment.") to the transcript before doing their real work."""
    text = config.get("agent_says")
    if not text:
        return state["transcript"]
    return state["transcript"] + [{"role": "agent", "text": _render(text, state)}]


def _primary_capture_key(config: dict) -> str | None:
    """The one capture key an LLM classification (THINK) or a loop condition
    (move_on_when) can target — first of the generic `captures` list, falling
    back to the older singular `capture_key` for steps saved before `captures`
    existed."""
    captures = config.get("captures") or []
    if captures:
        return captures[0]
    return config.get("capture_key")


def _apply_generic_captures(config: dict, state_captures: dict, primary_key: str | None,
                            primary_value: object, marker: object) -> dict:
    """Writes every configured `captures` key into the captures dict: the
    step's own primary key (if any) gets its real value, everything else just
    gets `marker` — an honest "this step ran" record, not a fabricated value,
    since only THINK's LLM classification produces a real per-key value today.
    `primary_key` is written even when it's the older singular `capture_key`
    rather than a member of the `captures` list (see _primary_capture_key)."""
    merged = dict(state_captures)
    if primary_key:
        merged[primary_key] = primary_value
    for key in config.get("captures") or []:
        if key != primary_key:
            merged[key] = marker
    return merged


def say_node_factory(config: dict, tenant_id: uuid.UUID):
    """SAY: render `agent_says` against call state -> transcript. No model call
    of its own — purely deterministic templating, matching the tasksheet's own
    description — but its "Uses" box can optionally ground that templating in
    retrieved content: a configured `kb_ids` runs the same lookup RETRIEVE
    does (see _maybe_retrieve) before rendering, so `agent_says` can reference
    `{retrieved_text}`. Any configured `captures` just get a `True` "this step
    ran" marker — SAY has no other result of its own to capture a real value
    from."""
    primary_key = _primary_capture_key(config)

    async def node(state: CallState) -> dict:
        retrieved = await _maybe_retrieve(config, state, tenant_id)
        render_state = state if retrieved is None else {**state, "retrieved_chunks": retrieved}
        transcript = _speak(config, render_state)
        captures = _apply_generic_captures(config, state.get("captures", {}), primary_key, True, marker=True)
        update: dict = {"transcript": transcript, "captures": captures}
        if retrieved is not None:
            update["retrieved_chunks"] = retrieved
        return update

    return node


def tool_node_factory(config: dict, tenant_id: uuid.UUID):
    """TOOL: guardrail check -> adapter.execute(action, payload). A guardrail
    denial sets handoff=True and records the reason — never an unhandled
    exception, per the tasksheet's own Definition of Done for this wave.

    `action` is optional in config now — the editor's settings panel only
    offers a single "Uses" picker (the tool connection), not a separate
    action field, so a missing action falls back to the connection's own
    catalog's first action (see ToolCatalogEntry.actions, seeded as either
    a list of plain action-key strings for a built-in tool or one
    {method, endpoint_url, description} dict for a custom tool — the string
    case is used as-is; the custom-tool case has no discrete action key, so
    "call" is used as a fixed placeholder key into tool_results/captures)."""
    connection_id = uuid.UUID(config["tool_connection_id"])
    configured_action = config.get("action")
    payload_template: dict = config.get("payload_template", {})
    primary_key = _primary_capture_key(config)

    async def node(state: CallState) -> dict:
        results = dict(state.get("tool_results") or {})
        transcript = _speak(config, state)
        async with get_session_factory()() as db:
            conn = (await db.execute(select(ToolConnection).where(
                ToolConnection.tenant_id == tenant_id, ToolConnection.id == connection_id)
            )).scalar_one_or_none()
            if conn is None or conn.status != "connected":
                action = configured_action or "call"
                results[action] = {"success": False, "error": "tool connection not found or not connected"}
                return {"tool_results": results, "transcript": transcript, "handoff": True}

            catalog = (await db.execute(select(ToolCatalogEntry).where(
                ToolCatalogEntry.id == conn.tool_catalog_id))).scalar_one_or_none()
            provider_name = catalog.provider_name if catalog else ""
            first_catalog_action = (catalog.actions[0] if catalog and catalog.actions else None)
            action = configured_action or (first_catalog_action if isinstance(first_catalog_action, str) else "call")
            creds = decrypt_credentials(conn.credentials_encrypted) if conn.credentials_encrypted else {}
            payload = {k: _render(str(v), state) for k, v in payload_template.items()}

            adapter = get_adapter(provider_name, conn.config, creds)
            denial = adapter.check_guardrails(action, payload, conn.guardrails)
            if denial is not None:
                results[action] = {"success": False, "denied": denial}
                return {"tool_results": results, "transcript": transcript, "handoff": True}

            result = adapter.execute(action, payload)
            results[action] = {"success": result.success, "data": result.data, "error": result.error}
            captures = _apply_generic_captures(config, state.get("captures", {}), primary_key,
                                               results[action], marker=result.success)
            return {"tool_results": results, "transcript": transcript, "captures": captures,
                    "handoff": not result.success}

    return node


def think_node_factory(config: dict, tenant_id: uuid.UUID):
    """THINK: the only probabilistic node type — a bounded LLM classification
    call writing captures[capture_key]. Recent transcript is the primary
    context given; a configured "Uses" (`kb_ids`) additionally grounds the
    classification in retrieved content (see _maybe_retrieve), appended to
    the same prompt rather than requiring a separate RETRIEVE step first.

    `capture_key` (singular) still works for steps saved before the editor's
    generic `captures` list existed; `_primary_capture_key` prefers the first
    entry of `captures` when both are present. `system_prompt` is optional —
    the settings panel no longer exposes it, so a step with only a captures
    list gets a plain auto-generated classification prompt instead of no
    prompt at all. A step with no capture key at all (nothing to classify
    into) skips the LLM call entirely — just speaks `agent_says`, if any."""
    primary_key = _primary_capture_key(config)
    system_prompt = config.get("system_prompt") or (
        f'Classify the caller\'s response for "{primary_key}" based on the conversation so far.'
        if primary_key else "")
    schema = config.get("schema")

    async def node(state: CallState) -> dict:
        retrieved = await _maybe_retrieve(config, state, tenant_id)
        speak_state = state if retrieved is None else {**state, "retrieved_chunks": retrieved}
        transcript = _speak(config, speak_state)
        if not primary_key:
            update: dict = {"transcript": transcript}
            if retrieved is not None:
                update["retrieved_chunks"] = retrieved
            return update

        recent = state["transcript"][-6:]
        user_text = "\n".join(f"{t['role']}: {t['text']}" for t in recent) or "(no transcript yet)"
        if retrieved:
            user_text += "\n\nRelevant context:\n" + " ".join(c.get("content", "") for c in retrieved)
        try:
            result = await get_llm().complete_json(system=system_prompt, user=user_text, schema=schema)
            value = result.data
        except LLMError as e:
            value = {"error": str(e)}

        captures = _apply_generic_captures(config, state.get("captures", {}), primary_key, value, marker=True)
        update: dict = {"transcript": transcript, "captures": captures}
        if retrieved is not None:
            update["retrieved_chunks"] = retrieved
        if primary_key in ("intent", "sentiment") and isinstance(value, dict) and primary_key in value:
            update[primary_key] = value[primary_key]
        return update

    return node


async def _maybe_retrieve(config: dict, state: CallState, tenant_id: uuid.UUID) -> list[dict] | None:
    """The shared internal-retrieve core (app/engine/rag/kb_retrieval.py),
    scoped to `config["kb_ids"]` — in-process, not over HTTP, since a graph
    running inside this same process has no reason to loop back through its
    own API (see kb_retrieval.py's docstring). `None` means "kb_ids wasn't
    configured at all" (a no-op, distinct from "[]" which means it WAS
    configured but found nothing), so a caller can tell "nothing to do" apart
    from "ran and came up empty".

    Originally RETRIEVE-only; every step type's "Uses" box now offers the
    same knowledge-base picker (a TOOL/THINK step can ground its own
    agent_says/system_prompt in retrieved content via `{retrieved_text}`,
    same as RETRIEVE's own consumers do), so this is shared rather than
    living only inside retrieve_node_factory."""
    kb_ids_raw = config.get("kb_ids") or []
    if not kb_ids_raw:
        return None
    kb_ids = [uuid.UUID(x) for x in kb_ids_raw]
    literal_query = config.get("query")
    if literal_query:
        query = _render(literal_query, state)
    else:
        customer_turns = [t for t in state["transcript"] if t["role"] == "customer"]
        query = customer_turns[-1]["text"] if customer_turns else ""
    if not query:
        return []

    async with get_session_factory()() as db:
        # conversation_id is deliberately NOT derived from call_id: it's a
        # real FK to conversations.id (see KnowledgeCitation), and call_id
        # is only a LangGraph thread identifier — it has no guaranteed
        # relationship to a conversations row. Wiring the two together is
        # P6's job, once the Call Runtime actually creates that row per call.
        chunks = await retrieve_for_kb_ids(
            db, tenant_id=tenant_id, query=query, kb_ids=kb_ids, top_k=config.get("top_k"),
            conversation_id=None)
        await db.commit()
    return [c.as_context() for c in chunks]


def retrieve_node_factory(config: dict, tenant_id: uuid.UUID):
    """RETRIEVE: kb_ids is this step's whole job (see _maybe_retrieve) — unlike
    SAY/THINK, where it's an optional grounding lookup alongside their own
    main behavior, RETRIEVE has no other behavior to speak of."""
    primary_key = _primary_capture_key(config)

    async def node(state: CallState) -> dict:
        transcript = _speak(config, state)
        retrieved = await _maybe_retrieve(config, state, tenant_id) or []
        captures = _apply_generic_captures(config, state.get("captures", {}), primary_key,
                                           retrieved, marker=bool(retrieved))
        return {"transcript": transcript, "retrieved_chunks": retrieved, "captures": captures}

    return node


NODE_FACTORIES = {
    "SAY": say_node_factory,
    "TOOL": tool_node_factory,
    "THINK": think_node_factory,
    "RETRIEVE": retrieve_node_factory,
}


def build_node(step_type: str, config: dict, tenant_id: uuid.UUID):
    factory = NODE_FACTORIES.get(step_type)
    if factory is None:
        raise ValueError(f"Unknown step_type '{step_type}' — must be one of {sorted(NODE_FACTORIES)}")
    return factory(config, tenant_id)
