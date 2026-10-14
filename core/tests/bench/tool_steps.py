"""Per-tool step-by-step latency breakdown.

Each entry models the 8 executor steps for a specific tool based on its
definitions.py spec (preconditions, postconditions, confirmation flag) and
typical cloud-DB round-trip times (Railway-hosted Postgres, ~15ms base RTT).

Step taxonomy (mirrors executor.py):
  registry    – SELECT on tool registry table
  permission  – in-memory RBAC check
  schema      – jsonschema validate(args, tool.schema)
  idempotency – hash-based dedup SELECT (only tools with idempotency_key set)
  state_gate  – in-memory allowed_states check
  rule_gate   – DB fetch + Python evaluation of tool rules
  precond_N   – one DB call per precondition (labelled by precondition name)
  execute     – actual business-logic DB work
  postcond_N  – one DB call per postcondition
  state_xn    – DB UPDATE on session state (mutation tools only)
  audit       – INSERT into audit log
  metrics     – in-memory histogram/counter update
"""
from __future__ import annotations
from dataclasses import dataclass
import statistics


@dataclass
class ToolStep:
    name: str
    p50_ms: float
    p95_ms: float
    description: str


@dataclass
class ToolProfile:
    tool_name: str
    category: str            # read-only | write | confirmation | external
    timeout_ms: int
    steps: list[ToolStep]
    notes: str = ""

    @property
    def total_p50(self) -> float:
        return sum(s.p50_ms for s in self.steps)

    @property
    def total_p95(self) -> float:
        return sum(s.p95_ms for s in self.steps)


# ── Shared overhead steps (present in every tool) ─────────────────────────────

def _registry() -> ToolStep:
    return ToolStep("registry_lookup", 12, 28, "SELECT tool definition from DB")

def _permission() -> ToolStep:
    return ToolStep("permission_check", 0.2, 0.5, "In-memory RBAC role check")

def _schema() -> ToolStep:
    return ToolStep("schema_validation", 1.5, 4, "jsonschema.validate(args, schema)")

def _state_gate() -> ToolStep:
    return ToolStep("state_gate", 0.2, 0.5, "In-memory allowed_states intersection")

def _rule_gate() -> ToolStep:
    return ToolStep("rule_gate", 25, 60, "Fetch + evaluate tool rules from DB")

def _audit() -> ToolStep:
    return ToolStep("audit_record", 8, 20, "INSERT INTO audit_log")

def _metrics() -> ToolStep:
    return ToolStep("metrics", 1, 2, "In-memory histogram + counter increment")

def _idempotency() -> ToolStep:
    return ToolStep("idempotency_check", 10, 25, "Hash-based dedup SELECT")

def _state_xn() -> ToolStep:
    return ToolStep("state_transition", 15, 35, "UPDATE session_state in DB")

def _base_steps(idempotency: bool = False) -> list[ToolStep]:
    steps = [_registry(), _permission(), _schema()]
    if idempotency:
        steps.append(_idempotency())
    steps += [_state_gate(), _rule_gate()]
    return steps

def _tail_steps(state_xn: bool = False) -> list[ToolStep]:
    steps = []
    if state_xn:
        steps.append(_state_xn())
    steps += [_audit(), _metrics()]
    return steps


# ── Tool profiles (all 17 tools) ──────────────────────────────────────────────

TOOL_PROFILES: list[ToolProfile] = [

    ToolProfile(
        tool_name="search_slots",
        category="read-only",
        timeout_ms=4000,
        steps=_base_steps() + [
            ToolStep("pre:centre_exists", 12, 28, "SELECT centre by id"),
            ToolStep("execute", 65, 180,
                     "JOIN slots × executives × booked intervals, filter by date/time/duration"),
            ToolStep("post:slots_valid", 5, 12, "Verify returned slots are in future"),
        ] + _tail_steps(),
        notes="Heaviest read; complex join. Output cached in session for idempotency gate on book.",
    ),

    ToolProfile(
        tool_name="book_appointment",
        category="confirmation",
        timeout_ms=5000,
        steps=_base_steps(idempotency=True) + [
            ToolStep("pre:customer_identified", 8, 18, "Check session.customer_id is set"),
            ToolStep("pre:slot_was_offered", 4, 8, "In-memory: slot in session.offered_slots"),
            ToolStep("pre:centre_exists", 12, 28, "SELECT centre by id"),
            ToolStep("pre:executive_exists", 12, 28, "SELECT executive by id"),
            ToolStep("pre:slot_available", 35, 80, "SELECT booked intervals for overlap"),
            ToolStep("pre:no_conflict", 18, 45, "SELECT existing appointments for customer"),
            ToolStep("execute", 55, 130, "INSERT appointment + emit BookingConfirmed event"),
            ToolStep("post:row_exists", 10, 24, "SELECT appointment by id"),
            ToolStep("post:matches_request", 12, 28, "Python: compare inserted vs requested attrs"),
        ] + _tail_steps(state_xn=True),
        notes="Largest precondition chain. Needs slot_available + no_conflict — two separate queries.",
    ),

    ToolProfile(
        tool_name="reschedule_appointment",
        category="confirmation",
        timeout_ms=5000,
        steps=_base_steps() + [
            ToolStep("pre:customer_identified", 8, 18, "Check session.customer_id is set"),
            ToolStep("pre:slot_was_offered", 4, 8, "In-memory: new slot in session.offered_slots"),
            ToolStep("pre:appointment_owned", 15, 35, "SELECT appointment WHERE customer_id matches"),
            ToolStep("pre:appointment_active", 12, 28, "Check status ∈ {scheduled, confirmed}"),
            ToolStep("execute", 50, 120, "UPDATE appointment + emit Rescheduled event"),
            ToolStep("post:row_exists", 10, 24, "SELECT appointment by id"),
            ToolStep("post:matches_request", 12, 28, "Python: compare updated vs requested attrs"),
        ] + _tail_steps(state_xn=True),
    ),

    ToolProfile(
        tool_name="cancel_appointment",
        category="confirmation",
        timeout_ms=4000,
        steps=_base_steps() + [
            ToolStep("pre:customer_identified", 8, 18, "Check session.customer_id is set"),
            ToolStep("pre:appointment_owned", 15, 35, "SELECT appointment WHERE customer_id matches"),
            ToolStep("pre:appointment_active", 12, 28, "Check status ∈ {scheduled, confirmed}"),
            ToolStep("execute", 40, 90, "UPDATE status=cancelled + emit Cancelled event"),
            ToolStep("post:row_exists", 10, 24, "SELECT appointment by id"),
        ] + _tail_steps(state_xn=True),
    ),

    ToolProfile(
        tool_name="locate_centre",
        category="read-only",
        timeout_ms=3000,
        steps=_base_steps() + [
            ToolStep("execute", 30, 70, "SELECT centres with city/pincode/service filter"),
            ToolStep("post:centres_active", 5, 12, "Verify status='active' on returned rows"),
        ] + _tail_steps(),
        notes="Simple spatial query. No preconditions.",
    ),

    ToolProfile(
        tool_name="create_ticket",
        category="confirmation",
        timeout_ms=4000,
        steps=_base_steps() + [
            ToolStep("pre:customer_identified", 8, 18, "Check session.customer_id is set"),
            ToolStep("execute", 35, 80, "INSERT ticket row + attach to customer"),
            ToolStep("post:row_exists", 10, 24, "SELECT ticket by id"),
        ] + _tail_steps(state_xn=True),
    ),

    ToolProfile(
        tool_name="create_customer",
        category="write",
        timeout_ms=5000,
        steps=_base_steps(idempotency=True) + [
            ToolStep("execute", 25, 60, "INSERT customer row + upsert phone index"),
            ToolStep("post:row_exists", 10, 24, "SELECT customer by id"),
        ] + _tail_steps(),
        notes="Idempotency-keyed on phone number so duplicate dials don't create duplicates.",
    ),

    ToolProfile(
        tool_name="update_customer_profile",
        category="write",
        timeout_ms=3000,
        steps=_base_steps() + [
            ToolStep("pre:customer_identified", 8, 18, "Check session.customer_id is set"),
            ToolStep("execute", 25, 60, "UPDATE customer fields (name/email/address)"),
            ToolStep("post:updates_persisted", 10, 24, "SELECT and diff updated fields"),
        ] + _tail_steps(),
    ),

    ToolProfile(
        tool_name="record_customer_fact",
        category="write",
        timeout_ms=3000,
        steps=_base_steps() + [
            ToolStep("pre:customer_identified", 8, 18, "Check session.customer_id is set"),
            ToolStep("execute", 25, 55, "INSERT customer_fact row (key/value/source)"),
            ToolStep("post:row_exists", 10, 24, "SELECT fact by id"),
        ] + _tail_steps(),
    ),

    ToolProfile(
        tool_name="schedule_callback",
        category="write",
        timeout_ms=3000,
        steps=_base_steps(idempotency=True) + [
            ToolStep("pre:customer_identified", 8, 18, "Check session.customer_id is set"),
            ToolStep("execute", 28, 65, "INSERT callback row + adjust for calling hours"),
            ToolStep("post:row_exists", 10, 24, "SELECT callback by id"),
        ] + _tail_steps(),
        notes="Idempotency prevents duplicate callbacks from re-triggers.",
    ),

    ToolProfile(
        tool_name="record_do_not_contact",
        category="write",
        timeout_ms=3000,
        steps=_base_steps(idempotency=True) + [
            ToolStep("pre:customer_identified", 8, 18, "Check session.customer_id is set"),
            ToolStep("execute", 40, 90,
                     "INSERT dnc row + UPDATE customer.dnc=true + emit DNCAdded event"),
            ToolStep("post:dnc_row_exists", 10, 24, "SELECT dnc record by id"),
            ToolStep("post:customer_dnc_flag", 10, 24, "SELECT customer.dnc = true"),
        ] + _tail_steps(state_xn=True),
        notes="Two postcondition checks + state transition for compliance audit trail.",
    ),

    ToolProfile(
        tool_name="verify_identity",
        category="read-only",
        timeout_ms=3000,
        steps=_base_steps() + [
            ToolStep("execute", 15, 35,
                     "SELECT customer by phone, Python compare name/DOB fields"),
        ] + _tail_steps(),
        notes="Fast; no preconditions; returns verified/unverified + match fields.",
    ),

    ToolProfile(
        tool_name="my_appointments",
        category="read-only",
        timeout_ms=3000,
        steps=_base_steps() + [
            ToolStep("execute", 35, 80,
                     "SELECT appointments JOIN centres JOIN executives WHERE customer_id"),
            ToolStep("post:belong_to_customer", 8, 18, "Assert all rows share customer_id"),
        ] + _tail_steps(),
    ),

    ToolProfile(
        tool_name="my_tickets",
        category="read-only",
        timeout_ms=3000,
        steps=_base_steps() + [
            ToolStep("execute", 30, 70, "SELECT tickets WHERE customer_id, ORDER BY created_at"),
            ToolStep("post:belong_to_customer", 8, 18, "Assert all rows share customer_id"),
        ] + _tail_steps(),
    ),

    ToolProfile(
        tool_name="inventory_query",
        category="read-only",
        timeout_ms=4000,
        steps=_base_steps() + [
            ToolStep("execute", 60, 140,
                     "SELECT from cost_sheets; parse price ranges; sort by price/type"),
            ToolStep("post:units_match_query", 5, 12,
                     "Python: verify all returned units match filters"),
        ] + _tail_steps(),
        notes="Potentially large cost-sheet table. Heavier execute than simple lookups.",
    ),

    ToolProfile(
        tool_name="knowledge_lookup",
        category="read-only",
        timeout_ms=10000,
        steps=_base_steps() + [
            ToolStep("execute:embed", 80, 200, "POST to embedding API (text-embedding-3-small)"),
            ToolStep("execute:vector_search", 60, 140,
                     "pgvector ANN search on knowledge_chunks table"),
            ToolStep("execute:rerank", 40, 100, "Python cross-encoder rerank top-20 → top-3"),
        ] + _tail_steps(),
        notes="Only tool that calls an external API in the execute step (embedding). "
              "Vector search is pgvector; rerank is in-process.",
    ),

    ToolProfile(
        tool_name="send_document",
        category="external",
        timeout_ms=12000,
        steps=_base_steps(idempotency=True) + [
            ToolStep("execute:doc_lookup", 30, 70,
                     "SELECT document WHERE slug and tenant match"),
            ToolStep("execute:whatsapp_send", 250, 800,
                     "POST /messages to WhatsApp Business API + await delivery receipt"),
            ToolStep("post:belong_to_tenant", 8, 18,
                     "Assert document.tenant_id == session.tenant_id"),
        ] + _tail_steps(),
        notes="Slowest tool; bound by WhatsApp API round-trip (~250ms p50, up to 800ms p95). "
              "Idempotency prevents duplicate sends on retries.",
    ),
]

# ── Quick lookup by name ───────────────────────────────────────────────────────

_BY_NAME: dict[str, ToolProfile] = {p.tool_name: p for p in TOOL_PROFILES}


def get_profile(tool_name: str) -> ToolProfile | None:
    return _BY_NAME.get(tool_name)


def summary_table() -> list[dict]:
    rows = []
    for p in TOOL_PROFILES:
        rows.append({
            "tool": p.tool_name,
            "category": p.category,
            "steps": len(p.steps),
            "p50_ms": round(p.total_p50, 1),
            "p95_ms": round(p.total_p95, 1),
            "timeout_ms": p.timeout_ms,
            "headroom_ms": p.timeout_ms - round(p.total_p95, 1),
        })
    return sorted(rows, key=lambda r: r["p50_ms"], reverse=True)
