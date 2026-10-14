"""The demo agent of a company record (docs/gtm-demo-links.md §7).

provision_demo_agent() makes a record's demo tenant a working, isolated copy of the
industry's engine pack, configured from the record's facts and dummy data:

  * tenant_pack_config  → the industry's engine pack (a reference; the pack is not changed)
  * tool_definitions    → the pack's standard tools, inside this tenant only
  * locations           → where visits happen (from the facts), with the sample booking
                           reference read back on every booking
  * tenant.settings     → project_facts (every fact, blanks as "the sales team will
                           confirm"), always in the planner's context
  * prompt_templates    → the persona: industry persona + instructions + guardrails + what the
                           agent knows before the call (the sample lead) + demo disclosure
  * agents              → a row describing the agent, for the admin screens

Idempotent: safe to run on every import, fact edit, dummy-data change and call start; it
only writes what changed. It never touches anything outside the record's own tenant.

start_demo_conversation() opens a call/chat in the demo tenant the way the engine expects:
a fresh sample lead (the caller the dummy data describes) and the demo greeting, so every
call starts from the same clean state.
"""
import copy
import re
import logging
import secrets
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (Agent, Conversation, Customer, CustomerFact, DemoProspect,
                           IndustryPackDef, Location, Message, PromptTemplate,
                           Tenant, TenantPackConfig, ToolDefinition)
from app.demo import records as svc
from app.demo.facts import FactValue
from app.demo.industries.definition import IndustryDefinition
from app.demo.sample.validate import PLACEHOLDER_RE

log = logging.getLogger(__name__)

PERSONA_NAME = "demo_persona"
CONFIRM = "not published; the sales team will confirm"

# Rules every demo agent follows whatever its industry says (§7). Code, not data, so no
# definition can leave them out.
ALWAYS_GUARDRAILS = [
    "This is a demo running on sample data. If the caller asks whether the details are real, "
    "say the company facts come from public information and the appointment slots, names and "
    "references are sample data.",
    "Never invent a price, number, name, date, policy or discount that is not in the project "
    "facts or the knowledge base. If something is not published, say the team will confirm it.",
    "Stay on this one use case. Politely decline unrelated requests.",
    "Never claim to be a human. If asked, say you are an AI assistant.",
]


# Demo visits are never tied to a named staff member. Without the field, a persona that
# names the sample sales contact can't lead the model to pass her NAME as executive_id (a
# live run did exactly that and the booking failed schema validation). Applied only to the
# demo tenant's own copies of the tool specs.
_DROP_PARAMS = {"executive_id"}


def demo_tool_spec(spec: dict) -> dict:
    out = copy.deepcopy(spec)
    props = (out.get("input_schema") or {}).get("properties") or {}
    for p in _DROP_PARAMS & set(props):
        del props[p]
    req = (out.get("input_schema") or {}).get("required")
    if isinstance(req, list):
        out["input_schema"]["required"] = [r for r in req if r not in _DROP_PARAMS]
    return out


def _fill(text: str, values: dict[str, str | None]) -> str:
    return PLACEHOLDER_RE.sub(lambda m: (values.get(m.group(1)) or "").strip(), text)


def strip_sample(name: str | None) -> str | None:
    """A sample person's name as shown and spoken: "Rahul (sample)" → "Rahul". Dummy data
    saved before names stopped carrying "(sample)" still has it."""
    if not name:
        return name
    return re.sub(r"\s*\(sample\)", "", name, flags=re.I).strip() or name


def project_facts(definition: IndustryDefinition, facts: dict[str, FactValue]) -> list[dict]:
    out = []
    for f in definition.facts_fields:
        fv = facts.get(f["key"])
        value = fv.value if fv and not fv.is_blank else None
        if value and f.get("type") == "unit_types" and fv.items:
            value = f"{value} (unit types: {', '.join(fv.items)})"
        out.append({"label": f["label"], "value": value or CONFIRM})
    return out


def render_persona(definition: IndustryDefinition, facts: dict[str, FactValue], data: dict) -> str:
    """The persona prompt. `data` is the record's effective dummy data (unresolved is fine:
    only names and references are used, never dates — dates come from the tools)."""
    values = {k: v.value for k, v in facts.items()}
    agent = definition.raw["agent"]
    lines = [_fill(agent["persona"], values).strip(), "", _fill(agent["instructions"], values).strip()]

    caller = data.get("caller") or {}
    lead_bits = []
    if caller.get("name"):
        lead_bits.append(f"The caller is {strip_sample(caller['name'])}, a sample lead.")
    if data.get("lead_source"):
        lead_bits.append(f"They came from: {_fill(data['lead_source'], values)}.")
    if isinstance(data.get("enquiry_minutes_before_call"), int):
        lead_bits.append(f"They enquired about {data['enquiry_minutes_before_call']} minutes "
                         "before this call.")
    if lead_bits:
        lines += ["", "What you know before the call: " + " ".join(lead_bits)]

    staff_field = (definition.engine or {}).get("staff_field")
    staff = data.get(staff_field) if staff_field else None
    if staff and staff.get("name"):
        lines.append(f"For anything you cannot answer, such as a discount or a price that is not "
                     f"published, offer a callback from the sales contact, "
                     f"{strip_sample(staff['name'])}. Site visits are booked for the caller "
                     "without naming any staff member.")
    ref_field = next((k for k, v in definition.sample_schema["properties"].items()
                      if v.get("x-demo-type") == "ref_code"), None)
    if ref_field and data.get(ref_field):
        lines.append(f"When a booking is confirmed, give the booking reference "
                     f"{data[ref_field]} along with the day, date and time.")
    questions = data.get("qualifying_questions") or []
    if questions:
        lines += ["", "Find out, naturally and one at a time (not as a checklist):"]
        lines += [f"- {q}" for q in questions]

    languages = ", ".join(agent.get("languages", []))
    if languages:
        lines += ["", f"Languages: {languages}. Reply in the language the caller uses; "
                      "Hinglish is fine when they speak Hinglish."]
    rules = ALWAYS_GUARDRAILS + list(agent.get("guardrails") or [])
    lines += ["", "Rules:"] + [f"- {_fill(r, values)}" for r in rules]
    return "\n".join(lines).strip() + "\n"


def render_greeting(definition: IndustryDefinition, facts: dict[str, FactValue]) -> str:
    return " ".join(_fill(definition.raw["agent"]["greeting"],
                          {k: v.value for k, v in facts.items()}).split())


def _visit_place(definition: IndustryDefinition, facts: dict[str, FactValue]) -> str | None:
    for key in (definition.engine or {}).get("visit_location_facts", []):
        fv = facts.get(key)
        if fv and not fv.is_blank:
            return fv.value
    return None


def _record_name(definition: IndustryDefinition, prospect: DemoProspect,
                 facts: dict[str, FactValue]) -> str:
    key = (definition.sheet or {}).get("record_name_field")
    fv = facts.get(key) if key else None
    return fv.value if fv and not fv.is_blank else prospect.company_name


async def provision_demo_agent(db: AsyncSession, prospect: DemoProspect) -> dict:
    """Bring the record's demo tenant in line with its facts and dummy data. Returns what
    was (re)written. Raises svc.Conflict when the record can't run an engine-pack demo."""
    if prospect.tenant_id is None:
        raise svc.Conflict("this record has no demo tenant")
    tenant = await db.get(Tenant, prospect.tenant_id)
    if tenant is None or tenant.kind != "demo":
        # Belt and braces: provisioning writes engine config, so it must never be pointed
        # at anything but a demo tenant.
        raise svc.Conflict("refusing to provision a tenant that is not a demo tenant")
    definition = await svc.load_definition(db, prospect.industry_key, prospect.industry_version)
    if not definition.has_agent:
        raise svc.Conflict(f"{definition.label} is a data-only industry: no demo agent until a "
                           "version with sample_data, sample_template, agent and prompts is "
                           "published")
    engine = definition.engine
    if not engine:
        raise svc.Conflict(f"{definition.label} has no engine pack; generic-engine demos are "
                           "not available yet")
    facts = await svc.load_facts(db, prospect, definition)
    _, data, _, _ = await svc.effective_dataset(db, prospect, definition)
    data = data or {}
    changed: list[str] = []

    # Pack: a reference inside this tenant only.
    pack_row = (await db.execute(select(IndustryPackDef).where(
        IndustryPackDef.slug == engine["pack"]))).scalar_one_or_none()
    if pack_row is None:
        raise svc.Conflict(f"engine pack {engine['pack']!r} is not installed")
    cfg = (await db.execute(select(TenantPackConfig).where(
        TenantPackConfig.tenant_id == tenant.id))).scalar_one_or_none()
    if cfg is None:
        db.add(TenantPackConfig(tenant_id=tenant.id, pack_id=pack_row.id, overrides={}))
        changed.append("pack")
    elif cfg.pack_id != pack_row.id:
        cfg.pack_id = pack_row.id
        changed.append("pack")

    # Tools: exactly the definition's allowlist of the pack's standard tools. Missing ones are
    # added; ones no longer allowed are disabled (kept, so past traces still resolve).
    from app.engine.tools.definitions import BUILTIN_TOOLS
    allowed = set(engine["tools"])
    specs = {t["name"]: demo_tool_spec(t) for t in BUILTIN_TOOLS if t["name"] in allowed}
    existing = {t.name: t for t in (await db.execute(select(ToolDefinition).where(
        ToolDefinition.tenant_id == tenant.id))).scalars()}
    for name, spec in specs.items():
        row = existing.get(name)
        if row is None:
            db.add(ToolDefinition(tenant_id=tenant.id, is_enabled=True, **spec))
            changed.append(f"tool:{name}")
        elif not row.is_enabled or row.input_schema != spec["input_schema"]:
            row.is_enabled = True
            row.input_schema = spec["input_schema"]
            changed.append(f"tool:{name}")
    for name, row in existing.items():
        if name not in allowed and row.is_enabled:
            row.is_enabled = False
            changed.append(f"tool-off:{name}")

    # Where visits happen, and who the sample sales contact is.
    name = _record_name(definition, prospect, facts)
    place = _visit_place(definition, facts)
    ref_field = next((k for k, v in definition.sample_schema["properties"].items()
                      if v.get("x-demo-type") == "ref_code"), None)
    loc_meta = {"demo": True, "booking_reference": data.get(ref_field) if ref_field else None}
    loc = (await db.execute(select(Location).where(Location.tenant_id == tenant.id)
                            .order_by(Location.created_at))).scalars().first()
    if loc is None:
        loc = Location(tenant_id=tenant.id, name=name[:200], address=place, city=prospect.city,
                       timezone=prospect.timezone, is_active=True, location_type=loc_meta)
        db.add(loc)
        changed.append("location")
    elif (loc.name, loc.address, loc.city, loc.timezone, loc.location_type) != (
            name[:200], place, prospect.city, prospect.timezone, loc_meta):
        loc.name, loc.address, loc.city, loc.timezone = name[:200], place, prospect.city, prospect.timezone
        loc.location_type = loc_meta
        changed.append("location")
    await db.flush()

    # Facts in every turn's context; the tenant's display name is the company's.
    settings = dict(tenant.settings or {})
    pf = project_facts(definition, facts)
    if settings.get("project_facts") != pf:
        settings["project_facts"] = pf
        changed.append("project_facts")
    settings.setdefault("demo", {}).update({"prospect_id": str(prospect.id),
                                            "industry": prospect.industry_key})
    tenant.settings = settings
    if tenant.name != prospect.company_name[:200]:
        tenant.name = prospect.company_name[:200]

    # Persona.
    persona = render_persona(definition, facts, data)
    current = (await db.execute(select(PromptTemplate).where(
        PromptTemplate.tenant_id == tenant.id, PromptTemplate.purpose == "persona",
        PromptTemplate.is_active.is_(True)))).scalars().all()
    if not (len(current) == 1 and current[0].template == persona):
        version = max([p.version for p in current] + [0]) + 1
        for p in current:
            p.is_active = False
        await db.flush()
        db.add(PromptTemplate(tenant_id=tenant.id, name=PERSONA_NAME, purpose="persona",
                              template=persona, version=version, is_active=True))
        changed.append("persona")

    # The agent row the admin screens show.
    greeting = render_greeting(definition, facts)
    agent = await db.get(Agent, prospect.agent_id) if prospect.agent_id else None
    if agent is None:
        agent = Agent(tenant_id=tenant.id, name=f"{name} demo"[:200], direction="inbound",
                      status="live", purpose_label=definition.raw["use_case"]["key"])
        db.add(agent)
        await db.flush()
        prospect.agent_id = agent.id
        changed.append("agent")
    agent.greeting_message = greeting[:500]
    agent.system_instructions = persona[:4000]
    agent.language = (definition.raw["agent"].get("languages") or ["en-IN"])[0]
    await db.flush()
    return {"changed": changed, "greeting": greeting}


async def start_demo_conversation(db: AsyncSession, prospect: DemoProspect, *, channel: str
                                  ) -> tuple[Conversation, Customer, str]:
    """Open a conversation in the record's demo tenant with a fresh sample lead and the
    demo greeting already spoken. Caller commits."""
    await provision_demo_agent(db, prospect)
    definition = await svc.load_definition(db, prospect.industry_key, prospect.industry_version)
    facts = await svc.load_facts(db, prospect, definition)
    ctx, _, _ = await svc.resolve_for_call(db, prospect, datetime.now(timezone.utc))
    caller = ctx.data.get("caller") or {}

    # A new sample lead per call: each call starts clean, and bookings made on one call
    # never appear on another. The phone is a placeholder in a range no real Indian mobile
    # uses (+91 0…), unique within the demo tenant.
    customer = None
    for _ in range(5):
        phone = "+910" + "".join(secrets.choice("0123456789") for _ in range(9))
        exists = (await db.execute(select(Customer.id).where(
            Customer.tenant_id == prospect.tenant_id, Customer.phone_normalized == phone))).first()
        if not exists:
            customer = Customer(tenant_id=prospect.tenant_id,
                                full_name=strip_sample(caller.get("name")) or "Sample caller",
                                phone=phone, phone_normalized=phone, current_state="lead",
                                lead_source=(ctx.data.get("lead_source") or "Demo")[:100],
                                tags=["demo_sample"], preferred_language="en")
            break
    if customer is None:
        raise svc.Conflict("could not allocate a sample caller; try again")
    db.add(customer)
    await db.flush()
    record_name_field = (definition.sheet or {}).get("record_name_field")
    if record_name_field and facts.get(record_name_field) and not facts[record_name_field].is_blank:
        db.add(CustomerFact(tenant_id=prospect.tenant_id, customer_id=customer.id,
                            category="product_interest", key="project",
                            value={"name": facts[record_name_field].value}, confidence=1.0))

    greeting = render_greeting(definition, facts)
    conv = Conversation(tenant_id=prospect.tenant_id, customer_id=customer.id, channel=channel,
                        direction="inbound", status="active", started_at=datetime.now(timezone.utc))
    db.add(conv)
    await db.flush()
    db.add(Message(tenant_id=prospect.tenant_id, conversation_id=conv.id, role="agent",
                   content=greeting))
    await db.flush()
    return conv, customer, greeting
