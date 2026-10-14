"""The data behind a prospect's demo page (docs/gtm-demo-links.md §8.2, the shape the
demo console in Generic-voice-agent-UI/src/demo/ renders).

Everything here is built from the record's own facts and dummy data. It never exposes
internal ids, document contents, tenant settings or anything about other records. Sample
sections are built generically from the industry's dummy-data schema, so a new industry's
page needs no code: objects and lists become sections, scalars are grouped under
"Details", labels come from the definition's `sample_labels` or the field name.

Each section carries a `kind` so the page can show it the way people read it:

  person   a sample person (name, phone); `you` marks the one the visitor plays
  slots    bookable times, in order
  list     a plain list (e.g. what the agent will ask), in order
  fields   label/value rows
"""
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import DemoLink, DemoProspect
from app.demo import records as svc
from app.demo.agent import strip_sample
from app.demo.industries.definition import IndustryDefinition

DISCLOSURE = "Built by Phonn from public information. Sample data only."
NOT_PUBLISHED = "Not published. The agent says the sales team will confirm."


def humanise(key: str) -> str:
    return key.replace("_", " ").strip().capitalize()


def _label(definition: IndustryDefinition, key: str) -> str:
    return (definition.raw.get("sample_labels") or {}).get(key) or humanise(key)


def _scalar(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, dict) and "spoken" in value:
        return value["spoken"]
    return str(value)


_UNIT_LABEL = re.compile(r"^(.+?)\s*\((.+)\)$")


def _detail(label: str, value) -> dict:
    """A number whose label carries its unit reads as a sentence: "Enquired (minutes before
    the call)" + 10 → "Enquired": "10 minutes before the call"."""
    m = _UNIT_LABEL.match(label)
    if m and isinstance(value, int) and not isinstance(value, bool):
        return {"label": m.group(1), "value": f"{value} {m.group(2)}"}
    return {"label": label, "value": _scalar(value)}


def sample_sections(definition: IndustryDefinition, data: dict) -> tuple[list[dict], list[dict]]:
    """(summary, sections) from RESOLVED dummy data."""
    props = definition.sample_schema["properties"]
    sections: list[dict] = []
    details: list[dict] = []
    summary: list[dict] = []
    for key in definition.raw["sample_data"]:
        value = data.get(key)
        node = props.get(key, {})
        kind = node.get("x-demo-type")
        label = _label(definition, key)
        if value is None or value == [] or value == "":
            continue
        if kind == "sample_caller":
            name = strip_sample(value.get("name"))
            items = [{"label": "Name", "value": name},
                     {"label": "Phone", "value": value.get("phone_masked")}]
            sections.append({"title": label, "kind": "person", "you": key == "caller", "items": items})
            if key == "caller":
                summary.append({"label": "You are", "value": name})
        elif node.get("type") == "array" and node.get("items", {}).get("x-demo-type") == "relative_slot":
            spoken = [s["spoken"] for s in value]
            sections.append({"title": label, "kind": "slots", "items": [
                {"label": f"Slot {i}", "value": v} for i, v in enumerate(spoken, 1)]})
            summary.append({"label": label, "value": " · ".join(spoken), "values": spoken})
        elif node.get("type") == "array":
            sections.append({"title": label, "kind": "list", "items": [
                {"label": str(i), "value": _scalar(v) if not isinstance(v, dict)
                 else ", ".join(f"{humanise(k)}: {_scalar(x)}" for k, x in v.items())}
                for i, v in enumerate(value, 1)]})
        elif node.get("type") == "object" and kind is None:
            sections.append({"title": label, "kind": "fields", "items": [
                {"label": humanise(k), "value": _scalar(v)} for k, v in value.items()]})
        else:
            details.append(_detail(label, value))
            if kind == "ref_code":
                summary.append({"label": label, "value": _scalar(value)})
    if details:
        sections.insert(1 if sections else 0, {"title": "Details", "kind": "fields", "items": details})
    return summary, sections


async def page_data(db: AsyncSession, link: DemoLink, *, now: datetime | None = None) -> dict:
    prospect = await db.get(DemoProspect, link.prospect_id)
    definition = await svc.load_definition(db, prospect.industry_key, prospect.industry_version)
    facts = await svc.load_facts(db, prospect, definition)
    record_key = (definition.sheet or {}).get("record_name_field")
    record_name = facts[record_key].value if record_key and not facts[record_key].is_blank else None

    summary, sections = [], []
    try:
        ctx, _, _ = await svc.resolve_for_call(db, prospect, now or datetime.now(timezone.utc))
        summary, sections = sample_sections(definition, ctx.data)
    except svc.DemoError:
        pass          # no usable dummy data: the page still shows the company and its facts

    docs = [{"title": d["title"], "kind": d.get("kind")}
            for d in await svc.list_kb_documents(db, prospect) if not d["generated"]]
    s = get_settings()
    tz = ZoneInfo(prospect.timezone or s.demo_default_timezone)
    use_case = definition.raw["use_case"]
    return {
        "company_name": prospect.company_name,
        "record_name": record_name,
        "industry_label": definition.label,
        "header": f"{prospect.company_name} × Phonn: a demo agent",
        "use_case": use_case.get("label"),
        # Who the visitor plays and what a good call ends with, for the page's scenario card.
        "scenario": {"you_play": use_case.get("prospect_plays"), "goal": use_case.get("goal_label")},
        "prompts": list(definition.raw.get("prompts") or []),
        "disclosure": DISCLOSURE,
        "expires_at": link.expires_at.isoformat(),
        "max_call_seconds": link.max_call_seconds or s.demo_max_call_seconds,
        "max_calls_per_day": link.max_calls_per_day or s.demo_max_calls_per_link_per_day,
        "sample": {"summary": summary, "sections": sections},
        "knowledge": {
            "facts": [{"label": f["label"],
                       "value": None if facts[f["key"]].is_blank else facts[f["key"]].value,
                       "source_url": facts[f["key"]].source_url,
                       # The date as the research team wrote it: in the record's timezone,
                       # not UTC (midnight 1 Oct in India is still 30 Sep in UTC).
                       "checked_on": facts[f["key"]].fetched_at.astimezone(tz).date().isoformat()
                       if facts[f["key"]].fetched_at else None}
                      for f in definition.facts_fields],
            "documents": docs,
            "not_published": NOT_PUBLISHED,
        },
    }
