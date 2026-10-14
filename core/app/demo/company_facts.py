"""The generated "Company facts" knowledge-base document (docs/gtm-demo-links.md §5.3).

Every fact of a company record becomes one line the agent can retrieve. A blank fact is
written as an explicit "not published, the sales team will confirm" line: retrieval then
finds a real answer to "what's the RERA number?" instead of nothing, which is what keeps
the agent from inventing one. Deterministic: the same facts always render the same text,
so an unchanged re-import doesn't re-embed anything.
"""
import hashlib

from app.demo.industries.definition import IndustryDefinition

ROLE = "company_facts"          # KnowledgeDocument.meta["demo_role"] of this document


def render_company_facts(definition: IndustryDefinition, *, company_name: str,
                         facts: dict[str, dict], confirm_phrase: str = "the sales team will confirm"
                         ) -> str:
    """facts: key → {"value", "items", "source_url", "fetched_at"} (fetched_at a date/ISO str)."""
    record_name_field = (definition.sheet or {}).get("record_name_field")
    title_bits = [company_name]
    if record_name_field and (facts.get(record_name_field) or {}).get("value"):
        title_bits.insert(0, facts[record_name_field]["value"])
    lines = [f"# {' by '.join(title_bits)}: company facts", ""]
    for f in definition.facts_fields:
        fact = facts.get(f["key"]) or {}
        value = (fact.get("value") or "").strip()
        label = f["label"]
        if not value:
            lines.append(f"- {label}: not published; {confirm_phrase}.")
            continue
        line = f"- {label}: {value}"
        if f.get("type") == "unit_types" and fact.get("items"):
            line += f" (unit types: {', '.join(fact['items'])})"
        lines.append(line)
    sources = sorted({(facts.get(f['key']) or {}).get("source_url") for f in definition.facts_fields}
                     - {None, ""})
    if sources:
        lines += ["", "Sources: " + ", ".join(sources)]
    return "\n".join(lines) + "\n"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
