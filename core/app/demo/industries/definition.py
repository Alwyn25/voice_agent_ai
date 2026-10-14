"""Validate and compile one industry definition (docs/gtm-demo-links.md §4.8).

Two layers, so an admin uploading a definition gets every problem in one go:

  1. META_SCHEMA — the shape of the document (required sections, field names, types).
  2. Cross-checks the schema can't express — fact keys unique, minimum rules naming real
     facts, tools naming known building blocks and real dummy-data lists, every
     {{placeholder}} naming a fact, and the industry's own sample_template passing its own
     sample_data schema.

compile_definition() returns an IndustryDefinition the rest of the demo code works from;
nothing downstream re-reads the raw YAML.

A definition has two parts:

  * data      — facts_fields, minimum and the workbook `sheet` mapping: enough to import the
                industry's tab, keep company records, people and facts, and build each
                record's knowledge base.
  * demo agent — sample_data, sample_template, agent and prompts (plus engine or tools):
                what a demo call runs on.

The agent part is all-or-nothing and optional. An industry without it is "data only": its
workbook tab imports normally, but it can't have demo links until a later version adds the
agent. So a new industry's research can start the day its tab exists.
"""
import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from jsonschema import Draft202012Validator

from app.demo.facts import FACT_TYPES
from app.demo.sample.compiler import SchemaDefinitionError, compile_sample_schema
from app.demo.sample.resolve import WEEKDAYS
from app.demo.sample.validate import PLACEHOLDER_RE, format_loc, validate_dataset

# Existing engine packs a demo may run on. Demo tenants only ever REFERENCE these packs
# (a tenant_pack_config row inside the demo tenant); the packs themselves are never changed.
ENGINE_PACKS = {"real_estate_india"}

# The building blocks a definition may use (§6.5). Adding one is a code change by design.
TOOL_BLOCKS = {"lookup", "list_slots", "book_slot", "create_record", "update_record"}
GOAL_RULES = {"slot_booked", "record_created", "lookup_answered", "order_read_back"}

KEY_PATTERN = r"^[a-z][a-z0-9_]{1,39}$"
_KEY = {"type": "string", "pattern": KEY_PATTERN}
_TEXT = {"type": "string", "minLength": 1}
_TEXT_LIST = {"type": "array", "items": _TEXT}

# The demo-agent part of a definition: all four, or none (a data-only industry).
AGENT_SECTIONS = ("sample_data", "sample_template", "agent", "prompts")
# Sections that only mean something with a demo agent.
AGENT_ONLY_SECTIONS = ("engine", "tools", "sample_labels", "qa_questions")
EXTRA_COLUMN_MODES = ("propose", "ignore")

META_SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["key", "label", "version", "facts_fields"],
    "properties": {
        "key": _KEY,
        "label": {"type": "string", "minLength": 1, "maxLength": 80},
        "version": {"type": "integer", "minimum": 1},
        "use_case": {
            "type": "object", "additionalProperties": False,
            "required": ["key", "label"],
            "properties": {"key": _KEY, "label": _TEXT, "prospect_plays": _TEXT,
                           "goal": {"enum": sorted(GOAL_RULES)}, "goal_label": _TEXT}},
        "trusted_sources": _TEXT,
        "facts_fields": {
            "type": "array", "minItems": 1,
            "items": {"type": "object", "additionalProperties": False,
                      "required": ["key", "label"],
                      "properties": {"key": _KEY, "label": _TEXT,
                                     "type": {"enum": list(FACT_TYPES)},
                                     "required": {"type": "boolean"},
                                     "sheet_column": _TEXT, "description": _TEXT}}},
        "minimum": {
            "type": "array",
            # {field, min_items?} — one fact; or {any_of: [...], min_items?} — "either 2 doctors
            # or services in total". Exactly one of field / any_of (checked below).
            "items": {"type": "object", "additionalProperties": False,
                      "properties": {"field": _KEY,
                                     "any_of": {"type": "array", "items": _KEY, "minItems": 2,
                                                "uniqueItems": True},
                                     "min_items": {"type": "integer", "minimum": 1},
                                     "message": _TEXT}}},
        "sheet": {
            "type": "object", "additionalProperties": False,
            "required": ["tab_names", "header_markers", "company"],
            "properties": {
                "tab_names": {**_TEXT_LIST, "minItems": 1},
                "header_markers": {**_TEXT_LIST, "minItems": 1},
                "record_name_field": _KEY,
                "skip_statuses": _TEXT_LIST,
                "example_markers": _TEXT_LIST,
                # Columns that are never facts (on top of DEMO_IMPORT_IGNORE_COLUMNS), and what
                # an import does with a column this definition doesn't map yet.
                "ignore_columns": _TEXT_LIST,
                "extra_columns": {"enum": list(EXTRA_COLUMN_MODES)},
                "company": {
                    "type": "object", "additionalProperties": False, "required": ["company_name"],
                    "properties": {k: _TEXT for k in (
                        "company_name", "city", "website_url", "trigger_note",
                        "research_verified", "research_status", "source_urls", "fetched_at")}},
                "contacts": {
                    "type": "object", "additionalProperties": False, "required": ["full_name"],
                    "properties": {k: _TEXT for k in ("full_name", "role", "email",
                                                      "linkedin_url")}}}},
        "sample_data": {"type": "object", "minProperties": 1},
        "sample_template": {"type": "object"},
        # Friendly titles for the dummy-data fields on the prospect's Sample data tab
        # ("free_visit_slots" → "Free site-visit slots"). Missing keys are humanised.
        "sample_labels": {"type": "object", "additionalProperties": _TEXT},
        "skip_weekdays": {"type": "array", "items": {"enum": list(WEEKDAYS)}, "maxItems": 6},
        # Run the demo on an existing engine pack (its tuned tools and planner guards) instead
        # of the generic tool blocks. The pack's own booking tools then read their slots from
        # `slot_collection` in the dummy data (app/demo/scheduling.py).
        "engine": {
            "type": "object", "additionalProperties": False,
            "required": ["pack", "slot_collection", "tools"],
            "properties": {"pack": {"enum": sorted(ENGINE_PACKS)},
                           # Allowlist: only these pack tools exist in a demo tenant. A pack
                           # tool that reads data a demo doesn't have (unit inventory) or acts
                           # outside the call (WhatsApp documents) must not be listed.
                           "tools": {"type": "array", "items": _KEY, "minItems": 1, "uniqueItems": True},
                           "slot_collection": _KEY,
                           "slot_minutes": {"type": "integer", "minimum": 15, "maximum": 240},
                           "visit_location_facts": {"type": "array", "items": _KEY, "minItems": 1},
                           "staff_field": _KEY}},
        "agent": {
            "type": "object", "additionalProperties": False,
            "required": ["persona", "languages", "greeting", "instructions"],
            "properties": {"persona": _TEXT, "languages": {**_TEXT_LIST, "minItems": 1},
                           "greeting": _TEXT, "instructions": _TEXT, "guardrails": _TEXT_LIST}},
        "tools": {
            "type": "array", "minItems": 1,
            "items": {"type": "object", "additionalProperties": False,
                      "required": ["block", "name", "description"],
                      "properties": {"block": {"enum": sorted(TOOL_BLOCKS)}, "name": _KEY,
                                     "description": _TEXT, "collection": _KEY,
                                     "ref_field": _KEY, "match_on": _TEXT_LIST,
                                     "fields": {"type": "object"}, "compute": _KEY,
                                     "ref_prefix": _TEXT}}},
        "prompts": {**_TEXT_LIST, "minItems": 3, "maxItems": 3},
        "qa_questions": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": False,
                      "required": ["say", "expect"],
                      "properties": {"say": _TEXT, "expect": _TEXT}}},
        "compliance": _TEXT_LIST,
    },
}


class DefinitionError(ValueError):
    """An industry definition is invalid. `errors` lists every problem found."""

    def __init__(self, key: str, errors: list[str]):
        self.key = key
        self.errors = errors
        super().__init__(f"industry {key!r} is invalid:\n  - " + "\n  - ".join(errors))


@dataclass(frozen=True)
class IndustryDefinition:
    key: str
    label: str
    version: int
    raw: dict                      # the definition as stored in demo_industry_versions
    content_hash: str              # sha256 of the canonical JSON; detects real changes
    facts_fields: list[dict]
    minimum: list[dict]
    sheet: dict | None
    sample_schema: dict            # compiled JSON Schema for the dummy data (empty: data only)
    sample_template: dict
    skip_weekdays: list[str]
    engine: dict | None = None
    has_agent: bool = True         # False = data-only industry: imports, but no demo links

    @property
    def fact_keys(self) -> set[str]:
        return {f["key"] for f in self.facts_fields}

    def fact_type(self, key: str) -> str:
        return next(f.get("type", "text") for f in self.facts_fields if f["key"] == key)


# The dummy-data schema of a data-only industry: nothing is allowed.
_NO_SAMPLE_SCHEMA = {"$schema": "https://json-schema.org/draft/2020-12/schema",
                     "type": "object", "additionalProperties": False, "properties": {}}


def with_added_facts(raw: dict, facts: list[dict]) -> dict:
    """A copy of `raw` with `facts` appended to facts_fields (used when an import adds the
    workbook's new columns to the industry). The caller compiles the result."""
    out = copy.deepcopy(raw)
    out["facts_fields"] = list(out.get("facts_fields") or []) + [dict(f) for f in facts]
    return out


def _canonical_hash(raw: dict) -> str:
    return hashlib.sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _placeholders_in(value: Any) -> set[str]:
    if isinstance(value, str):
        return set(PLACEHOLDER_RE.findall(value))
    if isinstance(value, dict):
        return set().union(*(_placeholders_in(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(_placeholders_in(v) for v in value)) if value else set()
    return set()


def _check_agent(raw: dict, known: set[str], errors: list[str]) -> dict:
    """Cross-checks of the demo-agent part; returns the compiled dummy-data schema."""
    schema = compile_sample_schema(raw["sample_data"])     # SchemaDefinitionError → caller
    top = schema["properties"]
    engine = raw.get("engine")
    if engine:
        node = top.get(engine["slot_collection"])
        if not (node and node.get("type") == "array"
                and node["items"].get("x-demo-type") == "relative_slot"):
            errors.append(f"engine.slot_collection: `{engine['slot_collection']}` must be a list "
                          "of relative_slot in sample_data")
        for k in engine.get("visit_location_facts", []):
            if k not in known:
                errors.append(f"engine.visit_location_facts: unknown fact {k!r}")
        from app.engine.tools.definitions import BUILTIN_TOOLS
        known_tools = {t["name"] for t in BUILTIN_TOOLS}
        for name in engine["tools"]:
            if name not in known_tools:
                errors.append(f"engine.tools: unknown pack tool {name!r}")
        for name in ("search_slots", "book_appointment"):
            if name not in engine["tools"]:
                errors.append(f"engine.tools: {name} is required for a slot-booking demo")
        staff = engine.get("staff_field")
        if staff and top.get(staff, {}).get("x-demo-type") != "sample_caller":
            errors.append(f"engine.staff_field: `{staff}` must be a sample_caller in sample_data")
    elif not raw.get("tools"):
        errors.append("tools: required unless the industry runs on an engine pack (`engine`)")
    for i, tool in enumerate(raw.get("tools", [])):
        coll = tool.get("collection")
        if tool["block"] in ("lookup", "list_slots", "book_slot", "create_record", "update_record"):
            if not coll:
                errors.append(f"tools[{i}] ({tool['name']}): needs a `collection`")
                continue
            node = top.get(coll)
            if tool["block"] in ("list_slots", "book_slot"):
                if not (node and node.get("type") == "array"
                        and node["items"].get("x-demo-type") == "relative_slot"):
                    errors.append(f"tools[{i}] ({tool['name']}): `{coll}` must be a list of "
                                  "relative_slot in sample_data")
            elif tool["block"] in ("lookup", "update_record") and not (node and node.get("type") == "array"):
                errors.append(f"tools[{i}] ({tool['name']}): `{coll}` must be a list in sample_data")
        ref = tool.get("ref_field")
        if ref and top.get(ref, {}).get("x-demo-type") != "ref_code":
            errors.append(f"tools[{i}] ({tool['name']}): ref_field `{ref}` must be a ref_code "
                          "field in sample_data")
    names = [t["name"] for t in raw.get("tools", [])]
    if len(names) != len(set(names)):
        errors.append("tools: tool names must be unique")

    use_case = raw.get("use_case")
    goal = (use_case or {}).get("goal")
    if use_case is None:
        errors.append("use_case: required for a demo agent")
    elif goal is None:
        errors.append("use_case.goal: required for a demo agent")
    elif engine:
        if goal != "slot_booked":
            errors.append(f"use_case.goal {goal!r}: an engine-pack demo books slots (slot_booked)")
    else:
        blocks = {t["block"] for t in raw.get("tools", [])}
        needs = {"slot_booked": "book_slot", "record_created": "create_record",
                 "order_read_back": "create_record", "lookup_answered": "lookup"}[goal]
        if needs not in blocks:
            errors.append(f"use_case.goal {goal!r} needs a `{needs}` tool")

    for section in ("agent", "prompts", "sample_template"):
        for name in sorted(_placeholders_in(raw.get(section)) - known):
            errors.append(f"{section}: unknown placeholder {{{{{name}}}}}")

    for k in raw.get("sample_labels", {}):
        if k not in top:
            errors.append(f"sample_labels: {k!r} is not a sample_data field")
    for e in validate_dataset(raw["sample_template"], schema, allowed_placeholders=known):
        errors.append(f"sample_template.{e}")
    return schema


def compile_definition(raw: dict) -> IndustryDefinition:
    """Validate `raw` (a parsed YAML/JSON definition) and compile it. Raises DefinitionError
    listing every problem."""
    key = raw.get("key", "?") if isinstance(raw, dict) else "?"
    errors = [f"{format_loc(e.absolute_path) or '(root)'}: {e.message}"
              for e in Draft202012Validator(META_SCHEMA).iter_errors(raw)]
    if errors:
        raise DefinitionError(key, sorted(errors))

    raw = copy.deepcopy(raw)
    facts = raw["facts_fields"]
    fact_keys = [f["key"] for f in facts]
    dupes = sorted({k for k in fact_keys if fact_keys.count(k) > 1})
    if dupes:
        errors.append(f"facts_fields: duplicate key(s) {dupes}")
    known = set(fact_keys)
    types = {f["key"]: f.get("type", "text") for f in facts}

    for i, rule in enumerate(raw.get("minimum", [])):
        if ("field" in rule) == ("any_of" in rule):
            errors.append(f"minimum[{i}]: needs exactly one of `field` or `any_of`")
            continue
        if "any_of" in rule:
            for k in rule["any_of"]:
                if k not in known:
                    errors.append(f"minimum[{i}]: unknown fact {k!r}")
        elif rule["field"] not in known:
            errors.append(f"minimum[{i}]: unknown fact {rule['field']!r}")
        elif "min_items" in rule and types[rule["field"]] not in ("list", "unit_types"):
            errors.append(f"minimum[{i}]: min_items needs a list or unit_types fact")

    sheet = raw.get("sheet")
    if sheet:
        if sheet.get("record_name_field") and sheet["record_name_field"] not in known:
            errors.append(f"sheet.record_name_field: unknown fact {sheet['record_name_field']!r}")
        cols = ([f["sheet_column"] for f in facts if "sheet_column" in f]
                + list(sheet["company"].values()) + list((sheet.get("contacts") or {}).values()))
        # One column may feed a company attribute AND a fact (Developer → company name and
        # developer_name), but two facts reading one column is almost always a typo.
        fact_cols = [f["sheet_column"].strip().lower() for f in facts if "sheet_column" in f]
        dup_cols = sorted({c for c in fact_cols if fact_cols.count(c) > 1})
        if dup_cols:
            errors.append(f"facts_fields: column(s) {dup_cols} mapped to more than one fact")
        if not cols:
            errors.append("sheet: maps no columns")
        mapped = {c.strip().lower() for c in cols}
        for c in sheet.get("ignore_columns", []):
            if c.strip().lower() in mapped:
                errors.append(f"sheet.ignore_columns: {c!r} is also mapped to a field")

    # The demo agent: all of its sections, or none of them.
    present = [s for s in AGENT_SECTIONS if s in raw]
    has_agent = len(present) == len(AGENT_SECTIONS)
    schema, template = _NO_SAMPLE_SCHEMA, {}
    if present and not has_agent:
        missing = [s for s in AGENT_SECTIONS if s not in raw]
        errors.append(f"a demo agent needs all of {', '.join(AGENT_SECTIONS)} (missing: "
                      f"{', '.join(missing)}); leave all of them out for a data-only industry")
    elif not has_agent:
        for s in AGENT_ONLY_SECTIONS:
            if s in raw:
                errors.append(f"{s}: only allowed with a demo agent "
                              f"({', '.join(AGENT_SECTIONS)})")
    else:
        try:
            schema = _check_agent(raw, known, errors)
        except SchemaDefinitionError as e:
            raise DefinitionError(key, errors + [str(e)]) from None
        template = raw["sample_template"]

    if errors:
        raise DefinitionError(key, errors)
    return IndustryDefinition(
        key=raw["key"], label=raw["label"], version=raw["version"], raw=raw,
        content_hash=_canonical_hash(raw), facts_fields=facts, minimum=raw.get("minimum", []),
        sheet=sheet, sample_schema=schema, sample_template=template,
        skip_weekdays=raw.get("skip_weekdays", []), engine=raw.get("engine") if has_agent else None,
        has_agent=has_agent)
