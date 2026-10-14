"""Industry `sample_data` block → JSON Schema (docs/gtm-demo-links.md §6.2).

An industry definition describes the shape of its dummy data with a small set of field
types written once here and shared by every industry. Adding an industry is a YAML upload;
only a genuinely new kind of value needs a new type in this file.

Grammar of a field spec (YAML):

    name: string                         # shorthand for {type: string}
    name: {type: integer, min: 0, max: 60, optional: true}
    name: {type: list, item: <spec>, min_items: 2}
    name: {type: object, fields: {a: <spec>, b: <spec>}}
    name: {a: <spec>, b: <spec>}         # shorthand for an object

Every field is required unless `optional: true`. Objects reject unknown keys, so a typo in
an upload ("free_vist_slots") is an error, not silently ignored data.
"""
from typing import Any

TIME_PATTERN = r"^([01]\d|2[0-3]):[0-5]\d$"
# +91 98XXX XXX01 — the masked form the industry sheets use. A real number can't pass.
MASKED_PHONE_PATTERN = r"^\+91 \d{2}XXX XXX\d{2}$"

SCALAR_TYPES = {"string", "text_template", "integer", "boolean", "money", "relative_slot",
                "relative_day", "sample_caller", "ref_code", "enum"}
CONTAINER_TYPES = {"list", "object"}
FIELD_TYPES = SCALAR_TYPES | CONTAINER_TYPES

# Options each type accepts (beyond `type`, `optional`, `description`, `example`).
_COMMON_OPTS = {"type", "optional", "description"}
_TYPE_OPTS: dict[str, set[str]] = {
    "string": {"max_length"},
    "text_template": {"max_length"},
    "integer": {"min", "max"},
    "boolean": set(),
    "money": {"max"},
    "relative_slot": set(),
    "relative_day": {"min", "max"},
    "sample_caller": set(),
    "ref_code": {"prefix"},
    "enum": {"values"},
    "list": {"item", "min_items", "max_items"},
    "object": {"fields"},
}


class SchemaDefinitionError(ValueError):
    """The sample_data block of an industry definition is malformed."""


def normalize_spec(spec: Any, path: str) -> dict:
    """Expand the YAML shorthands into a full spec dict and check its options."""
    if isinstance(spec, str):
        spec = {"type": spec}
    elif isinstance(spec, dict) and "type" not in spec:
        spec = {"type": "object", "fields": spec}
    if not isinstance(spec, dict):
        raise SchemaDefinitionError(f"{path}: expected a type name or a mapping, got {spec!r}")
    kind = spec.get("type")
    if kind not in FIELD_TYPES:
        raise SchemaDefinitionError(f"{path}: unknown field type {kind!r} "
                                    f"(known: {', '.join(sorted(FIELD_TYPES))})")
    unknown = set(spec) - _COMMON_OPTS - _TYPE_OPTS[kind]
    if unknown:
        raise SchemaDefinitionError(f"{path}: option(s) {sorted(unknown)} not valid for {kind}")
    if kind == "enum" and not (isinstance(spec.get("values"), list) and spec["values"]):
        raise SchemaDefinitionError(f"{path}: enum needs a non-empty `values` list")
    if kind == "ref_code" and not isinstance(spec.get("prefix"), str):
        raise SchemaDefinitionError(f"{path}: ref_code needs a `prefix` (e.g. VIS-)")
    if kind == "list":
        if "item" not in spec:
            raise SchemaDefinitionError(f"{path}: list needs an `item` spec")
        spec = {**spec, "item": normalize_spec(spec["item"], f"{path}[]")}
    if kind == "object":
        fields = spec.get("fields")
        if not isinstance(fields, dict) or not fields:
            raise SchemaDefinitionError(f"{path}: object needs a non-empty `fields` mapping")
        spec = {**spec, "fields": {k: normalize_spec(v, f"{path}.{k}") for k, v in fields.items()}}
    return spec


def _scalar_schema(spec: dict) -> dict:
    kind = spec["type"]
    if kind in ("string", "text_template"):
        return {"type": "string", "minLength": 1, "maxLength": spec.get("max_length", 500),
                "x-demo-type": kind}
    if kind == "integer":
        s: dict = {"type": "integer"}
        if "min" in spec:
            s["minimum"] = spec["min"]
        if "max" in spec:
            s["maximum"] = spec["max"]
        return s
    if kind == "boolean":
        return {"type": "boolean"}
    if kind == "money":
        # Whole rupees. Never a float: 1.04 Cr is 10400000, so rounding can't creep in.
        return {"type": "integer", "minimum": 0, "maximum": spec.get("max", 10**12),
                "x-demo-type": "money"}
    if kind == "relative_slot":
        return {"type": "object", "additionalProperties": False, "x-demo-type": "relative_slot",
                "required": ["day_offset", "time"],
                "properties": {
                    "day_offset": {"type": "integer", "minimum": 0, "maximum": 30},
                    "time": {"type": "string", "pattern": TIME_PATTERN},
                    "label": {"type": "string", "minLength": 1, "maxLength": 120}}}
    if kind == "relative_day":
        return {"type": "integer", "minimum": spec.get("min", -90), "maximum": spec.get("max", 90),
                "x-demo-type": "relative_day"}
    if kind == "sample_caller":
        return {"type": "object", "additionalProperties": False, "x-demo-type": "sample_caller",
                "required": ["name", "phone_masked"],
                "properties": {
                    # A plain first name ("Rahul"). The page and the agent already say it is
                    # sample data, and the masked phone keeps real contact details out.
                    "name": {"type": "string", "minLength": 1, "maxLength": 120},
                    "phone_masked": {"type": "string", "pattern": MASKED_PHONE_PATTERN}}}
    if kind == "ref_code":
        prefix = spec["prefix"]
        escaped = "".join("\\" + c if not c.isalnum() else c for c in prefix)
        return {"type": "string", "pattern": f"^{escaped}\\d{{3,8}}$", "x-demo-type": "ref_code"}
    if kind == "enum":
        return {"enum": list(spec["values"])}
    raise SchemaDefinitionError(f"no scalar schema for {kind}")   # pragma: no cover


def compile_spec(spec: dict) -> dict:
    """A normalized spec → JSON Schema."""
    kind = spec["type"]
    if kind == "list":
        s = {"type": "array", "items": compile_spec(spec["item"])}
        if "min_items" in spec:
            s["minItems"] = spec["min_items"]
        if "max_items" in spec:
            s["maxItems"] = spec["max_items"]
        return s
    if kind == "object":
        fields = spec["fields"]
        return {"type": "object", "additionalProperties": False,
                "properties": {k: compile_spec(v) for k, v in fields.items()},
                "required": [k for k, v in fields.items() if not v.get("optional")]}
    return _scalar_schema(spec)


def compile_sample_schema(block: dict) -> dict:
    """The whole `sample_data` block (an object at the top level) → a JSON Schema document."""
    if not isinstance(block, dict) or not block:
        raise SchemaDefinitionError("sample_data: must be a non-empty mapping of fields")
    spec = normalize_spec({"type": "object", "fields": block}, "sample_data")
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **compile_spec(spec)}
