"""Check a dummy-data upload against its industry's compiled schema plus the rules every
industry shares (docs/gtm-demo-links.md §6.2):

  * shape and types  — the compiled JSON Schema (unknown keys, missing fields, bad times)
  * no absolute dates — every date must be relative, because campaigns go out later than the
                        data is written; "2026-10-01" anywhere in the payload is refused
  * placeholders     — `{{field}}` only inside text_template fields, only naming a fact the
                        industry defines, and (when the record's facts are given) only one
                        that is filled in, so the agent never says "{{price}}" or a blank
  * sample people    — enforced by the schema: phones are masked

Errors carry a `loc` path ("free_visit_slots[1].time") so the admin screen and the XLSX
importer can point at the exact cell.
"""
import re
from dataclasses import dataclass
from typing import Any, Iterator

from jsonschema import Draft202012Validator

PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-z][a-z0-9_]*)\s*\}\}")
_ANY_BRACES_RE = re.compile(r"\{\{|\}\}")
# ISO dates, and d/m/y or d-m-y with a 4-digit year: the shapes an absolute date takes in a
# spreadsheet or a hand-written JSON. Times ("11:00") and offsets are not dates.
_ABSOLUTE_DATE_RE = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}[/.-]\d{1,2}[/.-]\d{4}\b")


@dataclass(frozen=True)
class DatasetError:
    loc: str
    msg: str

    def __str__(self) -> str:
        return f"{self.loc or '(root)'}: {self.msg}"


def format_loc(path) -> str:
    out = ""
    for part in path:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


def _schema_message(err) -> str:
    v = err.validator
    if v == "additionalProperties":
        extra = sorted(set(err.instance) - set(err.schema.get("properties", {})))
        return f"unknown field(s): {', '.join(extra)}"
    if v == "required":
        return err.message.replace(" is a required property", " is required")
    if v == "pattern":
        p = err.schema.get("pattern", "")
        if "XXX" in p:
            return "phone must be masked like +91 98XXX XXX01"
        if p.startswith("^([01]"):
            return "time must be HH:MM in 24-hour form, e.g. 11:00 or 16:30"
        return f"{err.instance!r} does not match the expected format"
    return err.message


def iter_strings(data: Any, schema: dict, path: tuple = ()) -> Iterator[tuple[tuple, str, dict]]:
    """Every string in `data` with its path and the schema node it sits under."""
    if isinstance(data, str):
        yield path, data, schema
    elif isinstance(data, dict):
        props = schema.get("properties", {})
        for k, v in data.items():
            yield from iter_strings(v, props.get(k, {}), path + (k,))
    elif isinstance(data, list):
        items = schema.get("items", {})
        for i, v in enumerate(data):
            yield from iter_strings(v, items, path + (i,))


def validate_dataset(data: Any, schema: dict, *, allowed_placeholders: set[str],
                     facts: dict[str, str | None] | None = None) -> list[DatasetError]:
    """All problems with `data`, sorted by location. Empty = valid.

    allowed_placeholders: the industry's facts_fields keys.
    facts: the record's facts (key → value or None). When given, every placeholder used must
           resolve to a non-blank value; when None (validating the industry-wide template),
           only the names are checked.
    """
    errors: list[DatasetError] = []
    for err in Draft202012Validator(schema).iter_errors(data):
        errors.append(DatasetError(format_loc(err.absolute_path), _schema_message(err)))

    for path, text, node in iter_strings(data, schema):
        loc = format_loc(path)
        if _ABSOLUTE_DATE_RE.search(text):
            errors.append(DatasetError(loc, "absolute dates are not allowed; use a relative "
                                            "slot or day offset so the demo stays current"))
        names = PLACEHOLDER_RE.findall(text)
        if not names:
            if _ANY_BRACES_RE.search(text):
                errors.append(DatasetError(loc, "malformed placeholder; use {{field_name}}"))
            continue
        if node.get("x-demo-type") != "text_template":
            errors.append(DatasetError(loc, "placeholders are only allowed in text fields "
                                            "declared as text_template"))
            continue
        for name in names:
            if name not in allowed_placeholders:
                errors.append(DatasetError(loc, f"unknown placeholder {{{{{name}}}}}"))
            elif facts is not None and not (facts.get(name) or "").strip():
                errors.append(DatasetError(loc, f"placeholder {{{{{name}}}}} has no value for "
                                                "this company; fill the fact or change the text"))
    return sorted(set(errors), key=lambda e: (e.loc, e.msg))
