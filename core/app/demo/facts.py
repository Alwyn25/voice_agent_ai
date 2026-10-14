"""Company facts: normalising spreadsheet cells into facts, and the go-live minimum check
(docs/gtm-demo-links.md §4.3, §5.4).

A fact's `type` in the industry definition decides how a cell is read:

  text        free text, whitespace tidied                     "Sumadhura Edition"
  list        comma/semicolon list (commas inside numbers kept) "Pool, gym, clubhouse"
  unit_types  BHK-style configurations, counted for the minimum "2BHK - 954 Sqft, 3BHK - …"
  month       a month/year (Excel dates become "December 2029") 2029-12-01 → "December 2029"
  url         a web address, scheme added if missing           "vgn.in" → "https://vgn.in/"

Blank-ish cells ("", "-", "—", "N/A", "TBD") become None: a fact nobody published, which
the agent answers with "the sales team will confirm". Spreadsheet error cells (#ERROR!,
#REF!) also become None but raise a warning so the admin can fix the source sheet.
"""
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

FACT_TYPES = ("text", "list", "unit_types", "month", "url")

_BLANK = {"", "-", "–", "—", "n/a", "na", "nil", "none", "tbd", "null", "not available"}
_ERROR_CELL_RE = re.compile(r"^#(ERROR!|REF!|VALUE!|NAME\?|N/A|DIV/0!|NUM!|NULL!)$", re.I)
_WS_RE = re.compile(r"[ \t\r\f\v]+")
# Split on ; or , — but not on a comma inside a number ("1,670 sq ft", "₹2,000").
_LIST_SPLIT_RE = re.compile(r";|,(?!\d)|\n")
_UNIT_RE = re.compile(
    r"((?:\d+(?:\.5)?\s*(?:,|&|and|/|\+)\s*)*\d+(?:\.5)?)\s*-?\s*"
    r"(?:bhk|br\b|beds?\b|bedrooms?\b)", re.I)
_NUM_RE = re.compile(r"\d+(?:\.5)?")


@dataclass
class FactValue:
    value: str | None
    items: list[str] | None = None          # list / unit_types only
    source_url: str | None = None
    fetched_at: datetime | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_blank(self) -> bool:
        return not (self.value or "").strip()


def clean_text(cell: Any) -> str | None:
    """A cell as tidy text, or None when blank. Integral floats lose their '.0'."""
    if cell is None:
        return None
    if isinstance(cell, float) and cell.is_integer():
        cell = int(cell)
    if isinstance(cell, datetime):
        cell = cell.date()
    if isinstance(cell, date):
        return cell.isoformat()
    lines = [_WS_RE.sub(" ", ln).strip() for ln in str(cell).splitlines()]
    text = "\n".join(ln for ln in lines if ln).strip()
    return None if text.lower() in _BLANK else text


def split_list(text: str) -> list[str]:
    seen, out = set(), []
    for part in _LIST_SPLIT_RE.split(text):
        item = part.strip(" .")
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


def parse_unit_types(text: str) -> list[str]:
    """Distinct configurations in the order given: "Studio, 1 & 2 BHK" → ["Studio", "1 BHK",
    "2 BHK"]; "3 BHK Type 1 …; 3 BHK Type 2 …; 4 BHK" → ["3 BHK", "4 BHK"]."""
    found: list[tuple[int, str]] = []
    for m in _UNIT_RE.finditer(text):
        for n in _NUM_RE.findall(m.group(1)):
            label = f"{float(n):g} BHK"
            found.append((m.start(), label))
    for m in re.finditer(r"\bstudios?\b", text, re.I):
        found.append((m.start(), "Studio"))
    seen, out = set(), []
    for _, label in sorted(found, key=lambda x: x[0]):
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def normalize_url(text: str) -> str | None:
    text = text.strip().split()[0] if text.strip() else ""
    if not text:
        return None
    if not re.match(r"^https?://", text, re.I):
        text = "https://" + text.lstrip("/")
    if re.fullmatch(r"https?://[^/]+", text, re.I):
        text += "/"
    return text


def normalize_fact(cell: Any, fact_type: str) -> FactValue:
    if fact_type not in FACT_TYPES:
        raise ValueError(f"unknown fact type {fact_type!r}")
    if isinstance(cell, str) and _ERROR_CELL_RE.match(cell.strip()):
        return FactValue(None, warnings=[f"spreadsheet error cell {cell.strip()} treated as blank"])

    if fact_type == "month":
        if isinstance(cell, (datetime, date)):
            d = cell.date() if isinstance(cell, datetime) else cell
            return FactValue(f"{d:%B} {d.year}")
        if isinstance(cell, (int, float)) and float(cell).is_integer() and 1900 < cell < 2200:
            return FactValue(str(int(cell)))
        return FactValue(clean_text(cell))

    text = clean_text(cell)
    if text is None:
        return FactValue(None)
    if fact_type == "list":
        items = split_list(text)
        return FactValue(", ".join(items) if items else None, items=items)
    if fact_type == "unit_types":
        items = parse_unit_types(text)
        warnings = [] if items else [f"no unit types (BHK) recognised in {text!r}"]
        return FactValue(text, items=items, warnings=warnings)
    if fact_type == "url":
        return FactValue(normalize_url(text))
    return FactValue(text)


def check_minimum(facts_fields: list[dict], minimum: list[dict],
                  facts: dict[str, FactValue]) -> list[str]:
    """What stops this record going live, as readable lines. Empty = minimum met.

    facts_fields: the definition's facts_fields (key, label, required, …)
    minimum:      extra rules, e.g. {field: configurations, min_items: 2, message: …}, or
                  {any_of: [doctors, services], min_items: 2, message: …}: met when those
                  facts together hold min_items (default 1) entries — a list fact counts its
                  items, any other filled-in fact counts one
    """
    labels = {f["key"]: f.get("label", f["key"]) for f in facts_fields}
    missing: list[str] = []
    for f in facts_fields:
        fv = facts.get(f["key"])
        if f.get("required") and (fv is None or fv.is_blank):
            missing.append(f"{labels[f['key']]} is missing")
    for rule in minimum:
        if "any_of" in rule:
            need = rule.get("min_items", 1)
            have = 0
            for key in rule["any_of"]:
                fv = facts.get(key)
                if fv is not None and not fv.is_blank:
                    have += len(fv.items) if fv.items is not None else 1
            if have < need:
                names = " or ".join(labels.get(k, k) for k in rule["any_of"])
                missing.append(f"{names}: needs {rule.get('message') or f'at least {need} in total'}"
                               f" (found {have})")
            continue
        fv = facts.get(rule["field"])
        if fv is None or fv.is_blank:
            continue   # already reported above if required
        if "min_items" in rule:
            have = len(fv.items or [])
            if have < rule["min_items"]:
                what = rule.get("message") or f"at least {rule['min_items']} {labels[rule['field']]}"
                missing.append(f"{labels[rule['field']]}: needs {what} (found {have}"
                               + (f": {', '.join(fv.items)}" if fv.items else "") + ")")
    return missing
