"""What is in an uploaded prospect workbook, before anything is imported
(docs/gtm-demo-links.md §5, §16.4).

inspect_workbook()  every tab, the industry it belongs to (a definition whose sheet.tab_names
                    names it), its header row and how many rows it has — so the admin picks
                    a tab and its industry is already selected.
draft_definition()  for a prospect tab no industry claims yet (a new vertical the research
                    team started), a data-only industry definition built from the tab: the
                    company and people columns are recognised from the existing industries'
                    mappings, every other column becomes a fact, and the workbook's
                    "Field guide" tab (when there is one) supplies what each field means,
                    which ones the go-live minimum needs, and the trusted-source rule. The
                    admin reviews it, publishes it, and imports the tab like any other.

Nothing here is specific to one workbook: "standard" columns are learnt from the industry
definitions in the database, not listed in code.
"""
import re
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO

import yaml

from app.demo.facts import clean_text
from app.demo.importer import (WorkbookError, fact_key_for, find_header, ignored_headers,
                               norm_header, open_workbook, read_rows, slugify)
from app.demo.industries.definition import IndustryDefinition

_SCAN = 30
_USE_CASE_RE = re.compile(r"^\s*demo use case\s*:\s*(.+)$", re.I)
# A header row of a prospect tab has at least this many columns the industries already map.
_MIN_KNOWN_HEADERS = 3
# The Field guide tab, as the research workbook lays it out. Matched by header text, so
# column order doesn't matter.
_GUIDE_FIELD_COLS = {"tab": "tab", "field": "field", "what to capture": "description",
                     "trusted source": "source", "in the minimum to go live?": "minimum"}
_GUIDE_TAB_COLS = {"tab": "tab", "minimum to go live": "minimum",
                   "trusted-source rule": "trusted_sources"}
_GUIDE_OPTIONAL = {"trusted source", "in the minimum to go live?"}
_DEFAULT_SKIP_STATUSES = ["Not a fit"]
_DEFAULT_EXAMPLE_MARKERS = ["Example"]


# ── what the existing industries teach us ────────────────────────────────────────────────

def _standard_columns(definitions: list[IndustryDefinition]) -> dict[str, tuple[str, str]]:
    """{normalised header: (group, role)} for the company and people columns the industries
    map, except company_name (each industry names its company column differently).
    group is "company" or "contacts"."""
    votes: Counter = Counter()
    for d in definitions:
        sheet = d.sheet or {}
        for group in ("company", "contacts"):
            for role, header in (sheet.get(group) or {}).items():
                if role != "company_name":
                    votes[(norm_header(header), group, role)] += 1
    out: dict[str, tuple[str, str]] = {}
    for (header, group, role), _ in votes.most_common():
        out.setdefault(header, (group, role))
    return out


def _common(definitions: list[IndustryDefinition], key: str, default: list[str]) -> list[str]:
    seen: dict[str, str] = {}
    for d in definitions:
        for v in (d.sheet or {}).get(key, []):
            seen.setdefault(v.lower(), v)
    return list(seen.values()) or list(default)


def _guess_header(rows: list[tuple], known: set[str]) -> int | None:
    best, best_n = None, 0
    for i, row in enumerate(rows[:_SCAN]):
        n = len({norm_header(v) for v in row if clean_text(v)} & known)
        if n > best_n:
            best, best_n = i, n
    return best if best_n >= _MIN_KNOWN_HEADERS else None


def _use_case(rows: list[tuple], header_row: int | None) -> str | None:
    for row in rows[:header_row if header_row is not None else _SCAN]:
        for v in row:
            m = _USE_CASE_RE.match(clean_text(v) or "")
            if m:
                return m.group(1).strip()
    return None


def _data_rows(rows: list[tuple], h: int, example_markers: list[str]) -> tuple[int, int]:
    """(rows with data, example rows) under header row h."""
    markers = {m.lower() for m in example_markers}
    data = examples = 0
    for row in rows[h + 1:]:
        cells = [clean_text(v) for v in row]
        if not any(cells):
            continue
        if any(c and c.lower() in markers for c in cells[:2]):
            examples += 1
        else:
            data += 1
    return data, examples


# ── inspect ──────────────────────────────────────────────────────────────────────────────

def inspect_workbook(source: str | Path | bytes | BinaryIO, definitions: list[IndustryDefinition],
                     *, source_name: str | None = None) -> dict:
    """Every tab of the workbook:

      kind "industry"  a definition claims it (industry_key set); header found or `problem`
      kind "new"       looks like a prospect tab but no industry claims it: draft one
      kind "other"     anything else (overview, notes, field guide)
    """
    wb, name = open_workbook(source, source_name)
    try:
        tabs = [(ws.title, read_rows(ws)) for ws in wb.worksheets]
    finally:
        wb.close()
    by_tab: dict[str, IndustryDefinition] = {}
    for d in definitions:
        for t in (d.sheet or {}).get("tab_names", []):
            by_tab.setdefault(t.strip().lower(), d)
    known = set(_standard_columns(definitions))
    example_markers = _common(definitions, "example_markers", _DEFAULT_EXAMPLE_MARKERS)

    out = []
    for title, rows in tabs:
        d = by_tab.get(title.strip().lower())
        entry: dict[str, Any] = {"tab": title, "kind": "other", "industry_key": None,
                                 "industry_label": None, "header_row": None, "columns": [],
                                 "rows": 0, "example_rows": 0, "use_case": None, "problem": None}
        h = None
        if d is not None and d.sheet:
            entry.update(kind="industry", industry_key=d.key, industry_label=d.label)
            try:
                h = find_header(rows, d.sheet["header_markers"])
            except WorkbookError as e:
                entry["problem"] = str(e)
        else:
            h = _guess_header(rows, known)
            if h is not None:
                entry["kind"] = "new"
        if h is not None:
            markers = (d.sheet.get("example_markers") if d and d.sheet else None) or example_markers
            entry["header_row"] = h + 1
            entry["columns"] = [clean_text(v) for v in rows[h] if clean_text(v)]
            entry["rows"], entry["example_rows"] = _data_rows(rows, h, markers)
        entry["use_case"] = _use_case(rows, h)
        out.append(entry)
    return {"source_name": name, "tabs": out}


# ── the Field guide tab ──────────────────────────────────────────────────────────────────

def _table(rows: list[tuple], wanted: dict[str, str]) -> list[dict]:
    """Rows of the first table in `rows` whose header has every column in `wanted` (by
    normalised header) → [{role: text}], until a blank row or a new header row."""
    need = set(wanted) - _GUIDE_OPTIONAL
    for i, row in enumerate(rows):
        cols = {norm_header(v): j for j, v in enumerate(row) if clean_text(v)}
        if not need <= set(cols):
            continue
        out = []
        for r in rows[i + 1:]:
            cells = [clean_text(v) for v in r]
            if not any(cells):
                break
            item = {role: (cells[cols[h]] if h in cols and cols[h] < len(cells) else None)
                    for h, role in wanted.items()}
            if item.get("tab") and norm_header(item["tab"]) == "tab":
                break                                       # the next table's header
            out.append(item)
        return out
    return []


def read_field_guide(tabs: list[tuple[str, list[tuple]]]) -> tuple[dict, dict]:
    """({tab: {field header: {description, source, minimum}}}, {tab: {minimum,
    trusted_sources}}) from whichever tab holds the field table; empty when none does."""
    fields: dict[str, dict] = {}
    per_tab: dict[str, dict] = {}
    for _, rows in tabs:
        found = _table(rows, _GUIDE_FIELD_COLS)
        if found:
            for item in found:
                if item.get("tab") and item.get("field"):
                    fields.setdefault(norm_header(item["tab"]), {})[norm_header(item["field"])] = item
            for item in _table(rows, _GUIDE_TAB_COLS):
                if item.get("tab"):
                    per_tab[norm_header(item["tab"])] = item
            break
    return fields, per_tab


# ── draft a definition for a new tab ─────────────────────────────────────────────────────

def _fact_type(header: str) -> str:
    h = norm_header(header)
    return "url" if re.search(r"\b(url|link|website|page)\b", h) else "text"


def draft_definition(source: str | Path | bytes | BinaryIO, tab: str,
                     definitions: list[IndustryDefinition], *, ignore_columns: list[str],
                     source_name: str | None = None) -> tuple[dict, list[str]]:
    """(definition, notes): a data-only industry definition for `tab`. Raises WorkbookError
    when the tab has no recognisable header row. `notes` tell the admin what to check."""
    wb, _ = open_workbook(source, source_name)
    try:
        tabs = [(ws.title, read_rows(ws)) for ws in wb.worksheets]
    finally:
        wb.close()
    rows = next((r for t, r in tabs if t.strip().lower() == tab.strip().lower()), None)
    if rows is None:
        raise WorkbookError(f"no tab named {tab!r}")
    title = next(t for t, _ in tabs if t.strip().lower() == tab.strip().lower())
    standard = _standard_columns(definitions)
    h = _guess_header(rows, set(standard))
    if h is None:
        raise WorkbookError(f"tab {title!r} has no header row with the usual company and people "
                            f"columns ({_MIN_KNOWN_HEADERS} or more of: "
                            f"{', '.join(sorted(standard)[:8])}…)")
    header = [clean_text(v) for v in rows[h]]
    guide_fields, guide_tabs = read_field_guide(tabs)
    guide = guide_fields.get(norm_header(title), {})
    tab_guide = guide_tabs.get(norm_header(title), {})
    # Bookkeeping columns: the global list plus every ignore_columns the industries use.
    ignore = {norm_header(c) for c in ignore_columns}
    for d in definitions:
        ignore |= ignored_headers(d, [])

    company: dict[str, str] = {}
    contacts: dict[str, str] = {}
    facts: list[dict] = []
    notes: list[str] = []
    taken: set[str] = set()
    for text in header:
        if not text or norm_header(text) in ignore:
            continue
        role = standard.get(norm_header(text))
        if role is not None:
            (company if role[0] == "company" else contacts).setdefault(role[1], text)
            continue
        if "company_name" not in company:
            # The first column that is neither bookkeeping nor standard names the company
            # (Developer, Brand, Hospital, …) — the same layout every industry tab uses.
            company["company_name"] = text
            continue
        key = fact_key_for(text, taken)
        taken.add(key)
        fact: dict[str, Any] = {"key": key, "label": text, "type": _fact_type(text),
                                "sheet_column": text}
        g = guide.get(norm_header(text))
        if g:
            if g.get("description"):
                fact["description"] = g["description"]
            if (g.get("minimum") or "").strip().lower().startswith("yes"):
                fact["required"] = True
        facts.append(fact)

    if "company_name" not in company:
        raise WorkbookError(f"tab {title!r}: no column left to use as the company name")
    if not facts:
        notes.append("the tab has no columns beyond the company and people ones; add facts "
                     "before importing")
        facts.append({"key": "company_summary", "label": "Company summary", "type": "text"})

    use_case = _use_case(rows, h)
    key = slugify(title).replace("-", "_")[:40].strip("_")
    if not re.match(r"^[a-z][a-z0-9_]{1,39}$", key or ""):
        key = f"industry_{key}"[:40].strip("_")
    if any(d.key == key for d in definitions):
        notes.append(f"an industry with key {key!r} already exists; change `key` to publish this "
                     "as a new industry, or add the tab name to that industry's sheet.tab_names")
    raw: dict[str, Any] = {"key": key, "label": title.strip()[:80], "version": 1}
    if use_case:
        words = slugify(re.sub(r"\s*\(.*?\)", "", use_case)).split("-")   # "(one branch per demo)"
        uc_key = ""
        for w in words:                                     # whole words, at most 40 characters
            if len(uc_key) + len(w) + 1 > 40:
                break
            uc_key = f"{uc_key}_{w}" if uc_key else w
        raw["use_case"] = {"key": uc_key if re.match(r"^[a-z][a-z0-9_]{1,39}$", uc_key) else "demo",
                           "label": use_case}
    if tab_guide.get("trusted_sources"):
        raw["trusted_sources"] = tab_guide["trusted_sources"]
    raw["facts_fields"] = facts
    raw["minimum"] = []
    raw["sheet"] = {
        "tab_names": [title],
        "header_markers": [company["company_name"], facts[0].get("sheet_column", company["company_name"])]
        if facts[0].get("sheet_column") else [company["company_name"]],
        "skip_statuses": _common(definitions, "skip_statuses", _DEFAULT_SKIP_STATUSES),
        "example_markers": _common(definitions, "example_markers", _DEFAULT_EXAMPLE_MARKERS),
        "company": company,
        **({"contacts": contacts} if "full_name" in contacts else {}),
    }
    if contacts and "full_name" not in contacts:
        notes.append("people columns found without a name column; people won't be imported")
    if not guide:
        notes.append("no Field guide rows for this tab: descriptions and required facts were "
                     "not set; mark the facts the go-live minimum needs as required")
    else:
        notes.append("required facts were set from the Field guide's 'In the minimum to go "
                     "live?' column")
    if tab_guide.get("minimum"):
        notes.append(f"Field guide minimum to go live: {tab_guide['minimum']!r}. Counts (\"at "
                     "least 2 …\") and either/or rules need a `minimum` entry "
                     "({field, min_items} or {any_of: [...], min_items})")
    notes.append("this is a data-only industry: records, people and facts import, but demo "
                 "links need a demo agent (sample_data, sample_template, agent, prompts) in a "
                 "later version")
    return raw, notes


def definition_yaml(raw: dict) -> str:
    return yaml.safe_dump(raw, sort_keys=False, allow_unicode=True, width=100)
