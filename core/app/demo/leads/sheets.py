"""The leads Google Sheet (docs/gtm-demo-links.md §10.3).

Two tabs in one spreadsheet (DEMO_LEADS_GSHEET_ID), both created with their headers if
missing (DEMO_SHEET_AUTO_CREATE_TABS):

  Leads          one row per form submission. Columns up to "Admin link" are written by
                 us; Status, Called by and Notes belong to the sales team and are read back
                 into demo_leads (the sheet wins for those three, the database for the rest).
  Demo activity  one row per company link (who it went to, who opened it, calls…),
                 rewritten on every sync.

Rules that keep the team's edits safe:
  * Rows are found by Lead ID, never by row number, so sorting, filtering and colouring
    rows is fine. Columns are found by header name, so moving or adding columns is fine.
    A missing required header stops the sync with an error instead of writing into the
    wrong column.
  * Appending is idempotent: a Lead ID already in the sheet is never appended again.
  * Values are written RAW, and any typed value starting with = + - @ is prefixed with ',
    so a form field like =IMPORTXML(...) is shown as text, never run as a formula.

The logic works on a small SheetBackend interface; GSpreadBackend is the real one (the
gspread calls are blocking, so the async wrapper runs them in a thread) and tests use an
in-memory fake.
"""
import asyncio
import base64
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Protocol

from app.config import get_settings
from app.demo.settings import csv_list

log = logging.getLogger(__name__)

LEAD_SYSTEM_COLUMNS = ["Lead ID", "Received at", "Kind", "Name", "Phone", "Email", "Company", "Role",
                       "Industry", "Preferred time", "Message", "Demo calls", "Longest call",
                       "Goal completed", "Questions asked", "Batch", "Admin link"]
LEAD_TEAM_COLUMNS = ["Status", "Called by", "Notes"]
LEAD_COLUMNS = LEAD_SYSTEM_COLUMNS + LEAD_TEAM_COLUMNS
ACTIVITY_COLUMNS = ["Company", "Industry", "Link status", "Expires on", "People sent", "People opened",
                    "Opened by", "Other visitors", "Calls", "Talk time", "Goal reached", "Questions asked",
                    "First opened", "Last activity", "Lead submitted"]
_FORMULA_START = ("=", "+", "-", "@")


class SheetError(RuntimeError):
    """The sheet can't be used as configured (unreachable, missing headers, no access)."""


_PLAIN_NUMBER = re.compile(r"^[+-]?[\d\s().-]+$")      # +91 98765 43210, -3, (080) 1234


def safe_cell(value) -> str:
    """A value as sheet text that can never run as a formula (also once the sheet is
    exported to CSV and opened in Excel). A plain number or phone number is left alone."""
    if value is None:
        return ""
    text = str(value)
    if text.startswith(_FORMULA_START) and not _PLAIN_NUMBER.match(text):
        return "'" + text
    return text


class SheetBackend(Protocol):
    def tabs(self) -> list[str]: ...
    def add_tab(self, title: str, header: list[str]) -> None: ...
    def values(self, title: str) -> list[list[str]]: ...
    def append_rows(self, title: str, rows: list[list[str]]) -> None: ...
    def update_cells(self, title: str, updates: list[tuple[int, int, str]]) -> None: ...
    def replace_all(self, title: str, rows: list[list[str]]) -> None: ...
    def set_header(self, title: str, header: list[str]) -> None: ...


# ── credentials and the real backend ─────────────────────────────────────────────────────

def service_account_info(raw: str) -> dict:
    """GOOGLE_SERVICE_ACCOUNT_JSON: base64 of the key file, the JSON itself, or a path."""
    raw = (raw or "").strip()
    if not raw:
        raise SheetError("GOOGLE_SERVICE_ACCOUNT_JSON is not set")
    candidates = []
    if raw.startswith("{"):
        candidates.append(raw)
    else:
        try:
            candidates.append(base64.b64decode(raw, validate=True).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            pass
        path = Path(raw)
        if len(raw) < 1024 and path.is_file():
            candidates.append(path.read_text(encoding="utf-8"))
    for text in candidates:
        try:
            info = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(info, dict) and info.get("client_email") and info.get("private_key"):
            return info
    raise SheetError("GOOGLE_SERVICE_ACCOUNT_JSON is not a service-account key "
                     "(base64 of the JSON file, the JSON itself, or a path to it)")


def _retry(fn, *args, attempts: int = 5):
    """Google allows ~60 writes a minute per service account: back off on 429/5xx."""
    from gspread.exceptions import APIError
    for i in range(attempts):
        try:
            return fn(*args)
        except APIError as e:
            code = getattr(e.response, "status_code", None)
            if code not in (429, 500, 502, 503) or i == attempts - 1:
                raise
            time.sleep(min(30, 2 ** i + random.random()))


class GSpreadBackend:
    def __init__(self, spreadsheet_id: str, info: dict):
        import gspread
        try:
            self._sh = gspread.service_account_from_dict(info).open_by_key(spreadsheet_id)
        except gspread.exceptions.SpreadsheetNotFound:
            raise SheetError(f"spreadsheet not found or not shared with {info['client_email']} "
                             "(share it with that address as Editor)") from None
        except gspread.exceptions.APIError as e:
            raise SheetError(f"Google Sheets refused access: {str(e)[:200]}") from None
        self.title = self._sh.title

    def _ws(self, title: str):
        return self._sh.worksheet(title)

    def tabs(self) -> list[str]:
        return [w.title for w in _retry(self._sh.worksheets)]

    def add_tab(self, title: str, header: list[str]) -> None:
        ws = _retry(lambda: self._sh.add_worksheet(title=title, rows=1000, cols=max(26, len(header))))
        _retry(lambda: ws.update([header], "A1", value_input_option="RAW"))
        _retry(lambda: ws.format(f"A1:{_col(len(header))}1", {"textFormat": {"bold": True}}))
        _retry(lambda: ws.freeze(rows=1))

    def set_header(self, title: str, header: list[str]) -> None:
        ws = self._ws(title)
        _retry(lambda: ws.update([header], "A1", value_input_option="RAW"))

    def values(self, title: str) -> list[list[str]]:
        return _retry(self._ws(title).get_all_values)

    def append_rows(self, title: str, rows: list[list[str]]) -> None:
        if rows:
            ws = self._ws(title)
            _retry(lambda: ws.append_rows(rows, value_input_option="RAW", table_range="A1"))

    def update_cells(self, title: str, updates: list[tuple[int, int, str]]) -> None:
        if updates:
            ws = self._ws(title)
            data = [{"range": f"{_col(c)}{r}", "values": [[v]]} for r, c, v in updates]
            _retry(lambda: ws.batch_update(data, value_input_option="RAW"))

    def replace_all(self, title: str, rows: list[list[str]]) -> None:
        ws = self._ws(title)
        _retry(ws.clear)
        if rows:
            _retry(lambda: ws.update(rows, "A1", value_input_option="RAW"))

    def add_status_dropdown(self, title: str, column: int, statuses: list[str]) -> None:
        """Best effort: a dropdown on the Status column (the team may also type other values)."""
        ws = self._ws(title)
        body = {"requests": [{"setDataValidation": {
            "range": {"sheetId": ws.id, "startRowIndex": 1, "startColumnIndex": column - 1,
                      "endColumnIndex": column},
            "rule": {"condition": {"type": "ONE_OF_LIST",
                                   "values": [{"userEnteredValue": s} for s in statuses]},
                     "showCustomUi": True, "strict": False}}}]}
        _retry(lambda: self._sh.batch_update(body))


def _col(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def sheet_url(spreadsheet_id: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"


def open_backend() -> GSpreadBackend:
    s = get_settings()
    if not s.demo_leads_gsheet_id.strip():
        raise SheetError("the leads sheet is off (DEMO_LEADS_GSHEET_ID is empty)")
    return GSpreadBackend(s.demo_leads_gsheet_id.strip(), service_account_info(s.google_service_account_json))


# ── the logic, on any backend ────────────────────────────────────────────────────────────

def ensure_tabs(backend: SheetBackend) -> None:
    """Both tabs exist and carry every required header (created when missing and allowed)."""
    s = get_settings()
    wanted = [(s.demo_sheet_leads_tab, LEAD_COLUMNS)]
    if s.demo_sheet_activity_tab.strip():
        wanted.append((s.demo_sheet_activity_tab, ACTIVITY_COLUMNS))
    existing = backend.tabs()
    for title, header in wanted:
        if title not in existing:
            if not s.demo_sheet_auto_create_tabs:
                raise SheetError(f"the sheet has no {title!r} tab (DEMO_SHEET_AUTO_CREATE_TABS=false)")
            backend.add_tab(title, header)
            if title == s.demo_sheet_leads_tab and hasattr(backend, "add_status_dropdown"):
                try:
                    backend.add_status_dropdown(title, header.index("Status") + 1,
                                                csv_list(s.demo_lead_statuses))
                except Exception:  # noqa: BLE001 — a nicety, never a reason to stop
                    log.warning("could not add the Status dropdown", exc_info=True)
            continue
        if title == s.demo_sheet_activity_tab:
            continue                       # rewritten whole on every sync
        rows = backend.values(title)
        if not rows or not any(c.strip() for c in rows[0]):
            backend.set_header(title, header)
            continue
        missing = [h for h in header if h not in [c.strip() for c in rows[0]]]
        if missing:
            raise SheetError(f"the {title!r} tab is missing the column(s) {', '.join(missing)}; "
                             "put the header back (exact name) or the sync can't place data safely")


def _header_map(header: list[str]) -> dict[str, int]:
    return {name.strip(): i for i, name in enumerate(header) if name.strip()}


def append_leads(backend: SheetBackend, leads: list[dict]) -> list[str]:
    """Append these leads (dicts keyed by LEAD_COLUMNS) to the Leads tab, skipping any Lead ID
    already there. Returns the Lead IDs now in the sheet (appended or already present)."""
    tab = get_settings().demo_sheet_leads_tab
    rows = backend.values(tab)
    cols = _header_map(rows[0])
    present = {r[cols["Lead ID"]] for r in rows[1:] if len(r) > cols["Lead ID"]}
    width = max(cols.values()) + 1
    new_rows = []
    for lead in leads:
        if lead["Lead ID"] in present:
            continue
        row = [""] * width
        for name, value in lead.items():
            if name in cols:
                row[cols[name]] = safe_cell(value)
        new_rows.append(row)
    backend.append_rows(tab, new_rows)
    return [lead["Lead ID"] for lead in leads]


def read_team_columns(backend: SheetBackend) -> dict[str, dict[str, str]]:
    """{Lead ID: {"Status", "Called by", "Notes"}} as the team last left them."""
    tab = get_settings().demo_sheet_leads_tab
    rows = backend.values(tab)
    if not rows:
        return {}
    cols = _header_map(rows[0])
    out = {}
    for r in rows[1:]:
        lid = r[cols["Lead ID"]].strip() if len(r) > cols["Lead ID"] else ""
        if lid:
            out[lid] = {name: (r[cols[name]].strip() if len(r) > cols[name] else "")
                        for name in LEAD_TEAM_COLUMNS}
    return out


def refresh_system_columns(backend: SheetBackend, leads: list[dict]) -> int:
    """Update our own columns (demo calls, longest call…) on rows already in the sheet, so a
    lead who keeps trying the demo shows it. Never touches the team's columns."""
    tab = get_settings().demo_sheet_leads_tab
    rows = backend.values(tab)
    cols = _header_map(rows[0])
    where = {r[cols["Lead ID"]]: i + 1 for i, r in enumerate(rows) if i and len(r) > cols["Lead ID"]}
    updates = []
    for lead in leads:
        row_no = where.get(lead["Lead ID"])
        if row_no is None:
            continue
        current = rows[row_no - 1]
        for name in ("Demo calls", "Longest call", "Goal completed", "Questions asked"):
            value = safe_cell(lead.get(name))
            idx = cols[name]
            if (current[idx] if len(current) > idx else "") != value:
                updates.append((row_no, idx + 1, value))
    backend.update_cells(tab, updates)
    return len(updates)


def write_activity(backend: SheetBackend, rows: list[dict]) -> None:
    tab = get_settings().demo_sheet_activity_tab
    if not tab.strip():
        return
    backend.replace_all(tab, [ACTIVITY_COLUMNS] + [[safe_cell(r.get(c)) for c in ACTIVITY_COLUMNS]
                                                    for r in rows])


async def run(fn, *args):
    """Run blocking sheet work off the event loop."""
    return await asyncio.to_thread(fn, *args)
