"""The leads sheet logic and the lead email, against an in-memory sheet (no Google, no DB):

  tabs        created with headers; an empty tab gets its header; a missing header stops
              the sync instead of writing into the wrong column
  leads       appended once (by Lead ID); columns matched by header after the team moves
              or adds columns; formulas typed into the form are written as text
  read back   Status / Called by / Notes found by Lead ID after the team sorts rows
  refresh     our columns update in place; the team's are never touched
  key         base64, JSON or a path; anything else refused
  email       what the prospect typed is escaped; Reply-To is the lead

    python -m pytest tests/test_demo_leads_sheet.py -q
"""
import base64
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    for k, v in {"DEMO_SHEET_LEADS_TAB": "Leads", "DEMO_SHEET_ACTIVITY_TAB": "Demo activity",
                 "DEMO_SHEET_AUTO_CREATE_TABS": "true", "DEMO_LEAD_NOTIFY_EMAILS": "sales@phonn.ai",
                 "DEMO_LEAD_FROM_EMAIL": "leads@phonn.ai", "DEMO_LEADS_GSHEET_ID": "sheet-id",
                 "DASHBOARD_URL": ""}.items():
        monkeypatch.setenv(k, v)
    from app.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeSheet:
    """Enough of a spreadsheet for the sync: tabs of rows of strings."""

    def __init__(self, tabs=None):
        self.data = {k: [list(r) for r in v] for k, v in (tabs or {}).items()}
        self.appends = 0

    def tabs(self):
        return list(self.data)

    def add_tab(self, title, header):
        self.data[title] = [list(header)]

    def set_header(self, title, header):
        rows = self.data[title]
        if rows:
            rows[0] = list(header)
        else:
            rows.append(list(header))

    def values(self, title):
        return [list(r) for r in self.data[title]]

    def append_rows(self, title, rows):
        if rows:
            self.appends += 1
        self.data[title].extend(list(r) for r in rows)

    def update_cells(self, title, updates):
        for r, c, v in updates:
            row = self.data[title][r - 1]
            while len(row) < c:
                row.append("")
            row[c - 1] = v

    def replace_all(self, title, rows):
        self.data[title] = [list(r) for r in rows]


def lead(lid="L-1", **kw):
    from app.demo.leads.sheets import LEAD_COLUMNS
    row = {c: "" for c in LEAD_COLUMNS}
    row.update({"Lead ID": lid, "Name": "Priya", "Email": "priya@x.in", "Status": "New",
                "Demo calls": "1"}, **kw)
    return row


def test_tabs_are_created_with_headers():
    from app.demo.leads.sheets import ACTIVITY_COLUMNS, LEAD_COLUMNS, ensure_tabs
    sheet = FakeSheet({"Sheet1": []})
    ensure_tabs(sheet)
    assert sheet.data["Leads"] == [LEAD_COLUMNS] and sheet.data["Demo activity"] == [ACTIVITY_COLUMNS]
    assert sheet.data["Sheet1"] == []                       # the team's own tab is left alone


def test_empty_tab_gets_header_missing_header_stops():
    from app.demo.leads.sheets import LEAD_COLUMNS, SheetError, ensure_tabs
    sheet = FakeSheet({"Leads": [], "Demo activity": []})
    ensure_tabs(sheet)
    assert sheet.data["Leads"][0] == LEAD_COLUMNS
    broken = FakeSheet({"Leads": [[c for c in LEAD_COLUMNS if c != "Status"]], "Demo activity": []})
    with pytest.raises(SheetError, match="Status"):
        ensure_tabs(broken)


def test_append_once_by_lead_id_and_by_header_name():
    from app.demo.leads.sheets import LEAD_COLUMNS, append_leads
    # The team moved Notes first and added their own column.
    header = ["Notes", "Priority"] + [c for c in LEAD_COLUMNS if c != "Notes"]
    sheet = FakeSheet({"Leads": [header]})
    append_leads(sheet, [lead("L-1"), lead("L-2", Name="Ravi")])
    append_leads(sheet, [lead("L-1"), lead("L-3")])          # L-1 again: not duplicated
    rows = sheet.data["Leads"][1:]
    assert [r[header.index("Lead ID")] for r in rows] == ["L-1", "L-2", "L-3"]
    assert rows[1][header.index("Name")] == "Ravi" and rows[0][header.index("Notes")] == ""
    assert rows[0][header.index("Priority")] == ""


def test_formulas_are_written_as_text():
    from app.demo.leads.sheets import LEAD_COLUMNS, append_leads, safe_cell
    sheet = FakeSheet({"Leads": [LEAD_COLUMNS]})
    append_leads(sheet, [lead("L-9", Name='=IMPORTXML("http://evil","//a")', Message="+1 call me",
                              Company="@acme", Role="-x")])
    row = dict(zip(LEAD_COLUMNS, sheet.data["Leads"][1]))
    assert row["Name"].startswith("'=") and row["Message"] == "'+1 call me"
    assert row["Company"] == "'@acme" and row["Role"] == "'-x"
    assert safe_cell(None) == "" and safe_cell("Priya") == "Priya"
    assert safe_cell("+91 98765 43210") == "+91 98765 43210" and safe_cell("-3") == "-3"
    assert safe_cell("+cmd|' /C calc'!A0") == "'+cmd|' /C calc'!A0"


def test_team_columns_read_back_after_sorting():
    from app.demo.leads.sheets import LEAD_COLUMNS, append_leads, read_team_columns
    sheet = FakeSheet({"Leads": [LEAD_COLUMNS]})
    append_leads(sheet, [lead("L-1"), lead("L-2")])
    st, cb, nt = (LEAD_COLUMNS.index(c) for c in ("Status", "Called by", "Notes"))
    sheet.data["Leads"][1][st], sheet.data["Leads"][1][cb] = "Contacted", "Asha"
    sheet.data["Leads"][2][nt] = "call after 4"
    sheet.data["Leads"][1:] = sheet.data["Leads"][1:][::-1]   # the team sorted the rows
    team = read_team_columns(sheet)
    assert team["L-1"] == {"Status": "Contacted", "Called by": "Asha", "Notes": ""}
    assert team["L-2"]["Notes"] == "call after 4"


def test_refresh_touches_only_our_columns():
    from app.demo.leads.sheets import LEAD_COLUMNS, append_leads, refresh_system_columns
    sheet = FakeSheet({"Leads": [LEAD_COLUMNS]})
    append_leads(sheet, [lead("L-1")])
    st = LEAD_COLUMNS.index("Status")
    sheet.data["Leads"][1][st] = "Converted"
    n = refresh_system_columns(sheet, [lead("L-1", **{"Demo calls": "3", "Status": "New"})])
    row = dict(zip(LEAD_COLUMNS, sheet.data["Leads"][1]))
    assert n == 1 and row["Demo calls"] == "3" and row["Status"] == "Converted"


def test_activity_tab_rewritten():
    from app.demo.leads.sheets import ACTIVITY_COLUMNS, write_activity
    sheet = FakeSheet({"Demo activity": [["old"], ["stale"]]})
    write_activity(sheet, [{"Company": "Acme", "Calls": 2}])
    assert sheet.data["Demo activity"][0] == ACTIVITY_COLUMNS
    assert sheet.data["Demo activity"][1][:1] == ["Acme"] and len(sheet.data["Demo activity"]) == 2


def test_service_account_key_forms(tmp_path):
    from app.demo.leads.sheets import SheetError, service_account_info
    info = {"type": "service_account", "client_email": "a@b.iam.gserviceaccount.com", "private_key": "k"}
    text = json.dumps(info)
    assert service_account_info(base64.b64encode(text.encode()).decode())["client_email"] == info["client_email"]
    assert service_account_info(text)["private_key"] == "k"
    path = tmp_path / "key.json"
    path.write_text(text)
    assert service_account_info(str(path))["client_email"] == info["client_email"]
    for bad in ("", "nope", base64.b64encode(b'{"type": "service_account"}').decode(), os.devnull):
        with pytest.raises(SheetError):
            service_account_info(bad)


def test_lead_email_is_escaped_and_replies_to_the_lead():
    from app.demo.leads.mailer import lead_mail
    m = lead_mail(lead("L-5", Name="<script>x</script>", Message="a & b\nnext", Company="Acme"),
                  subject_bits="Schedule a call · Acme")
    assert m.to == ["sales@phonn.ai"] and m.reply_to == "priya@x.in"
    assert "<script>" not in m.html and "&lt;script&gt;" in m.html and "a &amp; b<br>next" in m.html
    assert m.subject.startswith("[Demo lead] Schedule a call") and "L-5" in m.text
    assert "/t/" not in m.text and "/t/" not in m.html          # never the demo link
