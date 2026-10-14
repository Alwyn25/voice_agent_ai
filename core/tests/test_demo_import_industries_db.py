"""Multi-industry workbook import through the admin API against a scratch database:
every workbook tab has an industry, data-only industries import but can't have links, a
new column becomes the industry's next definition version on apply, a stale preview is
refused, and a brand-new tab is drafted, published and imported without code.

    TEST_ADMIN_DATABASE_URL=postgresql://aivoiceagent:aivoiceagent@127.0.0.1:5437/postgres \
        python -m pytest tests/test_demo_import_industries_db.py -q
"""
import yaml

from tests.test_demo_admin_api import ADMIN_URL, client, env, pytestmark, sql  # noqa: F401
from tests.test_demo_import_industries import (D2C_HEADER, D2C_READY, TAIL, full_workbook,
                                               tab_rows, xlsx)

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
HEADER_PLUS = [*D2C_HEADER[:-3], "Return policy", *TAIL]


def d2c_bytes(rows=None, header=HEADER_PLUS):
    return xlsx({"D2C": tab_rows(header, rows or [{**D2C_READY, "Return policy": "7-day returns"}])})


def preview(client, data, industry, **form):
    r = client.post("/admin/demo/imports", data={"industry_key": industry, **form},
                    files={"file": ("prospects.xlsx", data, XLSX)})
    assert r.status_code == 201, r.text
    return r.json()


def test_every_industry_listed_with_its_readiness(client, env):
    items = {i["key"]: i for i in client.get("/admin/demo/industries").json()}
    assert {"real_estate", "d2c", "hospitals", "restaurants", "hotels", "distributors"} <= set(items)
    assert items["real_estate"]["has_agent"] and items["real_estate"]["can_go_live"]
    d2c = items["d2c"]
    assert (d2c["has_agent"], d2c["can_go_live"], d2c["can_import"]) == (False, False, True)
    assert d2c["tab_names"] == ["D2C"] and "no demo agent yet" in d2c["problems"][0]
    # A data-only industry gets no dummy-data template.
    assert sql(env, "SELECT count(*) FROM demo_sample_datasets WHERE industry_key = 'd2c'")[0][0] == 0


def test_inspect_workbook(client):
    r = client.post("/admin/demo/imports/inspect", files={"file": ("w.xlsx", full_workbook(), XLSX)})
    assert r.status_code == 200, r.text
    tabs = {t["tab"]: t for t in r.json()["tabs"]}
    assert tabs["D2C"]["industry_key"] == "d2c" and tabs["Education"]["kind"] == "new"


def test_new_column_becomes_the_next_definition_version(client):
    pv = preview(client, d2c_bytes(), "d2c")
    rep = pv["report"]
    assert [(c["header"], c["action"], c["key"]) for c in rep["new_columns"]] == [
        ("Return policy", "add", "return_policy")]
    assert rep["records"][0]["facts"]["return_policy"]["value"] == "7-day returns"
    assert rep["records"][0]["record_name"] is None

    res = client.post(f"/admin/demo/imports/{pv['id']}/apply", json={}).json()
    assert res["created"] == 1 and res["agent_errors"] == [] and res["has_agent"] is False
    assert res["industry_version"] == 2
    assert res["columns_added"] == [{"header": "Return policy", "key": "return_policy", "action": "add"}]

    ind = client.get("/admin/demo/industries/d2c").json()
    assert ind["current_version"] == 2 and ind["versions"] == [1, 2]
    assert any(f["key"] == "return_policy" for f in ind["definition"]["facts_fields"])
    rid = client.get("/admin/demo/records", params={"industry_key": "d2c"}).json()["items"][0]["id"]
    rec = client.get(f"/admin/demo/records/{rid}").json()
    assert rec["industry_version"] == 2 and rec["has_agent"] is False
    assert rec["facts"]["return_policy"]["value"] == "7-day returns"
    assert rec["minimum_met"] is True
    # The fact is in the record's knowledge base like every other.
    text = client.state["jobs"][-1].text
    assert "- Return policy: 7-day returns" in text

    # Imported again: the column is a normal fact now, nothing is proposed.
    pv2 = preview(client, d2c_bytes(), "d2c")
    assert pv2["report"]["new_columns"] == [] and pv2["industry_version"] == 2


def test_data_only_record_has_no_agent_link_or_dummy_data(client):
    rid = client.get("/admin/demo/records", params={"industry_key": "d2c"}).json()["items"][0]["id"]
    r = client.post(f"/admin/demo/records/{rid}/link", json={})
    assert r.status_code == 409 and "no demo agent" in r.json()["detail"]["message"]
    assert client.post(f"/admin/demo/records/{rid}/agent").status_code == 409
    assert client.put(f"/admin/demo/records/{rid}/dummy-data", json={}).status_code == 409


def test_skipped_column_is_not_added(client):
    header = [*D2C_HEADER[:-3], "Instagram handle", *TAIL]
    data = d2c_bytes([{**D2C_READY, "Brand": "Other Brand", "Instagram handle": "@other"}], header)
    pv = preview(client, data, "d2c", skip_columns='["Instagram handle"]')
    assert pv["report"]["new_columns"] == [] and "Instagram handle" in pv["report"]["ignored_columns"]
    res = client.post(f"/admin/demo/imports/{pv['id']}/apply", json={}).json()
    assert res["columns_added"] == [] and res["industry_version"] == 2


def test_stale_preview_is_refused(client):
    first = preview(client, d2c_bytes([D2C_READY], [*D2C_HEADER[:-3], "Warranty", *TAIL]), "d2c")
    second = preview(client, d2c_bytes([D2C_READY], [*D2C_HEADER[:-3], "Warranty", *TAIL]), "d2c")
    assert client.post(f"/admin/demo/imports/{second['id']}/apply", json={}).status_code == 200
    r = client.post(f"/admin/demo/imports/{first['id']}/apply", json={})
    assert r.status_code == 409 and "changed since this preview" in r.json()["detail"]["message"]


def test_draft_publish_and_import_a_new_industry(client):
    r = client.post("/admin/demo/industries/draft", data={"tab": "Education"},
                    files={"file": ("w.xlsx", full_workbook(), XLSX)})
    assert r.status_code == 200, r.text
    draft = r.json()
    assert draft["valid"] and draft["key"] == "education" and draft["errors"] == []
    r = client.post("/admin/demo/industries", data={"text": draft["yaml"]})
    assert r.status_code == 201 and r.json()["created"] is True

    pv = preview(client, full_workbook(), "education")
    assert pv["report"]["tab"] == "Education" and pv["report"]["summary"]["ready"] == 1
    res = client.post(f"/admin/demo/imports/{pv['id']}/apply", json={}).json()
    assert res["created"] == 1 and res["agent_errors"] == []
    # Re-drafting the same tab now says the industry exists.
    again = client.post("/admin/demo/industries/draft", data={"tab": "Education"},
                        files={"file": ("w.xlsx", full_workbook(), XLSX)}).json()
    assert any("already exists" in n for n in again["notes"])


def test_real_estate_is_unchanged(client):
    from tests.test_demo_admin_api import re_rows, workbook_bytes
    pv = preview(client, workbook_bytes(re_rows()), "real_estate")
    assert pv["report"]["new_columns"] == []
    assert pv["report"]["ignored_columns"] == ["#", "Minimum met?"]
    assert yaml.safe_dump(pv["report"]["summary"])            # serialisable as stored
