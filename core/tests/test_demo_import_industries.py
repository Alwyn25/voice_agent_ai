"""Importing any industry's tab (docs/gtm-demo-links.md §4.8, §5): data-only industry
definitions, either/or minimum rules, columns the definition doesn't know yet (proposed as
new facts, re-connected after a rename, or skipped), picking the tab, inspecting a
workbook, and drafting a definition for a brand-new tab from the workbook's Field guide.

Synthetic workbooks laid out like the real prospect workbook; every name is made up.

    cd core && python -m pytest tests/test_demo_import_industries.py -q
"""
import sys
from io import BytesIO
from pathlib import Path

import openpyxl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.demo.facts import FactValue, check_minimum  # noqa: E402
from app.demo.importer import apply_column_proposals, parse_workbook  # noqa: E402
from app.demo.industries.definition import DefinitionError, compile_definition  # noqa: E402
from app.demo.industries.registry import load_seed_definitions  # noqa: E402
from app.demo.workbook import draft_definition, inspect_workbook  # noqa: E402

SEEDS = {d.key: d for d in load_seed_definitions()}
STD = ["City", "Official website URL", "Buyer name", "Buyer role", "Email", "LinkedIn URL",
       "Trigger / notes", "Verified", "Status"]
TAIL = ["Source URLs", "Date scraped", "Minimum met?"]
D2C_HEADER = ["#", "Brand", "Category", *STD, "Brand name", "Product catalogue", "Best-seller",
              "Shipping timelines", "Shipping charges", "Order tracking", "Payment options",
              "Support", *TAIL]


def tab_rows(header, rows, *, use_case="Order status check"):
    return [[header[1]], [f"Demo use case: {use_case}"], ["From general market knowledge."], [],
            header, *[[r.get(h) for h in header] for r in rows]]


def xlsx(tabs: dict[str, list[list]]) -> bytes:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for title, rows in tabs.items():
        ws = wb.create_sheet(title)
        for r in rows:
            ws.append(r)
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


D2C_READY = {"#": 1, "Brand": "Acme Sleep", "Category": "Mattresses", "City": "Pune",
             "Buyer name": "Asha Rao", "Buyer role": "Founder", "Email": "asha@acmesleep.in",
             "Status": "Reviewed", "Brand name": "Acme Sleep",
             "Product catalogue": "Ortho mattress (Queen) ₹18,999", "Shipping timelines": "3-5 days",
             "Payment options": "UPI, cards, COD", "Source URLs": "https://acmesleep.in",
             "Minimum met?": "Yes"}
EXAMPLE = {"#": "Example", "Brand": "Sample Brand (example)", "Brand name": "Sample"}


def d2c_report(rows=(D2C_READY,), header=D2C_HEADER, **kw):
    return parse_workbook(xlsx({"D2C": tab_rows(header, list(rows))}), SEEDS["d2c"],
                          source_name="t.xlsx", ignore_columns=["#", "Minimum met?"], **kw)


# ── seeds and data-only definitions ──────────────────────────────────────────────────────

def test_every_workbook_tab_has_an_industry():
    assert {"real_estate", "d2c", "hospitals", "restaurants", "hotels", "distributors"} <= set(SEEDS)
    tabs = [t.lower() for d in SEEDS.values() for t in d.sheet["tab_names"]]
    assert len(tabs) == len(set(tabs)), "two industries claim the same tab"
    assert SEEDS["real_estate"].has_agent
    for k in ("d2c", "hospitals", "restaurants", "hotels", "distributors"):
        assert SEEDS[k].has_agent is False and SEEDS[k].sample_template == {}


def test_data_only_needs_just_facts():
    d = compile_definition({"key": "education", "label": "Education", "version": 1,
                            "facts_fields": [{"key": "school_name", "label": "School name"}]})
    assert d.has_agent is False and d.engine is None and d.sheet is None


def test_agent_sections_are_all_or_nothing():
    base = {"key": "education", "label": "Education", "version": 1,
            "facts_fields": [{"key": "school_name", "label": "School name"}]}
    with pytest.raises(DefinitionError) as e:
        compile_definition({**base, "agent": {"persona": "x", "languages": ["en-IN"],
                                              "greeting": "Hi", "instructions": "x"}})
    assert "missing: sample_data, sample_template, prompts" in str(e.value)
    with pytest.raises(DefinitionError) as e:
        compile_definition({**base, "qa_questions": [{"say": "hi", "expect": "hello"}]})
    assert "qa_questions: only allowed with a demo agent" in str(e.value)


def test_minimum_rule_shapes_are_checked():
    base = {"key": "x_y", "label": "X", "version": 1,
            "facts_fields": [{"key": "a_a", "label": "A"}, {"key": "b_b", "label": "B"}]}
    with pytest.raises(DefinitionError) as e:
        compile_definition({**base, "minimum": [{"field": "a_a", "any_of": ["a_a", "b_b"]}]})
    assert "exactly one of `field` or `any_of`" in str(e.value)
    with pytest.raises(DefinitionError) as e:
        compile_definition({**base, "minimum": [{"any_of": ["a_a", "zz"]}]})
    assert "unknown fact 'zz'" in str(e.value)


def test_any_of_minimum():
    fields = [{"key": "menu", "label": "Menu highlights", "type": "list"},
              {"key": "price", "label": "Price range"}]
    rule = [{"any_of": ["menu", "price"], "min_items": 2, "message": "a menu or a price range"}]
    blank = {"menu": FactValue(None), "price": FactValue(None)}
    assert check_minimum(fields, rule, blank) == [
        "Menu highlights or Price range: needs a menu or a price range (found 0)"]
    one_each = {"menu": FactValue("Dosa", items=["Dosa"]), "price": FactValue("₹400 for two")}
    assert check_minimum(fields, rule, one_each) == []
    assert check_minimum(fields, [{"any_of": ["menu", "price"]}], blank)[0].endswith("(found 0)")


# ── importing a data-only industry ───────────────────────────────────────────────────────

def test_d2c_tab_imports_without_an_agent():
    rep = d2c_report([EXAMPLE, D2C_READY, {"#": 2, "Brand": "Bare Brand", "City": "Delhi",
                                            "Status": "Not started"}])
    assert rep.errors == [] and rep.new_columns == []
    assert rep.ignored_columns == ["#", "Minimum met?"]
    assert [s["reason"] for s in rep.skipped] == ["example row"]
    ready, bare = rep.records
    assert ready.status == "ready" and ready.dataset_errors == []
    assert ready.facts["payment_options"].items == ["UPI", "cards", "COD"]
    assert ready.facts["category"].value == "Mattresses"
    assert bare.status == "below_minimum"
    assert "Brand name is missing" in bare.minimum_missing


def test_hospital_branch_is_part_of_the_record_key():
    header = ["#", "Hospital", "Branch area", *STD, "Clinic name", "Branches", "Doctors",
              "Services or packages", *TAIL]
    rows = [{"Hospital": "Acme Hospital", "Branch area": "Whitefield", "Clinic name": "Acme",
             "Branches": "Whitefield, 9-9", "Doctors": "Dr A (Ortho)"},
            {"Hospital": "Acme Hospital", "Branch area": "Indiranagar"}]
    rep = parse_workbook(xlsx({"Hospitals": tab_rows(header, rows)}), SEEDS["hospitals"])
    keys = [r.record_key for r in rep.records]
    assert keys == ["acme-hospital--whitefield", "acme-hospital--indiranagar"]
    assert [r.record_name for r in rep.records] == ["Whitefield", "Indiranagar"]
    assert rep.records[0].status == "ready"
    assert "Doctors or Services or packages: needs doctors or services listed (found 0)" \
        in rep.records[1].minimum_missing


# ── columns the definition doesn't know yet ──────────────────────────────────────────────

def test_new_column_is_proposed_and_its_values_are_previewed():
    header = [*D2C_HEADER[:-3], "Return policy", "Instagram (followers)", *TAIL]
    rep = d2c_report([{**D2C_READY, "Return policy": "7-day returns"}], header=header)
    assert [(p["header"], p["action"], p["key"], p["filled"]) for p in rep.new_columns] == [
        ("Return policy", "add", "return_policy", 1),
        ("Instagram (followers)", "add", "instagram_followers", 0)]
    assert rep.new_columns[0]["column"] == "U" and rep.new_columns[0]["samples"] == ["7-day returns"]
    assert rep.records[0].facts["return_policy"].value == "7-day returns"
    assert rep.records[0].facts["instagram_followers"].is_blank
    assert rep.summary["new_columns"] == 2


def test_skipped_new_column_is_left_out():
    header = [*D2C_HEADER[:-3], "Return policy", *TAIL]
    rep = d2c_report([{**D2C_READY, "Return policy": "7-day returns"}], header=header,
                     skip_columns=["return policy"])
    assert rep.new_columns == [] and "Return policy" in rep.ignored_columns
    assert "return_policy" not in rep.records[0].facts


def test_renamed_column_reconnects_to_its_fact():
    header = [h if h != "Shipping timelines" else "Shipping timeline" for h in D2C_HEADER]
    row = {**{k: v for k, v in D2C_READY.items() if k != "Shipping timelines"},
           "Shipping timeline": "2 days in metros"}
    rep = d2c_report([row], header=header)
    assert rep.errors == []
    (p,) = rep.new_columns
    assert (p["action"], p["key"], p["previous_column"]) == ("map", "shipping_timelines",
                                                             "Shipping timelines")
    assert rep.records[0].facts["shipping_timelines"].value == "2 days in metros"
    assert rep.records[0].status == "ready"


def test_rename_is_an_error_when_new_columns_are_ignored():
    header = [h if h != "Shipping timelines" else "Shipping timeline" for h in D2C_HEADER]
    rep = d2c_report([D2C_READY], header=header, new_columns="ignore")
    assert rep.new_columns == []
    assert any("'Shipping timelines' (Shipping timelines) not found; renamed to "
               "'Shipping timeline'?" in e for e in rep.errors)


def test_new_column_feeds_a_fact_that_had_no_column():
    header = ["#", "Distributor", "Brands carried", "City", "Source page", *STD[1:], "Areas served"]
    rows = [{"Distributor": "Acme Agencies", "Brands carried": "Brand A, Brand B",
             "Areas served": "Jayanagar, JP Nagar"}]
    rep = parse_workbook(xlsx({"Distributors": tab_rows(header, rows)}), SEEDS["distributors"])
    (p,) = rep.new_columns
    assert (p["header"], p["action"], p["key"]) == ("Areas served", "map", "service_area")
    assert rep.records[0].facts["service_area"].value == "Jayanagar, JP Nagar"
    assert rep.records[0].facts["brands_carried"].items == ["Brand A", "Brand B"]


def test_accepted_columns_compile_into_the_next_version():
    header = [*D2C_HEADER[:-3], "Return policy", *TAIL]
    rep = d2c_report([D2C_READY], header=header)
    raw = {**apply_column_proposals(SEEDS["d2c"].raw, rep.new_columns), "version": 2}
    d = compile_definition(raw)
    added = next(f for f in d.facts_fields if f["key"] == "return_policy")
    assert added == {"key": "return_policy", "label": "Return policy", "type": "text",
                     "sheet_column": "Return policy",
                     "description": "Added from the workbook column 'Return policy'"}
    # Imported again with the new version, nothing is new any more.
    again = parse_workbook(xlsx({"D2C": tab_rows(header, [D2C_READY])}), d,
                           ignore_columns=["#", "Minimum met?"])
    assert again.new_columns == []


def test_definition_can_turn_new_columns_off():
    raw = {**SEEDS["d2c"].raw, "sheet": {**SEEDS["d2c"].raw["sheet"], "extra_columns": "ignore"}}
    header = [*D2C_HEADER[:-3], "Return policy", *TAIL]
    rep = parse_workbook(xlsx({"D2C": tab_rows(header, [D2C_READY])}), compile_definition(raw),
                         ignore_columns=["#", "Minimum met?"])
    assert rep.new_columns == []
    assert any("'Return policy'" in w and "not imported" in w for w in rep.warnings)


def test_admin_picked_tab():
    data = xlsx({"Brands Q4": tab_rows(D2C_HEADER, [D2C_READY])})
    rep = parse_workbook(data, SEEDS["d2c"], tab="brands q4", ignore_columns=["#", "Minimum met?"])
    assert rep.tab == "Brands Q4" and len(rep.records) == 1


# ── inspecting a workbook and drafting a new industry ────────────────────────────────────

EDU_HEADER = ["#", "School group", *STD, "School name", "Courses", "Fees", "Admission dates", *TAIL]
FIELD_GUIDE = [
    ["Field guide"], [],
    ["Tab", "Minimum to go live", "Trusted-source rule"],
    ["Education", "School name and courses", "The school's own website only."],
    [],
    ["Tab", "Field", "What to capture", "Trusted source", "In the minimum to go live?"],
    ["Education", "School name", "Name and pronunciation", "Official website", "Yes"],
    ["Education", "Courses", "Courses with durations", "Official website", "Yes (2 courses)"],
    ["Education", "Fees", "Fee per year, if published", "Official website", None],
]


def full_workbook() -> bytes:
    return xlsx({"Overview": [["Campaign"], ["Tab", "Companies"]],
                 "D2C": tab_rows(D2C_HEADER, [EXAMPLE, D2C_READY]),
                 "Education": tab_rows(EDU_HEADER, [{"School group": "Acme Schools",
                                                     "School name": "Acme Public School",
                                                     "Courses": "CBSE K-12, IB"}],
                                       use_case="Admission enquiry"),
                 "Field guide": FIELD_GUIDE})


def test_inspect_matches_tabs_to_industries():
    tabs = {t["tab"]: t for t in inspect_workbook(full_workbook(), list(SEEDS.values()))["tabs"]}
    assert (tabs["D2C"]["kind"], tabs["D2C"]["industry_key"]) == ("industry", "d2c")
    assert (tabs["D2C"]["rows"], tabs["D2C"]["example_rows"], tabs["D2C"]["header_row"]) == (1, 1, 5)
    assert tabs["D2C"]["use_case"] == "Order status check"
    assert tabs["Education"]["kind"] == "new" and tabs["Education"]["rows"] == 1
    assert tabs["Overview"]["kind"] == "other" and tabs["Field guide"]["kind"] == "other"


def test_draft_definition_for_a_new_tab():
    raw, notes = draft_definition(full_workbook(), "Education", list(SEEDS.values()),
                                  ignore_columns=["#", "Minimum met?"])
    d = compile_definition(raw)
    assert (d.key, d.label, d.has_agent) == ("education", "Education", False)
    assert raw["use_case"] == {"key": "admission_enquiry", "label": "Admission enquiry"}
    assert raw["trusted_sources"] == "The school's own website only."
    assert raw["sheet"]["company"]["company_name"] == "School group"
    assert raw["sheet"]["company"]["city"] == "City"
    assert raw["sheet"]["contacts"]["full_name"] == "Buyer name"
    facts = {f["key"]: f for f in raw["facts_fields"]}
    assert list(facts) == ["school_name", "courses", "fees", "admission_dates"]
    assert facts["school_name"]["required"] and facts["courses"]["required"]
    assert "required" not in facts["fees"]
    assert facts["courses"]["description"] == "Courses with durations"
    assert any("School name and courses" in n for n in notes)
    # The drafted definition imports its own tab.
    rep = parse_workbook(full_workbook(), d, ignore_columns=["#", "Minimum met?"])
    assert rep.new_columns == [] and rep.records[0].status == "ready"
    assert rep.records[0].facts["courses"].value == "CBSE K-12, IB"


def test_one_broken_seed_does_not_block_the_others(tmp_path):
    (tmp_path / "good_one.yaml").write_text(
        "key: good_one\nlabel: Good\nversion: 1\nfacts_fields: [{key: name_x, label: Name}]\n",
        encoding="utf-8")
    (tmp_path / "bad_one.yaml").write_text("key: bad_one\nlabel: Bad\n", encoding="utf-8")
    with pytest.raises(DefinitionError):
        load_seed_definitions(tmp_path)
    problems: list[str] = []
    assert [d.key for d in load_seed_definitions(tmp_path, errors=problems)] == ["good_one"]
    assert len(problems) == 1 and problems[0].startswith("bad_one.yaml")
