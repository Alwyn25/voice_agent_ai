"""Company facts (app/demo/facts.py) and contact pairing (app/demo/contacts.py).

The configuration strings below are the shapes found in the real-estate prospect sheet;
the people are made up.

    cd core && python -m pytest tests/test_demo_facts_contacts.py -q
"""
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.demo.contacts import parse_people, tidy_name  # noqa: E402
from app.demo.facts import (check_minimum, normalize_fact, normalize_url,  # noqa: E402
                            parse_unit_types, split_list)


# ── facts ──

@pytest.mark.parametrize("text, expected", [
    (" 2BHK - 954 to 1006 Sqft, 2.5BHK - 1104 to 1146 Sqft, 3BHK - 1136 to 1406 Sqft",
     ["2 BHK", "2.5 BHK", "3 BHK"]),
    ("3 BHK Type 1 – 1,670 sq. ft.; 3 BHK Type 2 – 1,785 sq. ft.; 4 BHK – 2,168 sq. ft. (SBUA)",
     ["3 BHK", "4 BHK"]),
    ("3 & 4 BHK", ["3 BHK", "4 BHK"]),
    ("3, 4 & 5 Bed Residences", ["3 BHK", "4 BHK", "5 BHK"]),
    ("3, 3.5 & 4.5 BHK", ["3 BHK", "3.5 BHK", "4.5 BHK"]),
    ("Studio, 1 & 2 BHK", ["Studio", "1 BHK", "2 BHK"]),
    ("2, 3, 4 BHK apartments; 4 BHK floor villas", ["2 BHK", "3 BHK", "4 BHK"]),
    ("3 & 4 bedroom residences", ["3 BHK", "4 BHK"]),
    ("3 & 4 BR", ["3 BHK", "4 BHK"]),
    ("5 BHK", ["5 BHK"]),
    ("Luxury residences", []),
])
def test_unit_types(text, expected):
    assert parse_unit_types(text) == expected


def test_month_from_excel_date_and_year():
    assert normalize_fact(datetime(2029, 12, 1), "month").value == "December 2029"
    assert normalize_fact(2030.0, "month").value == "2030"
    assert normalize_fact("Dec 2028; confirm", "month").value == "Dec 2028; confirm"


def test_blank_and_error_cells():
    for blank in (None, "", "  ", "-", "—", "N/A", "TBD"):
        assert normalize_fact(blank, "text").is_blank
    fv = normalize_fact("#ERROR!", "text")
    assert fv.is_blank and "spreadsheet error cell" in fv.warnings[0]


def test_list_keeps_commas_inside_numbers():
    assert split_list("Pool, gym; ₹2,000 off, clubhouse, Pool") == ["Pool", "gym", "₹2,000 off", "clubhouse"]


def test_text_tidies_whitespace_and_integral_floats():
    assert normalize_fact("  Sector   84,\n\n Gurugram ", "text").value == "Sector 84,\nGurugram"
    assert normalize_fact(12.0, "text").value == "12"


def test_unrecognised_units_warn():
    fv = normalize_fact("Premium residences", "unit_types")
    assert fv.items == [] and fv.warnings


@pytest.mark.parametrize("raw, url", [("vgn.in", "https://vgn.in/"),
                                      ("https://www.vgn.in/", "https://www.vgn.in/"),
                                      ("http://x.com/a b", "http://x.com/a")])
def test_normalize_url(raw, url):
    assert normalize_url(raw) == url


FIELDS = [{"key": "project_name", "label": "Project name", "required": True},
          {"key": "configurations", "label": "Configurations", "type": "unit_types", "required": True},
          {"key": "sales_office", "label": "Sales office", "required": True}]
MINIMUM = [{"field": "configurations", "min_items": 2, "message": "at least 2 BHK types"}]


def test_minimum_met():
    facts = {"project_name": normalize_fact("Green Acres", "text"),
             "configurations": normalize_fact("2 & 3 BHK", "unit_types"),
             "sales_office": normalize_fact("MG Road", "text")}
    assert check_minimum(FIELDS, MINIMUM, facts) == []


def test_minimum_reports_each_gap():
    facts = {"project_name": normalize_fact("Green Acres", "text"),
             "configurations": normalize_fact("3 BHK", "unit_types"),
             "sales_office": normalize_fact(None, "text")}
    assert check_minimum(FIELDS, MINIMUM, facts) == [
        "Sales office is missing",
        "Configurations: needs at least 2 BHK types (found 1: 3 BHK)"]


# ── contacts ──

def emails_by_name(people):
    return {c.full_name: c.email for c in people.contacts}


def test_positional_pairing_when_lists_line_up():
    p = parse_people("Asha Rao, Vikram Iyer", "Sales head, Manager",
                     "sales@acme.in, asha.rao@acme.in, vikram@acme.in",
                     "linkedin.com/in/asha-rao-1, https://www.linkedin.com/in/vikram-iyer/")
    assert [(c.full_name, c.role) for c in p.contacts] == [("Asha Rao", "Sales head"),
                                                          ("Vikram Iyer", "Manager")]
    assert emails_by_name(p) == {"Asha Rao": "asha.rao@acme.in", "Vikram Iyer": "vikram@acme.in"}
    assert p.company_emails == ["sales@acme.in"]
    assert p.contacts[0].linkedin_url == "https://www.linkedin.com/in/asha-rao-1/"
    assert not any(c.needs_review for c in p.contacts)


def test_best_match_beats_shared_surname():
    p = parse_people("Harish Sharma, Mamta Sharma", None, "mamta.sharma@acme.in", None)
    assert emails_by_name(p)["Mamta Sharma"] == "mamta.sharma@acme.in"
    assert emails_by_name(p)["Harish Sharma"] is None


def test_initial_and_reversed_forms():
    p = parse_people("Anurag Chauhan, Tejesh Gollapelli, Karthikeyan Vishwanathan", None,
                     "achauhan@acme.in, gollapellitejesh@acme.in, vkarthikeyan@acme.in", None)
    assert emails_by_name(p) == {"Anurag Chauhan": "achauhan@acme.in",
                                 "Tejesh Gollapelli": "gollapellitejesh@acme.in",
                                 "Karthikeyan Vishwanathan": "vkarthikeyan@acme.in"}


def test_single_letter_disambiguates():
    p = parse_people("Sreedhar G, S Sreedhar", None, "sreedhar.s@acme.in, sreedhar.g@acme.in", None)
    assert emails_by_name(p) == {"Sreedhar G": "sreedhar.g@acme.in", "S Sreedhar": "sreedhar.s@acme.in"}


def test_similar_spelling_pairs_but_flags_review():
    p = parse_people("Yuvraj Krishnan", None, "yuvaraj@acme.in", None)
    c = p.contacts[0]
    assert c.email == "yuvaraj@acme.in"
    assert c.needs_review and "similar spelling" in c.notes[0]


def test_unmatched_and_ambiguous_emails_are_reported_not_guessed():
    p = parse_people("Arun Kumar, Varun Kumar", None, "someone.else@acme.in, kumar@acme.in", None)
    assert emails_by_name(p) == {"Arun Kumar": None, "Varun Kumar": None}
    assert any("matches none" in w for w in p.warnings)
    assert any("matches 2 people equally" in w for w in p.warnings)
    assert all(c.needs_review for c in p.contacts)


def test_work_email_preferred_over_webmail_and_webmail_flagged():
    p = parse_people("Ritesh Singh", None, "ritesh606@gmail.com, ritesh.s@acme.in", None)
    assert p.contacts[0].email == "ritesh.s@acme.in"
    p = parse_people("Ritesh Singh", None, "ritesh606@gmail.com", None)
    assert any("webmail" in n for n in p.contacts[0].notes)


def test_role_count_mismatch_leaves_roles_for_review():
    p = parse_people("A One, B Two, C Three", "Manager, Head", None, None)
    assert all(c.role is None for c in p.contacts)
    assert any("roles for 3 names" in w for w in p.warnings)


def test_linkedin_in_email_column_is_rescued():
    p = parse_people("Rama Rao", None, "rama@acme.in, linkedin.com/in/rama-rao-77", None)
    assert p.contacts[0].linkedin_url == "https://www.linkedin.com/in/rama-rao-77/"
    assert p.contacts[0].email == "rama@acme.in"


def test_linkedin_by_slug_when_counts_differ():
    p = parse_people("Asha Rao, Vikram Iyer, Meena Das", None, None,
                     "linkedin.com/in/meena-das-9, linkedin.com/in/asha-rao")
    links = {c.full_name: c.linkedin_url for c in p.contacts}
    assert links["Meena Das"].endswith("/meena-das-9/")
    assert links["Asha Rao"].endswith("/asha-rao/")
    assert links["Vikram Iyer"] is None


def test_names_tidied_and_deduplicated():
    assert tidy_name("pavan kumar Nelakonda") == "Pavan Kumar Nelakonda"
    assert tidy_name("S  Sreedhar") == "S Sreedhar"
    p = parse_people("asha rao, Asha Rao", None, None, None)
    assert [c.full_name for c in p.contacts] == ["Asha Rao"]
