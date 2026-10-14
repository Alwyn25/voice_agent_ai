"""Dummy-data engine (app/demo/sample/): schema compiler, validator, and resolve_dataset.

Pure unit tests:

    cd core && python -m pytest tests/test_demo_sample_engine.py -q
"""
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.demo.sample.compiler import SchemaDefinitionError, compile_sample_schema  # noqa: E402
from app.demo.sample.resolve import resolve_dataset, spoken_time  # noqa: E402
from app.demo.sample.validate import validate_dataset  # noqa: E402

IST = ZoneInfo("Asia/Kolkata")

BLOCK = {
    "caller": "sample_caller",
    "lead_source": "text_template",
    "enquiry_minutes_before_call": {"type": "integer", "min": 0, "max": 1440},
    "free_visit_slots": {"type": "list", "min_items": 2, "item": "relative_slot"},
    "booking_reference": {"type": "ref_code", "prefix": "VIS-"},
    "placed": {"type": "relative_day", "optional": True},
    "status": {"type": "enum", "values": ["placed", "shipped"], "optional": True},
}
SCHEMA = compile_sample_schema(BLOCK)
FACTS_KEYS = {"project_name", "location", "price"}
GOOD = {
    "caller": {"name": "Rahul", "phone_masked": "+91 98XXX XXX01"},
    "lead_source": "Facebook lead form for {{project_name}}",
    "enquiry_minutes_before_call": 10,
    "free_visit_slots": [{"day_offset": 1, "time": "11:00"}, {"day_offset": 1, "time": "16:00"},
                         {"day_offset": 2, "time": "11:00"}],
    "booking_reference": "VIS-1187",
}


def errors(data, facts=None):
    return [str(e) for e in validate_dataset(data, SCHEMA, allowed_placeholders=FACTS_KEYS,
                                             facts=facts)]


def has(errs, fragment):
    assert any(fragment in e for e in errs), f"{fragment!r} not in {errs}"


# ── compiler ──

def test_unknown_type_rejected():
    with pytest.raises(SchemaDefinitionError, match="unknown field type 'datetime'"):
        compile_sample_schema({"when": "datetime"})


def test_bad_option_rejected():
    with pytest.raises(SchemaDefinitionError, match="not valid for string"):
        compile_sample_schema({"x": {"type": "string", "min_items": 2}})


def test_list_needs_item_and_enum_needs_values():
    with pytest.raises(SchemaDefinitionError, match="needs an `item`"):
        compile_sample_schema({"x": {"type": "list"}})
    with pytest.raises(SchemaDefinitionError, match="enum needs"):
        compile_sample_schema({"x": {"type": "enum"}})


def test_object_shorthand_and_optional():
    s = compile_sample_schema({"order": {"no": "string", "note": {"type": "string", "optional": True}}})
    order = s["properties"]["order"]
    assert order["required"] == ["no"]
    assert order["additionalProperties"] is False


# ── validator ──

def test_good_dataset_passes():
    assert errors(GOOD) == []
    assert errors(GOOD, facts={"project_name": "Green Acres"}) == []


def test_unknown_field_is_an_error():
    has(errors({**GOOD, "free_vist_slots": []}), "unknown field(s): free_vist_slots")


def test_missing_field_is_an_error():
    data = {k: v for k, v in GOOD.items() if k != "booking_reference"}
    has(errors(data), "'booking_reference' is required")


def test_plain_sample_name_ok_but_phone_must_be_masked():
    errs = errors({**GOOD, "caller": {"name": "Rahul", "phone_masked": "+91 9876543201"}})
    assert errs == ["caller.phone_masked: phone must be masked like +91 98XXX XXX01"]


def test_bad_time_and_too_few_slots():
    errs = errors({**GOOD, "free_visit_slots": [{"day_offset": 1, "time": "25:00"}]})
    has(errs, "free_visit_slots[0].time: time must be HH:MM")
    has(errs, "free_visit_slots: ")


def test_ref_code_prefix():
    has(errors({**GOOD, "booking_reference": "BK-1187"}), "booking_reference")


@pytest.mark.parametrize("text", ["Call on 2026-10-01", "visit 01/10/2026", "on 1-10-2026"])
def test_absolute_dates_refused_anywhere(text):
    has(errors({**GOOD, "lead_source": text}), "absolute dates are not allowed")


def test_unknown_placeholder():
    has(errors({**GOOD, "lead_source": "Ad for {{projct_name}}"}), "unknown placeholder {{projct_name}}")


def test_placeholder_must_have_a_value_for_this_record():
    has(errors(GOOD, facts={"project_name": "  "}), "has no value for this company")


def test_placeholder_only_in_text_templates():
    data = {**GOOD, "caller": {"name": "{{project_name}}", "phone_masked": "+91 98XXX XXX01"}}
    has(errors(data), "only allowed in text fields declared as text_template")


def test_malformed_placeholder():
    has(errors({**GOOD, "lead_source": "Ad for {{ Project Name }"}), "malformed placeholder")


# ── resolve ──

def resolve(data, now, skip=()):
    return resolve_dataset(data, SCHEMA, {"project_name": "Green Acres"}, now=now,
                           tz="Asia/Kolkata", skip_weekdays=skip).data


@pytest.mark.parametrize("now, expected", [
    # The spec's own examples: a call on 30 Sep offers 1 and 2 Oct; on 10 Oct, 11 and 12 Oct.
    (datetime(2026, 9, 30, 15, 0, tzinfo=IST), ["2026-10-01", "2026-10-01", "2026-10-02"]),
    (datetime(2026, 10, 10, 9, 30, tzinfo=IST), ["2026-10-11", "2026-10-11", "2026-10-12"]),
])
def test_slots_relative_to_call_time(now, expected):
    assert [s["date"] for s in resolve(GOOD, now)["free_visit_slots"]] == expected


def test_spoken_forms():
    out = resolve(GOOD, datetime(2026, 9, 30, 15, 0, tzinfo=IST))
    spoken = [s["spoken"] for s in out["free_visit_slots"]]
    assert spoken == ["tomorrow, Thursday 1 October at 11 am",
                      "tomorrow, Thursday 1 October at 4 pm",
                      "Friday 2 October at 11 am"]
    assert out["lead_source"] == "Facebook lead form for Green Acres"


def test_now_in_another_timezone_is_converted():
    # 22:00 UTC on 30 Sep is already 1 Oct 03:30 in India: "tomorrow" is 2 Oct.
    out = resolve(GOOD, datetime(2026, 9, 30, 22, 0, tzinfo=ZoneInfo("UTC")))
    assert out["free_visit_slots"][0]["date"] == "2026-10-02"


def test_past_same_day_slot_dropped():
    data = {**GOOD, "free_visit_slots": [{"day_offset": 0, "time": "11:00"},
                                         {"day_offset": 0, "time": "18:00"}]}
    out = resolve(data, datetime(2026, 10, 5, 12, 0, tzinfo=IST))
    assert [s["time"] for s in out["free_visit_slots"]] == ["18:00"]
    assert out["free_visit_slots"][0]["spoken"].startswith("today, ")


def test_skipped_weekday_moves_to_next_day_and_dedupes():
    # From Saturday 3 Oct: +1 is Sunday → Monday; +2 is already Monday → same slot, kept once.
    data = {**GOOD, "free_visit_slots": [{"day_offset": 1, "time": "11:00"},
                                         {"day_offset": 2, "time": "11:00"}]}
    out = resolve(data, datetime(2026, 10, 3, 10, 0, tzinfo=IST), skip=("sunday",))
    assert [s["date"] for s in out["free_visit_slots"]] == ["2026-10-05"]


def test_year_shown_only_when_it_changes():
    out = resolve(GOOD, datetime(2026, 12, 31, 10, 0, tzinfo=IST))
    assert out["free_visit_slots"][0]["spoken"] == "tomorrow, Friday 1 January 2027 at 11 am"


def test_relative_day():
    out = resolve({**GOOD, "placed": -3}, datetime(2026, 10, 5, 10, 0, tzinfo=IST))
    assert out["placed"] == {"offset": -3, "date": "2026-10-02", "spoken": "Friday 2 October"}


def test_naive_now_refused():
    with pytest.raises(ValueError, match="timezone-aware"):
        resolve(GOOD, datetime(2026, 10, 5, 10, 0))


@pytest.mark.parametrize("hhmm, spoken", [("11:00", "11 am"), ("16:30", "4:30 pm"),
                                          ("12:00", "12 noon"), ("00:00", "12 midnight"),
                                          ("00:15", "12:15 am")])
def test_spoken_time(hhmm, spoken):
    h, m = map(int, hhmm.split(":"))
    from datetime import time
    assert spoken_time(time(h, m)) == spoken


def test_old_sample_marker_is_not_shown_or_spoken():
    from app.demo.agent import strip_sample
    from app.demo.industries.registry import load_seed_definitions
    from app.demo.page import sample_sections
    assert strip_sample("Rahul (sample)") == "Rahul" and strip_sample("Priya (Sample)") == "Priya"
    assert strip_sample("Rahul") == "Rahul" and strip_sample(None) is None
    re_ = next(d for d in load_seed_definitions() if d.key == "real_estate")
    summary, sections = sample_sections(re_, {"caller": {"name": "Rahul (sample)",
                                                         "phone_masked": "+91 98XXX XXX01"}})
    assert {"label": "You are", "value": "Rahul"} in summary
    assert sections[0]["items"][0] == {"label": "Name", "value": "Rahul"}
