"""Industry definitions (app/demo/industries/): every launch seed compiles, a made-up
industry compiles end to end (nothing is tied to the known industries), and broken
definitions are refused with every problem listed.

    cd core && python -m pytest tests/test_demo_industry_definition.py -q
"""
import copy
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.demo.industries.definition import DefinitionError, compile_definition  # noqa: E402
from app.demo.industries.registry import load_seed_definitions, seed_dir  # noqa: E402
from app.demo.sample.resolve import resolve_dataset  # noqa: E402
from app.demo.sample.validate import validate_dataset  # noqa: E402

# A made-up industry using different building blocks than real estate, to prove the
# pipeline has no knowledge of specific industries.
FAKE = {
    "key": "pet_grooming",
    "label": "Pet grooming",
    "version": 1,
    "use_case": {"key": "grooming_booking", "label": "Grooming appointment", "goal": "slot_booked"},
    "facts_fields": [
        {"key": "salon_name", "label": "Salon name", "required": True},
        {"key": "services", "label": "Services", "type": "list"},
    ],
    "minimum": [{"field": "services", "min_items": 2}],
    "sample_data": {
        "caller": "sample_caller",
        "pet": {"name": "string", "breed": "string"},
        "slots": {"type": "list", "min_items": 1, "item": "relative_slot"},
        "ref": {"type": "ref_code", "prefix": "GRM-"},
        "greeting": "text_template",
    },
    "sample_template": {
        "caller": {"name": "Asha", "phone_masked": "+91 99XXX XXX10"},
        "pet": {"name": "Bruno", "breed": "Beagle"},
        "slots": [{"day_offset": 1, "time": "10:30"}],
        "ref": "GRM-2041",
        "greeting": "Thanks for calling {{salon_name}}",
    },
    "skip_weekdays": ["sunday"],
    "agent": {"persona": "Front desk for {{salon_name}}", "languages": ["en-IN"],
              "greeting": "Hi, {{salon_name}} here.", "instructions": "Book grooming slots."},
    "tools": [{"block": "list_slots", "name": "free_slots", "collection": "slots",
               "description": "List free slots."},
              {"block": "book_slot", "name": "book", "collection": "slots", "ref_field": "ref",
               "description": "Book a slot."}],
    "prompts": ["Kal grooming ho sakta hai?", "How much for a bath?", "Book Saturday"],
}


def broken(**changes):
    d = copy.deepcopy(FAKE)
    for path, value in changes.items():
        node = d
        *parents, leaf = path.split("__")
        for p in parents:
            node = node[int(p)] if p.isdigit() else node[p]
        node[int(leaf) if leaf.isdigit() else leaf] = value
    return d


def problems(raw) -> list[str]:
    with pytest.raises(DefinitionError) as exc:
        compile_definition(raw)
    return exc.value.errors


def test_every_seed_compiles_and_matches_its_file_name():
    defs = load_seed_definitions()
    assert {d.key for d in defs} >= {"real_estate"}
    for d in defs:
        assert (seed_dir() / f"{d.key}.yaml").exists()


def test_real_estate_seed_is_the_pdf_scenario():
    re_ = next(d for d in load_seed_definitions() if d.key == "real_estate")
    t = re_.sample_template
    assert t["caller"]["name"] == "Rahul"
    assert t["booking_reference"] == "VIS-1187"
    assert [(s["day_offset"], s["time"]) for s in t["free_visit_slots"]] == [
        (1, "11:00"), (1, "16:00"), (2, "11:00")]
    required = {f["key"] for f in re_.facts_fields if f.get("required")}
    assert required == {"project_name", "location", "configurations", "sales_office"}
    assert re_.minimum == [{"field": "configurations", "min_items": 2,
                            "message": "at least 2 BHK types"}]


def test_made_up_industry_compiles_and_resolves():
    d = compile_definition(FAKE)
    assert d.fact_keys == {"salon_name", "services"}
    facts = {"salon_name": "Happy Paws", "services": "Bath, Haircut"}
    assert validate_dataset(d.sample_template, d.sample_schema,
                            allowed_placeholders=d.fact_keys, facts=facts) == []
    # Saturday → +1 is Sunday, which this industry skips → Monday.
    ctx = resolve_dataset(d.sample_template, d.sample_schema, facts,
                          now=datetime(2026, 10, 3, 9, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
                          tz="Asia/Kolkata", skip_weekdays=d.skip_weekdays)
    assert ctx.data["slots"][0]["date"] == "2026-10-05"
    assert ctx.data["greeting"] == "Thanks for calling Happy Paws"


def test_hash_changes_only_with_content():
    a = compile_definition(FAKE)
    b = compile_definition(copy.deepcopy(FAKE))
    c = compile_definition(broken(label="Pet grooming & spa"))
    assert a.content_hash == b.content_hash != c.content_hash


def test_shape_errors_are_all_reported():
    errs = problems(broken(key="Pet Grooming", prompts=["only one"]))
    assert any(e.startswith("key:") for e in errs)
    assert any(e.startswith("prompts:") for e in errs)


def test_unknown_tool_block_and_goal():
    errs = problems(broken(tools__0__block="send_payment_link"))
    assert any("tools[0].block" in e or "tools.0.block" in e for e in errs)


def test_tool_must_point_at_a_slot_list():
    errs = problems(broken(tools__0__collection="pet"))
    assert any("must be a list of relative_slot" in e for e in errs)


def test_goal_needs_matching_tool():
    d = broken()
    d["tools"] = [d["tools"][0]]                      # list_slots only, no book_slot
    assert any("needs a `book_slot` tool" in e for e in problems(d))


def test_unknown_placeholders_anywhere():
    errs = problems(broken(agent__greeting="Hi from {{salon}}"))
    assert "agent: unknown placeholder {{salon}}" in errs


def test_template_must_pass_its_own_schema():
    errs = problems(broken(sample_template__ref="BOOK-1"))
    assert any(e.startswith("sample_template.ref") for e in errs)


def test_minimum_must_name_a_list_fact():
    errs = problems(broken(minimum=[{"field": "salon_name", "min_items": 2}]))
    assert any("min_items needs a list or unit_types fact" in e for e in errs)
    errs = problems(broken(minimum=[{"field": "nope"}]))
    assert any("unknown fact 'nope'" in e for e in errs)


def test_duplicate_fact_keys():
    d = broken()
    d["facts_fields"].append({"key": "salon_name", "label": "Again"})
    assert any("duplicate key" in e for e in problems(d))


def test_seed_file_name_must_match_key(tmp_path):
    (tmp_path / "wrong_name.yaml").write_text(yaml.safe_dump(FAKE), encoding="utf-8")
    with pytest.raises(ValueError, match="must match its key"):
        load_seed_definitions(tmp_path)
