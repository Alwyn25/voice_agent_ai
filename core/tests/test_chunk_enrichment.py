"""Unit tests for R5.08 — chunk metadata enrichment and the retrieval lifts built on it.

Pure tests: no DB, no embeddings, no network. The extraction cases are real sentences from
the seeded DSR / Lemon Tree documents (scripts/seed_knowledge_*.py), because the formats that
matter are the ones tenants actually write — "Flat cost 96,25,000. Total 1,03,93,750." —
not the "₹X lakh" the task note imagined.

What must hold, and is tested below:
  * every real price / rate / area / BHK format in the seed corpus is extracted, whole;
  * the numbers that are NOT prices (phones, dates, floors, unit codes, carpet splits) are not;
  * the query analyser reads English and Hinglish price / size / BHK questions correctly,
    and does not call a size, distance or availability question a price question;
  * the retrieval lift is bounded, never double-counts with PRICING_BOOST, is exact on BHK
    (2.5 != 2 != 3), and leaves untagged chunks untouched;
  * enrichment failure never fails ingestion.

Run: cd core && python -m pytest tests/test_chunk_enrichment.py -q
"""
import json
import sys
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

# core/ on the path so `import app...` resolves the same way the app does at runtime.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.rag import ingestion  # noqa: E402
from app.engine.rag.chunkers import chunk_text  # noqa: E402
from app.engine.rag.enrichment import (FACTS_VERSION, QueryIntent, analyze_query,  # noqa: E402
                                       enrich_chunk_meta, extract_facts, facts_of)
from app.engine.rag.retrieval import (META_BOOST_CAP, apply_metadata_boosts,  # noqa: E402
                                      metadata_filter_flag)


# ------------------------------------------------------------------ extraction: real formats

def test_cost_sheet_row_bare_indian_numbers():
    f = extract_facts("875 sq ft, East and West facing. Carpet 592. Flat cost 96,25,000. "
                      "Total 1,03,93,750.")
    assert f.areas == {875, 592}
    assert f.prices == {9_625_000, 10_393_750}
    assert not f.rates and not f.bhk


def test_three_bhk_row_ignores_unlabelled_small_numbers():
    # "Balcony and utility 99" and "External walls 61" are area splits with no unit — they
    # must not become prices (no money label) nor areas (below the sanity floor).
    f = extract_facts("1,239 sq ft, West facing. Carpet 808. Balcony and utility 99. "
                      "External walls 61. Flat cost 1,42,47,261. Infrastructure 3,09,750. "
                      "Total 1,54,32,011.")
    assert f.areas == {1239, 808}
    assert f.prices == {14_247_261, 309_750, 15_432_011}


def test_clubhouse_charges_both_figures_including_one_without_its_own_label():
    f = extract_facts("Clubhouse charges are 2,50,000 for a 2 BHK and 3,25,000 for a 2.5 BHK. "
                      "Car parking is 3,00,000 for all of these units.")
    assert f.prices == {250_000, 325_000, 300_000}
    assert f.bhk == {2.0, 2.5}          # "2.5" is a configuration, never ₹2.5


@pytest.mark.parametrize("text, rate", [
    ("Rate 11,000 rupees per square foot on super built-up area.", 11_000),
    ("3 BHK Cost Sheet — 11,499 per square foot", 11_499),
    ("priced at ₹11,499/sq ft", 11_499),
    ("Corner flats, 50 rupees per square foot", 50),
    ("Rs. 75 per sft for premium units", 75),
    ("Common area maintenance — 3.50 rupees per square foot per month", 3.5),
    ("infrastructure charge of 250 per sq. ft", 250),
])
def test_rates_per_square_foot(text, rate):
    f = extract_facts(text)
    assert f.rates == {rate}
    assert not f.areas, "a 'per square foot' unit must never be read as an area"
    assert f.has_price


@pytest.mark.parametrize("text, amount", [
    ("covered parking is 5.5 lakh", 550_000),
    ("starts at 1.54 crore all in", 15_400_000),
    ("budget of ₹60L", 6_000_000),
    ("around 60L", 6_000_000),
    ("Rs 1.2 Cr", 12_000_000),
    ("INR 45,00,000", 4_500_000),
    ("Legal fees — 30,000 rupees, excluding GST.", 30_000),
    ("Rs.3,25,000", 325_000),
    ("booking amount 1,000,000", 1_000_000),       # Western grouping, labelled
    ("80 लाख", 8_000_000),
])
def test_money_formats(text, amount):
    assert extract_facts(text).prices == {amount}


def test_area_lists_share_one_unit():
    f = extract_facts("1 BHK of 619, 625, 643 and 646 sq ft, and 1.5 BHK of 780 sq ft.")
    assert f.areas == {619, 625, 643, 646, 780}
    assert f.bhk == {1.0, 1.5}


def test_grouped_area_list_is_read_whole_not_as_fragments():
    # The regression that exposed the chunker bug: never "422" / "390".
    f = extract_facts("3 BHK of 1,422 and 1,390 sq ft")
    assert f.areas == {1422, 1390}


def test_number_is_never_truncated_to_satisfy_a_lookahead():
    f = extract_facts("charges 1,325 sq ft")         # label + area: must stay an area
    assert f.areas == {1325} and not f.prices


@pytest.mark.parametrize("text, bhk", [
    ("2.5 BHK, 1,325 sq ft", {2.5}),
    ("price of 3bhk", {3.0}),
    ("a 2-bhk flat", {2.0}),
    ("1, 2 and 3 BHK", {1.0, 2.0, 3.0}),
    ("2/3 BHK homes", {2.0, 3.0}),
    ("There is no 4 BHK", {4.0}),
    ("a three bedroom apartment", {3.0}),
    ("Is the rate the same for 2 and 3 bedroom?", {2.0, 3.0}),     # eval EN006
    ("ek BHK hai kya aapke paas", {1.0}),                          # eval HI025
    ("do bhk ka price", {2.0}),
    ("what do you have in 3 BHK", {3.0}),                           # verb "do" is not 2 BHK
])
def test_bhk_forms(text, bhk):
    assert extract_facts(text).bhk == bhk


@pytest.mark.parametrize("text", [
    "Home loan assistance — Sastry, 8792582433.",
    "Phone: +91 8884441863 and +91 8884441836.",
    "Ran 1 June 2026 to 31 July 2026 and has ended.",
    "Tower A wing 1 sr3, wing 2 sr6; Tower B wing 1 sr2, wing 2 sr2 and sr9.",
    "Floor rise applies from the 4th floor onwards.",
    "Basement plus ground plus 12 floors.",
    "RGA Tech Park 2.2 km away.",
    "External walls 6 inch solid blocks, internal walls 4 inch.",
    "GST at 5% applies.",
    "Clubhouse charges for a 2 BHK are listed separately.",   # "2" after a money label
])
def test_non_price_numbers_are_ignored(text):
    f = extract_facts(text)
    assert not f.prices and not f.rates and not f.areas, (text, f)


def test_to_meta_shape_and_version():
    meta = enrich_chunk_meta("2.5 BHK, 1,325 sq ft. Flat cost 1,45,75,000. "
                             "Rate 11,000 rupees per square foot.")
    assert meta["v"] == FACTS_VERSION
    assert meta["has_price"] and meta["has_rate"] and meta["has_area"]
    assert meta["bhk"] == [2.5]
    assert meta["prices"] == [14_575_000] and meta["rates"] == [11_000]
    assert meta["area_min"] == meta["area_max"] == 1325
    json.dumps(meta)                     # must be JSONB-serialisable as-is


def test_value_lists_are_capped_but_min_max_use_everything():
    text = " ".join(f"Total {i},00,000." for i in range(10, 60))    # 50 distinct prices
    meta = enrich_chunk_meta(text)
    assert len(meta["prices"]) <= 24
    assert meta["price_min"] == 1_000_000 and meta["price_max"] == 5_900_000


def test_empty_text():
    meta = enrich_chunk_meta("")
    assert meta["has_price"] is False and meta["bhk"] == []


def test_every_seed_pricing_chunk_is_tagged_as_priced():
    """Guard against a format regression on the real corpus: every chunk of the 2 BHK /
    2.5 BHK / 3 BHK cost sheets and the extra-charges sheet must come out priced."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import seed_knowledge_dsr as dsr
    sheets = [d for d in dsr.DOCUMENTS
              if d["title"].startswith(("2 BHK and 2.5 BHK Cost Sheet", "3 BHK Cost Sheet"))]
    assert len(sheets) == 2
    for d in sheets:
        for c in chunk_text(d["content"]):
            assert enrich_chunk_meta(c.embed_text)["has_price"], (d["title"], c.content[:80])


# ------------------------------------------------------------------ chunker regression

def test_chunker_keeps_grouped_numbers_intact_in_inline_lists():
    chunks = chunk_text("Tower A, wings 1 and 2, 3 BHK of 1,422 and 1,390 sq ft, 2 BHK of "
                        "1,099, 1,103 and 1,104, lift lobby, fire exit.", min_chars=5)
    joined = " ".join(c.content for c in chunks)
    for n in ("1,422", "1,390", "1,099", "1,103", "1,104"):
        assert n in joined
    assert "1, 422" not in joined


# ------------------------------------------------------------------ query analysis

@pytest.mark.parametrize("q", [
    "what is the price of a 3 BHK",
    "how much does the 2.5 bhk cost",
    "clubhouse ka charge kitna hai",
    "corner aur premium mein charge ka farak kya hai",
    "1239 wale flat ka total kitna padega sab milakar",
    "3 BHK ki keemat batao",
    "is it within 1.2 crore",
    "anything under 60L?",
    "what is the rate per sq ft",
    "is the corpus fund refundable",
    "Ballpark for a 2BHK?",                                  # eval EN004
    "Is floor rise charged on the 9th floor?",               # eval EN012
    "1390 square feet ka kitna banta hai",                   # eval HI008
    "registration ka kharcha kitna aayega",                  # eval HI059
])
def test_price_questions(q):
    assert analyze_query(q).wants_price, q


@pytest.mark.parametrize("q", [
    "is there a dog park",
    "OC mil gaya hai kya",
    "flat kitna bada hai",
    "kitne floors hai tower mein",
    "airport kitni door hai",
    "what amenities do you have",
    "is the EV charging point available",
    # Eval regression EN027: a COUNT question, not a price question.
    "How many flats in total?",
    "kitne units hain total",
    "number of towers in the project",
    # A/B Q38/Q39: RERA registration is compliance, not a fee.
    "What is the RERA registration number for this project?",
    "Where can I verify the RERA registration?",
])
def test_non_price_questions(q):
    assert not analyze_query(q).wants_price, q


def test_count_question_with_a_money_word_is_still_price():
    assert analyze_query("how many flats are under 1.2 crore").wants_price


def test_per_square_foot_is_a_rate_not_an_area_question():
    # Eval HI010: "per square feet" must not lift area chunks.
    i = analyze_query("saare charges milakar per square feet kitna padta hai")
    assert i.wants_price and i.wants_rate and not i.wants_area
    assert analyze_query("what is the rate per sq ft").wants_area is False
    assert analyze_query("1,239 sq ft flat ka total kitna").wants_area       # a real area


def test_common_area_maintenance_is_a_charge_not_a_size_question():
    # A/B regression: "common area" lifted area chunks and halved precision on this question.
    i = analyze_query("What are the common area maintenance (CAM) charges?")
    assert i.wants_price and not i.wants_area
    assert not analyze_query("is there a kids play area").wants_area
    assert analyze_query("what is the carpet area of the 2 BHK").wants_area
    assert analyze_query("What's the total area of the project?").wants_area


def test_rate_area_bhk_and_budget_intent():
    i = analyze_query("rate per sq ft for 2.5 bhk")
    assert i.wants_rate and i.wants_price and i.bhk == {2.5}
    assert analyze_query("carpet area of the 3 BHK").wants_area
    assert analyze_query("flat kitna bada hai").wants_area
    assert analyze_query("2 bhk under 1.2 crore").budget == [12_000_000]
    assert not analyze_query("").any


# ------------------------------------------------------------------ facts_of

def test_facts_of_tolerates_driver_shapes():
    facts = {"v": 1, "has_price": True}
    assert facts_of({"heading": "x", "facts": facts}) == facts
    assert facts_of(json.dumps({"facts": facts})) == facts       # JSONB delivered as str
    assert facts_of({"heading": "x"}) is None                    # untagged chunk
    assert facts_of(None) is None and facts_of("not json") is None
    assert facts_of({"facts": "garbage"}) is None


# ------------------------------------------------------------------ retrieval lifts

def _row(doc_type="brochure", **facts):
    base = {"v": 1, "has_price": False, "has_rate": False, "has_area": False, "bhk": []}
    base.update(facts)
    return {"doc_type": doc_type, "metadata": {"heading": "h", "facts": base}}


_BOOSTS = dict(price_boost=1.3, area_boost=1.2, bhk_boost=1.2)


def test_price_question_lifts_priced_chunks_only():
    rows = {"priced": _row(has_price=True), "prose": _row()}
    scores = {"priced": 1.0, "prose": 1.0}
    stats = apply_metadata_boosts(scores, rows, QueryIntent(wants_price=True),
                                  doc_price_boosted=set(), **_BOOSTS)
    assert scores == {"priced": 1.3, "prose": 1.0}
    assert stats["price"] == 1 and stats["lifts"] == {"priced": ["price"]}


def test_no_double_count_with_document_pricing_boost():
    rows = {"sheet": _row(doc_type="pricing", has_price=True)}
    scores = {"sheet": 1.6}                      # PRICING_BOOST already applied
    apply_metadata_boosts(scores, rows, QueryIntent(wants_price=True),
                          doc_price_boosted={"sheet"}, **_BOOSTS)
    assert scores["sheet"] == 1.6


def test_bhk_match_is_exact():
    rows = {"b25": _row(bhk=[2.5]), "b2": _row(bhk=[2]), "b3": _row(bhk=[3]),
            "both": _row(bhk=[2, 2.5])}
    scores = {k: 1.0 for k in rows}
    apply_metadata_boosts(scores, rows, QueryIntent(bhk={2.5}),
                          doc_price_boosted=set(), **_BOOSTS)
    assert scores == {"b25": 1.2, "b2": 1.0, "b3": 1.0, "both": 1.2}


def test_combined_lift_is_capped():
    rows = {"row": _row(has_price=True, has_area=True, bhk=[2.5])}
    scores = {"row": 1.0}
    apply_metadata_boosts(scores, rows, QueryIntent(wants_price=True, wants_area=True,
                                                    bhk={2.5}),
                          doc_price_boosted=set(), price_boost=1.5, area_boost=1.5,
                          bhk_boost=1.5)
    assert scores["row"] == pytest.approx(META_BOOST_CAP)


def test_untagged_chunks_are_neutral_and_counted():
    rows = {"old": {"doc_type": "brochure", "metadata": {"heading": "h"}},
            "new": _row(has_price=True)}
    scores = {"old": 1.0, "new": 1.0}
    stats = apply_metadata_boosts(scores, rows, QueryIntent(wants_price=True),
                                  doc_price_boosted=set(), **_BOOSTS)
    assert scores["old"] == 1.0
    assert stats["untagged"] == 1 and stats["tagged"] == 1


def test_rows_without_a_score_are_skipped():
    rows = {"a": _row(has_price=True)}
    scores: dict = {}
    apply_metadata_boosts(scores, rows, QueryIntent(wants_price=True),
                          doc_price_boosted=set(), **_BOOSTS)
    assert scores == {}


def test_filter_flag():
    price = QueryIntent(wants_price=True, wants_area=True)
    assert metadata_filter_flag("filter", price) == "has_price"      # price wins over area
    assert metadata_filter_flag("filter", QueryIntent(wants_area=True)) == "has_area"
    assert metadata_filter_flag("filter", QueryIntent(bhk={3.0})) is None   # bhk never filters
    assert metadata_filter_flag("boost", price) is None
    assert metadata_filter_flag("off", price) is None


# ------------------------------------------------------------------ ingestion safety

def test_enrichment_failure_never_fails_ingestion():
    with patch.object(ingestion, "enrich_chunk_meta", side_effect=RuntimeError("boom")):
        assert ingestion._chunk_facts("Total 1,00,000", uuid.uuid4(), 0) is None
    assert ingestion._chunk_facts("Total 1,00,000", uuid.uuid4(), 0)["has_price"]
