"""Per-tenant STT keyword boosting (app/engine/voice/keyterms.py): the pure cleanup/dedup
logic, which needs no DB. See the module docstring for why this exists and the real
Deepgram test (synthesised "DSR Highland Greenz"/"Chikkanayakanahalli") that motivated it.

    cd core && python -m pytest tests/test_keyterms.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.voice import keyterms as kt  # noqa: E402


def test_location_name_splits_on_dash():
    assert kt.split_location_name("DSR Highland Greenz — Site Office") == [
        "DSR Highland Greenz", "Site Office"]
    assert kt.split_location_name("DSR Techno Cube — Sales Gallery") == [
        "DSR Techno Cube", "Sales Gallery"]


def test_location_name_without_dash_is_kept_whole():
    assert kt.split_location_name("Downtown Clinic") == ["Downtown Clinic"]


def test_location_name_empty_is_empty():
    assert kt.split_location_name(None) == []
    assert kt.split_location_name("") == []
    assert kt.split_location_name("   ") == []


def test_dedupe_is_case_insensitive():
    assert kt.dedupe_and_cap(["DSR Infra Projects", "dsr infra projects", "Lemon Tree"]) == [
        "DSR Infra Projects", "Lemon Tree"]


def test_short_generic_terms_are_dropped():
    # None of these carry a word of MIN_WORD_LEN or more -- Deepgram already recognises
    # them reliably unaided, so boosting would burn a slot for no benefit (see the real
    # proof: the win was on "Chikkanayakanahalli"/"Greenz", not on short common words).
    assert kt.dedupe_and_cap(["AB", "ok", "3 BHK", ""]) == []


def test_real_term_kept_whole_not_reduced():
    assert kt.dedupe_and_cap(["DSR Infra Projects"]) == ["DSR Infra Projects"]


def test_cap_holds_and_earlier_entries_win():
    many = [f"Project{i:03d}word" for i in range(kt.MAX_KEYTERMS + 10)]
    capped = kt.dedupe_and_cap(many)
    assert len(capped) == kt.MAX_KEYTERMS
    assert capped[0] == "Project000word"


def test_demo_project_facts_supply_spoken_names():
    # A GTM demo tenant has no `projects` rows -- its project name lives only here.
    facts = [
        {"label": "Project name", "value": "Ajmera Manhattan — Buildings 3A & 3B"},
        {"label": "Developer name", "value": "Ajmera Realty & Infra"},
        {"label": "Location", "value": "Wadala East"},
        {"label": "Landmarks", "value": "Bhakti Park / Wadala"},
        {"label": "RERA number", "value": "not published; the sales team will confirm"},
        {"label": "Price", "value": "not published; the sales team will confirm"},
        {"label": "Location", "value": "Not published yet"},
        "malformed", {"label": "Project name"},
    ]
    assert kt.fact_keyterms(facts) == [
        "Ajmera Manhattan", "Buildings 3A & 3B", "Ajmera Realty & Infra", "Wadala East",
        "Bhakti Park", "Wadala"]
    assert kt.fact_keyterms(None) == []


def test_pack_terms_bypass_length_filter_and_dedupe():
    # "BHK"/"RERA" are short, but live calls mis-heard exactly these -- they must survive.
    out = kt.with_pack_terms(["Wadala East", "Site Visit Lounge"],
                             ["RERA", "BHK", "rera", "", "possession", "wadala east"])
    assert out == ["Wadala East", "Site Visit Lounge", "RERA", "BHK", "possession"]


def test_pack_terms_have_their_own_cap():
    proper = [f"Project{i:03d}word" for i in range(kt.MAX_KEYTERMS)]
    out = kt.with_pack_terms(proper, [f"term{i}" for i in range(kt.MAX_PACK_KEYTERMS + 5)])
    assert len(out) == kt.MAX_KEYTERMS + kt.MAX_PACK_KEYTERMS <= 100  # Deepgram's limit
    assert kt.with_pack_terms(proper, None) == proper


def test_real_estate_pack_boosts_the_misheard_domain_words():
    from app.packs.real_estate_india.pack import RealEstateIndiaPack
    terms = RealEstateIndiaPack().vocabulary.stt_keyterms
    for word in ("RERA", "BHK", "possession", "site visit"):
        assert word in terms
    assert len(terms) <= kt.MAX_PACK_KEYTERMS


def test_selftest_still_passes():
    """The module's own self-test (python -m app.engine.voice.keyterms) covers the same
    ground; running it here means a regression fails pytest too, not just the manual
    invocation."""
    kt._selftest()  # raises AssertionError on failure
