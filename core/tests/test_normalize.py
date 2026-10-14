"""Unit tests for extraction normalisation (normalize.py) and the chunker's extraction hardening.

What must hold (pure — no DB, no network):
  * running headers/footers repeated across pages are removed; page-number lines only when 2+;
    without page breaks only repeated lines that carry a page marker are removed (a repeated
    ordinary sub-heading is content);
  * hyphenated line wraps are joined; hard-wrapped lines are unwrapped ONLY with extraction
    evidence in "auto" mode (hand-wrapped markdown is untouched), always in "always" mode;
  * a space-aligned table under a digit-free header becomes a pipe table; ragged/short runs don't;
  * hygiene: NBSP/zero-width/soft hyphen/BOM handled, ZWJ/ZWNJ (Devanagari) preserved;
  * "off" returns the input unchanged; normalisation is idempotent; unknown mode raises;
  * chunker: commas inside numbers never split a list; the danda ends a sentence; unpunctuated
    runs are hard-split at word boundaries; page furniture and layout rows are never headings.

Run: cd core && python -m pytest tests/test_normalize.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from app.engine.rag import chunkers  # noqa: E402
from app.engine.rag.chunkers import (Chunk, chunk_text, hard_split, split_children,  # noqa: E402
                                     uncovered_text)
from app.engine.rag.normalize import normalize_extracted_text  # noqa: E402

HEADER = "Sunrise Meadows Brochure            Page {}"


def _paged(*bodies: str) -> str:
    return "\f".join(HEADER.format(i + 1) + "\n\n" + b for i, b in enumerate(bodies))


# ─────────────────────────────────────────────────────────────── boilerplate

def test_running_header_removed_across_pages():
    n = normalize_extracted_text(_paged("Alpha facts here.", "Beta facts here.",
                                        "Gamma facts here."))
    assert "Brochure" not in n.text and "Alpha facts here." in n.text
    assert n.report.boilerplate_lines_removed == 3 and n.report.pages == 3
    assert n.report.detected_extracted


def test_header_on_two_pages_only_is_kept():
    n = normalize_extracted_text(_paged("Alpha.", "Beta."))
    assert "Brochure" in n.text and n.report.boilerplate_lines_removed == 0


def test_page_number_lines_need_two():
    one = normalize_extracted_text("The tower count is below.\n\n12\n\nSix towers.")
    assert "12" in one.text and one.report.page_number_lines_removed == 0
    two = normalize_extracted_text("Intro text.\n\n1\n\nMore text.\n\nPage 2 of 9\n\nEnd.")
    assert "Page 2" not in two.text and two.report.page_number_lines_removed == 2


def test_unpaged_repeated_subheading_is_content():
    doc = ("# Towers\n\nPrice\nTower A starts at 1 crore.\n\nPrice\nTower B starts at 2 crore."
           "\n\nPrice\nTower C starts at 3 crore.")
    n = normalize_extracted_text(doc)
    assert n.text.count("Price") == 3 and n.report.boilerplate_lines_removed == 0


# ─────────────────────────────────────────────────────────────── wrapping

def test_hyphenated_wrap_joined():
    n = normalize_extracted_text("The develop-\ner has delivered 14 projects.")
    assert "developer has delivered" in n.text and n.report.hyphenations_joined == 1


def test_hand_wrapped_markdown_is_not_unwrapped_in_auto():
    md = ("# Notes\n\n1. The brochure gives the 3 BHK range as 1,239 to 1,422 square feet and\n"
          "the cost sheet prices a larger unit.\n\n2. Second point.")
    n = normalize_extracted_text(md)
    assert n.text == md.strip() and not n.report.detected_extracted
    assert n.report.lines_unwrapped == 0


def test_unwrap_with_extraction_evidence_or_always():
    body = "Sunrise Meadows is spread across twelve\nacres with six towers."
    auto = normalize_extracted_text(_paged(body, "B.", "C."))
    assert "twelve acres" in auto.text and auto.report.lines_unwrapped >= 1
    forced = normalize_extracted_text(body, mode="always")
    assert "twelve acres" in forced.text


def test_unwrap_never_joins_structure():
    txt = "- first item that wraps\nonto a second line\n- second item\n| a | b |\nlower row"
    n = normalize_extracted_text(txt, mode="always")
    assert "- first item that wraps onto a second line" in n.text
    assert "| a | b |\nlower row" in n.text, "never joined onto a table line"


# ─────────────────────────────────────────────────────────────── tables

TABLE = ("PRICE LIST\n"
         "Type        Carpet Area    Super Area    Price (Rs)\n"
         "2 BHK A     690 sq ft      1,050 sq ft   89,50,000\n"
         "2 BHK B     720 sq ft      1,110 sq ft   94,20,000\n"
         "3 BHK A     980 sq ft      1,480 sq ft   1,26,00,000\n")


def test_layout_table_becomes_pipe_table_and_chunks_as_table():
    n = normalize_extracted_text(TABLE)
    assert n.report.tables_converted == 1
    assert "| Type | Carpet Area | Super Area | Price (Rs) |" in n.text
    assert "| 2 BHK A | 690 sq ft | 1,050 sq ft | 89,50,000 |" in n.text
    parents = chunk_text(n.text, target_chars=900, overlap_chars=150, list_group_size=8)
    tables = [p for p in parents if p.metadata["kind"] == "table"]
    assert tables and tables[0].heading == "PRICE LIST"
    kids = split_children(tables[0], target_chars=100, min_chars=10)
    assert kids and all(k.content.startswith("| Type |") for k in kids), "header in every child"


def test_ragged_or_headerless_runs_are_not_tables():
    ragged = "Type    Area    Price\n2 BHK   690\n3 BHK   980    1,26,00,000\n"
    assert normalize_extracted_text(ragged).report.tables_converted == 0
    headerless = ("2 BHK A     690 sq ft      89,50,000\n2 BHK B     720 sq ft      94,20,000\n"
                  "3 BHK A     980 sq ft      1,26,00,000\n")
    assert normalize_extracted_text(headerless).report.tables_converted == 0


# ─────────────────────────────────────────────────────────────── hygiene / modes

def test_hygiene_and_devanagari_joiners():
    raw = "Club house​ with pool\r\nsoft­hyphen ﻿end. क्‍ष"
    n = normalize_extracted_text(raw)
    assert "Club house with pool" in n.text and "softhyphen" in n.text
    assert "\r" not in n.text and "﻿" not in n.text
    assert "‍" in n.text, "ZWJ is meaningful in Devanagari and must be kept"


def test_off_mode_and_idempotence_and_bad_mode():
    raw = _paged("Alpha facts.", "Beta facts.", "Gamma facts.")
    assert normalize_extracted_text(raw, mode="off").text == raw
    once = normalize_extracted_text(raw).text
    assert normalize_extracted_text(once).text == once
    with pytest.raises(ValueError):
        normalize_extracted_text("x", mode="sometimes")


# ─────────────────────────────────────────────────────────────── chunker hardening

def test_list_split_never_breaks_a_number():
    para = ("Tower A has 1,422 and 1,390 sq ft, clubhouse 3,25,000, parking 5,50,000, "
            "gym, pool, park")
    assert chunkers.ITEM_SEPARATOR.split("89,50,000, gym") == ["89,50,000", " gym"]
    chunks = chunk_text(para, target_chars=900)
    joined = " ".join(c.content for c in chunks)
    assert "1,422" in joined and "3,25,000" in joined and "5,50,000" in joined
    assert "1, 422" not in joined and "3, 25, 000" not in joined


def test_danda_ends_a_sentence():
    hindi = ("सनराइज़ मेडोज़ ओल्ड मद्रास रोड पर स्थित है। परियोजना में 1,100 घर हैं। "
             "2 बीएचके की कीमत 89 लाख से शुरू होती है। 3 बीएचके की कीमत 1.26 करोड़ से शुरू "
             "होती है। कब्ज़ा मार्च 2029 में होगा।")
    parent = Chunk(0, hindi, None, {"kind": "prose"})
    kids = split_children(parent, target_chars=100, min_chars=20, split_min_chars=100)
    assert len(kids) >= 2 and uncovered_text(parent, kids) == []
    assert all(k.content.endswith("।") for k in kids)


def test_hard_split_caps_unpunctuated_text_at_word_boundaries():
    run_on = " ".join(f"20 percent on slab {i}" for i in range(80))      # ~1,700 chars, no stop
    parents = chunk_text(run_on, target_chars=300, overlap_chars=0)
    assert len(parents) >= 5 and max(len(p.content) for p in parents) <= 300
    words = run_on.split()
    assert " ".join(" ".join(p.content for p in parents).split()) == " ".join(words)
    assert hard_split("a " * 3 + "https://example.com/" + "x" * 50, 20)[-1].startswith("https")


def test_page_furniture_and_layout_rows_are_not_headings():
    assert chunkers._classify_heading("Sunrise Meadows Brochure   Page 2") is None
    assert chunkers._classify_heading("3 of 12") is None
    assert chunkers._classify_heading("Type        Carpet Area    Price") is None
    assert chunkers._classify_heading("| Type | Area |") is None
    assert chunkers._classify_heading("Where it is") == (0, "Where it is")
