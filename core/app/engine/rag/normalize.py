"""Normalise EXTRACTED document text (PDF / DOCX / OCR output) before chunking.

WHY. The chunker (chunkers.py) reads structure from clean, markdown-like text: blank-line
paragraphs, headings, list markers, pipe tables. Text pulled out of a PDF does not look like that,
and measured on a realistic extraction the damage was concrete:

  * a space-aligned price table was read as a comma list, and the commas INSIDE the Indian-grouped
    numbers were split — "89,50,000" became "89, 50, 000" in the stored chunk (the number-safe
    split in chunkers.py fixes the corruption; converting the table here gives it the table path,
    so every row keeps its column names);
  * the page header "Sunrise Meadows Brochure   Page 2" was taken for a section heading and
    prefixed onto every chunk of that page;
  * "develop-\\ner" (a hyphenated line wrap) became "develop- er".

WHAT IT DOES, in order (each step counted in NormalizationReport, logged at ingestion):
  1. hygiene        CRLF -> LF, form feed -> page break, NBSP / narrow spaces -> space, strip
                    zero-width space / BOM / soft hyphen, tabs -> column gap, trailing spaces.
                    (ZWJ / ZWNJ are kept: they are meaningful in Devanagari.)
  2. boilerplate    drop standalone page-number lines ("12", "Page 3 of 9") and repeated running
                    headers/footers (the same line, digits ignored, on 3+ pages/places).
  3. de-hyphenate   "develop-\\ner" -> "developer" (only when the next line continues in lowercase).
  4. unwrap         join a hard-wrapped line with the next when the next continues in lowercase and
                    the current does not end a sentence — so a wrapped bullet stays one item and a
                    short wrapped fragment is never mistaken for a heading.
  5. tables         a run of 3+ lines that all split into the same number (3+) of cells on 2+
                    spaces, under a header line without digits, becomes a pipe table.
  6. tidy           collapse 3+ blank lines.

SAFE ON CLEAN INPUT. Steps 2, 3 and 5 key on shapes clean markdown does not have (form feeds,
repeated page furniture, "-" + newline + lowercase, 2+-space column gaps). Step 4 (unwrap) WOULD
fire on hand-wrapped markdown, so in the default "auto" mode it runs only when one of the others
found extraction evidence. Verified against every seed document (43 documents): identical chunks,
apart from the separately-fixed number-comma corruption. Domain-neutral: layout only, no
vocabulary.
Pure (stdlib ``re`` only), unit-tested in tests/test_normalize.py.
"""
import re
from dataclasses import dataclass, field

# Separator used in pipe tables this module writes (matches what chunkers._is_table_line reads).
_PIPE = " | "

# 1. hygiene
_SPACE_LIKE = re.compile(r"[    - ]")
_INVISIBLE = re.compile(r"[​﻿­]")        # NOT ‌/‍ (Devanagari)

# 2. boilerplate
_PAGE_NUMBER = re.compile(r"^\s*(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*$", re.I)
_SENTENCE_END = re.compile(r"[.!?।॥]\s*$")
_STRUCTURAL = re.compile(r"^\s*(?:#|[-*•]\s|\d{1,3}[.)]\s|\|)")

# 3/4. wrapping
_HYPHEN_WRAP = re.compile(r"([A-Za-z]{2,})-\n([a-z]{2,})")
_NO_CONTINUE_END = re.compile(r"[.!?:;।॥\"”')\]]\s*$")

# 5. tables
_COLUMN_GAP = re.compile(r"\s{2,}")


@dataclass
class NormalizationReport:
    """What normalisation changed — surfaced in the ingestion report and log line."""
    input_chars: int = 0
    output_chars: int = 0
    pages: int = 1
    page_number_lines_removed: int = 0
    boilerplate_lines_removed: int = 0
    boilerplate_samples: list[str] = field(default_factory=list)
    hyphenations_joined: int = 0
    lines_unwrapped: int = 0
    tables_converted: int = 0
    mode: str = "auto"
    # True when the text carried extraction signals (page breaks, page numbers, running headers,
    # hyphenated wraps, layout tables) — the condition for unwrapping in "auto" mode.
    detected_extracted: bool = False

    @property
    def changed(self) -> bool:
        return any((self.page_number_lines_removed, self.boilerplate_lines_removed,
                    self.hyphenations_joined, self.lines_unwrapped, self.tables_converted))

    def as_dict(self) -> dict:
        return {"changed": self.changed, "mode": self.mode,
                "detected_extracted": self.detected_extracted,
                "input_chars": self.input_chars,
                "output_chars": self.output_chars, "pages": self.pages,
                "page_number_lines_removed": self.page_number_lines_removed,
                "boilerplate_lines_removed": self.boilerplate_lines_removed,
                "boilerplate_samples": self.boilerplate_samples,
                "hyphenations_joined": self.hyphenations_joined,
                "lines_unwrapped": self.lines_unwrapped,
                "tables_converted": self.tables_converted}


@dataclass
class NormalizedText:
    text: str
    report: NormalizationReport


def _hygiene(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACE_LIKE.sub(" ", text)
    text = _INVISIBLE.sub("", text)
    text = text.replace("\t", "    ")                 # a tab separates columns: keep it a gap
    return "\n".join(ln.rstrip() for ln in text.split("\n"))


def _boilerplate_key(line: str) -> str:
    """A running header/footer differs only in its page number — compare with digits masked."""
    return re.sub(r"\d+", "#", _COLUMN_GAP.sub(" ", line.strip().lower()))


def _is_boilerplate_candidate(line: str) -> bool:
    s = line.strip()
    return (bool(s) and len(s) <= 100 and not _SENTENCE_END.search(s)
            and not _STRUCTURAL.match(s))


def _remove_boilerplate(pages: list[list[str]], report: NormalizationReport) -> list[list[str]]:
    # Page numbers: a line that is only a page number. Needs 2+ of them in the document, so a
    # lone "12" that happens to be content is never touched.
    page_number_lines = sum(1 for p in pages for ln in p if _PAGE_NUMBER.match(ln))
    if page_number_lines >= 2:
        pages = [[ln for ln in p if not _PAGE_NUMBER.match(ln)] for p in pages]
        report.page_number_lines_removed = page_number_lines

    # Running headers/footers: the same non-sentence line (digits masked) on 3+ pages. When the
    # extractor gave no page breaks, "3+ times in the document" alone is too loose — a clean
    # document may legitimately repeat a short sub-heading ("Price" under each section) — so
    # there a repeated line must ALSO carry a page marker (a number, or the word "page").
    paged = len(pages) > 1
    counts: dict[str, int] = {}
    for p in pages:
        keys = [_boilerplate_key(ln) for ln in p if _is_boilerplate_candidate(ln)]
        for key in (set(keys) if paged else keys):
            counts[key] = counts.get(key, 0) + 1
    repeated = {k for k, n in counts.items()
                if n >= 3 and k.strip("# ") and (paged or "#" in k or "page" in k)}
    if repeated:
        kept_pages = []
        for p in pages:
            kept = []
            for ln in p:
                if _is_boilerplate_candidate(ln) and _boilerplate_key(ln) in repeated:
                    report.boilerplate_lines_removed += 1
                    if len(report.boilerplate_samples) < 3 and ln.strip() not in \
                            report.boilerplate_samples:
                        report.boilerplate_samples.append(ln.strip())
                    continue
                kept.append(ln)
            kept_pages.append(kept)
        pages = kept_pages
    return pages


def _unwrap(lines: list[str], report: NormalizationReport) -> list[str]:
    out: list[str] = []
    for ln in lines:
        if (out and out[-1].strip() and ln.strip()
                and ln.lstrip()[:1].islower()
                and not _NO_CONTINUE_END.search(out[-1])
                and not _STRUCTURAL.match(ln)
                and not _COLUMN_GAP.search(ln.strip())
                and not out[-1].lstrip().startswith(("#", "|"))):
            out[-1] = f"{out[-1].rstrip()} {ln.strip()}"
            report.lines_unwrapped += 1
        else:
            out.append(ln)
    return out


def _cells(line: str) -> list[str]:
    return [c for c in _COLUMN_GAP.split(line.strip()) if c]


def _convert_tables(lines: list[str], report: NormalizationReport) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        head = lines[i]
        width = len(_cells(head)) if head.strip() and not _STRUCTURAL.match(head) else 0
        j = i + 1
        if width >= 3 and not re.search(r"\d", head):
            while (j < len(lines) and lines[j].strip() and not _STRUCTURAL.match(lines[j])
                   and len(_cells(lines[j])) == width):
                j += 1
        if width >= 3 and j - i >= 3 and not re.search(r"\d", head):   # header + 2+ rows
            if out and out[-1].strip():
                out.append("")                  # a table must be its own paragraph
            out.append("| " + _PIPE.join(_cells(head)) + " |")
            out.append("|" + "---|" * width)
            out.extend("| " + _PIPE.join(_cells(r)) + " |" for r in lines[i + 1:j])
            out.append("")
            report.tables_converted += 1
            i = j
            continue
        out.append(head)
        i += 1
    return out


MODES = ("auto", "always", "off")


def normalize_extracted_text(text: str, mode: str = "auto") -> NormalizedText:
    """Normalise extracted text for chunking.

    ``mode``:
      auto    (default) hygiene, boilerplate, de-hyphenation and layout tables always (none of
              them fires on clean markdown); LINE UNWRAPPING only when the text shows extraction
              signals. Unwrapping is the one step that also fires on hand-wrapped markdown (a
              wrapped numbered item would turn into a list chunk), so it waits for evidence.
      always  every step, unconditionally — for callers that KNOW the text is extracted (an
              ingestion service fed by a PDF/OCR extractor).
      off     returned unchanged.
    """
    if mode not in MODES:
        raise ValueError(f"normalize mode must be one of {MODES}, got {mode!r}")
    report = NormalizationReport(input_chars=len(text or ""), mode=mode)
    if mode == "off":
        report.output_chars = report.input_chars
        return NormalizedText(text=text or "", report=report)
    text = _hygiene(text or "")

    pages = [p.split("\n") for p in text.split("\f")]
    report.pages = len(pages)
    pages = _remove_boilerplate(pages, report)
    text = "\n\n".join("\n".join(p) for p in pages)

    text, n = _HYPHEN_WRAP.subn(r"\1\2", text)
    report.hyphenations_joined = n

    lines = _convert_tables(text.split("\n"), report)
    report.detected_extracted = bool(
        report.pages > 1 or report.page_number_lines_removed or report.boilerplate_lines_removed
        or report.hyphenations_joined or report.tables_converted)
    if mode == "always" or report.detected_extracted:
        lines = _unwrap(lines, report)

    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    report.output_chars = len(text)
    return NormalizedText(text=text, report=report)
