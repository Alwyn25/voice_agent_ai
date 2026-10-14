"""Structure-aware ("semantic") document chunking.

WHY THIS IS NOT NAIVE TEXT CHUNKING (Voice Agent Feedback #2–#7). A brochure is not a
wall of prose. It is prose *and* enumerations (amenity grids, specification lists) *and*
tables (cost sheets, floor plans). Splitting all of it on the same "pack N characters with
some overlap" rule quietly destroys the parts that are not prose:

  * An amenity list — "…play lawn, lawn mound, dog park, jogging track, cycle lane…" — packs
    into one chunk whose embedding is the *average* of twenty concepts. "dog park" is then
    ~1/20th of the vector's signal, so "do you have a dog park?" retrieves it weakly or not
    at all, and the agent falls back to "I'll find out and call you back". The fact was in
    the document the whole time; the chunk just could not surface it.
  * A cost-sheet row loses its column headers when the table is flattened, so "car parking
    charge for the 1,560 sq ft unit" no longer sits next to the words that identify it.
  * Only the *nearest* heading was carried, so a chunk under "Amenities › Outdoor" arrived
    at retrieval labelled just "Outdoor" — the word "amenities", which the caller actually
    said, was gone from both the embedding and the keyword index.

THE FIX — chunk each kind of content the way that kind of content wants to be chunked:

  1. **Heading breadcrumb.** Every chunk carries its full section path ("Amenities >
     Outdoor and landscape"), not just the leaf. That path is prepended to the embedded and
     indexed text, so the section's vocabulary is searchable and the model sees where the
     fact lives.
  2. **Enumeration splitting.** A long list — markdown bullets *or* an inline comma list —
     is broken into small item groups (``chunk_list_group_size``). Each group embeds as a
     tight cluster of a handful of related items, so any single item is a strong match.
  3. **Table row-with-header.** Table rows are chunked with their header row attached, so
     every row keeps the column names that give its numbers meaning.
  4. **Sentence-aware prose.** Ordinary paragraphs are packed to a target size and, when a
     chunk boundary falls inside long prose, the overlap is whole trailing *sentences*, not
     a mid-word character slice.

DOMAIN-NEUTRAL BY CONSTRUCTION. There is no real-estate vocabulary here — only structural
signals (headings, list markers, pipes, commas). The same code chunks a car spec sheet or
an insurance policy. Dependency-free (stdlib ``re`` only) and pure, so it unit-tests without
a database: run ``python -m app.engine.rag.chunkers``.
"""
import re
from dataclasses import dataclass, field

# Separator between levels of the heading breadcrumb. Kept ASCII on purpose: this string
# ends up in stored chunk content, in API responses and in seed-script console output that
# may be printed on a cp1252 Windows terminal, where a fancy chevron would raise.
_CRUMB = " > "

# --- heading detection -------------------------------------------------------------------
# ATX headings ("## Amenities") are unambiguous. A "bare" heading is a short, title-cased or
# colon-terminated line used as a section label in plain-text documents that never learned
# markdown ("Where it is", "Structure:"). Both are recognised; the bare heuristic is kept
# deliberately conservative so a normal capitalised sentence is not mistaken for a heading.
_ATX = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")

# --- list detection ----------------------------------------------------------------------
# A markdown list item: a bullet ("-", "*", "•") or an ordinal ("1.", "2)") then text.
_BULLET = re.compile(r"^\s*(?:[-*•]|\d{1,3}[.)])\s+(.+\S)\s*$")

# --- extraction hardening -------------------------------------------------------------------
# Sentence ends include the Devanagari danda (।) and double danda (॥): without them a Hindi
# paragraph is one unbreakable "sentence" and is never packed or split.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?।॥])\s+")
# List-item separator that never splits a NUMBER. "89,50,000" / "1,050" keep their grouping
# commas (a comma directly followed by a digit); "pool, gym, 2 courts" still splits. The plain
# [;,] split turned an extracted price row into "89, 50, 000" — a corrupted price in the KB.
ITEM_SEPARATOR = re.compile(r";|,(?!\d)")
# Lines that are never headings: page furniture ("Brochure   Page 2", "3 of 12"), table rows
# (pipes, or 3+-space column gaps from a PDF layout).
_PAGE_MARK = re.compile(r"\bpage\s*\d+\b|^\d{1,4}\s*(?:of|/)\s*\d{1,4}$", re.I)
_LAYOUT_GAP = re.compile(r"\S\s{3,}\S")


def hard_split(text: str, max_chars: int) -> list[str]:
    """Split text with no usable sentence boundary into <= max_chars pieces at WORD boundaries.

    The last-resort size cap for extracted text without punctuation (an OCR'd payment plan, a
    run-on spec line). Without it such a block became one arbitrarily large chunk — diluted
    beyond retrieval, and silently truncated by the embedding provider past 8,000 characters.
    Splits only on whitespace, so a number or word is never cut; a single token longer than
    max_chars (a URL) is kept whole.
    """
    words = text.split()
    pieces: list[str] = []
    cur = ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > max_chars:
            pieces.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        pieces.append(cur)
    return pieces


def sentences_of(text: str, max_chars: int) -> list[str]:
    """Sentences of ``text``; any sentence longer than ``max_chars`` is hard_split."""
    out: list[str] = []
    for sentence in SENTENCE_BOUNDARY.split(text.strip()):
        out.extend(hard_split(sentence, max_chars) if len(sentence) > max_chars else [sentence])
    return out


@dataclass
class Chunk:
    index: int
    content: str
    heading: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """What is embedded and stored. The breadcrumb leads so the section's own words
        ("amenities", "specifications") are part of both the vector and the keyword index."""
        return f"{self.heading}\n{self.content}" if self.heading else self.content


@dataclass
class Block:
    """One structurally-homogeneous span of a document, tagged with the section it lives in.

    A block is either prose, a list, or a table — never a mix — which is what lets each be
    chunked by its own rules downstream.
    """
    heading_path: list[str]
    kind: str                                  # "prose" | "list" | "table"
    text: str = ""                             # prose
    items: list[str] = field(default_factory=list)   # list
    marker: str = "inline"                     # "inline" | "bullet" (list only)
    header: str = ""                           # table header row
    sep: str = ""                              # table separator row ("|---|"), if any
    rows: list[str] = field(default_factory=list)    # table data rows

    @property
    def crumb(self) -> str | None:
        # Collapse repeated adjacent labels (a top-level "# Amenities" followed by nothing
        # else should not read "Amenities > Amenities").
        path = [p for i, p in enumerate(self.heading_path)
                if i == 0 or p != self.heading_path[i - 1]]
        return _CRUMB.join(path) if path else None


def _classify_heading(line: str) -> tuple[int, str] | None:
    """Return ``(level, text)`` if ``line`` is a heading, else ``None``.

    Level is the ATX depth (``#`` count). A bare heading has no intrinsic depth; the caller
    resolves it to one level below the most recent ATX heading so "Where it is" nests under
    "# DSR Highland Greenz". ``level`` is returned as 0 for bare headings to signal that.
    """
    m = _ATX.match(line)
    if m:
        return len(m.group(1)), m.group(2).strip()

    s = line.strip()
    if not s or len(s) > 90 or _BULLET.match(line):
        return None
    # Extraction furniture is never a section heading (see _PAGE_MARK / _LAYOUT_GAP).
    if "|" in s or _PAGE_MARK.search(s) or _LAYOUT_GAP.search(s):
        return None
    # "Structure:" — a short, non-sentence label ending in a colon.
    if s.endswith(":") and len(s.split()) <= 10:
        return 0, s.rstrip(":").strip()
    # "Doors and windows" — a short title-cased phrase that is not a sentence (no trailing
    # period), not mostly digits, and not a comma list.
    words = s.split()
    if (len(words) <= 8 and s[0].isupper() and not s.endswith(".")
            and "," not in s and sum(c.isdigit() for c in s) < len(s) / 2):
        return 0, s
    return None


def _is_table_line(line: str) -> bool:
    s = line.strip()
    return "|" in s and (s.count("|") >= 2 or (s.startswith("|") and s.endswith("|")))


def _is_sep_row(line: str) -> bool:
    s = line.strip()
    return "-" in s and re.fullmatch(r"[\s|:\-]+", s) is not None


def _looks_table(lines: list[str]) -> bool:
    return len(lines) >= 2 and all(_is_table_line(ln) for ln in lines)


def _looks_bullets(lines: list[str]) -> bool:
    return len(lines) >= 1 and all(_BULLET.match(ln) for ln in lines)


# An item separator in an inline list: a semicolon, or a comma that is NOT a digit-group
# separator. "1,422" / "1,03,93,750" are single numbers — a comma with a digit on BOTH sides
# is part of the number, never a list break.
#
# THE BUG THIS FIXES (found during R5.08). Splitting on every comma turned "3 BHK of 1,422
# and 1,390 sq ft; 2 BHK of 1,099, 1,103 and 1,104" into the items "1" / "422 and 1" /
# "390 sq ft" / ... — the stored chunk then read "1, 422 and 1, 390", and the trailing "104"
# fragment fell under min_chars and was silently dropped. The agent was reading corrupted
# areas, and the metadata extractor could not recover them.
_ITEM_SEP = re.compile(r";|,(?!\d)|(?<!\d),")


def _split_items(text: str) -> list[str]:
    return [p.strip() for p in _ITEM_SEP.split(text) if p.strip()]


def _is_inline_enumeration(text: str) -> bool:
    """True when a single paragraph is really a comma/semicolon list of short items.

    This is the case naive chunking most damages: the amenity and specification grids are
    written as prose commas, not markdown bullets, so nothing marks them as lists. We infer
    it from shape — many short, verbless segments and no interior sentence break:

      * at least five segments (four commas is a normal sentence, not a catalogue);
      * every segment short (a clause like "picnic lawn with hammocks" is fine; a full
        sentence is not);
      * no full stop except an optional trailing one — a period mid-paragraph means real
        sentences ("Flat cost is 1,42,47,261 rupees. Total is …"), which must stay prose.
    """
    body = text.strip()
    if not body:
        return False
    interior = body[:-1] if body.endswith(".") else body
    if "." in interior:                                   # a real sentence hides in here
        return False
    parts = _split_items(interior)
    parts = [p.strip() for p in ITEM_SEPARATOR.split(interior) if p.strip()]
    if len(parts) < 5:
        return False
    if any(len(p) > 70 for p in parts):
        return False
    return sum(len(p.split()) for p in parts) / len(parts) <= 7


def _paragraphs(lines: list[str]) -> list[list[str]]:
    """Group a section's body lines into paragraphs, split on blank lines."""
    paras: list[list[str]] = []
    buf: list[str] = []
    for ln in lines:
        if ln.strip():
            buf.append(ln)
        elif buf:
            paras.append(buf)
            buf = []
    if buf:
        paras.append(buf)
    return paras


def split_blocks(text: str) -> list[Block]:
    """Segment a document into typed :class:`Block` s, each tagged with its heading path.

    Headings maintain a stack keyed by level, so the path reflects real nesting; body lines
    between headings are grouped into paragraphs and each paragraph is classified as a
    table, a list, or prose.
    """
    blocks: list[Block] = []
    stack: list[tuple[int, str]] = []            # (level, text), outermost first
    last_atx_level = 0
    body: list[str] = []

    def current_path() -> list[str]:
        return [t for _, t in stack]

    def flush_body() -> None:
        if not body:
            return
        path = current_path()
        for para in _paragraphs(body):
            if _looks_table(para):
                sep = next((ln for ln in para if _is_sep_row(ln)), "")
                data = [ln for ln in para if ln is not sep and not _is_sep_row(ln)]
                header = data[0] if data else ""
                rows = data[1:] if len(data) > 1 else data
                blocks.append(Block(path, "table", header=header, sep=sep, rows=rows))
            elif _looks_bullets(para):
                items = [_BULLET.match(ln).group(1).strip() for ln in para]
                blocks.append(Block(path, "list", items=items, marker="bullet"))
            else:
                joined = " ".join(ln.strip() for ln in para)
                if _is_inline_enumeration(joined):
                    stripped = joined[:-1] if joined.endswith(".") else joined
                    items = _split_items(stripped)
                    items = [p.strip() for p in ITEM_SEPARATOR.split(stripped) if p.strip()]
                    blocks.append(Block(path, "list", items=items, marker="inline"))
                else:
                    blocks.append(Block(path, "prose", text=joined))
        body.clear()

    for raw in (text or "").splitlines():
        line = raw.rstrip()
        head = _classify_heading(line)
        if head is not None:
            flush_body()
            level, htext = head
            if level == 0:                       # bare heading nests under the last ATX one
                level = last_atx_level + 1
            else:
                last_atx_level = level
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, htext))
            continue
        body.append(line)

    flush_body()
    return blocks


def _tail_sentences(text: str, max_chars: int) -> str:
    """Whole trailing sentences of ``text`` up to ``max_chars`` — a clean overlap seam that
    never cuts a word or a number in half (a mid-word slice was corrupting "11,499")."""
    if max_chars <= 0:
        return ""
    sentences = SENTENCE_BOUNDARY.split(text.strip())
    out: list[str] = []
    total = 0
    for s in reversed(sentences):
        if out and total + len(s) + 1 > max_chars:
            break
        out.insert(0, s)
        total += len(s) + 1
    return " ".join(out).strip()


def _pack_prose(text: str, target: int, overlap: int) -> list[str]:
    """Pack paragraphs to ~``target`` chars; overflowing prose overlaps by whole sentences."""
    bodies: list[str] = []
    buffer = ""
    for para in [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]:
        # A single oversized paragraph is split on sentence boundaries so no chunk balloons.
        if len(para) > target * 1.6:
            # sentences_of() hard-splits a sentence with no usable boundary (extracted text).
            for sentence in sentences_of(para, target):
                if buffer and len(buffer) + len(sentence) + 1 > target:
                    bodies.append(buffer.strip())
                    buffer = _tail_sentences(buffer, overlap)
                buffer = f"{buffer} {sentence}".strip()
            continue
        if buffer and len(buffer) + len(para) + 2 > target:
            bodies.append(buffer.strip())
            buffer = _tail_sentences(buffer, overlap)
        buffer = f"{buffer}\n\n{para}".strip()
    if buffer.strip():
        bodies.append(buffer.strip())
    return bodies


def _pack_list(items: list[str], marker: str, group_size: int, target: int) -> list[str]:
    """Split a list into item groups so each embeds as a focused cluster.

    A group closes when it reaches ``group_size`` items or would exceed ``target`` chars,
    whichever comes first. Bullets are rejoined as bullets (structure the model reads well);
    an inline comma list stays a comma list.
    """
    def render(group: list[str]) -> str:
        if marker == "bullet":
            return "\n".join(f"- {it}" for it in group)
        return ", ".join(group) + "."

    bodies: list[str] = []
    group: list[str] = []
    size = 0
    for it in items:
        add = len(it) + 2
        if group and (len(group) >= group_size or size + add > target):
            bodies.append(render(group))
            group, size = [], 0
        group.append(it)
        size += add
    if group:
        bodies.append(render(group))
    return bodies


def _pack_table(header: str, sep: str, rows: list[str], target: int) -> list[str]:
    """Chunk table rows with the header attached, so every row keeps its column names."""
    prefix = "\n".join(x for x in (header, sep) if x.strip())
    bodies: list[str] = []
    batch: list[str] = []
    size = len(prefix)
    for row in rows:
        if batch and size + len(row) + 1 > target:
            bodies.append("\n".join(x for x in (prefix, *batch) if x))
            batch, size = [], len(prefix)
        batch.append(row)
        size += len(row) + 1
    if batch:
        bodies.append("\n".join(x for x in (prefix, *batch) if x))
    return bodies or ([prefix] if prefix else [])


def chunk_text(text: str, *, target_chars: int = 900, overlap_chars: int = 150,
               list_group_size: int = 8, min_chars: int = 20) -> list[Chunk]:
    """Turn a document into retrieval chunks, chunking each block by its structural kind.

    Prose from adjacent paragraphs under one heading is merged and size-packed; a list or a
    table is emitted as its own chunk(s) and never fused with surrounding prose, because
    fusing is exactly what dilutes a structured fact past the point of retrieval.
    """
    chunks: list[Chunk] = []

    def emit(body: str, crumb: str | None, kind: str) -> None:
        body = body.strip()
        if len(body) < min_chars:
            return
        chunks.append(Chunk(len(chunks), body, crumb,
                            {"kind": kind, "heading_path": crumb}))

    prose_buf = ""
    prose_crumb: str | None = None

    def flush_prose() -> None:
        nonlocal prose_buf
        for body in _pack_prose(prose_buf, target_chars, overlap_chars):
            emit(body, prose_crumb, "prose")
        prose_buf = ""

    for block in split_blocks(text):
        crumb = block.crumb
        if block.kind == "prose":
            if prose_buf and crumb != prose_crumb:
                flush_prose()
            prose_crumb = crumb
            prose_buf = f"{prose_buf}\n\n{block.text}".strip()
            continue

        # A structured block ends the current prose run and is chunked on its own terms.
        flush_prose()
        prose_crumb = crumb
        if block.kind == "list":
            for body in _pack_list(block.items, block.marker, list_group_size, target_chars):
                emit(body, crumb, "list")
        elif block.kind == "table":
            for body in _pack_table(block.header, block.sep, block.rows, target_chars):
                emit(body, crumb, "table")

    flush_prose()
    return chunks


# ------------------------------------------------------------------ R5.03 child chunks
#
# SMALL-TO-BIG ("parent-child") RETRIEVAL. A ~900-char chunk is the right size to SHOW the model
# (a whole cost-sheet paragraph, the table header, the neighbouring sentences) but the wrong size
# to SEARCH with: its embedding is the average of every fact inside it, so "carpet area of the
# 1,390 sq ft unit" is a small fraction of a vector that also means "1,239 sq ft", "1,560 sq ft",
# "clubhouse charges" and "car parking". The fix is to search with small pieces and answer with
# the big one: every chunk produced by chunk_text() becomes a PARENT, and is split here into
# ~200-char CHILDREN. Retrieval matches children, then hands the model the parent
# (app/engine/rag/parent_child.py).
#
# Two invariants this code must keep — both are unit-tested:
#   * COVERAGE: every piece of the parent's text is inside at least one child. In parent-child
#     mode the parent itself is never searched, so text that no child carries is unreachable.
#     That is why an undersized trailing fragment is MERGED into its neighbour, never dropped.
#   * NO CROSS-RECORD OVERLAP: overlap is only added between children cut from ONE oversized
#     paragraph. Cost sheets are written one unit per paragraph; carrying "Total 1,54,32,011."
#     from unit one into unit two's child would teach that child a price it does not own.


def _unpack_list(body: str) -> tuple[list[str], str]:
    """Recover (items, marker) from a list body rendered by :func:`_pack_list`."""
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if lines and all(ln.lstrip().startswith("- ") for ln in lines):
        return [ln.lstrip()[2:].strip() for ln in lines], "bullet"
    stripped = body.strip()
    stripped = stripped[:-1] if stripped.endswith(".") else stripped
    return [p.strip() for p in ITEM_SEPARATOR.split(stripped) if p.strip()], "inline"


def _unpack_table(body: str) -> tuple[str, str, list[str]]:
    """Recover (header, sep, rows) from a table body rendered by :func:`_pack_table`."""
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if not lines:
        return "", "", []
    header, rest = lines[0], lines[1:]
    sep = rest[0] if rest and _is_sep_row(rest[0]) else ""
    rows = rest[1:] if sep else rest
    return header, sep, rows


_NUMBER_LED = re.compile(r"^\d")


def _records(sentences: list[str]) -> list[str] | None:
    """Group a paragraph's sentences into RECORDS, or return None if it is not a record run.

    A record run is a paragraph written as repeated entries that each open with a number —
    "875 sq ft, East facing. Carpet 592. Total 1,03,93,750. 880 sq ft, North facing. ..." — the
    shape of a cost sheet, a spec list or a timetable flattened into prose. Structural signal
    only (a sentence that starts with a digit), no domain vocabulary. Each record is one
    number-led sentence plus the sentences after it up to the next number-led one. Requires the
    paragraph to OPEN with a number-led sentence and hold at least three of them, so ordinary
    prose that happens to contain a sentence like "3 BHK is 11,499 per sq ft." is unaffected.
    """
    if not sentences or not _NUMBER_LED.match(sentences[0]):
        return None
    if sum(bool(_NUMBER_LED.match(x)) for x in sentences) < 3:
        return None
    records: list[str] = []
    for sentence in sentences:
        if _NUMBER_LED.match(sentence) or not records:
            records.append(sentence)
        else:
            records[-1] = f"{records[-1]} {sentence}"
    return records


def _pack_child_prose(text: str, target: int, overlap: int,
                      record_split: bool = False) -> list[str]:
    """Pack prose into child-sized pieces.

    Differs from :func:`_pack_prose` (parent packing) in one deliberate way: small adjacent
    paragraphs are merged while they fit, but NOTHING is carried across a paragraph boundary.
    Only a paragraph too long for one child is cut on sentence boundaries, and only those cuts
    overlap (by whole trailing sentences) so a fact straddling the seam is whole on one side.

    ``record_split``: an oversized paragraph that is a run of number-led records (see
    :func:`_records`) is instead packed whole-record by whole-record, with no overlap — every
    record is intact in exactly one child, where sentence packing would cut a record's "Total"
    off into the next child.
    """
    bodies: list[str] = []
    buffer = ""
    for para in [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]:
        if len(para) <= target:
            if buffer and len(buffer) + len(para) + 2 > target:
                bodies.append(buffer)
                buffer = ""
            buffer = f"{buffer}\n\n{para}".strip()
            continue
        # Oversized paragraph: close what we have, then sentence-pack it with in-paragraph overlap.
        if buffer:
            bodies.append(buffer)
            buffer = ""
        # A child may hold one long sentence whole (up to 1.5 x target); only a run-on block
        # beyond that is hard-split — cutting ordinary sentences mid-way would hurt more.
        sentences = sentences_of(para, int(target * 1.5))
        records = _records(sentences) if record_split else None
        if records:
            piece = ""
            for rec in records:
                if piece and len(piece) + len(rec) + 1 > target:
                    bodies.append(piece)
                    piece = ""
                piece = f"{piece} {rec}".strip()
            if piece:
                bodies.append(piece)
            continue
        piece = ""
        for sentence in sentences:
            if piece and len(piece) + len(sentence) + 1 > target:
                bodies.append(piece)
                seam = _tail_sentences(piece, overlap)
                # A seam as long as the piece would re-emit the same child forever-ish; drop it.
                piece = seam if seam and len(seam) < len(piece) else ""
            piece = f"{piece} {sentence}".strip()
        if piece:
            bodies.append(piece)
    if buffer:
        bodies.append(buffer)
    return bodies


def _merge_small(bodies: list[str], min_chars: int, joiner: str) -> list[str]:
    """Fold any body shorter than ``min_chars`` into its predecessor (or successor, for the
    first one). Never drops text — see the COVERAGE invariant above."""
    out: list[str] = []
    for b in bodies:
        if out and len(b) < min_chars:
            out[-1] = f"{out[-1]}{joiner}{b}"
        else:
            out.append(b)
    if len(out) > 1 and len(out[0]) < min_chars:
        out[1] = f"{out[0]}{joiner}{out[1]}"
        out.pop(0)
    return out


def split_children(parent: Chunk, *, target_chars: int = 200, overlap_chars: int = 100,
                   list_group_size: int = 3, min_chars: int = 40,
                   split_min_chars: int | None = None, title: str | None = None,
                   record_split: bool = False) -> list[Chunk]:
    """Split one parent chunk into child chunks, by the parent's structural kind.

    * table  -> header (+ separator) + as many rows as fit; every child keeps the column names.
    * list   -> groups of ``list_group_size`` items (vs 8 for parents), same marker style.
    * prose  -> see :func:`_pack_child_prose`.

    Returns ``[]`` when splitting would not help — the parent is already child-sized or yields a
    single piece. The caller then stores it as an ordinary STANDALONE chunk: a parent whose only
    child is a copy of itself would double the rows and change nothing.

    Children inherit the parent's heading breadcrumb, so a child's ``embed_text`` still carries
    "3 BHK cost sheet" / "Amenities > Outdoor" — the section vocabulary the caller actually uses.
    With ``title`` (the document's title) the breadcrumb is prefixed with it too, unless the
    heading already contains it — a ~200-char slice has little context of its own, and the title
    is often the most specific label it has ("Location and Connectivity", "3 BHK Cost Sheet").
    ``index`` is the child's position within its parent (0-based).

    ``split_min_chars``: a parent body at or below this is left whole (default 1.3 x target).
    ``record_split``: see :func:`_pack_child_prose`.
    """
    # ``content`` is the body WITHOUT the breadcrumb; each child re-adds it via embed_text.
    body = parent.content.strip()
    # Small enough already: a child could not be meaningfully more specific than this.
    threshold = int(target_chars * 1.3) if split_min_chars is None else split_min_chars
    if len(body) <= threshold:
        return []

    kind = parent.metadata.get("kind") or "prose"
    if kind == "list":
        items, marker = _unpack_list(body)
        bodies = _pack_list(items, marker, list_group_size, target_chars)
        joiner = "\n" if marker == "bullet" else " "
    elif kind == "table":
        header, sep, rows = _unpack_table(body)
        bodies = _pack_table(header, sep, rows, target_chars) if rows else [body]
        joiner = "\n"
    else:
        bodies = _pack_child_prose(body, target_chars, overlap_chars, record_split)
        joiner = " "

    bodies = _merge_small([b.strip() for b in bodies if b.strip()], min_chars, joiner)
    if len(bodies) <= 1:
        return []
    heading = parent.heading
    if title and title.strip() and title.strip().lower() not in (heading or "").lower():
        heading = f"{title.strip()}{_CRUMB}{heading}" if heading else title.strip()
    return [Chunk(i, b, heading,
                  {"kind": kind, "heading_path": parent.heading, "child_index": i})
            for i, b in enumerate(bodies)]


def uncovered_text(parent: Chunk, children: list[Chunk] | list[str]) -> list[str]:
    """Pieces of the parent's text that NO child contains — must be empty (COVERAGE invariant).

    The unit of comparison follows the parent's kind, because children re-render their piece:
    list -> each item, table -> each line, prose -> each sentence / line. Used by the unit tests
    and by ``scripts/backfill_parent_child.py --verify`` on live data.
    """
    texts = [c if isinstance(c, str) else c.content for c in children]
    union = "\n".join(texts)
    body = parent.content.strip()
    kind = parent.metadata.get("kind") or "prose"
    if kind == "list":
        pieces, _ = _unpack_list(body)
    elif kind == "table":
        pieces = [ln.strip() for ln in body.splitlines()]
    else:
        pieces = [p.strip() for line in body.split("\n")
                  for p in SENTENCE_BOUNDARY.split(line)]
    return [p for p in pieces if p and p not in union]


# ------------------------------------------------------------------ self-test (no DB, no net)

def _selftest() -> int:
    """Run with ``python -m app.engine.rag.chunkers``."""
    failures = 0

    def check(name: str, cond: bool) -> None:
        nonlocal failures
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        if not cond:
            failures += 1

    # A real slice of the DSR amenities + specs + a price table, exercising every block kind.
    doc = """# DSR Highland Greenz

## Where it is
DSR Highland Greenz is at Chikkanayakanahalli, off Sarjapur Road, Bengaluru. It is a
gated community of luxury apartments by DSR Infra Projects.

# Amenities

## Outdoor and landscape
Amphitheatre, party lawn with stage, barbeque zone, informal seating, youth hangout zone,
picnic lawn, play lawn, lawn mound, dog park, jogging track, cycle lane, pedestrian bridge,
bird watching deck, seasonal stream, elders' seating plaza.

## Indoor and clubhouse
Swimming pool, reception, multipurpose hall, squash court, badminton court, indoor games,
yoga room, creche, guest bedroom, indoor gym, library, banquet hall.

# Cost Sheet

| Unit | Area | Total |
|------|------|-------|
| Unit 1 | 1,239 | 1,54,32,011 |
| Unit 2 | 1,390 | 1,72,06,110 |
"""
    chunks = chunk_text(doc, target_chars=400, overlap_chars=60, list_group_size=6)
    by_kind = {c.metadata.get("kind") for c in chunks}
    check("produces prose, list and table chunks", {"prose", "list", "table"} <= by_kind)

    # Breadcrumb: a chunk carries its full section path, not just the leaf.
    where = [c for c in chunks if "Where it is" in (c.heading or "")]
    check("breadcrumb carries the document root",
          bool(where) and where[0].heading == "DSR Highland Greenz > Where it is")

    # THE POINT OF THIS FILE: "dog park" lands in a small, focused enumeration chunk rather
    # than one averaged blob of twenty amenities.
    dog = [c for c in chunks if "dog park" in c.content]
    check("dog park is in exactly one chunk", len(dog) == 1)
    check("that chunk is a focused group (<= group_size items)",
          bool(dog) and dog[0].content.count(",") + 1 <= 6)
    check("dog park chunk keeps its Amenities breadcrumb",
          bool(dog) and (dog[0].heading or "").startswith("Amenities"))
    check("creche is retrievable (survives chunking)",
          any("creche" in c.content for c in chunks))

    # Table rows keep their header, so a row's numbers stay attached to the column names.
    table_chunks = [c for c in chunks if c.metadata.get("kind") == "table"]
    check("table row keeps its header",
          bool(table_chunks) and all("Unit" in c.content and "Total" in c.content
                                     for c in table_chunks))
    check("table data survives", any("1,54,32,011" in c.content for c in table_chunks))

    # Prose is not shattered into the enumeration path and its sentences stay whole.
    prose = [c for c in chunks if c.metadata.get("kind") == "prose"]
    check("prose sentence stays intact",
          any("gated community of luxury apartments" in c.content for c in prose))

    # Overlap seam never splits a number (the "11,499" corruption class).
    packed = _pack_prose("The rate is 11,499 rupees per square foot on super built-up area. "
                         "Three layouts are offered. Unit one is 1,239 square feet. Unit two "
                         "is 1,390 square feet. Unit three is 1,560 square feet in total area.",
                         target=90, overlap=40)
    check("no chunk splits a grouped number",
          all(not re.search(r"\d,\s*$|^\s*\d{3}\b", b) for b in packed))

    # A digit-group comma is part of the number, never a list break (the "1, 422" bug).
    mixed = chunk_text("Tower A, wings 1 and 2, 3 BHK of 1,422 and 1,390 sq ft, 2 BHK of "
                       "1,099, 1,103 and 1,104, lift lobby, fire exit.", min_chars=5)
    joined = " ".join(c.content for c in mixed)
    check("grouped numbers survive list splitting",
          all(n in joined for n in ("1,422", "1,390", "1,099", "1,103", "1,104")))
    check("no digit-group comma turned into a list break", "1, 422" not in joined)

    # Empty / tiny input degrades quietly.
    check("empty document -> no chunks", chunk_text("") == [])
    check("chunk indexes are contiguous from 0",
          [c.index for c in chunks] == list(range(len(chunks))))

    # R5.03 children: a cost-sheet parent (one unit per paragraph) splits into per-unit
    # children, nothing is lost, and no child carries another unit's figures.
    sheet = Chunk(0, "Rate 11,499 rupees per square foot on super built-up area. Clubhouse "
                     "charges 3,25,000 on all three.\n\n"
                     "1,239 sq ft, West facing. Carpet 808. Flat cost 1,42,47,261. "
                     "Total 1,54,32,011.\n\n"
                     "1,390 sq ft, East and West facing. Carpet 923. Flat cost 1,59,83,610. "
                     "Total 1,72,06,110.", "3 BHK cost sheet", {"kind": "prose"})
    kids = split_children(sheet, target_chars=120, overlap_chars=40, min_chars=30)
    check("parent splits into several children", len(kids) >= 3)
    check("children keep the parent breadcrumb",
          all(k.heading == "3 BHK cost sheet" for k in kids))
    check("no child mixes two units' totals",
          not any("1,54,32,011" in k.content and "1,72,06,110" in k.content for k in kids))
    check("every figure survives in some child",
          all(any(n in k.content for k in kids)
              for n in ("11,499", "1,54,32,011", "1,72,06,110", "Carpet 923")))
    check("child-sized parent is not split", split_children(
        Chunk(0, "Swimming pool and gym.", None, {"kind": "prose"})) == [])

    print("CHUNKERS SELFTEST:", "PASS" if not failures else f"{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
