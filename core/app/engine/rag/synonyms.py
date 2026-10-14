"""Tenant-supplied synonym / alias bridging for the keyword arm of retrieval.

WHY THIS EXISTS (Voice Agent Feedback #3, #5, #6). The knowledge base held the answers
the caller wanted — "creche" is named in the clubhouse list, the "RCC framed structure" is
spelled out, power backup is described — yet the agent said it would "find out and call
back". The gap was vocabulary, not content: the customer said *daycare*, *generator*,
*bricks*; the document said *creche*, *DG set / power backup*, *RCC structure*. The
semantic (vector) arm bridges some of this on its own, but the lexical (``to_tsquery``) arm
matches exact lexemes only, so a keyword-dense chunk that never contains the customer's
literal word ranks below the fold and never reaches the model.

THE FIX. Before the keyword query is built, expand the customer's significant words with
their tenant-declared equivalents, so a search for *generator* also matches *DG* and *power
backup*. This is recall-first and deliberately biased that way: the vector arm and the
relative-score cut in :func:`retrieve` discard anything an over-broad OR pulls in, so the
cost of one spurious alias is a chunk that loses on ranking, while the cost of the missing
alias was an invented "I'll call you back".

DOMAIN-NEUTRAL BY CONSTRUCTION. The engine ships zero real-estate vocabulary. Aliases are
tenant DATA, read from ``tenant.settings["retrieval_synonyms"]`` (see ``seed_dsr_settings``
for the DSR set) — so Motorworks can bridge ``bhp ↔ horsepower`` with the identical
mechanism and this file never changes. The declared shape is a list of groups, each group a
list of surface phrases that mean the same thing::

    "retrieval_synonyms": [
        ["generator", "dg", "dg set", "diesel generator", "power backup", "backup power"],
        ["creche", "daycare", "day care", "childcare", "child care"]
    ]

Dependency-free (stdlib ``re`` only) and pure, so it unit-tests without a database.
"""
import re

# Alias groups are matched at the WORD level, because ``to_tsquery`` ranks on lexemes, not
# phrases. A multi-word member like "power backup" therefore contributes the words "power"
# and "backup" to the group's vocabulary; searching any one word in the group can pull in
# every other word. That is what lets the single token "generator" reach a chunk whose only
# relevant text is "power backup". Words shorter than two characters carry no signal and are
# dropped so a stray "a"/"of" in a phrase cannot wire unrelated groups together.
_WORD_RE = re.compile(r"[a-z0-9]+")

# A hard cap on how many alias words one query may gain. Real groups are small, but a
# malformed tenant config (one giant group, or accidental transitive overlap) must never
# turn a three-word question into a hundred-term ``to_tsquery`` — that is both slow and
# precision-destroying. Well past what any genuine expansion needs.
_MAX_ADDED_TERMS = 24


def normalize_synonyms(raw: object) -> list[list[str]]:
    """Coerce whatever is stored in ``tenant.settings["retrieval_synonyms"]`` into clean
    groups of lowercase phrases.

    Tenant settings are operator-edited JSON, so this is defensive on purpose: a non-list,
    a group with a single member, blank strings and duplicates are all dropped rather than
    trusted. A malformed config yields an empty list and retrieval simply runs without
    expansion — never an exception on the hot path.
    """
    if not isinstance(raw, list):
        return []
    groups: list[list[str]] = []
    for group in raw:
        if not isinstance(group, (list, tuple)):
            continue
        members: list[str] = []
        for member in group:
            if not isinstance(member, str):
                continue
            cleaned = member.strip().lower()
            if cleaned and cleaned not in members:
                members.append(cleaned)
        # A group of one bridges nothing — it needs at least two surface forms to be a
        # synonym relationship at all.
        if len(members) >= 2:
            groups.append(members)
    return groups


def build_alias_index(groups: list[list[str]]) -> dict[str, set[str]]:
    """Flatten synonym groups into a word→equivalent-words map, the form the hot path uses.

    Each group's members are tokenised to their constituent words, and every word in the
    group is mapped to all the OTHER words in it. So the DSR generator group yields, among
    others, ``generator → {dg, set, diesel, power, backup, ...}`` and, symmetrically,
    ``backup → {generator, dg, power, ...}`` — expansion works whichever synonym the
    customer happens to say.

    Built once per turn from the tenant's (small, static) config; cheap enough not to cache.
    """
    index: dict[str, set[str]] = {}
    for members in groups:
        words: set[str] = set()
        for member in members:
            words.update(w for w in _WORD_RE.findall(member) if len(w) > 1)
        for w in words:
            index.setdefault(w, set()).update(words - {w})
    return index


def expand_terms(terms: list[str], index: dict[str, set[str]]) -> list[str]:
    """Return ``terms`` followed by their alias words, order preserved and de-duplicated.

    The original terms always come first and are never dropped, so expansion can only add
    recall — it cannot change what the unexpanded query already matched. Additions stop at
    :data:`_MAX_ADDED_TERMS` so a pathological config cannot bloat the ``to_tsquery``.
    """
    if not index:
        return terms
    out = list(terms)
    seen = {t.lower() for t in terms}
    added = 0
    for term in terms:
        if added >= _MAX_ADDED_TERMS:
            break
        for alias in sorted(index.get(term.lower(), ())):  # sorted → deterministic query
            if alias not in seen:
                seen.add(alias)
                out.append(alias)
                added += 1
                if added >= _MAX_ADDED_TERMS:
                    break
    return out


def _selftest() -> int:
    """Run with ``python -m app.engine.rag.synonyms``. No DB, no network."""
    failures = 0

    def check(name: str, cond: bool) -> None:
        nonlocal failures
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        if not cond:
            failures += 1

    raw = [
        ["generator", "dg set", "power backup"],
        ["creche", "daycare", "child care"],
        ["x"],                 # too small — dropped
        "not a group",         # wrong type — dropped
        ["Bricks", "bricks", "RCC structure"],  # dedupe + case-fold
    ]
    groups = normalize_synonyms(raw)
    check("drops undersized/invalid groups", len(groups) == 3)
    check("case-folds and dedupes", groups[2] == ["bricks", "rcc structure"])

    index = build_alias_index(groups)
    # A single spoken word reaches every other word in its group, across member boundaries.
    check("generator bridges to backup", "backup" in index.get("generator", set()))
    check("generator bridges to dg", "dg" in index.get("generator", set()))
    check("daycare bridges to creche", "creche" in index.get("daycare", set()))
    check("bricks bridges to rcc", "rcc" in index.get("bricks", set()))
    # Groups stay isolated — "generator" must not leak into the creche group.
    check("groups isolated", "creche" not in index.get("generator", set()))

    expanded = expand_terms(["generator"], index)
    check("original term kept first", expanded[0] == "generator")
    check("expansion adds aliases", {"dg", "power", "backup"} <= set(expanded))
    check("no-index is identity", expand_terms(["generator"], {}) == ["generator"])
    check("empty raw -> no groups", normalize_synonyms(None) == [])

    print("SYNONYMS SELFTEST:", "PASS" if not failures else f"{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
