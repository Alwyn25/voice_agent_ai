"""Per-tenant STT keyword boosting (Deepgram's ``keyterm``, nova-3 only).

WHY THIS EXISTS. Deepgram transcribes by default against common English, so a proper
noun specific to one tenant -- a project name, a site address's locality -- is exactly
the kind of word it gets wrong, confidently, with no way to know it should have listened
harder for it. A real test proved this directly: synthesised audio saying "DSR Highland
Greenz" and "Chikkanayakanahalli" came back as "DSR Highland Greens" and "chicken naya
kanajali" with no boosting, and correctly both times with it -- same audio, same model,
the only difference was telling Deepgram these words were expected.

WHY IT IS NOT A HARDCODED LIST. This platform is not domain-specific -- a real estate
tenant's important words (project names, site localities) are nothing like a different
tenant's (clinic names, doctors, treatments). Hardcoding "BHK"/"crore" here would help
one tenant and do nothing for any other, in a codebase that otherwise keeps every
tenant's vocabulary in ITS OWN data, never in platform code. So this reads the words
from exactly where a tenant already keeps them -- their own name, their projects' names
and caller-facing aliases, and their visit locations -- fetched fresh per call, nothing
to maintain by hand, and automatically relevant to whichever tenant is calling.

DOMAIN WORDS COME FROM THE INDUSTRY PACK. Live calls later proved proper nouns are not the
whole problem: every real-estate voice call on record mis-heard the domain's own short words
("RERA" -> "rare"/"herrera", "3 BHK" -> "3 bsk", "possession" -> "position"), and the planner
acted on the wrong word. Those belong to the industry, not to one tenant, so they live on the
pack (Vocabulary.stt_keyterms) and are appended here -- still nothing hardcoded in this file.

Compatibility note: Deepgram's REST test (see the real connection check that motivated
this file) confirmed ``keyterm`` works alongside ``language=multi`` on ``model=nova-3``;
the older ``keywords`` parameter does not (HTTP 400) -- this module only ever builds a
``keyterm`` list, never ``keywords``.

The DB-querying half (``build_keyterms``) needs a live database and isn't exercised by
the self-test below; the cleanup/dedup logic it depends on (``split_location_name``,
``dedupe_and_cap``) is pure and IS covered -- run ``python -m app.engine.voice.keyterms``.
"""
import re
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Location, Project, Tenant

# Deepgram accepts up to 100 keyterms per request; capped well under that so the list
# stays made of genuinely distinctive words (the ones worth a model's limited attention)
# rather than padding with every short, common word a large tenant might have on file.
MAX_KEYTERMS = 40
# A keyterm shorter than this is almost always a common word ("ABC") that Deepgram already
# recognises reliably on its own -- boosting burns a slot without the payoff seen on genuine
# proper nouns in testing (place names, project names). Applies to the tenant's proper nouns
# only: the industry pack's curated terms (below) are exempt, because live calls proved the
# short DOMAIN acronyms are exactly what gets misheard ("BHK" -> "bsk", "RERA" -> "rare").
MIN_WORD_LEN = 4
# Domain terms (the call lexicon's pack vocabulary, derived action phrases and tenant
# intents -- app/engine/voice/lexicon.py) get their own budget on top of MAX_KEYTERMS, so a
# tenant with many projects can never crowd them out. 40 + 40 stays under Deepgram's 100.
MAX_PACK_KEYTERMS = 40

# project_facts labels (tenant.settings, as a GTM demo tenant keeps them -- it has no
# `projects` rows) whose values are spoken proper nouns worth boosting. Matched lowercase.
_FACT_NAME_LABELS = ("project name", "developer name", "location", "landmarks")
# A fact still waiting on the sales team reads "not published; ..." -- never a spoken name.
_UNPUBLISHED_RE = re.compile(r"^\s*not published\b", re.I)


def _words(phrase: str | None) -> list[str]:
    """The distinctive words in a phrase -- short common ones dropped. Used only to test
    whether a candidate term carries at least one word worth boosting; the term itself is
    kept whole (see dedupe_and_cap), not reduced to this list."""
    if not phrase:
        return []
    return [w for w in phrase.replace(",", " ").split() if len(w) >= MIN_WORD_LEN]


def split_location_name(name: str | None) -> list[str]:
    """A location's admin label is often not a spoken phrase -- "DSR Highland Greenz —
    Site Office" has an em-dash nobody says aloud. Each side of a dash/slash becomes its
    own clean, speakable term instead of one glued, unspeakable string."""
    if not name:
        return []
    return [seg.strip() for seg in re.split(r"[—–/]", name) if seg.strip()]


def dedupe_and_cap(terms: list[str]) -> list[str]:
    """Case-insensitive de-dup, drop anything with no word meeting MIN_WORD_LEN (so a lone
    "DSR" survives inside "DSR Infra Projects" but a term that is ONLY short words like "A
    B" does not), cap at MAX_KEYTERMS. Order-preserving, so earlier/more authoritative
    sources (tenant name, project name) win a slot before later ones if the cap is hit."""
    seen: set[str] = set()
    deduped: list[str] = []
    for term in terms:
        term = (term or "").strip()
        key = term.lower()
        if not key or key in seen or not _words(term):
            continue
        seen.add(key)
        deduped.append(term)
        if len(deduped) >= MAX_KEYTERMS:
            break
    return deduped


def fact_keyterms(facts) -> list[str]:
    """Spoken proper nouns from a tenant's curated ``project_facts`` (tenant.settings) --
    the project name, developer, locality and landmarks. A GTM demo tenant keeps its whole
    project here and has no ``projects`` rows, so without this its own project name was
    never boosted at all. Unpublished placeholders and malformed entries are skipped."""
    terms: list[str] = []
    for f in facts or []:
        if not isinstance(f, dict):
            continue
        label = str(f.get("label") or "").strip().lower()
        value = str(f.get("value") or "").strip()
        if label in _FACT_NAME_LABELS and value and not _UNPUBLISHED_RE.match(value):
            terms.extend(split_location_name(value))
    return terms


def with_pack_terms(proper_nouns: list[str], pack_terms: list[str] | None) -> list[str]:
    """Append the industry pack's curated terms (Vocabulary.stt_keyterms) after the
    tenant's proper nouns: de-duplicated against them and each other, capped at
    MAX_PACK_KEYTERMS, and exempt from MIN_WORD_LEN -- see that constant for why."""
    out = list(proper_nouns)
    seen = {t.lower() for t in out}
    added = 0
    for term in pack_terms or []:
        term = (term or "").strip()
        if not term or term.lower() in seen:
            continue
        seen.add(term.lower())
        out.append(term)
        added += 1
        if added >= MAX_PACK_KEYTERMS:
            break
    return out


async def build_keyterms(db: AsyncSession, tenant_id: uuid.UUID,
                         pack_terms: list[str] | None = None) -> list[str]:
    """This tenant's own proper nouns, fresh from their data, plus domain terms
    (``pack_terms``) -- nothing hardcoded, nothing tenant-specific in this file itself. Safe
    to call on every call setup: a handful of cheap, indexed queries, and an empty result
    (e.g. a brand-new tenant with no projects yet) simply means no boosting this call, not
    an error. Call setup uses app/engine/voice/lexicon.py, which builds the domain terms
    dynamically and reuses ``tenant_proper_nouns`` below.
    """
    return with_pack_terms(dedupe_and_cap(await tenant_proper_nouns(db, tenant_id)),
                           pack_terms)


async def tenant_proper_nouns(db: AsyncSession, tenant_id: uuid.UUID) -> list[str]:
    """The tenant's spoken proper nouns, in authority order, not yet de-duplicated or
    capped: its name, its project_facts names, its projects and their aliases, and its
    visit locations and cities."""
    terms: list[str] = []

    tenant_name, tenant_settings = (await db.execute(
        select(Tenant.name, Tenant.settings).where(Tenant.id == tenant_id))).one_or_none() \
        or (None, None)
    if tenant_name:
        terms.append(tenant_name)
    terms.extend(fact_keyterms((tenant_settings or {}).get("project_facts")))

    projects = (await db.execute(select(Project.name, Project.aliases).where(
        Project.tenant_id == tenant_id, Project.status == "active"))).all()
    for name, aliases in projects:
        if name:
            terms.append(name)
        terms.extend(a for a in (aliases or []) if a)

    locations = (await db.execute(select(Location.name, Location.city).where(
        Location.tenant_id == tenant_id, Location.is_active.is_(True)))).all()
    for name, city in locations:
        terms.extend(split_location_name(name))
        if city:
            terms.append(city)

    return terms


def _selftest() -> None:
    assert split_location_name("DSR Highland Greenz — Site Office") == [
        "DSR Highland Greenz", "Site Office"]
    assert split_location_name("DSR Techno Cube — Sales Gallery") == [
        "DSR Techno Cube", "Sales Gallery"]
    assert split_location_name("Downtown Clinic") == ["Downtown Clinic"]
    assert split_location_name(None) == []
    assert split_location_name("") == []

    # Case-insensitive de-dup, order preserved, original casing of the FIRST occurrence kept.
    assert dedupe_and_cap(["DSR Infra Projects", "dsr infra projects", "Lemon Tree"]) == [
        "DSR Infra Projects", "Lemon Tree"]

    # Short/generic terms (no word meeting MIN_WORD_LEN) are dropped entirely -- these would
    # burn a boosting slot on a word Deepgram already gets right unaided.
    assert dedupe_and_cap(["AB", "ok", "3 BHK", ""]) == []
    # ...but a real term is kept whole, not reduced to just its long words.
    assert dedupe_and_cap(["DSR Infra Projects"]) == ["DSR Infra Projects"]

    # Cap holds, earlier entries win the available slots.
    many = [f"Project{i:03d}word" for i in range(MAX_KEYTERMS + 10)]
    capped = dedupe_and_cap(many)
    assert len(capped) == MAX_KEYTERMS
    assert capped[0] == "Project000word"

    print("keyterms selftest: all passed")


if __name__ == "__main__":
    _selftest()
