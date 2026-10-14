"""Decide WHICH project a call is about — deterministically, before any model call.

The engine was one-project-per-tenant; a tenant with several developments needs every turn
scoped to one of them, or a Lemon Tree caller gets answered from a DSR cost sheet and "what
projects do you have?" returns one hardcoded set of facts. This module resolves the project
from the strongest signal available and PINS it to the conversation, so retrieval, the prompt
facts and any booking all agree for the rest of the call.

RESOLUTION ORDER (strongest, most deterministic first — the model never guesses here):

  1. Explicit mention — the caller named a project in THIS utterance ("about Highland
     Greenz"). Their own words win, and win even over an existing pin, because naming a
     different project is how a caller switches. Matched against name + aliases.
  2. Existing pin — conversation.project_id, set on an earlier turn. Resolve once, carry it.
  3. Dialled number — Call.to_number == projects.phone_did. The best signal for an unknown
     inbound caller when each project has its own line; null when they share a hotline.
  4. Caller history — a known caller's most recent visit's project. A returning buyer is
     almost always calling about the project they already engaged with.
  4b. Enquiry interest (OUTBOUND ONLY) — the project the lead actually enquired about
     (CustomerProductInterest, level "interested"). A marketing/service call WE placed is about
     the customer's stated interest, so leading with "which of our projects?" when their enquiry
     already names one is a needless cold question. Consulted only on an outbound call and only
     after 1–4 have missed, so inbound resolution runs the exact same queries it did before.
  5. Tenant default — the single project flagged is_default, if the tenant set one.

If none of these resolve AND the tenant has more than one active project, the result is
AMBIGUOUS: the agent must ASK which project rather than answer product facts from a mixed
pool. A tenant with zero or one active project never reaches that branch — with zero it is
tenant-wide (unchanged behaviour), with one that project is always the answer.

Only signals 1–4 (evidence the caller actually meant this project) PIN the conversation; a
bare tenant-default is used for the turn but not pinned, so a later real signal can still take
over. Mention re-pins whenever it fires.
"""
import re
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Appointment, Call, CustomerProductInterest, Project
from app.services.phone import normalize_phone

# A project_id that matches NO chunk, so retrieval passing it returns only tenant-wide
# (project_id IS NULL) documents. Used when the project is ambiguous: shared company-wide
# material is safe to surface, but no single project's cost sheet is, until the caller picks.
SHARED_ONLY_PROJECT_ID = uuid.UUID(int=0)

# Statuses that mean a visit still stands — the same set turn_context uses for "on file".
_LIVE_APPT_STATUSES = ("booked", "rescheduled")

# An alias/name shorter than this is too generic to match on as a bare substring (it would
# fire on ordinary words). Longer phrases are matched on word boundaries.
_MIN_MENTION_LEN = 4


@dataclass
class ProjectResolution:
    """The outcome of resolving a call to a project.

    ``project`` is the resolved project (or None). ``ambiguous`` is True only when the tenant
    has >1 active project and nothing selected one — the signal the agent must ask. ``pin`` is
    True when the resolution rests on real caller evidence and should be written to the
    conversation. ``signal`` names the winning rule, for the audit trail.
    """
    project: Project | None = None
    ambiguous: bool = False
    pin: bool = False
    signal: str = "none"
    candidates: list[Project] = field(default_factory=list)

    @property
    def project_id(self) -> uuid.UUID | None:
        return self.project.id if self.project else None

    @property
    def retrieval_project_id(self) -> uuid.UUID | None:
        """What to pass to retrieve(): the resolved project, or the shared-only sentinel when
        ambiguous (so a mixed pool is never surfaced), or None for a tenant-wide tenant."""
        if self.project is not None:
            return self.project.id
        if self.ambiguous:
            return SHARED_ONLY_PROJECT_ID
        return None


def project_location(project: Project) -> str | None:
    """This project's location, pulled from its curated facts — the value of the fact whose
    label names a place ("Location", "Address", "Where"). Used so the disambiguation question
    can tell projects apart by WHERE they are, not just by name. None if no such fact."""
    for f in (project.facts or []):
        if not isinstance(f, dict):
            continue
        label = str(f.get("label") or "").strip().lower()
        if any(w in label for w in ("location", "address", "where", "place")):
            value = str(f.get("value") or "").strip()
            if value:
                return value
    return None


def project_choice_label(project: Project) -> str:
    """How a project is offered to the caller when asking which one they mean: its name and,
    when known, its place — "DSR Highland Greenz (off Sarjapur Road, Bengaluru)"."""
    loc = project_location(project)
    return f"{project.name} ({loc})" if loc else project.name


def _mention_phrases(project: Project) -> list[str]:
    """The surface strings that count as naming this project: its name and declared aliases,
    plus the distinctive tail of the name (so "DSR Highland Greenz" is matched by "highland
    greenz"), all lowercased and de-duplicated."""
    phrases = {(project.name or "").strip().lower()}
    for a in (project.aliases or []):
        if isinstance(a, str) and a.strip():
            phrases.add(a.strip().lower())
    # Drop a leading developer prefix token from the name so the project's own words match
    # even when the caller omits the brand ("Highland Greenz" for "DSR Highland Greenz").
    words = (project.name or "").strip().lower().split()
    if len(words) >= 3:
        phrases.add(" ".join(words[1:]))
    return [p for p in phrases if len(p) >= _MIN_MENTION_LEN]


def match_project_mention(text: str, projects: list[Project]) -> Project | None:
    """The project the caller named in ``text``, or None. Pure — no DB, unit-testable.

    Matches a project's name/aliases on word boundaries. Returns a project only when EXACTLY
    one matches: if the utterance names two projects (or an ambiguous phrase hits several),
    the caller has to disambiguate, so this declines to guess.
    """
    low = f" {(text or '').lower()} "
    hits: list[Project] = []
    for project in projects:
        for phrase in _mention_phrases(project):
            if re.search(rf"\b{re.escape(phrase)}\b", low):
                hits.append(project)
                break
    return hits[0] if len(hits) == 1 else None


# --- softer selectors, used only among the OFFERED candidates when a clarify is open ---
#
# match_project_mention above needs the caller to actually name the project. But a caller
# answering "which one?" rarely repeats the whole name — they point at it: by its position in
# the two options they just heard ("the second one"), by the brand the name leads with ("DSR",
# which the mention matcher deliberately drops as too generic to scan for in free text), or by
# where it is ("the one off Sarjapur Road"). Each selector below returns a project ONLY when
# it picks out exactly one of the candidates, so it can select but never invent.

# Spoken ordinals → 1-based position, including the ASR misfire we actually see ("send" for
# "second"). Only positions 1–5 — a caller is choosing from a short spoken list, and a stray
# "third" against two options simply resolves to nothing.
#
# NOTE deliberately NO bare cardinals ("one", "two"): in "the second one" the word "one" is a
# NOUN, not the position, and mapping it to 1 turned every "the second one" into the ambiguous
# set {1, 2}. Digit tokens ("1", "2") are kept — a caller who says a digit in a clarify answer
# means the position, and a digit never plays the noun role "one" does.
_ORDINAL_WORDS: dict[str, int] = {
    "first": 1, "1st": 1, "1": 1,
    "second": 2, "2nd": 2, "2": 2, "send": 2, "seconds": 2,
    "third": 3, "3rd": 3, "3": 3,
    "fourth": 4, "4th": 4, "4": 4,
    "fifth": 5, "5th": 5, "5": 5,
}


def match_project_ordinal(text: str, projects: list[Project]) -> Project | None:
    """The project the caller pointed at BY POSITION in the offered list ("the second one",
    "the 2nd", "number one"), or None. ``projects`` must be in the SAME order the caller was
    offered them — which is the order resolve_project enumerates and project_candidates is
    built in. "last" resolves to the final option. Returns None unless exactly one position is
    named, so "the first or the second" stays ambiguous."""
    low = (text or "").lower()
    positions: set[int] = set()
    if re.search(r"\blast\b", low):
        positions.add(len(projects))
    for word in re.findall(r"[a-z0-9]+", low):
        pos = _ORDINAL_WORDS.get(word)
        if pos is not None:
            positions.add(pos)
    if len(positions) != 1:
        return None
    idx = positions.pop() - 1
    return projects[idx] if 0 <= idx < len(projects) else None


def match_project_prefix(text: str, projects: list[Project]) -> Project | None:
    """The project a leading brand token uniquely names ("DSR" → DSR Highland Greenz), or None.

    The mention matcher drops this token because scanning free text for a 3-letter brand would
    fire on ordinary words. Among the OFFERED candidates it is safe: a caller who says "DSR"
    when the two options are "DSR Highland Greenz" and "Lemon Tree" clearly means the first.
    Declines when the token is not the head word, is too short, or two candidates share it (two
    DSR projects → still ambiguous, as it must be)."""
    tokens = set(re.findall(r"[a-z0-9]+", (text or "").lower()))
    if not tokens:
        return None
    hits: list[Project] = []
    for project in projects:
        head = ((project.name or "").strip().lower().split() or [""])[0]
        if len(head) >= 3 and head in tokens:
            hits.append(project)
    return hits[0] if len(hits) == 1 else None


def match_project_location(text: str, projects: list[Project]) -> Project | None:
    """The project a place reference uniquely names ("the Sarjapur one", "the Whitefield
    project"), or None. Matches distinctive words from each project's curated location against
    the utterance; a place word shared by both (e.g. "Bengaluru") is ignored because it cannot
    tell them apart. Unique-match only."""
    low = f" {(text or '').lower()} "
    # Distinctive location words per project — drop generic/common tokens and any word that
    # appears in more than one project's location (it discriminates nothing).
    generic = {"road", "off", "main", "the", "bengaluru", "bangalore", "junction", "near",
               "bengaluru.", "cross", "layout", "phase", "block"}
    words_by_project: list[tuple[Project, set[str]]] = []
    seen: dict[str, int] = {}
    for project in projects:
        loc = (project_location(project) or "").lower()
        words = {w for w in re.findall(r"[a-z]+", loc) if len(w) >= 4 and w not in generic}
        words_by_project.append((project, words))
        for w in words:
            seen[w] = seen.get(w, 0) + 1
    hits: list[Project] = []
    for project, words in words_by_project:
        distinctive = {w for w in words if seen.get(w, 0) == 1}
        if any(re.search(rf"\b{re.escape(w)}\b", low) for w in distinctive):
            hits.append(project)
    return hits[0] if len(hits) == 1 else None


async def _active_projects(db: AsyncSession, tenant_id: uuid.UUID) -> list[Project]:
    return list((await db.execute(select(Project).where(
        Project.tenant_id == tenant_id, Project.status == "active")
        .order_by(Project.created_at.asc()))).scalars().all())


async def _project_by_did(db: AsyncSession, conversation_id: uuid.UUID,
                          projects: list[Project]) -> Project | None:
    """The project whose phone_did the caller dialled (Call.to_number), or None."""
    dialled = (await db.execute(
        select(Call.to_number).where(Call.conversation_id == conversation_id)
        .order_by(Call.created_at.desc()).limit(1))).scalars().first()
    if not dialled:
        return None
    want = normalize_phone(dialled)
    for project in projects:
        if project.phone_did and normalize_phone(project.phone_did) == want:
            return project
    return None


async def _project_from_history(db: AsyncSession, tenant_id: uuid.UUID,
                                customer_id: uuid.UUID,
                                by_id: dict[uuid.UUID, Project]) -> Project | None:
    """The project of the caller's most recent visit that still names one, or None."""
    rows = (await db.execute(select(Appointment.project_id).where(
        Appointment.tenant_id == tenant_id,
        Appointment.customer_id == customer_id,
        Appointment.project_id.isnot(None),
        Appointment.status.in_(_LIVE_APPT_STATUSES))
        .order_by(Appointment.starts_at.desc()))).scalars().all()
    for pid in rows:
        if pid in by_id:
            return by_id[pid]
    return None


# Interest levels that indicate a LIVE enquiry worth leading an outbound call with — the lead
# said they WANT it, not that they already bought ("purchased") or walked away ("rejected").
_LIVE_INTEREST_LEVELS = ("interested",)


async def _project_from_interest(db: AsyncSession, tenant_id: uuid.UUID,
                                 customer_id: uuid.UUID,
                                 by_id: dict[uuid.UUID, Project]
                                 ) -> tuple[Project, str | None] | None:
    """The customer's most recent LIVE product interest that names an active project, as a
    ``(project, product_ref)`` pair — or None.

    This is the enquiry signal behind an outbound call: a lead marked "interested" in a 3 BHK at
    DSR Highland Greenz is calling-worthy precisely about that project, and the "3 BHK" is what
    they enquired about. Ordered by most-recently recorded so a lead whose interest moved from
    one project to another leads with the latest. ``product_ref`` rides along so the outbound
    opener can recall and verify the stated unit before pitching; it may be an empty/whitespace
    string in older data, normalised to None here so callers need not re-check."""
    rows = (await db.execute(select(
        CustomerProductInterest.project_id, CustomerProductInterest.product_ref).where(
        CustomerProductInterest.tenant_id == tenant_id,
        CustomerProductInterest.customer_id == customer_id,
        CustomerProductInterest.interest_level.in_(_LIVE_INTEREST_LEVELS))
        .order_by(CustomerProductInterest.created_at.desc()))).all()
    for pid, product_ref in rows:
        if pid is not None and pid in by_id:
            return by_id[pid], ((product_ref or "").strip() or None)
    # FALLBACK — THE BUG THIS FIXES (P3-F1/F2/F3): every "interested" row on file has no
    # project_id at all, an incomplete/legacy record from before interest rows were linked
    # to a project. The enquiry itself is still real; only the project link is missing. The
    # old behaviour discarded it entirely here, so load_registered_interest returned None,
    # the outbound opener fell back to a generic "our DSR Highland Greenz project" line with
    # no unit mentioned, and "what was I looking for again?" had nothing to answer from —
    # the agent asked the CALLER to remind it instead. Recorded product_ref is still worth
    # recalling as long as there is exactly one project it could plausibly be about (the
    # tenant's default) — never guessed when more than one active project exists, same
    # restraint as signal 5 in resolve_project.
    unlinked = [product_ref for pid, product_ref in rows if pid is None]
    if unlinked:
        defaults = [p for p in by_id.values() if p.is_default]
        if len(defaults) == 1:
            return defaults[0], ((unlinked[0] or "").strip() or None)
    return None


async def load_registered_interest(db: AsyncSession, *, tenant_id: uuid.UUID,
                                   customer_id: uuid.UUID
                                   ) -> tuple[str, str | None] | None:
    """The interest THIS customer registered with us, as ``(project_name, unit)`` — or None.

    This is the DURABLE, resolution-independent read of the enquiry behind an outbound call: the
    project the lead is "interested" in and the unit ("3 BHK") they enquired about. Unlike the
    ``enquiry_interest`` project-resolution SIGNAL — which only wins on the first turns and is
    then shadowed by the conversation's project pin — this answers "what did I register interest
    in?" on EVERY turn of the call, so the agent can state it back whenever the caller asks
    rather than asking them to remind it. Caller (turn_context) gates it to OUTBOUND calls, so
    an inbound turn never runs this query and its cost/behaviour are unchanged.

    Never raises on the hot path: any failure degrades to None and the openers/prompt simply
    omit the recall, exactly as when no enquiry is on file."""
    projects = await _active_projects(db, tenant_id)
    if not projects:
        return None
    hit = await _project_from_interest(db, tenant_id, customer_id, {p.id: p for p in projects})
    if hit is None:
        return None
    project, unit = hit
    return project.name, unit


async def resolve_project(db: AsyncSession, *, tenant_id: uuid.UUID,
                          conversation, customer, user_input: str,
                          awaiting_project_choice: bool = False) -> ProjectResolution:
    """Resolve the call to a project. See module docstring for the signal order.

    ``awaiting_project_choice`` is True when the agent asked "which project?" on the previous
    turn and is now reading the answer. It unlocks the softer POSITIONAL selector (ordinals),
    which is only meaningful as a reply to that question — "the second one" mid-booking must
    not select a project. The prefix and location selectors are safe either way (they name a
    project), but are kept in the same clarify-answer branch for one clear resolution path.

    Never raises on the hot path: any lookup that fails degrades to the next signal.
    """
    projects = await _active_projects(db, tenant_id)
    if not projects:
        return ProjectResolution(signal="tenant_wide")          # nothing to resolve
    if len(projects) == 1:
        return ProjectResolution(project=projects[0], pin=True, signal="only_project")

    by_id = {p.id: p for p in projects}

    # 1 — explicit mention (wins even over a pin: this is how a caller switches project).
    mentioned = match_project_mention(user_input, projects)
    if mentioned is not None:
        return ProjectResolution(project=mentioned, pin=True, signal="mention")

    # 1b — the caller is ANSWERING the "which project?" question. Accept the softer ways a
    # person points at one of the two options they just heard: by position, by the brand the
    # name leads with, or by where it is. Each returns a project only when it uniquely picks
    # one candidate, so this can select but never guess. Tried in this order because position
    # is the most common spoken answer ("the second one") and the least ambiguous.
    if awaiting_project_choice:
        for selector, signal in ((match_project_ordinal, "ordinal"),
                                 (match_project_prefix, "prefix"),
                                 (match_project_location, "location")):
            picked = selector(user_input, projects)
            if picked is not None:
                return ProjectResolution(project=picked, pin=True, signal=signal)

    # 2 — existing pin from an earlier turn.
    pinned_id = getattr(conversation, "project_id", None)
    if pinned_id and pinned_id in by_id:
        return ProjectResolution(project=by_id[pinned_id], pin=False, signal="pinned")

    # 3 — dialled number.
    try:
        by_did = await _project_by_did(db, conversation.id, projects)
    except Exception:
        by_did = None
    if by_did is not None:
        return ProjectResolution(project=by_did, pin=True, signal="dialled_number")

    # 4 — the caller's own history.
    if customer is not None:
        try:
            from_hist = await _project_from_history(db, tenant_id, customer.id, by_id)
        except Exception:
            from_hist = None
        if from_hist is not None:
            return ProjectResolution(project=from_hist, pin=True, signal="caller_history")

    # 4b — enquiry interest, OUTBOUND ONLY. We placed this call about the lead's stated
    # interest; if their enquiry already names a project, lead with it instead of asking. Gated
    # on direction so the query NEVER runs for an inbound call — inbound resolution is unchanged
    # in both behaviour and cost. Runs after 1–4, so a mention / pin / dialled DID / real visit
    # still wins; pins on success, so it costs at most one query per outbound call.
    if customer is not None and getattr(conversation, "direction", None) == "outbound":
        try:
            from_interest = await _project_from_interest(db, tenant_id, customer.id, by_id)
        except Exception:
            from_interest = None
        if from_interest is not None:
            _proj, _ref = from_interest
            return ProjectResolution(project=_proj, pin=True, signal="enquiry_interest")

    # 5 — tenant default (used for the turn, but NOT pinned: a real signal later still wins).
    defaults = [p for p in projects if p.is_default]
    if len(defaults) == 1:
        return ProjectResolution(project=defaults[0], pin=False, signal="tenant_default")

    # Nothing selected one, and there are several — the agent must ask.
    return ProjectResolution(ambiguous=True, signal="ambiguous", candidates=projects)


def _selftest() -> int:
    """Pure checks on mention matching — no DB. Run with
    ``python -m app.engine.planner.project_resolution``."""
    failures = 0

    def check(name: str, cond: bool) -> None:
        nonlocal failures
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")
        if not cond:
            failures += 1

    class _P:
        def __init__(self, name, aliases=(), facts=()):
            self.name = name
            self.aliases = list(aliases)
            self.facts = list(facts)

    dsr = _P("DSR Highland Greenz", ["highland", "greenz"],
             [{"label": "Location", "value": "off Sarjapur Road, Bengaluru"}])
    lemon = _P("Lemon Tree", ["lemontree"],
               [{"label": "Location", "value": "Whitefield, Bengaluru"}])
    parkway = _P("DSR Parkway", [])
    projects = [dsr, lemon, parkway]

    check("matches full name", match_project_mention("tell me about DSR Highland Greenz", projects) is dsr)
    check("matches name tail", match_project_mention("is highland greenz ready?", projects) is dsr)
    check("matches alias", match_project_mention("what about greenz", projects) is dsr)
    check("matches other project", match_project_mention("i want lemon tree details", projects) is lemon)
    check("no mention -> None", match_project_mention("what is the price of a 3 bhk", projects) is None)
    check("two mentions -> None (ambiguous)",
          match_project_mention("compare lemon tree and highland", projects) is None)
    check("short/substring safe", match_project_mention("i have a treehouse", projects) is None)

    # Location-aware disambiguation labels.
    check("location pulled from facts", project_location(dsr) == "off Sarjapur Road, Bengaluru")
    check("no location -> None", project_location(parkway) is None)
    check("choice label carries place",
          project_choice_label(lemon) == "Lemon Tree (Whitefield, Bengaluru)")
    check("choice label without place is just the name",
          project_choice_label(parkway) == "DSR Parkway")

    # Softer clarify-answer selectors — the ways a caller points at one of the OFFERED options
    # without naming it. Order here is the order the caller heard them (Lemon first, as in the
    # reported call), so "the second one" must land on DSR.
    offered = [lemon, dsr]
    check("ordinal 'the second one' -> 2nd offered",
          match_project_ordinal("the second one", offered) is dsr)
    check("ordinal '2nd' -> 2nd offered", match_project_ordinal("2nd", offered) is dsr)
    check("ordinal ASR 'the send one' -> 2nd offered",
          match_project_ordinal("i want the send one mentioned", offered) is dsr)
    check("ordinal 'first' -> 1st offered", match_project_ordinal("the first please", offered) is lemon)
    check("ordinal 'last' -> final offered", match_project_ordinal("the last one", offered) is dsr)
    check("ordinal none named -> None", match_project_ordinal("what are the prices", offered) is None)
    check("ordinal two named -> None",
          match_project_ordinal("the first or the second", offered) is None)
    check("ordinal out of range -> None", match_project_ordinal("the fourth", offered) is None)

    # Brand prefix — bare "DSR" uniquely names DSR Highland Greenz here (Lemon Tree does not
    # lead with it). With two DSR projects the head word no longer discriminates.
    check("prefix 'DSR' -> DSR (unique head word)",
          match_project_prefix("DSR", offered) is dsr)
    check("prefix 'lemon' is not a 3+ head collision with dsr",
          match_project_prefix("dsr please", offered) is dsr)
    check("prefix shared head -> None",
          match_project_prefix("dsr", [dsr, parkway]) is None)
    check("prefix not head word -> None",
          match_project_prefix("greenz", offered) is None)   # 'greenz' is a tail, not the head

    # Location reference — a distinctive place word picks the project; a shared word (Bengaluru)
    # discriminates nothing and must not match.
    check("location 'Sarjapur' -> DSR",
          match_project_location("the one off Sarjapur road", offered) is dsr)
    check("location 'Whitefield' -> Lemon",
          match_project_location("the whitefield project", offered) is lemon)
    check("location shared word only -> None",
          match_project_location("the one in Bengaluru", offered) is None)
    check("location none named -> None",
          match_project_location("tell me the price", offered) is None)

    print("PROJECT RESOLUTION SELFTEST:", "PASS" if not failures else f"{failures} FAILURE(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
