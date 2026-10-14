"""Split a sheet row's people columns into contacts (docs/gtm-demo-links.md §4.2a).

The prospect workbook lists people as parallel comma lists — "Buyer name", "Buyer role",
"Email", "LinkedIn URL" — that often don't line up: a shared inbox (sales@, enquiries@)
sits among personal addresses, one list is longer than another, a LinkedIn URL lands in
the Email column. So the importer never pairs by position alone:

  * names, roles, LinkedIn URLs are paired by position only when the lists are the same
    length; otherwise LinkedIn URLs are matched to names by their slug, and roles are left
    for an admin to fill in.
  * emails are matched to a name by the name's tokens in the address ("manoj.gowda@" →
    Manoj Bache Gowda, "achauhan@" → Anurag Chauhan). Shared inboxes are kept as company
    emails, never assigned to a person. An address that matches no one, or more than one
    person, is reported, never guessed.
  * a contact without an email, or with anything uncertain, is marked needs_review so the
    admin confirms it before a link is generated.
"""
import re
from dataclasses import dataclass, field

_SPLIT_RE = re.compile(r"[,;\n]+")
_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+'-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_LINKEDIN_RE = re.compile(r"(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/in/([A-Za-z0-9_%-]+)", re.I)
_TOKEN_RE = re.compile(r"[a-z]+")

# Local parts of shared inboxes: a company address, not a person.
SHARED_INBOXES = {"info", "sales", "enquiries", "enquiry", "enquire", "marketing", "corporate",
                  "investors", "investor", "contact", "contactus", "hello", "support", "admin",
                  "crm", "care", "customercare", "bookings", "booking", "reservations", "office",
                  "mail", "team", "connect"}
WEBMAIL = {"gmail.com", "yahoo.com", "yahoo.co.in", "outlook.com", "hotmail.com",
           "rediffmail.com", "icloud.com"}


@dataclass
class ParsedContact:
    full_name: str
    first_name: str | None
    role: str | None = None
    email: str | None = None
    linkedin_url: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        return bool(self.notes)


@dataclass
class ParsedPeople:
    contacts: list[ParsedContact]
    company_emails: list[str]        # shared inboxes (sales@, info@, …)
    warnings: list[str]


def _split(cell: str | None) -> list[str]:
    return [p.strip() for p in _SPLIT_RE.split(cell or "") if p.strip()]


def tidy_name(name: str) -> str:
    """Collapse spaces; capitalise words typed all-lowercase ("pavan kumar Nelakonda" →
    "Pavan Kumar Nelakonda"). Words with any capital ("McKenzie", "S") are left alone."""
    words = name.split()
    return " ".join(w.capitalize() if w.islower() else w for w in words)


def _name_tokens(name: str) -> list[str]:
    return _TOKEN_RE.findall(name.lower())


def normalize_linkedin(text: str) -> str | None:
    m = _LINKEDIN_RE.search(text)
    return f"https://www.linkedin.com/in/{m.group(1)}/" if m else None


def _within_one_edit(a: str, b: str) -> bool:
    """True when a and b differ by at most one insertion, deletion or substitution."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = j = edits = 0
    while i < len(a) and j < len(b):
        if a[i] != b[j]:
            edits += 1
            if edits > 1:
                return False
            if len(a) == len(b):
                i += 1
            j += 1
        else:
            i += 1
            j += 1
    return edits + (len(b) - j) <= 1


def fuzzy_score(handle: str, name: str) -> int:
    """Second pass for spelling variants common in transliterated names ("yuvaraj" ~
    "Yuvraj", "santhosh" ~ "Santosh"): a handle part within one edit of a name token of 5+
    letters. Only used when the exact pass found no one, and always flagged for review."""
    tokens = [t for t in _name_tokens(name) if len(t) >= 5]
    parts = [p for p in re.split(r"[._\-+\d%]+", handle.lower()) if len(p) >= 5]
    return sum(2 for p in parts if any(_within_one_edit(p, t) for t in tokens))


def match_score(handle: str, name: str) -> int:
    """How strongly an email local part or LinkedIn slug points at this person; 0 = not at
    all. Scored rather than yes/no so "mamta.sharma" picks Mamta Sharma over Harish Sharma,
    and "sreedhar.g" picks Sreedhar G over S Sreedhar."""
    tokens = _name_tokens(name)
    if not tokens:
        return 0
    parts = [p for p in re.split(r"[._\-+\d%]+", handle.lower()) if p]
    joined = "".join(parts)
    first, last = tokens[0], tokens[-1]
    score = 0
    for p in parts:
        if p in tokens:
            score += 3 if len(p) >= 3 else 1                     # manoj.gowda, sreedhar.g
    if len(tokens) > 1:
        whole = {first + last, last + first}                     # parthgadhvi, gollapellitejesh
        initials = {first[0] + last, last + first[0], last[0] + first, first + last[0]}
        if joined in whole:
            score += 6
        elif len(last) >= 3 and len(first) >= 3 and joined in initials:
            score += 4                                           # achauhan, vkarthikeyan
    if score == 0 and len(first) >= 3 and joined.startswith(first):
        score = 2                                                # ritesh606
    return score


def best_match(handle: str, candidates: list["ParsedContact"], *,
               scorer=None) -> tuple[int | None, list[int]]:
    """Index of the single best-scoring candidate (None if no one, or a tie), plus every
    index that scored at all (for the warning)."""
    scorer = scorer or match_score
    scores = [(scorer(handle, c.full_name), i) for i, c in enumerate(candidates)]
    hits = [i for s, i in scores if s > 0]
    if not hits:
        return None, []
    top = max(s for s, _ in scores)
    winners = [i for s, i in scores if s == top]
    return (winners[0] if len(winners) == 1 else None), hits


def parse_people(names_cell: str | None, roles_cell: str | None, emails_cell: str | None,
                 linkedin_cell: str | None) -> ParsedPeople:
    warnings: list[str] = []
    contacts: list[ParsedContact] = []
    seen: set[str] = set()
    for raw in _split(names_cell):
        name = tidy_name(raw)
        if name.lower() in seen:
            warnings.append(f"duplicate name {name!r} ignored")
            continue
        seen.add(name.lower())
        tokens = name.split()
        contacts.append(ParsedContact(full_name=name, first_name=tokens[0] if tokens else None))

    # ── roles: by position only when the counts agree ──
    roles = _split(roles_cell)
    if roles and len(roles) == len(contacts):
        for c, r in zip(contacts, roles):
            c.role = r
    elif roles:
        warnings.append(f"{len(roles)} roles for {len(contacts)} names; roles left for review")
        for c in contacts:
            c.notes.append("role not paired (role and name lists differ in length)")

    # ── LinkedIn: by position when counts agree and the slug doesn't contradict, else by slug ──
    raw_links = [normalize_linkedin(x) for x in re.split(r"[,;\s]+", linkedin_cell or "") if x.strip()]
    links = [x for x in raw_links if x]
    # LinkedIn URLs pasted into the Email column are rescued, not treated as bad emails.
    email_tokens = _split(emails_cell)
    for tok in list(email_tokens):
        if "linkedin.com/in/" in tok.lower():
            email_tokens.remove(tok)
            if (u := normalize_linkedin(tok)) and u not in links:
                links.append(u)
            warnings.append("LinkedIn URL found in the Email column; moved to LinkedIn")
    if links:
        if len(links) == len(contacts):
            pairs = list(zip(contacts, links))
        else:
            pairs = []
            for url in links:
                slug = url.rstrip("/").rsplit("/", 1)[-1]
                free = [c for c in contacts if c.linkedin_url is None]
                win, hits = best_match(slug, free)
                if win is not None:
                    pairs.append((free[win], url))
                    free[win].linkedin_url = url     # claim now so a later URL can't take it
                else:
                    warnings.append(f"LinkedIn {url} matched {len(hits)} people equally; "
                                    "not assigned" if hits else
                                    f"LinkedIn {url} matches none of the listed people")
        for c, url in pairs:
            c.linkedin_url = url

    # ── emails: shared inboxes to the company, personal ones by name ──
    company_emails: list[str] = []
    claimed: dict[int, str] = {}
    fuzzy_paired: set[int] = set()
    for tok in email_tokens:
        addr = tok.strip().strip(".").lower()
        if not _EMAIL_RE.match(addr):
            warnings.append(f"not an email address: {tok!r}")
            continue
        local, domain = addr.split("@", 1)
        if local in SHARED_INBOXES:
            if addr not in company_emails:
                company_emails.append(addr)
            continue
        win, hits = best_match(local, contacts)
        fuzzy = False
        if win is None and not hits:
            win, hits = best_match(local, contacts, scorer=fuzzy_score)
            fuzzy = win is not None
        if win is not None:
            prev = claimed.get(win)
            # Prefer the company-domain address over a personal webmail one.
            if prev is None or (prev.split("@")[1] in WEBMAIL and domain not in WEBMAIL):
                claimed[win] = addr
                if fuzzy:
                    fuzzy_paired.add(win)
            else:
                warnings.append(f"{addr} also matches {contacts[win].full_name}; kept {prev}")
        elif not hits:
            warnings.append(f"email {addr} matches none of the listed people")
        else:
            warnings.append(f"email {addr} matches {len(hits)} people equally "
                            f"({', '.join(contacts[i].full_name for i in hits)}); not assigned")
    for i, addr in claimed.items():
        contacts[i].email = addr
        if i in fuzzy_paired:
            contacts[i].notes.append(f"email {addr} paired by similar spelling; please confirm")
        if addr.split("@")[1] in WEBMAIL:
            contacts[i].notes.append("personal webmail address; prefer a work email for outreach")
    for c in contacts:
        if not c.email:
            c.notes.append("no email matched")
    return ParsedPeople(contacts=contacts, company_emails=company_emails, warnings=warnings)
