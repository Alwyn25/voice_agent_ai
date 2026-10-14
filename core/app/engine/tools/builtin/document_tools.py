"""send_document — deliver a shareable document (floor plan, brochure, price sheet, …) to
the caller over WhatsApp.

TWO CONCERNS, KEPT APART:

  1. WHICH document the caller means. ``resolve_documents`` is a PURE function over
     document-like rows — a coarse doc_type filter plus a token match on title/description —
     so the whole selection is unit-tested without a database (see the self-check at the
     bottom). It never guesses: an empty match returns nothing, and the tool says so honestly
     rather than sending the wrong file.

  2. HOW it is delivered. That is the messaging provider (app/services/messaging), chosen from
     config. Whether a real WhatsApp send happens or the stub simulates one, this tool calls it
     the same way and reports what the provider said.

RECIPIENT is never asked for. Asking a caller to recite the number they are calling from is the
invariant this project fights hardest (see never-ask-caller-for-own-number): the destination is
taken from the customer's record, then the caller-ID resolved onto the session, and only an
explicit ``to_phone`` the caller volunteered overrides that.

DELIVERY STATUS is honest in the record even when the spoken line says "sent": the stub returns
``simulated``, which lands verbatim in the tool_executions audit row, so an operator can always
tell a rehearsed send from a real one.
"""
from __future__ import annotations

import re

from sqlalchemy import or_, select

from app.db.models import ShareableDocument
from app.engine.tools.types import ToolContext
from app.services.messaging import OutboundDocument, get_whatsapp_provider
from app.services.messaging.base import STATUS_FAILED
from app.services.phone import normalize_phone

DEFAULT_LIMIT = 1
MAX_LIMIT = 5

# Words that carry no signal for matching a document by name — dropped before token overlap so
# "can you send me the floor plan" scores on "floor"/"plan", not on "send"/"me"/"the".
_STOPWORDS = {
    "the", "a", "an", "me", "my", "your", "please", "can", "you", "could", "would",
    "send", "share", "whatsapp", "document", "documents", "doc", "docs", "file", "files",
    "over", "to", "of", "for", "on", "it", "that", "this", "and", "with", "get", "want",
    "have", "show", "give", "us", "our", "their",
}


def _tokens(*parts: object) -> set[str]:
    text = " ".join(str(p or "") for p in parts).lower()
    return {t for t in re.split(r"[^a-z0-9]+", text) if t and t not in _STOPWORDS}


def _query_tokens(text: object) -> set[str]:
    """Query tokens, keeping a bare configuration like '3bhk' aligned to how titles read
    ('3 BHK'): split the digits from 'bhk' so '3bhk' contributes both '3' and 'bhk'."""
    normalised = re.sub(r"(\d(?:\.\d)?)\s*bhk", r"\1 bhk", str(text or "").lower())
    return _tokens(normalised)


def canonical_doc_type(value: object) -> str | None:
    """Loose wording -> a ShareableDocument.doc_type. 'floorplan'/'floor plan' -> 'floor_plan',
    'price'/'cost sheet' -> 'price_sheet'. None when nothing recognisable is named — the tool
    then leans on the query text alone rather than filtering to a guess."""
    if not value:
        return None
    key = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    aliases = {
        "floorplan": "floor_plan", "floor_plan": "floor_plan", "floor_plans": "floor_plan",
        "layout": "floor_plan", "plan": "floor_plan",
        "brochure": "brochure", "brochures": "brochure", "e_brochure": "brochure",
        "price": "price_sheet", "price_sheet": "price_sheet", "price_list": "price_sheet",
        "pricing": "price_sheet", "cost_sheet": "cost_sheet", "costsheet": "cost_sheet",
        "payment_plan": "payment_plan", "payment_schedule": "payment_plan",
        "payment": "payment_plan",
        "location": "location_map", "location_map": "location_map", "map": "location_map",
        "amenities": "amenities", "master_plan": "master_plan", "masterplan": "master_plan",
        "specification": "specification", "specifications": "specification",
        "specs": "specification", "spec": "specification",
        "legal": "legal", "agreement": "agreement", "offer": "offer",
    }
    return aliases.get(key)


def _shape(d) -> dict:
    """One document as the fields the tool reports. ``id`` feeds the belongs-to-tenant
    postcondition and is never spoken; the rest is what the agent may read back."""
    return {"id": str(d.id), "title": d.title, "doc_type": d.doc_type}


def resolve_documents(rows, *, doc_type: str | None, query: object,
                      limit: int = DEFAULT_LIMIT) -> tuple[list, list]:
    """Pick the document(s) the caller means, best first. Returns (selected, other_matches).

    Pure and DB-free so it is exhaustively testable. Ranking, highest wins:
      * a doc_type filter, when given, is a HARD narrow — only rows of that type survive it
        (unless it eliminates everything, in which case it is treated as a soft signal so a
        mis-typed category never silently sends nothing when a clear title match exists);
      * within the survivors, score by how many query tokens hit the title/description/type;
      * a doc_type-only request (no query) keeps the type's rows in a stable order.
    A row with zero signal (no type match, no token overlap) is never selected.
    """
    active = [d for d in rows if getattr(d, "is_active", True)]
    q_tokens = _query_tokens(query)

    typed = [d for d in active if doc_type and d.doc_type == doc_type]
    pool = typed if typed else active
    # If a doc_type was asked for and matched some rows, that filter already IS the signal — a
    # type-only request ("send the brochure") should return those even with no query tokens.
    type_satisfied = bool(typed)

    scored = []
    for d in pool:
        overlap = len(q_tokens & _tokens(d.title, d.description, d.doc_type.replace("_", " ")))
        # Base credit when the hard type filter matched, so type-only requests rank above noise.
        score = overlap + (2 if (doc_type and d.doc_type == doc_type) else 0)
        if score <= 0 and not (type_satisfied and not q_tokens):
            continue
        scored.append((score, d))

    # Stable: preserve input order (seed/priority order) among equal scores.
    scored.sort(key=lambda s: s[0], reverse=True)
    ordered = [d for _, d in scored]

    n = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
    return ordered[:n], ordered[n:]


def _mask(phone: str) -> str:
    digits = re.sub(r"\D", "", phone or "")
    return f"ending {digits[-4:]}" if len(digits) >= 4 else "on file"


def _recipient(ctx: ToolContext, params: dict) -> str | None:
    """The number to deliver to, strongest first, WITHOUT asking the caller:
      1. an explicit to_phone the caller volunteered (a different number they gave);
      2. the identified customer's own number;
      3. the caller-ID resolved onto this session / the durable conversation record.
    """
    explicit = (params.get("to_phone") or "").strip()
    if explicit and len(re.sub(r"\D", "", explicit)) >= 8:
        return normalize_phone(explicit)
    if ctx.customer is not None and getattr(ctx.customer, "phone_normalized", None):
        return ctx.customer.phone_normalized
    caller = (ctx.extra or {}).get("caller_phone")
    return normalize_phone(caller) if caller else None


async def send_document(ctx: ToolContext, params: dict) -> dict:
    doc_type = canonical_doc_type(params.get("document_type"))
    query = params.get("query")
    if not doc_type and not (query and str(query).strip()):
        # Nothing to resolve against. Returned, not raised — the agent should ask which
        # document, not apologise for a system fault.
        return {"sent": False, "found": False, "needs": "document",
                "guidance": ("No document was named. Ask the caller which document they want "
                             "(floor plan, brochure, price sheet, …) before sending.")}

    # Scope to this project plus any tenant-wide documents, mirroring retrieval and inventory.
    q = select(ShareableDocument).where(
        ShareableDocument.tenant_id == ctx.tenant_id,
        ShareableDocument.is_active.is_(True))
    if ctx.project_id is not None:
        q = q.where(or_(ShareableDocument.project_id == ctx.project_id,
                        ShareableDocument.project_id.is_(None)))
    rows = list((await ctx.db.execute(q)).scalars().all())

    selected, others = resolve_documents(
        rows, doc_type=doc_type, query=query, limit=int(params.get("limit") or DEFAULT_LIMIT))

    if not selected:
        return {"sent": False, "found": False, "query": query, "document_type": doc_type,
                "available": sorted({d.title for d in rows})[:8],
                "guidance": ("No matching document is on file for this project. Do NOT invent "
                             "one or claim it was sent. Tell the caller plainly you don't have "
                             "that document to hand; if the list of available titles helps, "
                             "offer one of those, or offer a site visit where they can see it.")}

    to = _recipient(ctx, params)
    if not to:
        # No number from any source. This is an upstream gap (caller-ID should have reached the
        # session); do not ask the caller to recite their own number — say a human will follow up.
        return {"sent": False, "found": True, "documents": [_shape(d) for d in selected],
                "guidance": ("The document was found but there is no phone number on file to "
                             "send it to. Do NOT ask the caller to read out their own number. "
                             "Tell them the team will send it across shortly.")}

    provider = get_whatsapp_provider()
    sent_docs: list[dict] = []
    failures: list[str] = []
    last_status = None
    for d in selected:
        result = await provider.send_document(to=to, document=OutboundDocument(
            url=d.file_url,
            file_name=d.file_name or f"{d.title}.pdf",
            caption=d.title,
            mime_type=d.mime_type))
        last_status = result.status
        if result.ok:
            sent_docs.append({**_shape(d), "delivery_status": result.status,
                              "provider_message_id": result.provider_message_id})
        else:
            failures.append(result.error or "delivery failed")

    if not sent_docs:
        # Every attempt failed at the provider — a real error, reported honestly.
        return {"sent": False, "found": True, "delivery_status": STATUS_FAILED,
                "documents": [_shape(d) for d in selected], "error": "; ".join(failures)[:300],
                "guidance": ("The document could not be sent right now (a delivery error, not "
                             "the caller's fault). Tell them plainly it didn't go through and "
                             "that the team will follow up — do NOT claim it was sent.")}

    other_titles = [d.title for d in others]
    return {
        "sent": True,
        "found": True,
        "channel": "WhatsApp",
        # The tool reports 'documents' (the audited, postcondition-checked list) and a spoken
        # summary the responder grounds on.
        "documents": sent_docs,
        "sent_titles": [d["title"] for d in sent_docs],
        "to_masked": _mask(to),
        "delivery_status": last_status,
        "other_available": other_titles,
        "partial_failures": failures or None,
        "guidance": (
            f"Confirm to the caller that the document"
            f"{'s' if len(sent_docs) > 1 else ''} "
            f"({', '.join(d['title'] for d in sent_docs)}) "
            f"{'have' if len(sent_docs) > 1 else 'has'} been sent to their WhatsApp "
            f"on the number {_mask(to)}. Keep it to one short line."
            + (f" If they want more, these are also available: {', '.join(other_titles[:4])}."
               if other_titles else "")),
    }


# --- self-check -------------------------------------------------------------------------
if __name__ == "__main__":
    import uuid as _uuid
    from dataclasses import dataclass, field

    @dataclass
    class D:
        title: str
        doc_type: str
        description: str | None = None
        is_active: bool = True
        id: _uuid.UUID = field(default_factory=_uuid.uuid4)

    ROWS = [
        D("DSR Highland Greenz — 2 BHK Floor Plan", "floor_plan", "2 bedroom layout"),
        D("DSR Highland Greenz — 3 BHK Floor Plan", "floor_plan", "3 bedroom layout"),
        D("DSR Highland Greenz Brochure", "brochure", "the full project brochure"),
        D("DSR Price Sheet", "price_sheet", "current cost sheet and rates"),
        D("DSR Location Map", "location_map", "how to reach the site"),
        D("Retired 3 BHK Plan", "floor_plan", "old", is_active=False),
    ]

    failures = 0

    def check(label, cond):
        global failures
        print(("ok  " if cond else "FAIL") + "  " + label)
        if not cond:
            failures += 1

    # The scenario in the brief: "can you send the floor plan first?" then narrowed to 3 BHK.
    sel, _ = resolve_documents(ROWS, doc_type="floor_plan", query="3 bhk floor plan")
    check("3 BHK floor plan wins", sel and sel[0].title.endswith("3 BHK Floor Plan"))

    # Type-only request with no configuration returns the type's docs, not nothing.
    sel, other = resolve_documents(ROWS, doc_type="floor_plan", query=None, limit=1)
    check("floor-plan-only returns a floor plan", sel and sel[0].doc_type == "floor_plan")
    check("floor-plan-only offers the other as alternative",
          any(d.doc_type == "floor_plan" for d in other))

    # Free-text with no type still resolves via title/description tokens.
    sel, _ = resolve_documents(ROWS, doc_type=None, query="send me the brochure please")
    check("brochure by text", sel and sel[0].doc_type == "brochure")

    sel, _ = resolve_documents(ROWS, doc_type=None, query="what is the price")
    check("price by text", sel and sel[0].doc_type == "price_sheet")

    # A retired document is never selected.
    sel, other = resolve_documents(ROWS, doc_type="floor_plan", query="old plan", limit=5)
    check("retired doc excluded", all(d.is_active for d in sel + other))

    # No signal at all -> no selection (the tool then asks / declines honestly).
    sel, _ = resolve_documents(ROWS, doc_type=None, query="do you have a swimming pool")
    check("no-signal query selects nothing", sel == [])

    # A mistyped/unknown type falls back to title matching rather than sending nothing.
    sel, _ = resolve_documents(ROWS, doc_type="nonexistent_type", query="brochure")
    check("unknown type falls back to text match", sel and sel[0].doc_type == "brochure")

    # canonicalisation
    check("canonical floorplan", canonical_doc_type("floor plan") == "floor_plan")
    check("canonical price", canonical_doc_type("price list") == "price_sheet")
    check("canonical unknown", canonical_doc_type("swimming pool") is None)

    print("\nDOCUMENT SELFTEST:", "PASS" if not failures else f"{failures} FAILURE(S)")
    import sys
    sys.exit(1 if failures else 0)
