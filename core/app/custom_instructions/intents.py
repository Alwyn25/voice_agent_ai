"""Intent vocabulary — the readable, versioned contract between the planner (which EMITS an
intent) and a routing rule (which KEYS on one). Phase 2, spec §4.3.

The planner labels intents freely ("loan", "loan_question", "finance_query"); a routing
rule that keys on ``turn.intent == loan_enquiry`` only fires if the planner's label maps to
exactly that. ``confidence.normalize_intent`` is the mapping, driven by ``INTENT_ALIASES``.
This module makes that alias set:

  * READABLE — ``BASE_INTENTS`` exposes the code defaults as a structured artifact the
    compiler grounds against and an admin can inspect;
  * EXTENSIBLE — a tenant adds intents (``intent_vocabulary`` rows) with the aliases the
    planner's label is normalised through, so a new routing intent fires reliably.

The compiler may ONLY key a rule on an intent present here; an unknown intent is returned to
the admin as a grounding gap ("add 'loan_enquiry' to the vocabulary first"), never silently
compiled onto a label the planner never produces.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import IntentVocabularyEntry
from app.engine.planner.confidence import INTENT_ALIASES, _needle_matches

# Human descriptions for the canonical code-default intents. Aliases come from
# INTENT_ALIASES (the live normalisation contract), so the two never drift.
_BASE_DESCRIPTIONS: dict[str, str] = {
    "request_human": "Caller asks to speak to a human, manager or supervisor.",
    "dnc_request": "Caller asks to stop being contacted / opt out.",
    "legal_complaint": "Caller raises a legal threat (lawyer, consumer court).",
    "payment_dispute": "Caller disputes a payment, refund or charge.",
    "complaint": "Caller raises a grievance or complaint.",
    "book_appointment": "Caller wants to book a site visit or appointment.",
    "reschedule": "Caller wants to move an existing appointment.",
    "cancel": "Caller wants to cancel an appointment.",
    "call_later": "Caller is busy / asks to be called back later.",
}


@dataclass
class IntentEntry:
    name: str
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    source: str = "default"                 # default | tenant


def base_intents() -> list[IntentEntry]:
    """The code-default canonical intents (from INTENT_ALIASES), as an artifact."""
    return [IntentEntry(name=name, description=_BASE_DESCRIPTIONS.get(name, ""),
                        aliases=list(needles), source="default")
            for name, needles in INTENT_ALIASES.items()]


async def load_vocabulary(db: AsyncSession, tenant_id: uuid.UUID) -> list[IntentEntry]:
    """The full vocabulary for a tenant: code defaults merged with active tenant rows.
    A tenant row with the same ``name`` overrides the default (lets a tenant enrich the
    aliases/description of a base intent)."""
    merged: dict[str, IntentEntry] = {e.name: e for e in base_intents()}
    rows = (await db.execute(select(IntentVocabularyEntry).where(
        IntentVocabularyEntry.tenant_id == tenant_id,
        IntentVocabularyEntry.is_active.is_(True)))).scalars().all()
    for r in rows:
        merged[r.name] = IntentEntry(
            name=r.name, description=(r.description or ""),
            aliases=list(r.aliases or []), examples=list(r.examples or []), source="tenant")
    return list(merged.values())


def extra_aliases(vocab: list[IntentEntry]) -> dict[str, list[str]]:
    """The alias map to hand ``normalize_intent`` so tenant intents normalise on the hot
    path. Base intents are already in INTENT_ALIASES; only tenant-sourced entries need
    injecting (a tenant override of a base intent is included so enriched aliases apply)."""
    return {e.name: e.aliases for e in vocab if e.source == "tenant" and e.aliases}


def names(vocab: list[IntentEntry]) -> list[str]:
    return [e.name for e in vocab]


def detect_from_text(text: str | None, vocab: list[IntentEntry]) -> str | None:
    """Deterministic safety net: the canonical name of the FIRST TENANT intent whose alias (or
    its own name) appears in the caller's utterance as whole word(s), else None.

    Why this exists: a routing rule keys on ``turn.intent``, but that label is produced by the
    planner LLM and is context-sensitive — mid-pitch on an OUTBOUND call the model may label a
    "what are the loan options?" turn as a product/pricing question and answer it, so the label
    never becomes ``loan_enquiry`` and the rule silently does not fire (the exact producer/
    consumer gap of spec §4.3). Matching the tenant's own aliases against what the caller
    actually SAID makes the rule fire on the request regardless of the model's labelling.

    Scoped to tenant intents only — base safety intents (request_human, dnc_request, …) keep
    their existing, stronger model-signal paths and are never overridden by a keyword here.
    Uses the same word-boundary matcher as ``normalize_intent`` so "loan" matches "loans" but
    not "sloane", and a multi-word alias like "home loan" needs both words in order.
    """
    low = (text or "").lower()
    if not low.strip():
        return None
    for e in vocab:
        if e.source != "tenant":
            continue
        needles = list(e.aliases or [])
        needles.append(e.name.replace("_", " "))  # the canonical name itself, spaced
        if any(_needle_matches(low, n) for n in needles):
            return e.name
    return None


def render_for_prompt(vocab: list[IntentEntry]) -> str:
    """A compact block listing every groundable intent, for the compiler's grounding."""
    lines = []
    for e in vocab:
        alias = f" (aka {', '.join(e.aliases[:6])})" if e.aliases else ""
        desc = f" — {e.description}" if e.description else ""
        lines.append(f"- {e.name}{alias}{desc}")
    return "\n".join(lines)
