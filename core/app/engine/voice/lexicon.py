"""The call lexicon: every term THIS call's caller is likely to say that speech-to-text can
get wrong -- assembled per call from data, never hardcoded per industry.

WHY. Recorded calls showed the domain's own words misheard on every call ("RERA" ->
"rare"/"herrera", "possession" -> "position", "3 BHK" -> "3 bsk", "book a visit" -> "mokai
visit"/"bouquet visser"), with the planner acting on the wrong word. Patching one industry's
words does nothing for the next tenant -- a clinic's "consultation", a SaaS seller's "demo
call", a tenant's own custom intent like "loan enquiry". So the lexicon is derived from the
same sources the engine already uses to understand a tenant:

  1. the tenant's proper nouns           (keyterms.tenant_proper_nouns: name, projects,
                                          aliases, project_facts names, locations)
  2. the industry pack's vocabulary      (Vocabulary.stt_keyterms, technical_terms, the
                                          commitment / product nouns) -- with any per-tenant
                                          override from tenant_pack_config already applied
  3. action phrases derived from it      ("book a site visit", "book a visit", "reschedule
                                          my consultation", ...) -- what callers actually say
                                          to trigger a tool, built from the pack's own noun
  4. the tenant's custom intents         (intent_vocabulary: names + aliases) -- so a rule
                                          keyed on a tenant intent is as hearable as a core one

It is used twice: as Deepgram's ``keyterm`` boost list (so the words are heard right in the
first place) and as the reference set app/engine/voice/asr_hints.py matches each transcript
against (so a word still misheard is caught and flagged to the planner).
"""
import logging
import uuid
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.engine.voice.keyterms import dedupe_and_cap, tenant_proper_nouns, with_pack_terms

logger = logging.getLogger(__name__)

# Vocabulary's own placeholder defaults (app/packs/base.py). A pack that leaves them unset has
# no real noun to derive phrases from, so they never become lexicon terms.
_PLACEHOLDER_NOUNS = {"commitment", "conversion", "product", "products"}

# How callers ask for the engine's core actions, as templates over the pack's commitment
# noun. Engine-level (every pack has booking / rescheduling / cancelling tools), so this is
# the generic shape of the request, not any industry's wording.
_ACTION_TEMPLATES = ("book a {n}", "schedule a {n}", "reschedule my {n}", "cancel my {n}")


@dataclass
class CallLexicon:
    proper_nouns: list[str] = field(default_factory=list)
    domain_terms: list[str] = field(default_factory=list)

    def keyterms(self) -> list[str]:
        """Deepgram's keyterm list: proper nouns (length-filtered, capped) then domain terms
        in their own reserved budget -- see keyterms.with_pack_terms."""
        return with_pack_terms(dedupe_and_cap(self.proper_nouns), self.domain_terms)

    def terms(self) -> list[str]:
        """Every term, de-duplicated (case-insensitive), for transcript matching. Uncapped:
        matching is local and cheap, so a term that missed Deepgram's cap can still be
        caught after the fact."""
        return _dedupe(self.proper_nouns + self.domain_terms)


def _dedupe(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        t = " ".join(str(t or "").split())
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def action_phrases(commitment_noun: str | None) -> list[str]:
    """The ways a caller asks to book / move / cancel the pack's commitment ("site visit",
    "consultation", "demo call"). A multi-word noun also yields its head-noun form, because
    callers shorten it: "book a site visit" is just as often "book a visit"."""
    noun = " ".join(str(commitment_noun or "").lower().split())
    if not noun or noun in _PLACEHOLDER_NOUNS:
        return []
    nouns = [noun]
    head = noun.split()[-1]
    if head != noun:
        nouns.append(head)
    return [tpl.format(n=n) for n in nouns for tpl in _ACTION_TEMPLATES]


def domain_terms(vocabulary) -> list[str]:
    """The pack's vocabulary as lexicon terms: curated STT terms first (most-misheard
    order), then the action phrases, technical terms and nouns. Tolerates a missing
    vocabulary (no pack resolved) by returning nothing."""
    if vocabulary is None:
        return []
    nouns = [n for n in (getattr(vocabulary, "commitment_noun", None),
                         getattr(vocabulary, "product_noun", None),
                         getattr(vocabulary, "product_noun_plural", None))
             if n and str(n).lower() not in _PLACEHOLDER_NOUNS]
    return _dedupe([*(getattr(vocabulary, "stt_keyterms", None) or []),
                    *action_phrases(getattr(vocabulary, "commitment_noun", None)),
                    *(getattr(vocabulary, "technical_terms", None) or []),
                    *nouns])


def intent_terms(entries) -> list[str]:
    """A tenant's own custom intents (intent_vocabulary rows) as spoken terms: the intent
    name in words ("loan_enquiry" -> "loan enquiry") and each alias. Code-default intents
    are skipped -- their aliases are planner LABELS ("dnc", "opt_out"), not caller speech."""
    out: list[str] = []
    for e in entries or []:
        if getattr(e, "source", None) != "tenant":
            continue
        out.append(str(e.name or "").replace("_", " "))
        out.extend(str(a).replace("_", " ") for a in (e.aliases or []))
    return _dedupe(out)


async def build_call_lexicon(db: AsyncSession, tenant_id: uuid.UUID) -> CallLexicon:
    """Assemble this call's lexicon. Every source is best-effort: a failure in one just
    leaves its terms out -- a call is never refused or delayed for want of a lexicon. Each
    source runs in its own SAVEPOINT so a failed query cannot poison the call's long-lived
    session (the same guard retrieval uses)."""
    from app.custom_instructions.intents import load_vocabulary
    from app.packs import PackRegistry

    async def _safe(what: str, loader):
        try:
            async with db.begin_nested():
                return await loader()
        except Exception:
            logger.exception("call lexicon: %s unavailable (tenant=%s)", what, tenant_id)
            return None

    proper = await _safe("proper nouns", lambda: tenant_proper_nouns(db, tenant_id)) or []
    vocab = await _safe("pack vocabulary",
                        lambda: PackRegistry.vocabulary_for_tenant(db, tenant_id))
    intents = await _safe("intent vocabulary", lambda: load_vocabulary(db, tenant_id))
    return CallLexicon(proper_nouns=proper,
                       domain_terms=_dedupe(domain_terms(vocab) + intent_terms(intents)))
