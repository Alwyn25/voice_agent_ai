"""IndustryPack abstract base + the dataclasses a pack fills in.

A pack is the seam between the generic call-handling engine and one industry's
domain knowledge: its customer-journey state machine, the domain-specific chunks of
the planner's system prompt, the vocabulary customers actually hear, the schema of
its catalog-item attributes, and how it phrases money. Exactly one pack is active
per tenant (see app/packs/__init__.py's PackRegistry) — the engine itself never
branches on industry, it only ever asks the active pack.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PromptSections:
    """The RE-specific paragraphs extracted out of engine/planner/prompts.py's
    BASE_SYSTEM, one per named placeholder token it now contains. A pack that
    doesn't need a given section (e.g. a services business with no per-unit
    pricing table) leaves it "" and that part of the prompt is simply blank."""
    pricing_and_quoting: str = ""
    price_change_objection: str = ""
    scope_boundaries: str = ""
    booking_requirements: str = ""
    booking_sequence: str = ""


@dataclass(frozen=True)
class Vocabulary:
    """The nouns a caller actually hears, so the generic engine never has to say
    "BHK" or "site visit" itself. Wired into the planner's literals as of P6's
    Generic Sprint 3 (outbound_flow.py / preference.py).

    ``unit_patterns`` convention, established by that wiring (nothing consumed this
    field before P6, so this is a free choice, not a constraint inherited from
    elsewhere) — position-independent, so patterns of both kinds can appear in any
    order: a pattern WITH a capture group is a NUMERIC-configuration regex (the
    group captures the number) and canonicalises to "{number} {technical_terms[0]}"
    (see app.engine.planner.preference._canonical_unit); a pattern with NO capture
    group is a whole-match NAMED-unit regex and canonicalises to its own matched
    text, title-cased. A pack with no unit concept at all just leaves the list
    empty (unit detection is then honestly a no-op for that pack's tenants,
    matching "no stand-in for a feature that doesn't apply")."""
    commitment_noun: str = "commitment"
    conversion_action: str = "conversion"
    product_noun: str = "product"
    product_noun_plural: str = "products"
    discovery_question: str = "What are you looking for?"
    technical_terms: list[str] = field(default_factory=list)
    unit_patterns: list[str] = field(default_factory=list)
    # Domain words callers SAY that speech-to-text gets wrong without being told to expect
    # them -- boosted on every voice call of this pack's tenants (Deepgram ``keyterm``, see
    # app/engine/voice/keyterms.py). Curated, so it bypasses keyterms' length filter: short
    # acronyms are exactly what live calls mis-transcribed ("RERA" -> "rare"/"herrera",
    # "3 BHK" -> "3 bsk"). Empty for a pack with no such vocabulary.
    stt_keyterms: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LocaleConfig:
    code: str = "en_US"  # en_IN | en_US | en_GB | en_AE
    currency_symbol: str = "$"


GENERIC_DEFAULT_PERSONA = """You are a polite and efficient customer care agent.
Speak naturally and warmly. Be brief — this is a phone call, not an email. Never oversell.
If the customer is annoyed, acknowledge it before anything else.

Default to English unless the caller's own messages establish a different language; once
they do, mirror it back turn by turn, including natural code-mixing with English if that is
how they write. Keep numbers, prices and technical terms the way the caller themselves says
them, rather than translating them into a textbook form nobody actually speaks."""


class IndustryPack(ABC):
    """One concrete subclass per industry, registered once at import time via
    app.packs.PackRegistry.register(). Every property is read-only; a pack is
    configuration, not mutable state — per-tenant customisation is
    tenant_pack_config.overrides layered on top by the registry, never a pack
    subclass per tenant."""

    slug: str
    name: str

    @property
    @abstractmethod
    def state_machine(self) -> dict:
        """The default customer-journey definition (see
        app/engine/state_machine/machine.py's docstring for the dict shape) — used
        to seed a brand-new tenant and as machine.load_definition()'s fallback when
        no StateMachineDef row exists yet for a tenant."""

    @property
    def tools(self) -> list[str]:
        """Extra tool names this pack's tenants get on top of the engine's
        CORE_TOOLS (app/engine/tools/definitions.py's `tools_for_pack()`, wired in
        P7's Generic Sprint 4) — looked up in that module's PACK_TOOL_SPECS."""
        return []

    @property
    def conversion_state(self) -> str:
        """Vocal Backend Tasksheet P7 Generic Sprint 4: the state-machine state name
        this pack considers a successful conversion — used wherever the engine needs
        to count "converted" customers (app/api/analytics.py's overview) without
        hardcoding a real-estate-specific state name. Defaults to "converted" (the
        real-estate pack's own state, and the literal this replaces), so a pack that
        doesn't override it keeps today's exact behaviour; a pack whose journey ends
        somewhere else (the generic pack's "closed") overrides it."""
        return "converted"

    @property
    @abstractmethod
    def prompt_sections(self) -> PromptSections: ...

    @property
    def default_persona(self) -> str:
        return GENERIC_DEFAULT_PERSONA

    @property
    @abstractmethod
    def vocabulary(self) -> Vocabulary: ...

    @property
    def catalog_schema(self) -> dict:
        """Attribute-key declarations for this pack's catalog items (e.g. RE's
        bhk_type/sqft/facing/tower/rate_psf). Fully wired once P4's
        inventory→catalog rename lands; empty until then."""
        return {}

    @property
    @abstractmethod
    def locale(self) -> LocaleConfig: ...

    def format_currency(self, amount: int | float | None) -> str | None:
        """The exact figure, grouped and symboled for this pack's locale. See
        app/packs/locale.py for the shared per-locale formatters; a pack overrides
        this only if it needs something the shared formatters don't cover."""
        from app.packs.locale import format_currency as _format
        return _format(amount, self.locale.code)

    def speak_currency(self, amount: int | float | None) -> str | None:
        """The rounded, spoken-aloud phrasing ("1.1 crore", "$1.2 million")."""
        from app.packs.locale import speak_currency as _speak
        return _speak(amount, self.locale.code)
