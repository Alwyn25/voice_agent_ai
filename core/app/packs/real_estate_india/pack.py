from app.packs.base import IndustryPack, LocaleConfig, PromptSections, Vocabulary
from app.packs.real_estate_india import prompts as re_prompts
from app.packs.real_estate_india.state_machine import DEFAULT_STATE_MACHINE


class RealEstateIndiaPack(IndustryPack):
    slug = "real_estate_india"
    name = "Real Estate (India)"

    @property
    def state_machine(self) -> dict:
        return DEFAULT_STATE_MACHINE

    @property
    def prompt_sections(self) -> PromptSections:
        return PromptSections(
            pricing_and_quoting=re_prompts.PRICING_AND_QUOTING,
            price_change_objection=re_prompts.PRICE_CHANGE_OBJECTION,
            scope_boundaries=re_prompts.SCOPE_BOUNDARIES,
            booking_requirements=re_prompts.BOOKING_REQUIREMENTS,
            booking_sequence=re_prompts.BOOKING_SEQUENCE,
        )

    @property
    def default_persona(self) -> str:
        return re_prompts.PERSONA

    @property
    def vocabulary(self) -> Vocabulary:
        return Vocabulary(
            commitment_noun="site visit",
            conversion_action="booking",
            product_noun="project",
            product_noun_plural="projects",
            discovery_question="What configuration are you looking for?",
            technical_terms=["BHK", "floor plan", "token amount", "sqft", "carpet area"],
            # Vocal Backend Tasksheet P6 Generic Sprint 3 (app/engine/planner/preference.py):
            # by convention (documented on Vocabulary.unit_patterns itself), pattern [0] is the
            # NUMERIC-configuration regex — exactly the old module-level `_BHK_RE`, one capture
            # group for the number, paired with technical_terms[0] ("BHK") as its suffix — and
            # every later pattern is a whole-match NAMED-unit regex, exactly the old
            # `_NAMED_UNITS` tuple. Kept byte-identical to the pre-Sprint-3 behaviour.
            unit_patterns=[r"\b(\d(?:\.\d)?)\s*(?:bhk|bedroom(?:s)?|bed)\b",
                          r"\b(?:studio|penthouse|duplex|villa|plot|showflat)\b"],
            # Every voice call on record mis-heard these: RERA as "rare"/"herrera"/"raidra",
            # possession as "position" (routed to a booking-status lookup), BHK as "bsk",
            # "book a visit" as "bouquet visite". Ordered most-misheard first, in case a
            # tenant's own proper nouns leave little room under Deepgram's keyterm cap.
            stt_keyterms=["RERA", "BHK", "possession", "site visit", "carpet area",
                          "floor plan", "token amount", "occupancy certificate", "OC",
                          "sqft", "crore", "lakh", "amenities", "clubhouse"],
        )

    @property
    def locale(self) -> LocaleConfig:
        return LocaleConfig(code="en_IN", currency_symbol="₹")

    @property
    def tools(self) -> list[str]:
        # inventory_query's description/schema (bhk_type, sqft, tower, rate_psf, …) is
        # real-estate-only — see app/engine/tools/definitions.py's own module docstring.
        return ["inventory_query"]

    # conversion_state left at the base class's "converted" default — this pack's own
    # state machine literally has a "converted" state, so there is nothing to override.
