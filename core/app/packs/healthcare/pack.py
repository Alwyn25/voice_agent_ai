from app.packs.base import IndustryPack, LocaleConfig, PromptSections, Vocabulary
from app.packs.healthcare.state_machine import HEALTHCARE_STATE_MACHINE


class HealthcarePack(IndustryPack):
    """Vocal Backend Tasksheet P9 Generic Sprint 5: a minimal second vertical (on
    top of real_estate_india and generic) proving the pack abstraction actually
    generalises — a seed/demo stub, not a tuned production pack the way
    real_estate_india is. No custom prompt_sections (blank, like the generic pack)
    since nothing here has been tuned against real conversations yet."""
    slug = "healthcare"
    name = "Healthcare"

    @property
    def state_machine(self) -> dict:
        return HEALTHCARE_STATE_MACHINE

    @property
    def prompt_sections(self) -> PromptSections:
        return PromptSections()

    @property
    def vocabulary(self) -> Vocabulary:
        return Vocabulary(
            commitment_noun="consultation",
            conversion_action="booking",
            product_noun="service",
            product_noun_plural="services",
            discovery_question="What symptoms or concerns would you like to discuss?",
            technical_terms=["patient", "clinic", "consultation", "provider"],
            unit_patterns=[],
        )

    @property
    def locale(self) -> LocaleConfig:
        return LocaleConfig(code="en_US", currency_symbol="$")

    @property
    def conversion_state(self) -> str:
        return "closed"
