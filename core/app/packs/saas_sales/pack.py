from app.packs.base import IndustryPack, LocaleConfig, PromptSections, Vocabulary
from app.packs.saas_sales.state_machine import SAAS_SALES_STATE_MACHINE


class SaasSalesPack(IndustryPack):
    """Vocal Backend Tasksheet P9 Generic Sprint 5 — same seed-stub status as
    app.packs.healthcare.pack.HealthcarePack."""
    slug = "saas_sales"
    name = "SaaS Sales"

    @property
    def state_machine(self) -> dict:
        return SAAS_SALES_STATE_MACHINE

    @property
    def prompt_sections(self) -> PromptSections:
        return PromptSections()

    @property
    def vocabulary(self) -> Vocabulary:
        return Vocabulary(
            commitment_noun="demo call",
            conversion_action="signup",
            product_noun="plan",
            product_noun_plural="plans",
            discovery_question="What are you hoping to solve with our product?",
            technical_terms=["trial", "seat", "contract", "onboarding"],
            unit_patterns=[],
        )

    @property
    def locale(self) -> LocaleConfig:
        return LocaleConfig(code="en_US", currency_symbol="$")

    @property
    def conversion_state(self) -> str:
        return "closed_won"
