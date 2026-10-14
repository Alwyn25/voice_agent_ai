from app.packs.base import IndustryPack, LocaleConfig, PromptSections, Vocabulary
from app.packs.generic.state_machine import GENERIC_STATE_MACHINE


class GenericPack(IndustryPack):
    """The platform default for a tenant with no industry pack assigned — a
    lowest-common-denominator sales funnel with no domain-specific prompt content
    at all (every PromptSections field is blank, so BASE_SYSTEM's engine-owned
    rules alone govern the call)."""
    slug = "generic"
    name = "Generic"

    @property
    def state_machine(self) -> dict:
        return GENERIC_STATE_MACHINE

    @property
    def prompt_sections(self) -> PromptSections:
        return PromptSections()

    @property
    def vocabulary(self) -> Vocabulary:
        return Vocabulary()

    @property
    def locale(self) -> LocaleConfig:
        return LocaleConfig(code="en_US", currency_symbol="$")

    @property
    def conversion_state(self) -> str:
        # GENERIC_STATE_MACHINE's own terminal success state is "closed", not "converted".
        return "closed"
