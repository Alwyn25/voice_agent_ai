"""Generic currency formatting (Vocal Backend Tasksheet P0, Generic Sprint 1).

Real-estate-India's crore/lakh phrasing (app/services/money.py) is deliberately left
exactly where it is and exactly how it works — it backs a heavily-tuned budget-parsing
pipeline (parse_budget) that several planner prompt blocks depend on verbatim, and a
"move" that touched those call sites for zero behavioural gain would be pure risk. What
this module adds is the generic dispatcher the tasksheet actually asks for: a
locale-keyed `format_currency`/`speak_currency` pair that en_IN satisfies by delegating
to money.py, and en_US/en_GB/en_AE satisfy with plain thousands-grouping — so a new pack
never has to reach for India-specific parsing to format its own currency.
"""
from app.services import money

SUPPORTED_LOCALES = ("en_IN", "en_US", "en_GB", "en_AE")

_SYMBOLS = {"en_US": "$", "en_GB": "£", "en_AE": "AED "}


def format_currency(amount: int | float | None, locale: str = "en_IN") -> str | None:
    """The exact figure, grouped and symboled for `locale` — the written, not
    spoken-aloud, form. Use speak_currency() for what an agent should say."""
    if amount is None:
        return None
    if locale == "en_IN":
        grouped = money.indian_format(int(amount))
        return f"₹{grouped}" if grouped is not None else None
    symbol = _SYMBOLS.get(locale, "$")
    return f"{symbol}{amount:,.2f}" if isinstance(amount, float) and amount % 1 else f"{symbol}{int(amount):,}"


def speak_currency(amount: int | float | None, locale: str = "en_IN") -> str | None:
    """Rounded, spoken-aloud phrasing: "1.1 crore" for en_IN, "$1.2 million"
    elsewhere. Never speak the exact figure digit by digit on a call."""
    if amount is None:
        return None
    if locale == "en_IN":
        return money.crore_phrase(int(amount))
    symbol = _SYMBOLS.get(locale, "$")
    value = float(amount)
    if value >= 1_000_000:
        text = f"{value / 1_000_000:.2f}".rstrip("0").rstrip(".")
        return f"{symbol}{text} million"
    if value >= 1_000:
        text = f"{value / 1_000:.1f}".rstrip("0").rstrip(".")
        return f"{symbol}{text}k"
    return f"{symbol}{value:,.0f}"
