"""Inventory lookup as an explicit tool — the structured answer to price/budget/size
questions, so the sentence-writer never has to do arithmetic over prose.

WHY A TOOL. See app/db/models/inventory.py. A caller asking "anything around 1.1 cr" is
stating a numeric band; the honest answer is a comparison against the real cost-sheet rows
(which units sit near it, and where the next one jumps to). This tool does that comparison
deterministically and returns FACTS — the responder grounds on them like any other
VERIFIED RESULT, and a result-correctness postcondition (inventory_units_match_query) proves
the rows it returned actually satisfy the constraint. No model number-crunching anywhere.

The numeric selection lives in `build_result`, a PURE function over unit-like objects, so it
is unit-tested end to end without a database (see the self-check at the bottom). `inventory_query`
is only the DB-loading, project-scoping shell around it.
"""
import re
import uuid

from sqlalchemy import or_, select

from app.db.models import Unit
from app.engine.tools.types import ToolContext
from app.services.money import Budget, crore_phrase, parse_budget

DEFAULT_LIMIT = 6


def canonical_bhk(value) -> str | None:
    """"3bhk", "3 bhk", "3-BHK", "2.5bhk" -> "3 BHK" / "2.5 BHK". None when no config is named.

    Matches how seed_units.py stores bhk_type, so a caller's loose wording lines up with the
    row. Returns None (not a guess) when there is no recognisable "<n> BHK" in the text.
    """
    if not value:
        return None
    low = str(value).lower()
    m = re.search(r"(\d(?:\.\d)?)\s*(?:-|\s)?\s*bhk", low)
    if not m:
        # A bare number the planner put in a bhk field ("3") — accept only a small integer/half.
        m2 = re.fullmatch(r"\s*(\d(?:\.\d)?)\s*", low)
        if not m2:
            return None
        num = m2.group(1)
    else:
        num = m.group(1)
    num = num.rstrip("0").rstrip(".") if "." in num else num   # "2.0" -> "2"
    return f"{num} BHK"


def _shape(u) -> dict:
    """One unit as speakable facts. `id`/`price_exact` are for the correctness check and are
    never spoken; `price_phrase` (rounded, e.g. "1.1 crore") is the only price field handed to
    the responder — a raw digit-grouped rupee figure ("1,09,85,999") used to sit alongside it
    here as `price_inr` and got read out verbatim on calls instead of the rounded phrase, so it
    is deliberately not exposed to the responder at all now."""
    return {
        "id": str(u.id),
        "bhk_type": u.bhk_type,
        "sqft": u.super_builtup_sqft,
        "facing": u.facing,
        "tower": u.tower,
        "price_phrase": crore_phrase(u.all_in_total),
        "price_exact": u.all_in_total,
        "availability": u.availability,
    }


def _closest(units, target, *, below):
    """The priced unit closest to `target` on one side, or None. `below` True → the dearest at
    or under target; False → the cheapest at or over it."""
    side = [u for u in units if (u.all_in_total <= target if below else u.all_in_total >= target)]
    if not side:
        return None
    return min(side, key=lambda u: abs(u.all_in_total - target))


def budget_from_params(params: dict) -> Budget | None:
    """A Budget from either the free-text `budget` field (parsed deterministically) or the
    structured min_price/max_price. The text path is preferred — it carries "around"/"under"
    intent the two numbers cannot."""
    parsed = parse_budget(params.get("budget"))
    if parsed is not None:
        return parsed
    lo = params.get("min_price")
    hi = params.get("max_price")
    lo = int(lo) if isinstance(lo, (int, float)) and lo > 0 else None
    hi = int(hi) if isinstance(hi, (int, float)) and hi > 0 else None
    if lo and hi:
        return Budget(kind="range", minimum=min(lo, hi), maximum=max(lo, hi), raw="")
    if hi:
        return Budget(kind="max", maximum=hi, raw="")
    if lo:
        return Budget(kind="min", minimum=lo, raw="")
    return None


def build_result(rows, *, config: str | None, budget: Budget | None,
                 sort: str | None = None, limit: int = DEFAULT_LIMIT) -> dict:
    """The pure comparison: given the project's unit rows (already scoped and sold-filtered),
    the asked configuration and budget, return the speakable facts. No DB, no I/O — the whole
    numeric decision, so it can be tested exhaustively without a database."""
    if config:
        rows = [u for u in rows if u.bhk_type.lower() == config.lower()]

    unpriced_configs = sorted({u.bhk_type for u in rows
                               if not u.price_published or u.all_in_total is None})
    priced = sorted([u for u in rows if u.price_published and u.all_in_total is not None],
                    key=lambda u: u.all_in_total)

    # The asked-for configuration exists but nothing under it is priced — the 1 BHK / Lemon
    # Tree case. Answer honestly; do NOT fall through to a nearest-price search.
    if config and not priced:
        exists = bool(rows)
        return {
            "config": config, "budget": budget.as_dict() if budget else None,
            "price_published": False, "count": 0, "units": [], "cheapest": None,
            "nearest_below": None, "nearest_above": None,
            "unpriced_configs": unpriced_configs, "applied_constraint": {},
            "guidance": (
                f"{config} exists at this project but no cost sheet is published for it, so "
                f"there is no price to quote. Say that plainly, do not estimate one from "
                f"another configuration's rate, and offer a site visit or a callback where the "
                f"sales team gives current pricing."
                if exists else
                f"There is no {config} at this project. Say so plainly and, if useful, mention "
                f"the configurations that do exist.")}

    cheapest = priced[0] if priced else None
    dearest = priced[-1] if priced else None
    matches = priced
    nearest_below = nearest_above = None
    applied: dict = {}
    guidance = ""

    if budget is not None:
        if budget.kind == "max":
            applied = {"max": budget.maximum}
            matches = [u for u in priced if u.all_in_total <= budget.maximum]
            over = [u for u in priced if u.all_in_total > budget.maximum]
            nearest_above = over[0] if over else None
            cap = crore_phrase(budget.maximum)
            if matches:
                guidance = (f"These are within {cap}. Lead with the closest to their ceiling."
                            + (f" The next one up is {crore_phrase(nearest_above.all_in_total)} "
                               f"if they can stretch." if nearest_above else ""))
            else:
                guidance = (f"Nothing published is within {cap}. The entry point is "
                            f"{crore_phrase(cheapest.all_in_total)} — say so honestly "
                            f"and do NOT imply something cheaper exists."
                            if cheapest else f"No published units to compare against {cap}.")
        elif budget.kind == "min":
            applied = {"min": budget.minimum}
            matches = [u for u in priced if u.all_in_total >= budget.minimum]
            floor = crore_phrase(budget.minimum)
            guidance = (f"These are at or above {floor}, cheapest first."
                        if matches else
                        f"Nothing published reaches {floor}; the dearest is "
                        f"{crore_phrase(dearest.all_in_total)}." if dearest else "")
        elif budget.kind == "range":
            applied = {"min": budget.minimum, "max": budget.maximum}
            matches = [u for u in priced
                       if budget.minimum <= u.all_in_total <= budget.maximum]
            lo, hi = crore_phrase(budget.minimum), crore_phrase(budget.maximum)
            below = _closest(priced, budget.minimum, below=True)
            above = _closest(priced, budget.maximum, below=False)
            guidance = (f"These fall between {lo} and {hi}."
                        if matches else
                        f"Nothing published falls between {lo} and {hi}. "
                        + (f"Closest below is {crore_phrase(below.all_in_total)}; " if below else "")
                        + (f"closest above is {crore_phrase(above.all_in_total)}." if above else ""))
        else:  # point — "around X"
            target = budget.target
            applied = {"min": budget.minimum, "max": budget.maximum}
            matches = [u for u in priced
                       if budget.minimum <= u.all_in_total <= budget.maximum]
            nearest_below = _closest(priced, target, below=True)
            nearest_above = _closest(priced, target, below=False)
            tphrase = crore_phrase(target)
            if matches:
                guidance = f"These sit near {tphrase}. Offer them."
                # Name the jump only when the next unit above is OUTSIDE the band they asked
                # around — that gap is the honest, useful thing to tell them.
                if nearest_above and nearest_above.all_in_total > budget.maximum:
                    guidance += (f" The next size up jumps to "
                                 f"{crore_phrase(nearest_above.all_in_total)}, so there's a gap "
                                 f"above what they asked for.")
            else:
                near = nearest_below or nearest_above
                guidance = (f"Nothing is priced near {tphrase}. The closest is "
                            f"{crore_phrase(near.all_in_total)} — be honest about the "
                            f"gap rather than pretending it fits." if near else
                            f"No published units to compare against {tphrase}.")
    else:
        if str(sort or "").lower() == "price_desc":
            matches = list(reversed(priced))
        guidance = ("Cheapest first. Give the range or the specific one they asked for; do not "
                    "read out the whole list.") if priced else "No published prices to give."

    shown = matches[:max(1, int(limit))]
    return {
        "config": config,
        "budget": budget.as_dict() if budget else None,
        "price_published": bool(priced),
        "count": len(matches),
        "units": [_shape(u) for u in shown],
        "cheapest": _shape(cheapest) if cheapest else None,
        "nearest_below": _shape(nearest_below) if nearest_below else None,
        "nearest_above": _shape(nearest_above) if nearest_above else None,
        "unpriced_configs": unpriced_configs,
        # Consumed by the inventory_units_match_query postcondition, not spoken.
        "applied_constraint": applied,
        "guidance": guidance,
    }


async def inventory_query(ctx: ToolContext, params: dict) -> dict:
    # Project scope, mirroring retrieval: this project's own units plus any tenant-wide ones.
    q = select(Unit).where(Unit.tenant_id == ctx.tenant_id)
    if ctx.project_id is not None:
        q = q.where(or_(Unit.project_id == ctx.project_id, Unit.project_id.is_(None)))
    rows = list((await ctx.db.execute(q)).scalars().all())
    # A caller cannot buy a sold unit; 'hold' is still shown (it may free up and is useful for
    # a price answer).
    rows = [u for u in rows if u.availability != "sold"]

    return build_result(
        rows,
        config=canonical_bhk(params.get("bhk_type")),
        budget=budget_from_params(params),
        sort=params.get("sort"),
        limit=int(params.get("limit") or DEFAULT_LIMIT))


# --- self-check -------------------------------------------------------------------------
if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class U:
        bhk_type: str
        all_in_total: int | None
        super_builtup_sqft: int | None = None
        facing: str | None = None
        tower: str | None = None
        availability: str = "available"
        price_published: bool = True
        id: uuid.UUID = None  # type: ignore

        def __post_init__(self):
            self.id = uuid.uuid4()
            if self.all_in_total is None:
                self.price_published = False

    # The real DSR cost-sheet totals.
    ROWS = [
        U("2 BHK", 10393750, 875), U("2 BHK", 10450000, 880),
        U("2 BHK", 12913750, 1099), U("2 BHK", 12925000, 1100),
        U("2 BHK", 12958750, 1103), U("2 BHK", 12970000, 1104),
        U("2 BHK", 12981250, 1105), U("2 BHK", 13768750, 1175),
        U("2 BHK", 14162500, 1210),
        U("2.5 BHK", 15531250, 1325), U("2.5 BHK", 15587500, 1330),
        U("3 BHK", 15432011, 1239), U("3 BHK", 17206110, 1390), U("3 BHK", 19253440, 1560),
        U("1 BHK", None, 619), U("1 BHK", None, 646), U("1.5 BHK", None, 780),
    ]

    failures = 0

    def check(label, cond):
        global failures
        print(("ok  " if cond else "FAIL") + "  " + label)
        if not cond:
            failures += 1

    def q(text, config=None):
        return build_result(ROWS, config=canonical_bhk(config), budget=parse_budget(text))

    # THE BUG: "around 1.1 cr" must surface the ~1.04 cr 2 BHKs and flag the jump to ~1.29 cr,
    # never "nothing under a crore".
    r = q("do you have anything around 1.1 cr")
    prices = {u["price_exact"] for u in r["units"]}
    check("around 1.1cr returns the two ~1.04cr units", prices == {10393750, 10450000})
    check("around 1.1cr nearest_above is the ~1.29cr unit",
          r["nearest_above"]["price_exact"] == 12913750)
    check("around 1.1cr guidance names the jump", "jumps to" in r["guidance"])
    check("around 1.1cr is NOT the 'nothing under a crore' script",
          "nothing under" not in r["guidance"].lower())

    # "under 1 crore" — nothing qualifies; honest entry point, no invented cheaper unit.
    r = q("anything under 1 crore")
    check("under 1cr has no matches", r["count"] == 0)
    check("under 1cr states the real entry point",
          "1.04 crore" in r["guidance"] and "entry point" in r["guidance"])

    # cheapest 3 BHK
    r = build_result(ROWS, config="3 BHK", budget=None, sort="price_asc")
    check("cheapest 3 BHK first is 1.54cr", r["cheapest"]["price_exact"] == 15432011)
    check("3 BHK count is 3", r["count"] == 3)

    # between 1.2 and 1.4 cr
    r = q("between 1.2 and 1.4 crore")
    check("range 1.2-1.4 matches the mid 2 BHKs",
          all(12000000 <= u["price_exact"] <= 14000000 for u in r["units"]) and r["count"] >= 4)

    # 1 BHK — exists but unpriced: honest, no price, no fallback to 2 BHK rate.
    r = build_result(ROWS, config="1 BHK", budget=None)
    check("1 BHK price_published False", r["price_published"] is False)
    check("1 BHK guidance says not published", "no cost sheet is published" in r["guidance"])
    check("1 BHK returns no priced units", r["units"] == [])

    # constraint recorded for the correctness check
    r = q("around 1.1 cr")
    check("applied_constraint carries the band",
          r["applied_constraint"].get("max") == 12650000)

    # config canonicalisation
    check("canonical 3bhk", canonical_bhk("3bhk") == "3 BHK")
    check("canonical 2.5-BHK", canonical_bhk("2.5-BHK") == "2.5 BHK")
    check("canonical bare 2", canonical_bhk("2") == "2 BHK")
    check("canonical none", canonical_bhk("do you have a garden") is None)

    print("\nINVENTORY SELFTEST:", "PASS" if not failures else f"{failures} FAILURE(S)")
    import sys
    sys.exit(1 if failures else 0)
