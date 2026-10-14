"""The platform-fallback customer-journey definition (Vocal Backend Tasksheet P0,
Generic Sprint 1) — used whenever a tenant has no industry pack assigned
(no tenant_pack_config row) and no unrecognised slug falls through to a registered
pack either. Deliberately far shorter than real_estate_india's 17-state journey:
this is a lowest-common-denominator sales funnel, not a tuned vertical flow.
"""

GENERIC_STATE_MACHINE: dict = {
    "initial": "new",
    "states": {
        "new":         {"goal": "make_first_contact"},
        "engaged":     {"goal": "capture_requirement"},
        "qualified":   {"goal": "match_offering_to_need"},
        "committed":   {"goal": "close_the_deal"},
        "closed":      {"goal": "handover_and_referrals", "sticky": True, "terminal": True},
        "lost":        {"goal": "none", "terminal": True},
    },
    "transitions": [
        {"from": ["new"], "to": "engaged", "on": ["system:call_connected"]},
        {"from": ["engaged"], "to": "qualified",
         "on": ["signal:requirement_captured", "signal:qualified"]},
        {"from": ["qualified"], "to": "committed",
         "on": ["signal:agreed", "tool:book_appointment"]},
        {"from": ["committed"], "to": "closed", "on": ["signal:converted"]},
        {"from": "*", "to": "lost", "on": ["signal:not_interested_final", "signal:dnc_request"]},
    ],
}
