"""Vocal Backend Tasksheet P9 Generic Sprint 5 — a minimal healthcare-vertical
customer journey (patient/consultation/clinic), deliberately as short as the
generic pack's own journey: this is a seed stub proving the pack abstraction
genuinely supports a second vertical, not a tuned production flow the way
real_estate_india's 17-state journey is.
"""

HEALTHCARE_STATE_MACHINE: dict = {
    "initial": "new",
    "states": {
        "new":          {"goal": "make_first_contact"},
        "scheduled":    {"goal": "confirm_the_consultation",
                         "entry_actions": ["schedule_reminders"]},
        "consulted":    {"goal": "capture_outcome_and_next_steps"},
        "follow_up":    {"goal": "check_on_recovery_or_next_steps"},
        "closed":       {"goal": "none", "sticky": True, "terminal": True},
        "no_show":      {"goal": "none", "terminal": True},
    },
    "transitions": [
        {"from": ["new"], "to": "scheduled",
         "on": ["signal:consultation_booked", "tool:book_appointment"]},
        {"from": ["scheduled"], "to": "consulted", "on": ["signal:consultation_completed"]},
        {"from": ["scheduled"], "to": "no_show", "on": ["signal:consultation_missed"]},
        {"from": ["consulted"], "to": "follow_up", "on": ["signal:follow_up_needed"]},
        {"from": ["consulted", "follow_up"], "to": "closed", "on": ["signal:care_complete"]},
        {"from": "*", "to": "closed", "on": ["signal:not_interested_final", "signal:dnc_request"]},
    ],
}
