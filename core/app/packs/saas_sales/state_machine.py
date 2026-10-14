"""Vocal Backend Tasksheet P9 Generic Sprint 5 — a minimal SaaS-sales-vertical
customer journey (demo/trial/contract), same "seed stub, not a tuned production
flow" status as app/packs/healthcare/state_machine.py.
"""

SAAS_SALES_STATE_MACHINE: dict = {
    "initial": "new",
    "states": {
        "new":              {"goal": "make_first_contact"},
        "demo_scheduled":   {"goal": "confirm_the_demo", "entry_actions": ["schedule_reminders"]},
        "trial_active":     {"goal": "drive_activation"},
        "contract_sent":    {"goal": "get_a_decision", "entry_actions": ["notify_supervisor"]},
        "closed_won":       {"goal": "none", "sticky": True, "terminal": True},
        "closed_lost":      {"goal": "none", "terminal": True},
    },
    "transitions": [
        {"from": ["new"], "to": "demo_scheduled",
         "on": ["signal:demo_booked", "tool:book_appointment"]},
        {"from": ["demo_scheduled"], "to": "trial_active", "on": ["signal:trial_started"]},
        {"from": ["trial_active"], "to": "contract_sent", "on": ["signal:contract_sent"]},
        {"from": ["contract_sent"], "to": "closed_won", "on": ["signal:converted"]},
        {"from": "*", "to": "closed_lost", "on": ["signal:not_interested_final", "signal:dnc_request"]},
    ],
}
