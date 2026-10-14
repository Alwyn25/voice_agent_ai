"""Customer journey definition, seeded per tenant into state_machine_defs.

Structure:
  initial: str
  states: {name: {goal, sticky?, terminal?, entry_actions?: [str], interrupting?: bool}}
  transitions: [{from: [str]|"*", to: str|"_return", on: [str], guard?: str}]

Trigger kinds: "tool:<name>", "signal:<name>", "system:<name>"
"_return" pops the customer's state stack (used by interrupting states).

REAL ESTATE JOURNEY
A property sale is a long, high-value, multi-visit process, so the middle of this journey
is longer than a service booking's. Two things worth knowing:

  - "site_visit_scheduled" and "site_visit_done" replace the generic appointment states,
    and they are driven by the same booking tools — a site visit IS an appointment against
    a project, so book_appointment/cancel_appointment carry it.
  - Nothing after "booking_token" is reversible by the agent. Documentation and conversion
    are human-confirmed steps; the agent can report on them but a signal has to come from
    outside to move them.
"""

DEFAULT_STATE_MACHINE: dict = {
    "initial": "lead",
    "states": {
        "lead":                 {"goal": "make_first_contact"},
        "contacted":            {"goal": "capture_requirement"},
        "qualified":            {"goal": "match_projects_to_budget"},
        "site_visit_scheduled": {"goal": "ensure_the_visit_happens",
                                 "entry_actions": ["schedule_reminders"]},
        "site_visit_done":      {"goal": "capture_feedback_and_shortlist"},
        "offer_shared":         {"goal": "get_a_decision_on_the_unit"},
        "negotiation":          {"goal": "close_on_price_and_terms"},
        "booking_token":        {"goal": "collect_token_and_hold_the_unit",
                                 "entry_actions": ["notify_supervisor"]},
        "documentation":        {"goal": "complete_agreement_and_registration"},
        "converted":            {"goal": "handover_and_referrals", "sticky": True},

        # Interrupting — these pause the journey and return to it afterwards.
        "call_later":           {"goal": "call_at_requested_time", "interrupting": True,
                                 "entry_actions": ["schedule_callback"]},
        "busy":                 {"goal": "retry_per_policy", "interrupting": True},
        "support_active":       {"goal": "resolve_issue", "interrupting": True},
        "complaint_raised":     {"goal": "resolve_complaint", "interrupting": True,
                                 "entry_actions": ["notify_supervisor"]},

        "dormant":              {"goal": "re_engage_when_inventory_changes"},
        "closed_lost":          {"goal": "none"},
        "do_not_contact":       {"goal": "none", "terminal": True},
    },
    "transitions": [
        {"from": ["lead"], "to": "contacted", "on": ["system:call_connected"]},
        {"from": ["contacted"], "to": "qualified",
         "on": ["signal:requirement_captured", "signal:qualified"]},

        # A site visit is an appointment booked against a project.
        {"from": ["lead", "contacted", "qualified", "site_visit_done", "dormant"],
         "to": "site_visit_scheduled", "on": ["tool:book_appointment"]},
        {"from": ["site_visit_scheduled"], "to": "site_visit_done",
         "on": ["signal:appointment_attended", "signal:site_visit_done"]},
        {"from": ["site_visit_scheduled"], "to": "qualified",
         "on": ["signal:no_show", "tool:cancel_appointment"]},

        {"from": ["site_visit_done", "qualified"], "to": "offer_shared",
         "on": ["signal:cost_sheet_shared", "signal:proposal_shared"]},
        {"from": ["offer_shared"], "to": "negotiation", "on": ["signal:negotiating"]},
        {"from": ["offer_shared", "negotiation"], "to": "booking_token",
         "on": ["signal:token_paid", "tool:collect_token"]},
        {"from": ["booking_token"], "to": "documentation", "on": ["signal:agreement_started"]},
        {"from": ["documentation"], "to": "converted",
         "on": ["signal:converted", "signal:registration_done"]},

        {"from": "*", "to": "call_later", "on": ["signal:call_me_later"]},
        {"from": "*", "to": "busy", "on": ["system:call_busy"]},
        {"from": "*", "to": "support_active", "on": ["tool:create_ticket"]},
        {"from": "*", "to": "complaint_raised", "on": ["signal:complaint"]},
        {"from": "*", "to": "dormant", "on": ["signal:gone_quiet"]},
        {"from": "*", "to": "closed_lost", "on": ["signal:not_interested_final"]},
        {"from": "*", "to": "do_not_contact", "on": ["signal:dnc_request"]},  # never guarded

        {"from": ["call_later"], "to": "_return", "on": ["system:callback_placed"]},
        {"from": ["busy"], "to": "_return", "on": ["system:call_connected"]},
        {"from": ["support_active"], "to": "_return", "on": ["signal:ticket_resolved"]},
        {"from": ["complaint_raised"], "to": "_return", "on": ["signal:complaint_resolved"]},
        {"from": ["dormant"], "to": "contacted", "on": ["system:call_connected"]},
    ],
}
