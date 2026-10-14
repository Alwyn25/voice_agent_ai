"""Default rule sets seeded per tenant. Authored YAML — editable via the dashboard."""

CALLING_POLICY = """
rule_set: calling_policy
description: When the platform may place an outbound call.
applies_to: [outbound_call]
rules:
  - id: dnc_absolute
    priority: 0
    terminal: true
    when:
      any:
        - {field: customer.on_dnc_list, op: eq, value: true}
        - {field: customer.current_state, op: eq, value: do_not_contact}
    then:
      - block: {scope: all, reason: "Customer is on the Do-Not-Contact list"}

  - id: marketing_dnc
    priority: 5
    when:
      all:
        - {field: customer.on_dnc_marketing, op: eq, value: true}
        - {field: session.direction, op: eq, value: outbound}
        - {field: customer.current_state, op: not_in, value: [appointment_booked, support_active]}
    then:
      - block: {scope: marketing, reason: "Customer opted out of promotional calls"}

  - id: business_calling_hours
    priority: 10
    when:
      all:
        - {field: customer.has_contact_window, op: eq, value: false}
        - not:
            field: now.local_time
            op: between
            value: ["09:00", "21:00"]
    then:
      - block: {scope: outbound_call, reason: "Outside permitted calling hours (09:00-21:00)"}

  - id: customer_preferred_hours
    priority: 11
    when:
      all:
        - {field: customer.has_contact_window, op: eq, value: true}
        - not:
            field: now.local_time
            op: between
            value: ["$customer.preferred_contact_start", "$customer.preferred_contact_end"]
    then:
      - block: {scope: outbound_call, reason: "Outside the customer's preferred contact hours"}

  - id: daily_attempt_cap
    priority: 20
    when: {field: counters.attempts_today, op: gte, value: 3}
    then:
      - block: {scope: outbound_call, reason: "Daily call attempt limit reached"}

  - id: stop_after_repeated_no_answer
    priority: 21
    when: {field: counters.consecutive_no_answer, op: gte, value: 5}
    then:
      - block: {scope: outbound_call, reason: "Five consecutive unanswered calls"}

  - id: closed_lost_no_outreach
    priority: 30
    when: {field: customer.current_state, op: eq, value: closed_lost}
    then:
      - block: {scope: marketing, reason: "Customer marked closed-lost"}
"""

ACTION_POLICY = """
rule_set: action_policy
description: Guards applied before the agent may execute a tool.
applies_to: ["tool:*"]
rules:
  - id: no_actions_for_dnc
    priority: 0
    terminal: true
    when:
      all:
        - {field: customer.current_state, op: eq, value: do_not_contact}
        - {field: proposal.tool, op: not_in, value: [record_customer_fact, create_ticket]}
        - {field: session.direction, op: eq, value: outbound}
    then:
      - block: {scope: all, reason: "Customer is on Do-Not-Contact; only records may be updated"}

  - id: verify_identity_before_changes
    priority: 10
    when:
      all:
        - {field: proposal.tool, op: in, value: [reschedule_appointment, cancel_appointment]}
        - {field: session.identity_verified, op: eq, value: false}
        - {field: session.direction, op: eq, value: inbound}
    then:
      - require: {condition: identity_verified, prompt_hint: verify_identity_flow,
                  reason: "Verify the caller's identity before changing their appointments"}

  - id: verify_identity_for_profile_changes_inbound
    priority: 11
    when:
      all:
        - {field: proposal.tool, op: eq, value: update_customer_profile}
        - {field: session.identity_verified, op: eq, value: false}
        - {field: session.direction, op: eq, value: inbound}
    then:
      - require: {condition: identity_verified, prompt_hint: verify_identity_flow,
                  reason: "Verify the caller's identity before changing their details"}

  - id: no_booking_for_unknown_caller
    priority: 20
    when:
      all:
        - {field: proposal.tool, op: eq, value: book_appointment}
        - {field: customer.id, op: exists, value: false}
    then:
      - block: {scope: "tool:book_appointment", reason: "Customer must be identified before booking"}
"""

ESCALATION_POLICY = """
rule_set: escalation_policy
description: Situations that must reach a human.
applies_to: [turn, "tool:*"]
rules:
  - id: legal_or_payment_dispute
    priority: 0
    when:
      any:
        - {field: turn.intent, op: in, value: [legal_complaint, payment_dispute, regulator_threat]}
    then:
      - escalate: {type: immediate, reason: "Legal or payment dispute raised"}

  - id: vip_negative_sentiment
    priority: 10
    when:
      all:
        - {field: customer.tags, op: contains, value: vip}
        - {field: turn.sentiment, op: in, value: [negative, angry]}
    then:
      - escalate: {type: immediate, reason: "VIP customer with negative sentiment"}

  # Keyed SOLELY on the model's explicit human_handoff decision, not on a keyword match of
  # its free-form intent label. Inferring the handover from the intent string is what
  # transferred a caller who said "I'll come by in person" (the substring "person") in the
  # middle of a reschedule -- that was fixed by making the match word-boundary-aware rather
  # than substring, but the intent-label OR-branch itself remained, on the assumption
  # (recorded in confidence.py's own comments) that "a spurious match here can no longer
  # transfer one" since human_handoff was now the primary signal.
  #
  # THE BUG THIS FIXES: that assumption was live-disproven. A real call had the model
  # correctly set human_handoff_requested=False and action="respond" (reasoning: "customer
  # wants details via email") -- genuinely the right call, nothing about asking for an
  # email implies wanting a person -- but it ALSO free-text-labelled the turn's intent
  # "request_human" as a topic tag (this is ABOUT getting human-channel help, e.g. the
  # sales team's email), not a literal "transfer me now". normalize_intent's word-boundary
  # match still finds "human" inside "request_human" and resolves it to the canonical
  # request_human intent, and this rule's OR-branch fired on that alone, overriding the
  # model's own correct human_handoff=False decision and forcing an unwanted escalate.
  #
  # human_handoff_requested is the model's single, explicit, directly-asked-for decision on
  # this exact question ("does the caller want a person now?") -- every live case observed
  # had it right. The intent-label alias was meant as a safety net for the opposite failure
  # (missing a genuine request), but intent is a FREE-FORM descriptive tag the model uses
  # for topic/category purposes, not a commitment to "transfer now", so matching it against
  # the same question the boolean already answers just reintroduces the false-positive risk
  # the boolean was added to remove. Dropped; human_handoff is authoritative alone.
  - id: explicit_human_request
    priority: 15
    when:
      {field: turn.human_handoff, op: eq, value: true}
    then:
      - escalate: {type: immediate, reason: "Customer asked to speak with a person"}

  - id: repeated_failed_interactions
    priority: 20
    when: {field: counters.failed_interactions, op: gte, value: 4}
    then:
      - escalate: {type: scheduled, reason: "Repeated unsuccessful interactions"}

  - id: complaint_state
    priority: 25
    when: {field: customer.current_state, op: eq, value: complaint_raised}
    then:
      - escalate: {type: ticket, reason: "Open complaint on record"}
"""

APPOINTMENT_POLICY = """
rule_set: appointment_policy
description: Booking-specific constraints beyond the scheduling engine.
applies_to: ["tool:book_appointment", "tool:reschedule_appointment"]
rules:
  - id: no_weekend_booking_for_restricted_regions
    priority: 10
    when:
      all:
        - {field: customer.region_code, op: in, value: []}
        - {field: now.is_weekend, op: eq, value: true}
    then:
      - block: {scope: "tool:book_appointment", reason: "Weekend booking not available in this region"}

  - id: reschedule_limit
    priority: 20
    when: {field: counters.reschedules, op: gte, value: 3}
    then:
      - require: {condition: supervisor_approved,
                  reason: "More than three reschedules — supervisor approval required"}
"""

DEFAULT_RULE_SETS = [
    ("calling_policy", "calling_policy", CALLING_POLICY),
    ("action_policy", "action_policy", ACTION_POLICY),
    ("escalation_policy", "escalation", ESCALATION_POLICY),
    ("appointment_policy", "appointment", APPOINTMENT_POLICY),
]
