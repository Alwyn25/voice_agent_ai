"""Built-in tool contracts, seeded per tenant into tool_definitions.

These are DATA: a pack (or the dashboard) can edit them without engine changes.

Vocal Backend Tasksheet P7 Generic Sprint 4 (`pack.register_tools()`): every tenant used
to get all 17 of these regardless of industry pack, including `inventory_query`, whose
description/schema (bhk_type, sqft, tower, rate_psf, …) is real-estate-only — a
generic-pack tenant would see a tool it can't sensibly use. `CORE_TOOLS`/`PACK_TOOL_SPECS`
below split that out WITHOUT renaming or restructuring `inventory_query` itself. `BUILTIN_TOOLS`
itself is left untouched, still all 17, since scripts/apply_inventory_update.py still reads the
whole list unfiltered.
"""

BUILTIN_TOOLS: list[dict] = [
    {
        "name": "search_slots",
        "category": "scheduling",
        "description": ("Find available appointment slots at a centre. Use before offering "
                        "times to the customer. Never invent times — only offer what this returns. "
                        "centre_id must be a UUID: if you do not already have one, call "
                        "locate_centre FIRST and use the centre_id it returns. If the customer "
                        "named a day, pass date_from AND date_to as that same date. If they "
                        "named a time of day or an hour, pass part_of_day or time_from/time_to "
                        "— do not filter the results yourself and do not ask them to pick from "
                        "a list that does not contain what they asked for."),
        "input_schema": {
            "type": "object",
            # date_from used to be REQUIRED with only "YYYY-MM-DD" as guidance. With no date
            # from the customer the model had to invent one, and it reliably picked a week
            # out — so a caller who said nothing about timing was offered slots seven days
            # away while tomorrow was wide open. Optional, with an explicit instruction to
            # omit it, is the fix; the tool defaults to today.
            "required": ["centre_id"],
            "properties": {
                "centre_id": {"type": "string", "description": (
                    "The centre_id returned by locate_centre. A centre name is accepted "
                    "and resolved, but the id is preferred.")},
                "date_from": {"type": "string", "description": (
                    "YYYY-MM-DD. Omit unless the customer named a specific day — leaving it "
                    "out searches from today onwards. But if they DID name a day, you must "
                    "send it, and send date_to as the same date: without it the search "
                    "starts today and you will read back the wrong day. Never guess a date.")},
                "date_to": {"type": "string", "description": (
                    "YYYY-MM-DD. Set this equal to date_from when the customer named one "
                    "specific day. Omit only when they gave no day at all.")},
                # Added because the customer asking for "the evening" had no way through
                # this contract at all: the tool returned the first few slots of the day and
                # the agent could only ever offer the morning, however many times she asked.
                "part_of_day": {"type": "string", "enum": ["morning", "afternoon", "evening"],
                                "description": (
                    "Use when the customer says morning / afternoon / evening rather than "
                    "an hour. Evening means from 4 PM onwards.")},
                "time_from": {"type": "string", "description": (
                    "Earliest acceptable start, e.g. \"16:00\" or \"4pm\". Use when the "
                    "customer named an hour.")},
                "time_to": {"type": "string", "description": (
                    "Latest acceptable start, e.g. \"18:00\". Pair with time_from for a "
                    "range; send time_from alone for \"after four\".")},
                "executive_id": {"type": "string", "format": "uuid"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 12},
            },
            "additionalProperties": False,
        },
        "allowed_states": [],
        "preconditions": [{"check": "centre_exists", "args": {"centre_id": "$param.centre_id"}}],
        "postconditions": [
            {"check": "slots_are_valid", "args": {"slots": "$result.slots"}},
        ],
        "effects": {},
        "failure_policy": {"timeout_ms": 4000,
                           "on_execution_error": {"action": "respond",
                                                  "template": "scheduling_unavailable"}},
        "requires_confirmation": False,
        "idempotency_key": None,
        "permission_level": "agent",
        "implementation": "builtin:scheduling_tools.search_slots",
    },
    {
        "name": "book_appointment",
        "category": "scheduling",
        "description": ("Book an appointment for the current customer at a centre. Use only "
                        "after the customer has explicitly agreed to a specific slot that "
                        "search_slots returned. centre_id must be a UUID from locate_centre or "
                        "search_slots; slot_start must be copied exactly from search_slots."),
        "input_schema": {
            "type": "object",
            "required": ["centre_id", "slot_start"],
            "properties": {
                "centre_id": {"type": "string", "description": (
                    "The centre_id returned by locate_centre. A centre name is accepted "
                    "and resolved, but the id is preferred.")},
                "slot_start": {"type": "string", "description": "ISO-8601 datetime with timezone"},
                "executive_id": {"type": "string", "format": "uuid"},
                "notes": {"type": "string", "maxLength": 500},
            },
            "additionalProperties": False,
        },
        # Nearly everywhere, because wanting to see the property is not a privilege of
        # one journey stage.
        #
        # The old list named seven states and silently excluded call_later, busy,
        # support_active, complaint_raised, booking_token, documentation and converted. A
        # customer in any of those who asked to book got "Model proposed unavailable tool"
        # — and the model, unable to say why, invented its way around it: vague replies,
        # a repeated question, and finally a claim that a booking existed. The whole call
        # was the model working around a tool it was not allowed to use.
        #
        # The two genuine exclusions are kept. site_visit_scheduled has a visit already
        # (reschedule is the right tool), and do_not_contact must not be actioned at all.
        # closed_lost is left out too: reviving a dead lead is a sales decision.
        "allowed_states": ["lead", "contacted", "qualified", "site_visit_done",
                           "offer_shared", "negotiation", "dormant",
                           "booking_token", "documentation", "converted",
                           "call_later", "busy", "support_active", "complaint_raised"],
        "preconditions": [
            {"check": "customer_identified"},
            # FIRST, deliberately. A time nobody looked up is rejected before anything
            # touches the database, and the message names the times that are actually
            # free — so the agent has something true to offer instead of guessing again.
            {"check": "slot_was_offered", "args": {"slot_start": "$param.slot_start"}},
            {"check": "centre_exists", "args": {"centre_id": "$param.centre_id"}},
            {"check": "executive_exists", "args": {"executive_id": "$param.executive_id"}},
            {"check": "slot_available", "args": {"centre_id": "$param.centre_id",
                                                 "slot_start": "$param.slot_start",
                                                 "executive_id": "$param.executive_id"}},
            {"check": "no_conflicting_appointment", "args": {"slot_start": "$param.slot_start"}},
        ],
        "postconditions": [
            {"check": "row_exists", "args": {"table": "appointments",
                                             "where": {"id": "$result.appointment_id",
                                                       "status": "booked"}}},
            {"check": "appointment_matches_request",
             "args": {"appointment_id": "$result.appointment_id",
                      "slot_start": "$param.slot_start", "centre_id": "$param.centre_id"}},
        ],
        "effects": {"state_transition": "appointment_booked",
                    "emit_events": ["appointment_booked"]},
        "failure_policy": {
            "timeout_ms": 5000,
            "on_precondition_fail": {
                "slot_available": {"action": "respond", "template": "slot_gone_offer_alternatives"},
                "no_conflicting_appointment": {"action": "respond",
                                               "template": "already_has_appointment"},
                "default": {"action": "clarify"},
            },
            "on_execution_error": {"action": "retry", "max_retries": 1, "then": "escalate"},
        },
        "requires_confirmation": True,
        # Scoped to the CALL. Without the conversation in the key this matched every past
        # successful execution for all time: a customer who booked Friday at one, cancelled
        # it, and later asked for Friday at one again was told "your appointment is already
        # confirmed" — and no appointment existed, because the executor returned the old
        # result and never ran the tool. The state transition never fired either, so the
        # CRM disagreed with what the customer had just been told.
        #
        # Per-conversation still does the job this key exists for, which is stopping a
        # retry inside one call from booking the same slot twice.
        "idempotency_key": "book:{customer_id}:{centre_id}:{slot_start}:{conversation_id}",
        "permission_level": "agent",
        "implementation": "builtin:scheduling_tools.book_appointment",
    },
    {
        "name": "reschedule_appointment",
        "category": "scheduling",
        "description": ("Move an existing appointment to a different slot. Call my_appointments first if you do not already know which booking they mean — never guess an id."),
        "input_schema": {
            "type": "object",
            # appointment_id is NOT format:uuid and NOT required. Schema validation runs
            # before the tool, so a short reference or a placeholder was rejected there and
            # the resolver never got the chance to sort it out — costing the customer four
            # turns of apology on a call that should have taken two.
            "required": ["slot_start"],
            "properties": {
                "appointment_id": {"type": "string", "description": (
                    "The appointment_id from my_appointments. Omit it if you have already "
                    "looked up their booking on this call — the one under discussion is "
                    "used. Never invent one.")},
                "slot_start": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "allowed_states": ["site_visit_scheduled", "offer_shared", "negotiation",
                           "booking_token", "documentation"],
        "preconditions": [
            {"check": "customer_identified"},
            # Same reasoning as booking: the NEW time has to be one we offered. A
            # reschedule to an invented slot produced the worst bug of the lot — a customer
            # was moved to a different day and told it was confirmed.
            {"check": "slot_was_offered", "args": {"slot_start": "$param.slot_start"}},
            {"check": "appointment_owned_by_customer",
             "args": {"appointment_id": "$param.appointment_id"}},
            {"check": "appointment_active", "args": {"appointment_id": "$param.appointment_id"}},
        ],
        # $result, NOT $param. appointment_id is deliberately optional here — the tool falls
        # back to the booking already under discussion — so on the common path the param is
        # absent, this checked `id IS NULL`, found nothing, and marked a reschedule that had
        # actually succeeded as failed. failure_policy then escalated it.
        "postconditions": [
            {"check": "row_exists", "args": {"table": "appointments",
                                             "where": {"id": "$result.appointment_id",
                                                       "status": "rescheduled"}}},
            {"check": "appointment_matches_request",
             "args": {"appointment_id": "$result.appointment_id",
                      "slot_start": "$param.slot_start"}},
        ],
        "effects": {},
        "failure_policy": {"timeout_ms": 5000,
                           "on_precondition_fail": {"default": {"action": "clarify"}},
                           "on_execution_error": {"action": "respond",
                                                  "template": "slot_gone_offer_alternatives"}},
        "requires_confirmation": True,
        "idempotency_key": None,
        "permission_level": "agent",
        "implementation": "builtin:scheduling_tools.reschedule_appointment",
    },
    {
        "name": "cancel_appointment",
        "category": "scheduling",
        "description": ("Cancel an existing appointment for the current customer. If the "
                        "booking has already been read out on this call you do not need to "
                        "name it again — the tool uses the one under discussion."),
        "input_schema": {
            "type": "object",
            # NOT required, matching reschedule_appointment and matching what the
            # description below has always said. The two contradicted each other: the
            # description invited the model to omit the id, the schema then rejected the
            # call for omitting it, and the customer was asked "do you have the appointment
            # ID handy?" — about a booking the agent had read aloud one sentence earlier.
            # Nobody has that. The tool resolves it from the call.
            "required": [],
            "properties": {
                "appointment_id": {"type": "string", "description": (
                    "The appointment_id from my_appointments. Omit it if the booking has "
                    "already been looked up on this call — never ask the customer for it.")},
                "reason": {"type": "string", "maxLength": 300},
            },
            "additionalProperties": False,
        },
        "allowed_states": ["site_visit_scheduled", "offer_shared", "negotiation",
                           "booking_token", "documentation"],
        "preconditions": [
            {"check": "customer_identified"},
            # NO slot_was_offered here. Cancelling a visit does not involve choosing a time,
            # and this tool has no slot_start parameter — so the check (copy-pasted from
            # reschedule/book, whose comment about "the NEW time" was pasted with it) received
            # slot_start=None every time and rejected EVERY cancellation with "slot_start is
            # required", from the initial commit onward. A caller who said "cancel my visit"
            # then "yes" was answered "I haven't confirmed a free slot yet — which day?" about
            # a booking already read back to them. The only real preconditions for a cancel are:
            # the caller is identified, the appointment is theirs, and it is still active.
            {"check": "appointment_owned_by_customer",
             "args": {"appointment_id": "$param.appointment_id"}},
            {"check": "appointment_active", "args": {"appointment_id": "$param.appointment_id"}},
        ],
        # $result for the same reason as reschedule: the model often sends the short
        # reference (ADED37CF) that my_appointments reads out rather than the UUID, and the
        # tool resolves it — but this check compared the raw string against the id column
        # and failed a cancellation that had already happened.
        "postconditions": [
            # customer_id catches the resolver landing on the wrong appointment (a short
            # reference matched ambiguously, or the "already under discussion" fallback
            # picked up a stale one) -- ownership was already checked as a precondition,
            # but that was against the id the model SENT, not the id the write actually hit.
            {"check": "row_exists", "args": {"table": "appointments",
                                             "where": {"id": "$result.appointment_id",
                                                       "status": "cancelled",
                                                       "customer_id": "$customer.id"}}},
        ],
        "effects": {"state_transition": "follow_up"},
        "failure_policy": {"timeout_ms": 4000,
                           "on_precondition_fail": {"default": {"action": "clarify"}},
                           "on_execution_error": {"action": "escalate"}},
        "requires_confirmation": True,
        "idempotency_key": "cancel:{appointment_id}",
        "permission_level": "agent",
        "implementation": "builtin:scheduling_tools.cancel_appointment",
    },
    {
        "name": "locate_centre",
        "category": "utility",
        "description": ("Find centres near a city or region and return their centre_id, name, "
                        "address and phone. Call this only when a SITE VISIT booking, "
                        "reschedule, or availability search needs a centre_id you don't already "
                        "have. A callback request (schedule_callback) has no centre — never call "
                        "this for one."),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "region_code": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "additionalProperties": False,
        },
        "allowed_states": [],
        "preconditions": [],
        "postconditions": [
            {"check": "centres_are_active", "args": {"centres": "$result.centres"}},
        ],
        "effects": {},
        "failure_policy": {"timeout_ms": 3000},
        "requires_confirmation": False,
        "idempotency_key": None,
        "permission_level": "agent",
        "implementation": "builtin:scheduling_tools.locate_centre",
    },
    {
        "name": "create_ticket",
        "category": "support",
        "description": ("Raise a support ticket for a customer complaint or issue that cannot "
                        "be resolved in the conversation."),
        "input_schema": {
            "type": "object",
            "required": ["subject"],
            "properties": {
                "subject": {"type": "string", "maxLength": 300},
                "description": {"type": "string", "maxLength": 2000},
                "category": {"type": "string"},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
            },
            "additionalProperties": False,
        },
        "allowed_states": [],
        "preconditions": [{"check": "customer_identified"}],
        "postconditions": [{"check": "row_exists",
                            "args": {"table": "tickets",
                                     "where": {"id": "$result.ticket_id",
                                              "subject": "$param.subject",
                                              "status": "$result.status"}}}],
        "effects": {"state_transition": "support_active", "emit_events": ["ticket_created"]},
        "failure_policy": {"timeout_ms": 4000, "on_execution_error": {"action": "escalate"}},
        "requires_confirmation": True,
        "idempotency_key": None,
        "permission_level": "agent",
        "implementation": "builtin:crm_tools.create_ticket",
    },
    {
        "name": "create_customer",
        "category": "utility",
        "description": ("Register a caller who is not in our records yet. Use it as soon as "
                        "an unknown caller wants something that needs an account — a site "
                        "visit, a callback, a ticket. Ask for their name and confirm the "
                        "number they are calling from, then call this. Do NOT ask an unknown "
                        "caller to verify their identity: there is no record to check them "
                        "against, and no answer they give can succeed."),
        "input_schema": {
            "type": "object",
            "required": ["full_name", "phone"],
            "properties": {
                "full_name": {"type": "string", "description": "As they gave it."},
                "phone": {"type": "string", "description": (
                    "The number they are calling from, or one they give. Any common format "
                    "is accepted and normalised.")},
                "email": {"type": "string"},
                "preferred_language": {"type": "string"},
            },
            "additionalProperties": False,
        },
        "preconditions": [],
        "postconditions": [
            # phone/full_name catch the write silently persisting something other than what
            # the tool's own result claims -- e.g. a normalisation bug that saves one phone
            # format but reports another.
            {"check": "row_exists", "args": {"table": "customers",
                                             "where": {"id": "$result.customer_id",
                                                       "phone_normalized": "$result.phone",
                                                       "full_name": "$result.full_name"}}},
        ],
        # Idempotent by phone within the call: a retry must not create a second record for
        # the same person, which would split their history in two.
        "idempotency_key": "create_customer:{phone}:{conversation_id}",
        "implementation": "builtin:crm_tools.create_customer",
        "effects": {},
        "requires_confirmation": False,
        "allowed_states": [],
        # permission_level, not "permissions" — the column is a single string. Getting the
        # key wrong meant ToolDefinition(**spec) raised on an unexpected keyword, the whole
        # seeding loop aborted, and the tool never reached the database. The code had it,
        # the planner did not, and the seeder printed nothing about the failure.
        "permission_level": "agent",
        "failure_policy": {"timeout_ms": 5000,
                           "on_precondition_fail": {"default": {"action": "clarify"}},
                           "on_execution_error": {"action": "clarify"}},
    },
    {
        "name": "update_customer_profile",
        "category": "utility",
        "description": ("Update the customer's own details (name, email, location, language, "
                        "preferred contact hours) when they tell you a correction."),
        "input_schema": {
            "type": "object",
            "properties": {
                "full_name": {"type": "string"},
                "email": {"type": "string"},
                "location": {"type": "string"},
                "region_code": {"type": "string"},
                "preferred_language": {"type": "string"},
                "preferred_contact_start": {"type": "string", "description": "HH:MM"},
                "preferred_contact_end": {"type": "string", "description": "HH:MM"},
            },
            "additionalProperties": False,
        },
        "allowed_states": [],
        "preconditions": [{"check": "customer_identified"}],
        "postconditions": [
            {"check": "customer_profile_updates_persisted",
             "args": {"customer_id": "$customer.id", "updated_fields": "$result.updated_fields"}},
        ],
        "effects": {},
        "failure_policy": {"timeout_ms": 3000},
        "requires_confirmation": False,
        "idempotency_key": None,
        "permission_level": "agent",
        "implementation": "builtin:crm_tools.update_customer_profile",
    },
    {
        "name": "record_customer_fact",
        "category": "utility",
        "description": ("Record something important learned about the customer during the call "
                        "(a preference, objection, pain point, or buying signal)."),
        "input_schema": {
            "type": "object",
            "required": ["category", "key", "value"],
            "properties": {
                "category": {"type": "string",
                             "enum": ["pain_point", "objection", "preference", "buying_intent",
                                      "decision_maker", "product_interest"]},
                "key": {"type": "string", "maxLength": 100},
                "value": {"type": "string", "maxLength": 500},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "additionalProperties": False,
        },
        "allowed_states": [],
        "preconditions": [{"check": "customer_identified"}],
        "postconditions": [
            {"check": "row_exists", "args": {"table": "customer_facts",
                                             "where": {"id": "$result.fact_id",
                                                      "category": "$param.category",
                                                      "key": "$param.key"}}},
        ],
        "effects": {},
        "failure_policy": {"timeout_ms": 3000},
        "requires_confirmation": False,
        "idempotency_key": None,
        "permission_level": "agent",
        "implementation": "builtin:crm_tools.record_customer_fact",
    },
    {
        "name": "schedule_callback",
        "category": "utility",
        "description": ("Schedule a callback ONLY when the customer explicitly asks to be "
                        "called back at another time. It is NOT a fallback for a question you "
                        "could not answer — if a lookup came up empty, say so honestly; do "
                        "not paper over it by offering a callback nobody requested. When they "
                        "genuinely do ask, call it as soon as they ask — do not tell them you "
                        "will arrange one and then stop, which leaves them expecting a call "
                        "that was never booked. If the time they want is outside calling hours "
                        "it is moved to the next permitted slot automatically; say so plainly "
                        "rather than refusing or asking them to pick again."),
        "input_schema": {
            "type": "object",
            "required": ["requested_for"],
            "properties": {
                # "ISO-8601 datetime" alone was not enough: asked to call back "tomorrow at
                # 6am" the model could not commit to a value, so it promised a callback and
                # called nothing. Say where the date comes from and it fills it in.
                "requested_for": {"type": "string", "description": (
                    "ISO-8601 local datetime, e.g. 2026-08-14T09:00:00. Work it out from "
                    "today's date given in your prompt — \"tomorrow at 6am\" is the next "
                    "day at 06:00. Always send a full date and time, never a description.")},
                "reason": {"type": "string", "maxLength": 300},
            },
            "additionalProperties": False,
        },
        "allowed_states": [],
        "preconditions": [{"check": "customer_identified"}],
        "postconditions": [
            # Not checking requested_for here: the tool deliberately MOVES a time outside
            # 9am-9pm into the next legal slot, so the persisted value is expected to
            # differ from what was asked for whenever moved_because is present -- comparing
            # against it would need the same naive/aware normalisation risk as
            # appointment_matches_request for a lower-stakes tool (a callback, not a
            # booking). customer_id alone still catches it landing on the wrong record.
            {"check": "row_exists", "args": {"table": "callback_requests",
                                             "where": {"id": "$result.callback_id",
                                                      "customer_id": "$customer.id"}}},
        ],
        "effects": {},
        "failure_policy": {"timeout_ms": 3000},
        "requires_confirmation": False,
        "idempotency_key": "callback:{customer_id}:{requested_for}",
        "permission_level": "agent",
        "implementation": "builtin:crm_tools.schedule_callback",
    },
    {
        "name": "record_do_not_contact",
        "category": "utility",
        "description": ("Honour a customer's request to stop being contacted. Use as soon as they "
                        "ask to be removed, to stop receiving calls, or say they are not "
                        "interested in ever being contacted again. This outranks every other goal."),
        "input_schema": {
            "type": "object",
            "properties": {
                "scope": {"type": "string", "enum": ["all", "marketing"]},
                "reason": {"type": "string", "maxLength": 300},
            },
            "additionalProperties": False,
        },
        "allowed_states": [],
        "preconditions": [{"check": "customer_identified"}],
        "postconditions": [
            {"check": "row_exists", "args": {"table": "customers",
                                             "where": {"id": "$customer.id",
                                                      "current_state": "do_not_contact"}}},
            # The state machine hook that writes this is a separate code path
            # (state_machine.machine._sync_dnc_list, called from apply_trigger) from the
            # state transition itself -- one succeeding does not guarantee the other ran.
            # A customer who said "stop calling me" and moved to do_not_contact but never
            # made it onto this list would still get called by anything that reads the DNC
            # list directly instead of customer state, which is exactly the gap this closes.
            {"check": "row_exists", "args": {"table": "dnc_list",
                                             "where": {"phone_normalized": "$customer.phone_normalized",
                                                      "scope": "all"}}},
        ],
        # No state_transition effect here on purpose: record_do_not_contact fires the
        # signal:dnc_request trigger itself, inside the tool. Declaring the effect as well
        # would fire a second, non-existent tool:record_do_not_contact trigger — harmless,
        # but it makes the outcome report state_changed_to=None on a turn that did move the
        # customer, which is exactly the sort of thing that sends you looking for a bug in
        # the wrong place.
        "effects": {"emit_events": ["do_not_contact_recorded"]},
        "failure_policy": {"timeout_ms": 3000, "on_execution_error": {"action": "escalate"}},
        "requires_confirmation": False,
        # Scoped to the CALL, not to the customer for all time. "dnc:{customer_id}" meant
        # the first opt-out was remembered forever: if staff later took someone off the
        # list at their request, the agent could never put them back on, because the
        # executor found the old successful execution and returned skipped_duplicate
        # without running anything. A second genuine opt-out would be silently ignored.
        # Per-conversation still stops a double-record inside one call, which is all this
        # key was ever needed for.
        "idempotency_key": "dnc:{customer_id}:{conversation_id}",
        "permission_level": "agent",
        "implementation": "builtin:crm_tools.record_do_not_contact",
    },
    {
        "name": "verify_identity",
        "category": "utility",
        "description": ("Check who you are speaking to before discussing their records, "
                        "payments or bookings. Ask for THE LAST FOUR DIGITS of the mobile "
                        "number registered with us — not the full number, not their email, "
                        "not their address. The moment they say four digits, call this with "
                        "them; do not repeat the question or ask for something else instead, "
                        "which strands the customer answering the same thing twice. Never "
                        "say what the correct digits are, and never guess on their behalf."),
        "input_schema": {
            "type": "object",
            "required": ["last_four"],
            "properties": {
                "last_four": {"type": "string",
                              "description": "The four digits the caller said, digits only."},
            },
            "additionalProperties": False,
        },
        "allowed_states": [],
        "preconditions": [],
        "postconditions": [],
        "effects": {},
        "failure_policy": {"timeout_ms": 3000},
        "requires_confirmation": False,
        "idempotency_key": None,
        "permission_level": "agent",
        "implementation": "builtin:lookup_tools.verify_identity",
    },
    {
        "name": "my_appointments",
        "category": "scheduling",
        "description": ("Look up the caller's own site visits — what is coming up, and what "
                        "has already happened. Use this whenever they ask about their "
                        "booking rather than guessing or asking them to remind you."),
        "input_schema": {
            "type": "object",
            "properties": {
                "include_past": {"type": "boolean",
                                 "description": "Include visits that have already happened."},
            },
            "additionalProperties": False,
        },
        "allowed_states": [],
        "preconditions": [],
        "postconditions": [
            {"check": "appointments_belong_to_customer",
             "args": {"customer_id": "$customer.id", "live": "$result.live",
                      "past": "$result.past"}},
        ],
        "effects": {},
        "failure_policy": {"timeout_ms": 3000},
        "requires_confirmation": False,
        "idempotency_key": None,
        "permission_level": "agent",
        "implementation": "builtin:lookup_tools.my_appointments",
    },
    {
        "name": "my_tickets",
        "category": "support",
        "description": ("Look up the caller's own support tickets and their current status. "
                        "Use this when they ask what happened to a complaint or request, "
                        "instead of raising a duplicate."),
        "input_schema": {
            "type": "object",
            "properties": {
                "include_closed": {"type": "boolean",
                                   "description": "Include tickets already closed or resolved."},
            },
            "additionalProperties": False,
        },
        "allowed_states": [],
        "preconditions": [],
        "postconditions": [
            {"check": "tickets_belong_to_customer",
             "args": {"customer_id": "$customer.id", "tickets": "$result.tickets"}},
        ],
        "effects": {},
        "failure_policy": {"timeout_ms": 3000},
        "requires_confirmation": False,
        "idempotency_key": None,
        "permission_level": "agent",
        "implementation": "builtin:lookup_tools.my_tickets",
    },
    {
        "name": "inventory_query",
        "category": "inventory",
        "description": (
            "Answer any question about which UNITS exist and what they COST — the structured "
            "price/inventory source. Use it, not knowledge_lookup, whenever the caller asks "
            "about price, budget, or what fits a figure: 'anything around 1.1 cr', 'under "
            "90 lakh', 'between 1.2 and 1.4 crore', 'cheapest 3 BHK', 'what does a 2.5 BHK "
            "cost', 'do you have something in my budget', 'what's the entry price'. It "
            "compares against the real cost-sheet rows and returns the nearest units, the "
            "cheapest, and where the price jumps — never estimate or add up a price yourself. "
            "Pass the caller's budget words verbatim in `budget` ('around 1.1 cr', 'under 1 "
            "crore') and the configuration in `bhk_type` if they named one. If a configuration "
            "has no published cost sheet the result says so (price_published false) — relay "
            "that honestly, do not reach for another configuration's rate. Speak from `units`, "
            "`cheapest`, `nearest_above` and `guidance`; read prices exactly as given."),
        "input_schema": {
            "type": "object",
            "properties": {
                "bhk_type": {"type": "string", "description": (
                    "The configuration the caller named, e.g. '3 BHK', '2.5 BHK'. Omit if they "
                    "asked about price/budget without naming one.")},
                "budget": {"type": "string", "description": (
                    "The caller's budget in their own words — 'around 1.1 cr', 'under 90 lakh', "
                    "'between 1.2 and 1.4 crore', 'at least 1.5 cr'. Copy their phrasing; the "
                    "tool parses crore/lakh itself. Omit if they gave no figure.")},
                "min_price": {"type": "integer", "description": (
                    "Lower bound in whole rupees, only if you already have an exact number "
                    "rather than words. Prefer `budget`.")},
                "max_price": {"type": "integer", "description": (
                    "Upper bound in whole rupees, only if you already have an exact number. "
                    "Prefer `budget`.")},
                "sort": {"type": "string", "enum": ["price_asc", "price_desc"],
                         "description": "Cheapest-first by default; price_desc for 'most "
                                        "expensive' / 'top end'."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 12},
            },
            "additionalProperties": False,
        },
        "allowed_states": [],
        "preconditions": [],
        "postconditions": [
            {"check": "inventory_units_match_query",
             "args": {"units": "$result.units",
                      "constraint": "$result.applied_constraint"}},
        ],
        "effects": {},
        "failure_policy": {"timeout_ms": 4000,
                           "on_execution_error": {"action": "respond",
                                                  "template": "knowledge_unavailable"}},
        "requires_confirmation": False,
        "idempotency_key": None,
        "permission_level": "agent",
        "implementation": "builtin:inventory_tools.inventory_query",
    },
    {
        "name": "knowledge_lookup",
        "category": "utility",
        "description": ("Search the company knowledge base (pricing, policies, product "
                        "details, FAQs) for the answer to a customer question. Never answer "
                        "product, price or policy questions from memory. Do NOT call this "
                        "when the RETRIEVED KNOWLEDGE block already covers the question — it "
                        "was fetched from this same knowledge base for this turn, so answer "
                        "from it. Use this only when that block is empty or does not cover "
                        "what they asked, or to try a better-phrased query."),
        "input_schema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "maxLength": 300,
                          "description": "The question in the customer's own terms"},
                "doc_type": {"type": "string",
                             "enum": ["brochure", "faq", "pricing", "policy", "promo",
                                      "playbook", "sop", "compliance"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 8},
            },
            "additionalProperties": False,
        },
        "allowed_states": [],
        "preconditions": [],
        "postconditions": [],
        "effects": {},
        "failure_policy": {"timeout_ms": 10000,
                           "on_execution_error": {"action": "respond",
                                                  "template": "knowledge_unavailable"}},
        "requires_confirmation": False,
        "idempotency_key": None,
        "permission_level": "agent",
        "implementation": "builtin:knowledge_tools.knowledge_lookup",
    },
    {
        "name": "send_document",
        "category": "utility",
        "description": (
            "Send a document to the caller on WhatsApp — a floor plan, brochure, price/cost "
            "sheet, payment plan, location map, or any other document on file. Use this "
            "whenever the caller asks you to send, share, or WhatsApp them something. Also "
            "use it proactively, without waiting to be asked, when a caller asks for "
            "directions, the site address, or how to reach the location: offer to WhatsApp "
            "them the map pin ('I'll send you the map location on WhatsApp so it's easy to "
            "navigate') and call this tool with document_type=location_map. You "
            "CANNOT show or read a document out; this delivers the actual file. Name what "
            "they asked for in `query` in their own words ('the 3 BHK floor plan', 'the "
            "brochure') and, when it is clear, also set `document_type`. Do NOT ask the "
            "caller for their phone number — it is taken from the number they are calling "
            "on; only pass `to_phone` if they explicitly give a DIFFERENT number. If nothing "
            "matches, the tool says so — relay that honestly and never claim a document was "
            "sent when it was not."),
        "input_schema": {
            "type": "object",
            # Neither field is individually required — the tool needs at least one and returns
            # a 'needs' asking which document when both are absent, which is a better caller
            # experience than a schema rejection. Kept out of `required` for that reason.
            "properties": {
                "query": {"type": "string", "maxLength": 200, "description": (
                    "What the caller asked for, in their words: 'the 3 BHK floor plan', 'the "
                    "brochure', 'the payment schedule'. Matched against the document titles.")},
                "document_type": {"type": "string", "description": (
                    "The kind of document, when clear: floor_plan, brochure, price_sheet, "
                    "payment_plan, cost_sheet, location_map, amenities, master_plan, "
                    "specification, legal, agreement, offer. Loose wording ('floor plan', "
                    "'price list') is accepted and normalised.")},
                "to_phone": {"type": "string", "description": (
                    "ONLY if the caller asks you to send it to a different number than the one "
                    "they are calling from. Otherwise omit it — never ask for their number.")},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5, "description": (
                    "How many matching documents to send. Defaults to 1 (the best match); raise "
                    "it only if the caller clearly wants several.")},
            },
            "additionalProperties": False,
        },
        # Wanting a document sent is not a privilege of any journey stage — allowed everywhere,
        # matching knowledge_lookup / inventory_query.
        "allowed_states": [],
        "preconditions": [],
        "postconditions": [
            {"check": "documents_belong_to_tenant",
             "args": {"documents": "$result.documents"}},
        ],
        "effects": {},
        "failure_policy": {"timeout_ms": 12000,
                           "on_execution_error": {"action": "respond",
                                                  "template": "knowledge_unavailable"}},
        # The caller asking IS the confirmation — like schedule_callback, this does not stop to
        # ask "shall I send it?" before sending what they just requested.
        "requires_confirmation": False,
        # Scoped to the CALL, so a retry inside one turn cannot double-send the same document,
        # while a caller who genuinely asks again later still gets it. document_type/query are
        # part of the key so two different documents in one call are not collapsed.
        "idempotency_key": "send_document:{conversation_id}:{document_type}:{query}",
        "permission_level": "agent",
        "implementation": "builtin:document_tools.send_document",
    },
]

# Universal across every industry pack — search_slots/book_appointment/create_ticket/
# knowledge_lookup and the rest of BUILTIN_TOOLS minus the one pack-specific exception.
CORE_TOOLS: list[dict] = [s for s in BUILTIN_TOOLS if s["name"] != "inventory_query"]

# Tool specs a pack opts into by name via IndustryPack.tools (app/packs/base.py) — looked
# up here rather than duplicated per pack. Currently just the one real-estate exception;
# a future pack with its own tool would add an entry here (or the spec could move to live
# beside that pack instead, once there's a second one to prove out the right split).
PACK_TOOL_SPECS: dict[str, dict] = {
    s["name"]: s for s in BUILTIN_TOOLS if s["name"] == "inventory_query"
}


def tools_for_pack(pack) -> list[dict]:
    """CORE_TOOLS + this pack's own extra tools, resolved to full specs — what
    scripts/seed.py seeds a new tenant with instead of the old unconditional
    BUILTIN_TOOLS. Silently skips a pack-declared tool name with no matching spec
    (a pack should never crash a tenant's seeding over a typo in `tools`)."""
    extra = [PACK_TOOL_SPECS[name] for name in pack.tools if name in PACK_TOOL_SPECS]
    return CORE_TOOLS + extra