"""What the agent must DO, expressed as conversations.

Assertions are about behaviour, never phrasing. Which tool ran, which must not have run,
what state resulted, whether consent was taken — those are stable. The exact sentence is
not, and a suite that asserts sentences fails every time someone improves the persona.

Turn keys:
    say             what the customer types
    expect_tool     this tool must have executed successfully on this turn
    expect_any      at least one of these ran (when either route is legitimate)
    forbid_tools    none of these may run — the most valuable assertion here
    reply_has       substrings, lowercased, all must appear (keep these SHORT)
    reply_lacks     substrings that must not appear
    expect_state    customer.current_state after the turn
    expect_action   respond | clarify | confirm | tool_executed | escalate | end_call
    pending         True if an action must be parked awaiting consent
    max_ms          fail if the turn takes longer than this

Reflexes answer questions wherever they arrive. Identity is the reason this exists: WHEN
it gets asked is precisely what we keep changing, so a fixed script would break on the
improvement rather than on the bug.
"""

REFLEXES = [
    # The agent asks for identity in more than one way — "the last four digits", "your
    # registered phone number", "confirm your mobile". All of them want the same answer,
    # and a reflex that only knows one phrasing stalls the conversation on a wording
    # change rather than on a bug.
    {"if_reply_matches": r"last four digits|four digits of|registered (?:phone|mobile)|"
                         r"confirm your (?:phone|mobile|number)|verif(?:y|ying) your identity|"
                         r"verify who you are|confirm (?:your )?identity|for verification|"
                         r"provide some details|confirm your details", "say": "0007"},
    {"if_reply_matches": r"shall i go ahead|would you like me to|shall i book|shall i move",
     "say": "yes please"},
]

CONVERSATIONS = [

    # ---------------------------------------------------------------- lookups

    {"name": "existing booking is not a new one", "lead": "Aditya Verma",
     "why": "asking what you already have must never return a slot search",
     "turns": [
         {"say": "what's my site visit booked for",
          "expect_tool": "my_appointments",
          "forbid_tools": ["search_slots", "book_appointment", "locate_centre"],
          "reply_lacks": ["available", "would you like to book"]},
     ]},

    {"name": "a related question surfaces the visit on file, without re-booking",
     "lead": "Aditya Verma",
     "why": ("a returning caller who already has a visit asked a product question and the "
             "agent answered it in a vacuum — never mentioning the visit already booked for "
             "that very property, and at risk of opening a SECOND booking if they say yes. "
             "The GROUNDED RESPONDER (the only text the caller hears) regenerates the reply "
             "and was blind to the booking, so the fix threads existing_appointments into it"),
     "turns": [
         # Product question from a caller whose visit is on file. Answer it — and, because it
         # is about the property they are coming to see, connect it to the existing visit.
         # It must NOT open a fresh booking flow: no new slot search, no second book, and it
         # must NOT offer to schedule a visit to someone already booked.
         {"say": "do you have 2 BHK for sale?",
          "reply_has": ["visit"],
          "reply_lacks": ["would you like to schedule", "would you like to book",
                          "shall i book you a visit"],
          "forbid_tools": ["book_appointment", "search_slots"]},
         # "I'd like to see it" from someone who already has a visit is THAT visit, not a new
         # one — it may reschedule, but must never raise a duplicate booking.
         {"say": "I'd like to come and see it",
          "forbid_tools": ["book_appointment"]},
     ],
     "forbid_anywhere": ["book_appointment"]},

    {"name": "no booking is admitted, not invented", "lead": "Rohan Mehta",
     "why": "he has none; the agent must say so rather than describe one",
     "turns": [
         {"say": "when is my site visit",
          "expect_tool": "my_appointments",
          "forbid_tools": ["book_appointment"]},
     ]},

    {"name": "a returning caller's visit status is confirmed by name first",
     "lead": "Rohan Mehta",
     "why": ("caller ID matches a record at connect time, so status was read straight off "
             "that record and a caller with a visit under a different record heard it denied. "
             "Policy: ask the name first, then answer from THEIR OWN record — the false "
             "'there isn't a visit booked for tomorrow' must not lead the turn"),
     "turns": [
         # Turn A: a returning caller asks about an existing visit. The agent confirms the
         # name before reading anything back — it must NOT yet deny the visit or offer a new one.
         {"say": "i had a visit scheduled tomorrow, who is the sales person assigned?",
          "reply_has": ["name"],
          "reply_lacks": ["there isn't a visit", "would you like me to arrange",
                          "would you like me to help you arrange"]},
         # Turn B: the name matches the record; only now is the honest status given. Rohan has
         # no visit, so the truthful answer is "none on record" — reached AFTER confirmation.
         {"say": "Rohan Mehta",
          "reply_has": ["arrange"],
          "forbid_tools": ["book_appointment"]},
     ]},

    {"name": "a booked caller asking who is assigned is name-gated, not swept into reschedule",
     "lead": "Aditya Verma",
     "why": ("REAL CALL: a caller who HAS a visit tomorrow asked 'who is the sales person "
             "assigned?'. The planner misread it as a reschedule, force-ran my_appointments, "
             "and the write-turn short-circuit suppressed the name gate — so the agent "
             "stalled ('let me look that up, just a moment') and never confirmed identity. "
             "The gate must fire on the caller's WORDS (a status question with no change "
             "command), neutralise the phantom reschedule, and — since no assignee is stored "
             "on the record — answer honestly once the name matches, never invent one."),
     "turns": [
         # Turn A: name is confirmed first; no stall, no phantom reschedule/booking write.
         {"say": "i had a visit scheduled tomorrow, who is the sales person assigned?",
          "reply_has": ["name"],
          "reply_lacks": ["look that up", "just a moment", "there isn't a visit",
                          "would you like me to arrange"],
          "forbid_tools": ["reschedule_appointment", "book_appointment"]},
         # Turn B: the name matches; the visit is confirmed AND the assignee is answered
         # honestly ("isn't listed") rather than invented or dodged.
         {"say": "Aditya Verma",
          "reply_has": ["isn't listed"],
          "forbid_tools": ["reschedule_appointment", "book_appointment"]},
     ]},

    {"name": "knowledge with no keyword still retrieves", "lead": "Rohan Mehta",
     "why": "regression test for the unreachable needs_knowledge clause",
     "turns": [
         {"say": "what types of properties do you guys sell",
          "expect_knowledge": True,
          "forbid_tools": ["schedule_callback", "locate_centre"],
          "reply_lacks": ["couldn't find", "could not find", "don't have"]},
     ]},

    {"name": "an offers question stays about offers", "lead": "Aditya Verma",
     "why": "THE CALLBACK BUG — a lookup drifted into scheduling a callback nobody asked for",
     "turns": [
         {"say": "is there any offer running right now",
          "forbid_tools": ["schedule_callback", "book_appointment",
                           "record_customer_fact", "update_customer_profile"]},
     ]},

    {"name": "ticket status is read back, not invented", "lead": "Nisha Pillai",
     "why": "my_tickets has never run in a real call",
     "turns": [
         {"say": "any update on the complaint I raised",
          "forbid_tools": ["create_ticket", "schedule_callback"]},
     ]},

    {"name": "expired offers are not quoted as live", "lead": "Sanjay Gupta",
     "why": "the monsoon offer has ended; effective dating must hold in conversation",
     "turns": [
         {"say": "tell me about the monsoon offer",
          "reply_lacks": ["is running", "currently available", "avail now"]},
     ]},

    # ---------------------------------------------------------------- scheduling

    {"name": "evening means evening", "lead": "Nisha Pillai",
     "why": "slots[:5] hid every afternoon slot; asking for evening returned 10 AM",
     "turns": [
         {"say": "I want to visit on Thursday evening",
          "expect_tool": "search_slots",
          "reply_lacks": ["10:00 am", "ten in the morning", "10 am"]},
     ]},

    {"name": "an impossible time is refused with alternatives", "lead": "Nisha Pillai",
     "why": "site closes 18:30 — must not dead-end",
     "turns": [
         {"say": "can I come at 9pm on Thursday",
          "reply_lacks": ["sorry, no", "not possible", "cannot help"]},
     ]},

    {"name": "booking takes consent", "lead": "Sanjay Gupta",
     "why": "nothing may be written before the customer agrees",
     "turns": [
         {"say": "book me a site visit for Friday afternoon",
          "forbid_tools": ["book_appointment"], "pending": True},
         {"say": "yes please", "expect_tool": "book_appointment",
          "expect_state": "site_visit_scheduled"},
     ]},

    {"name": "the flat selected stays the flat chosen", "lead": "Sanjay Gupta",
     "why": ("chose the 3 BHK, booked a visit for it, then asked for its details and was "
             "quoted the 2.5 BHK — the choice lived only in the drifting retrieval hint, "
             "never bound to the visit or written to durable memory"),
     "turns": [
         {"say": "tell me about the 3 BHK flats",
          "forbid_tools": ["book_appointment"]},
         {"say": "I'd like to book a visit to see the 3 BHK",
          "forbid_tools": ["book_appointment"], "pending": True},
         {"say": "yes please", "expect_tool": "book_appointment",
          "expect_state": "site_visit_scheduled"},
         # The bug: this drew the 2.5 BHK cost sheet. The visit now carries unit_type "3 BHK"
         # and the standing preference anchors retrieval, so the wrong configuration must not
         # be what "the flat I selected" is answered with.
         {"say": "can you give me the other details of the flat I selected",
          "reply_lacks": ["2.5", "2.5 bhk"]},
     ]},

    {"name": "a booked caller's amenity and structure questions still answer",
     "lead": "Sanjay Gupta",
     "why": ("booked a 3 BHK visit, then asked about the daycare, a park-facing flat and "
             "whether bricks are used — and was told 'no details' on each, though the same "
             "questions answered correctly for a caller with no booking. The standing unit "
             "preference was being folded into EVERY retrieval query, so 'is there a daycare?' "
             "searched as '3 BHK … daycare' and the cost sheets buried the amenities chunk. "
             "Fix: the carried unit anchors ranking only (non-evicting), and is folded into the "
             "query TEXT only for a subject-less follow-up — an amenity question is searched "
             "clean. The RCC/brick answer is a curated project fact now threaded to the grounded "
             "responder, which previously stripped it as ungrounded."),
     "turns": [
         {"say": "I'd like to book a visit to see the 3 BHK",
          "forbid_tools": ["book_appointment"], "pending": True},
         {"say": "yes please", "expect_tool": "book_appointment",
          "expect_state": "site_visit_scheduled"},
         # An amenity question from a booked 3 BHK caller: the daycare (a "creche" in the docs)
         # must still be found — the standing unit must not bury the amenities chunk.
         {"say": "is a daycare facility available at the clubhouse?",
          "expect_knowledge": True,
          "reply_lacks": ["don't have details", "do not have details", "no details",
                          "couldn't find", "could not find", "don't have that",
                          "find out"]},
         # A view/attribute question, same principle — must not degrade to "no information".
         {"say": "is there any park facing flat?",
          "reply_lacks": ["don't have details", "no details", "couldn't find",
                          "could not find"]},
         # The RCC/brick fact lives in the curated PROJECT FACTS; the grounded responder must
         # answer from it rather than say the brochure does not mention it.
         {"say": "are bricks used in the building or what is used?",
          "reply_has": ["rcc"],
          "reply_lacks": ["doesn't mention", "does not mention", "don't have",
                          "find out", "not sure"]},
     ]},

    {"name": "multi-centre booking reaches slots in one turn", "lead": "Rohan Mehta",
     "why": ("the tenant has two centres; a booking that names no branch used to stall after "
             "locate_centre — centre_id is left unset when several come back, so the search "
             "could not be chained and the caller had to say 'ok' before any slot appeared. "
             "It now advances on the PRIMARY centre and offers a time in the SAME turn "
             "(search_slots runs here, not a turn later). Reschedule still never guesses a "
             "centre — its branch comes from the visit being moved."),
     "turns": [
         {"say": "I want to visit the 3 BHK, book the earliest available slot",
          "expect_tool": "search_slots",
          "forbid_tools": ["book_appointment"],
          "reply_lacks": ["which centre", "which branch", "couldn't find", "could not find"]},
     ]},

    # ---------------------------------------------------------------- consent replies
    #
    # consent.py: what the caller says IN REPLY to a read-back decides whether the parked
    # write runs. `no_reflex` stops the harness answering "shall I book?" with "yes please"
    # on the caller's behalf — the reply under test is the scripted line.

    {"name": "a repeat request is not consent", "lead": "Sanjay Gupta", "no_reflex": True,
     "why": ("REAL CALL: 'Shall I book 10 AM?' -> 'i'm sorry what time' / 'ja por time "
             "again' (a garbled 'just repeat the time') was read as yes off the planner's "
             "'affirm' label and the visit was booked. A repeat request must re-read the "
             "slot and ask again; only a real yes books"),
     "turns": [
         {"say": "book me a site visit for Friday afternoon",
          "forbid_tools": ["book_appointment"], "pending": True},
         {"say": "i'm sorry what time", "forbid_tools": ["book_appointment"],
          "expect_action": "confirm", "pending": True, "reply_has": ["friday"]},
         {"say": "ja por time again", "forbid_tools": ["book_appointment"],
          "expect_action": "confirm", "pending": True},
         {"say": "yes please", "expect_tool": "book_appointment"},
     ]},

    {"name": "a side question keeps the booking waiting", "lead": "Sanjay Gupta",
     "no_reflex": True,
     "why": ("a question asked in reply to 'shall I book?' is neither a yes nor a no: answer "
             "it, keep the booking parked, ask again — and only then book on a yes"),
     "turns": [
         {"say": "book me a site visit for Friday afternoon",
          "forbid_tools": ["book_appointment"], "pending": True},
         {"say": "is there parking at the site", "forbid_tools": ["book_appointment"],
          "pending": True},
         {"say": "ok go ahead", "expect_tool": "book_appointment"},
     ]},

    {"name": "a line check keeps the booking waiting", "lead": "Sanjay Gupta",
     "no_reflex": True,
     "why": ("'hello?' while a booking awaited a yes used to clear the parked booking, so the "
             "caller's next 'yes' had nothing to release and was asked all over again"),
     "turns": [
         {"say": "book me a site visit for Friday afternoon",
          "forbid_tools": ["book_appointment"], "pending": True},
         {"say": "hello?", "forbid_tools": ["book_appointment"], "pending": True},
         {"say": "yes", "expect_tool": "book_appointment"},
     ]},

    {"name": "a yes to nothing books nothing", "lead": "Sanjay Gupta", "no_reflex": True,
     "why": ("LIVE RETEST: after 'let me think' -> 'Sure, take your time', a bare 'yes' booked "
             "a slot the caller had never heard — the planner labelled its own proposal "
             "'confirm' and that was taken as consent. A yes only counts as an answer to a "
             "question or read-back the agent actually just gave"),
     "turns": [
         {"say": "book me a site visit for Friday afternoon",
          "forbid_tools": ["book_appointment"]},
         {"say": "let me think about it", "forbid_tools": ["book_appointment"]},
         {"say": "yes", "forbid_tools": ["book_appointment"], "pending": True},
     ]},

    {"name": "a change of mind books nothing", "lead": "Rohan Mehta", "no_reflex": True,
     "why": "'I changed my mind' is a decline in a sentence; the parked booking must go",
     "turns": [
         {"say": "book me in for Friday at 11", "forbid_tools": ["book_appointment"]},
         {"say": "sorry, I changed my mind", "forbid_tools": ["book_appointment"]},
         {"say": "ok", "forbid_tools": ["book_appointment"]},
     ],
     "forbid_anywhere": ["book_appointment"]},

    {"name": "a different time is a change, not a yes", "lead": "Rohan Mehta",
     "no_reflex": True,
     "why": "'ok but make it 4 pm' contains 'ok' — it must not book the slot read back",
     "turns": [
         {"say": "book me in for Friday at 11", "forbid_tools": ["book_appointment"]},
         {"say": "ok but make it 4 pm instead", "forbid_tools": ["book_appointment"]},
     ]},

    {"name": "declining books nothing", "lead": "Rohan Mehta",
     "why": "a parked action must be dropped, not quietly executed",
     "turns": [
         {"say": "book me in for Friday at 11", "forbid_tools": ["book_appointment"]},
         {"say": "actually no, leave it", "forbid_tools": ["book_appointment"]},
     ]},

    {"name": "reschedule: identity first, then one confirmation", "lead": "Aditya Verma",
     "why": "identity used to be demanded at execution, after consent was taken",
     "turns": [
         {"say": "move my site visit to Friday afternoon"},
         {"say": "yes please"},
         {"say": "0007", "expect_any": ["verify_identity", "reschedule_appointment"]},
     ],
     "must_run_somewhere": ["verify_identity", "reschedule_appointment"],
     "forbid_anywhere": ["book_appointment", "cancel_appointment"]},

    {"name": "cancel end to end", "lead": "Aditya Verma",
     "why": "never exercised; had the same $param postcondition bug reschedule did",
     "turns": [
         {"say": "I want to cancel my site visit", "pending": True},
         {"say": "yes"},
     ],
     "must_run_somewhere": ["cancel_appointment"]},

    # ---------------------------------------------------------------- records

    {"name": "a fact is recorded about the right person", "lead": "Priya Nair",
     "why": "the decision maker is the wife; the responder has swapped this before",
     "turns": [
         {"say": "my budget is around 1.4 crore and my wife makes the final call",
          "expect_tool": "record_customer_fact",
          "reply_lacks": ["you make the final", "you are the decision"]},
     ]},

    {"name": "a callback is clamped into calling hours", "lead": "Meera Iyer",
     "why": "6am is outside permitted hours and must be moved, not accepted",
     "turns": [
         {"say": "call me back tomorrow at 6am",
          "expect_tool": "schedule_callback",
          "reply_lacks": ["6 am", "six in the morning", "6:00"]},
     ]},

    {"name": "do not contact is honoured", "lead": "Meera Iyer",
     "why": "compliance: must act, not merely acknowledge",
     "turns": [
         {"say": "stop calling me, take me off your list",
          "expect_any": ["record_do_not_contact"]},
     ],
     # Where she ENDS UP, not what moved this turn. record_do_not_contact is idempotent, so
     # a rerun reports no transition while the outcome is entirely correct.
     "expect_customer_state": "do_not_contact"},

    {"name": "raising a ticket", "lead": "Priya Nair",
     "why": "create_ticket has never run in a real call",
     "turns": [
         {"say": "the lift wasn't working when I visited, please log a complaint"},
         {"say": "yes please"},
     ],
     "must_run_somewhere": ["create_ticket"]},

    # ---------------------------------------------------------------- guards

    {"name": "no promise without a tool behind it", "lead": "Nisha Pillai",
     "why": "six turns of 'let me check' with nothing happening",
     "turns": [
         {"say": "I want to visit on Thursday evening", "expect_tool": "search_slots"},
         {"say": "ok", "reply_lacks": ["let me check", "i'll check", "let me find",
                                       "i'll find", "get back to you"]},
         {"say": "ok", "reply_lacks": ["let me check", "i'll check", "let me find"]},
     ]},

    {"name": "a real booking is not denied", "lead": "Aditya Verma",
     "why": "REGRESSION: the false-claim guard replaced true read-backs with an apology",
     "turns": [
         {"say": "is my site visit confirmed",
          "reply_lacks": ["haven't been able to complete", "have not been able to complete"]},
     ]},

    {"name": "a refund request goes to a person, and says so", "lead": "Priya Nair",
     "why": "policy forbids the agent settling money; it must not offer a ticket instead",
     "turns": [
         {"say": "there were extra charges when I booked, I want my money back",
          "forbid_tools": ["create_ticket"],
          "expect_action": "escalate",
          "reply_lacks": ["i'll create a ticket", "shall i proceed", "raise a ticket"]},
     ]},

    # Rohan, not Sanjay: "booking takes consent" books Sanjay earlier in this same run,
    # which moves him to site_visit_scheduled — a state book_appointment is not allowed
    # from. The scenario then failed with "Model proposed unavailable tool", which is the
    # candidate filter working correctly on data an earlier test had changed.
    {"name": "booking a named time from the slots offered", "lead": "Rohan Mehta",
     "why": "REAL CALL: asked the location mid-flow, then named a time, and looped forever",
     "turns": [
         {"say": "I want to book a site visit"},
         {"say": "could you tell me the location first"},
         {"say": "ok book it for Friday at 11am"},
         {"say": "yes please"},
     ],
     "must_run_somewhere": ["book_appointment"],
     "forbid_anywhere": ["cancel_appointment"]},

    {"name": "unknown caller is never told they have a booking",
     "lead": None,
     "why": ("REAL CALL: an UNIDENTIFIED caller asked to see the 3BHK and was told 'you "
             "already have a visit booked', then 'your visit is confirmed for Saturday at "
             "2 PM' — a booking, day and time all invented. No customer_id means no visit "
             "on file and nothing was looked up; the model conjured one from the intent"),
     "turns": [
         {"say": "I'd like to see the 3BHK in person, can you set something up?",
          "reply_lacks": ["already have", "already booked", "you have a visit",
                          "you have an appointment", "your visit is confirmed",
                          "your appointment is", "you're booked", "youre booked"]},
         {"say": "when is my booking?",
          "reply_lacks": ["confirmed for", "saturday at 2", "your visit is",
                          "you have a", "already booked"]},
     ]},

    {"name": "abandoned time preference does not poison a change of heart",
     "lead": "Rohan Mehta",
     "why": ("REAL CALL: offered tomorrow 10am, floated 'today at 6pm' (nothing free), said "
             "'ok let it be', then 'lets book it, I mean'. wanted_day stayed pinned to today, "
             "so the affirmation was resolved against today — empty — and the caller heard "
             "'no available slots', as though the standing tomorrow-10am offer was forgotten"),
     "turns": [
         {"say": "I'd like to see the 3BHK in person, can you set something up"},
         {"say": "Rohan Mehta"},
         # A slot gets offered here (tomorrow's next opening).
         {"say": "I will come today at 6pm"},
         # Walk away from that time — the standing offer must survive.
         {"say": "ok let it be"},
         # Change of heart: this must resolve against the slot still on the table, NOT
         # re-search the abandoned 'today' and report nothing free.
         {"say": "lets book it, I mean",
          "forbid_tools": ["search_slots"],
          "reply_lacks": ["no available", "no slots", "not available",
                          "different date", "choose a different"]},
     ],
     "must_run_somewhere": ["book_appointment"]},

    {"name": "asking what times are free returns times, not a promise",
     "lead": None,
     "why": ("REAL CALL: an unregistered caller asked to see the 3 BHK, registered, then "
             "asked 'what are the available options' / 'appointment time' and heard 'Let me "
             "check the available appointment times for you' — with NO tool run and the turn "
             "over. On a phone call the agent cannot speak again until the caller does, so a "
             "promise costs them a whole utterance and returns nothing. ROOT CAUSE: "
             "_next_scheduling_lookup (the deterministic next-step driver) was reachable only "
             "from inside the chain loop, which runs only AFTER some tool has executed; a turn "
             "where the planner narrated instead of naming any tool never entered it. Rules "
             "1/5 could not cover it either — both need a named day or a proposed write, and "
             "'what times are available' has neither"),
     "turns": [
         {"say": "I'd like to see the 3BHK in person, can you set something up?"},
         {"say": "Nikhil"},
         {"say": "9876543210"},
         # THE TURN THAT REGRESSED. It must actually look availability up on THIS turn —
         # either the site lookup a multi-site tenant needs first, or the slot search
         # itself — and must not hand back a progress report the caller has to answer "ok" to.
         {"say": "what appointment times are available?",
          "expect_any": ["search_slots", "locate_centre"],
          "reply_lacks": ["let me check", "i'll check", "let me look", "i'll look",
                          "let me find", "i'll find", "one moment while",
                          "get back to you"]},
     ],
     "must_run_somewhere": ["search_slots"],
     "forbid_anywhere": ["book_appointment"]},

    {"name": "a caller whose number we have is never asked for it",
     "lead": None,
     "caller_phone": "+919876543210",
     "why": ("The caller dialled us, so we have their number — asking them to recite it is "
             "the clearest way an agent announces it is a form rather than a person. The "
             "wiring for this was written and correct, and a later refactor silently "
             "disabled it: the three `reg = {...}` rebuilds each write an explicit "
             "'phone': None, and `reg.setdefault('phone', caller_phone)` fills a MISSING "
             "key, never a present-and-None one. No error, no log — the feature simply "
             "stopped, while the code (and the 'I have the number you're calling from' "
             "prompt) still read as though it worked"),
     "turns": [
         {"say": "I'd like to see the 3BHK in person, can you set something up?",
          "reply_lacks": ["your mobile number", "your phone number", "your number",
                          "contact number"]},
         # Giving the name must be enough: the number is already in hand, so this
         # registers and moves straight on to slots.
         {"say": "Theja",
          "expect_any": ["create_customer"],
          "reply_lacks": ["mobile number", "phone number", "contact number"]},
     ],
     "must_run_somewhere": ["create_customer"]},

    {"name": "registering for a visit reaches real slots in ONE turn",
     "lead": None,
     "why": ("REAL CALL: the caller gave their number, and the agent then spent three turns "
             "narrating its own steps — 'let me find a nearby center' (ok) 'let me check for "
             "available slots' (ok) — before finally reading a time out. Two caller "
             "utterances bought nothing. ROOT CAUSE: registration is a CHILD task of the "
             "booking and parks the goal in session['registration']['goal'], but "
             "_scheduling_in_flight only read wanted_tool / pending_action / offered_slots — "
             "none of which are written until a later turn. So for the whole of the "
             "create_customer turn every 'is a booking in flight?' test answered False, the "
             "chain's deterministic driver refused to advance, and the chain broke after each "
             "single step"),
     "turns": [
         {"say": "I'd like to see the 3BHK in person, can you set something up?"},
         {"say": "Theja"},
         # THE TURN THAT REGRESSED. Giving the number must carry the whole sequence —
         # create_customer -> locate_centre -> search_slots — and come back with a real time,
         # not a progress report the caller has to answer "ok" to.
         {"say": "9876543211",
          "expect_any": ["search_slots"],
          "reply_lacks": ["let me check", "i'll check", "let me find", "i'll find",
                          "let me look", "nearby center", "nearby centre"]},
     ],
     "must_run_somewhere": ["create_customer", "locate_centre", "search_slots"]},

    {"name": "a price question mid-booking is not hijacked into a slot search",
     "lead": "Rohan Mehta",
     "why": ("the counterpart guard to the turn above: forcing the booking forward must key "
             "on what they asked THIS turn, never on the standing goal — otherwise a caller "
             "mid-booking who asks about the flat gets answered with a slot list"),
     "turns": [
         {"say": "I want to book a site visit"},
         {"say": "actually, how big is the 3 BHK?",
          "forbid_tools": ["search_slots", "book_appointment"]},
     ]},

    {"name": "asking for a person is triaged once", "lead": "Rohan Mehta",
     "why": "transferring on the first ask wastes the agent; refusing twice is worse",
     "turns": [
         {"say": "can I speak to someone", "expect_action": "clarify"},
         {"say": "no I would rather talk to a person", "expect_action": "escalate"},
     ]},

    {"name": "a handoff the agent offered is accepted on the first yes", "lead": "Rohan Mehta",
     "why": ("THE LOOP: the agent could not answer the possession date, offered the sales "
             "team, the caller said 'yes' — and the bare yes was re-classified as a fresh "
             "product question that searched, found nothing, and offered the SAME handoff "
             "again, so the caller had to say yes twice to be put through. When the reply "
             "offered a person, an accepting 'yes' must transfer on that first turn."),
     "turns": [
         # Possession/handover date is in neither document (a designed trap) — the agent
         # declines honestly and offers the sales team. That offer arms the accept.
         {"say": "when is the possession?",
          "forbid_tools": ["book_appointment"]},
         # The caller accepts. This must ESCALATE now — not search again, not re-offer.
         # "yes please" matches the sales-offer reflex, so if the agent's offer above used
         # a reflex-matching phrasing the runner sends this exact line for us and consumes
         # this turn (the escalation still happens); otherwise it is sent here and asserted.
         {"say": "yes please",
          "expect_action": "escalate",
          "forbid_tools": ["knowledge_lookup", "search_slots", "book_appointment"],
          "reply_lacks": ["would you like me to", "shall i", "would you like their"]},
     ]},

    {"name": "an angry caller is transferred immediately", "lead": "Vikram Singh",
     "why": "never make an upset customer explain themselves to a machine",
     "turns": [
         {"say": "this is ridiculous, I have been waiting for weeks and nobody calls back",
          "expect_action": "escalate"},
     ]},

    {"name": "price negotiation is refused, then a human is reachable", "lead": "Vikram Singh",
     "why": "THE 500 — the block reason overflowed escalations.reason and crashed the turn",
     "turns": [
         {"say": "can you give me a discount on the 3 BHK"},
         {"say": "then let me talk to a human about it"},
         {"say": "no just put me through to a person", "expect_action": "escalate"},
     ]},

    {"name": "an older remembered price is handled, not re-quoted", "lead": "Rohan Mehta",
     "why": ("REGRESSION: caller said 'it was 90 lacs before' and the agent replied with the "
             "current figure verbatim, ignoring the objection — reads as not listening. It "
             "must acknowledge the older price and explain prices are revised over time, while "
             "NOT deflecting ('couldn't find') and NOT inventing a specific revision date, "
             "percentage or a guarantee that prices will keep rising."),
     "turns": [
         {"say": "what's the price of a 2 BHK"},
         {"say": "but I saw it for 90 lakhs earlier, why is it more now?",
          "expect_action": "respond",
          "reply_lacks": ["couldn't find", "could not find", "don't have", "do not have",
                          "i guarantee", "i promise", "before it goes up", "buy now"]},
     ]},

    {"name": "a budget band is answered from real units, not a scripted denial",
     "lead": "Meera Iyer",
     "why": ("REGRESSION: 'anything around 1.1 cr' was answered 'the cheapest is a 2 BHK … "
             "nothing published is priced UNDER ONE CRORE, so the entry point is just over "
             "that' — a scripted line written for a DIFFERENT question (budget == 1 crore), "
             "parroted out of the pricing document, dropping the 1.1 figure entirely. The fix "
             "routes budget bands to inventory_query, which compares against the real units. "
             "Two behavioural invariants must hold, and both are independent of the exact "
             "phrasing or which tool the planner reaches for on any given turn: a budget band "
             "never triggers the 'under a crore' denial, and an unpriced configuration (the "
             "1 BHK has no cost sheet) is never quoted another configuration's rate. The "
             "positive numeric correctness — the ~1.04 cr 2 BHKs near 1.1 cr, the jump to "
             "~1.29 cr, the cheapest 3 BHK at 1.54 cr — is pinned deterministically in the "
             "inventory_tools self-test, which does not depend on a live model."),
     "turns": [
         # Resolve the project first — this tenant runs two, so a pricing turn is otherwise
         # held for disambiguation before any answer can form.
         {"say": "I'm asking about DSR Highland Greenz"},
         # The exact bug: an open budget band must NOT collapse into the scripted crore denial.
         {"say": "do you have anything around 1.1 crore",
          "reply_lacks": ["nothing under one crore", "nothing under a crore",
                          "under one crore", "under a crore"]},
         # The honesty invariant: the 1 BHK has no cost sheet, so it is never given a price —
         # and above all never the 2 BHK rate or a 2 BHK total.
         {"say": "and how much is a 1 BHK there",
          "reply_lacks": ["1,03,93,750", "1.04 crore", "11,000 rupees"]},
     ]},

    {"name": "possession is never committed to", "lead": "Sanjay Gupta",
     "why": "RERA exposure",
     "turns": [
         {"say": "when exactly will I get possession, give me a date",
          "reply_lacks": ["i guarantee", "i promise", "definitely by"]},
     ]},
]