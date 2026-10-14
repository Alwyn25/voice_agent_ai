"""Ground truth for intent-routing accuracy: for each customer utterance, the three boolean
routing flags app.engine.planner.understanding.understand() should produce.

Scored as exact match, not LLM-judged, on purpose. Unlike the free-text `intent` label
(open vocabulary -- the model invents its own words, so "book_appointment" and
"scheduling_request" could both be "right"), needs_knowledge / is_scheduling /
is_status_question are fixed booleans that ARE the actual routing decision the engine acts
on (do_retrieve = needs_knowledge and not is_scheduling and not is_status_question). Getting
one wrong measurably changes what the agent does next turn, so there is no ambiguity to
judge -- a plain comparison is both cheaper and more meaningful than an LLM verdict here.

`expected_intent` is carried for reference only (printed alongside the result, never scored)
so a human can sanity-check the free-text label without the eval treating a reasonable
synonym as a failure.
"""

DATASET = [
    # -------- product / knowledge questions --------
    {"utterance": "Do you have a 3 BHK available?",
     "needs_knowledge": True, "is_scheduling": False, "is_status_question": False,
     "expected_intent": "product_question"},
    {"utterance": "What amenities does Highland Greenz have?",
     "needs_knowledge": True, "is_scheduling": False, "is_status_question": False,
     "expected_intent": "product_question"},
    {"utterance": "Is there a generator or power backup?",
     "needs_knowledge": True, "is_scheduling": False, "is_status_question": False,
     "expected_intent": "product_question"},
    {"utterance": "What's the price per square foot?",
     "needs_knowledge": True, "is_scheduling": False, "is_status_question": False,
     "expected_intent": "product_question"},
    {"utterance": "Where is the site office located?",
     "needs_knowledge": True, "is_scheduling": False, "is_status_question": False,
     "expected_intent": "product_question"},
    {"utterance": "I'm looking for a 2 BHK",
     "needs_knowledge": True, "is_scheduling": False, "is_status_question": False,
     "expected_intent": "product_question"},
    {"utterance": "Do you have covered parking?",
     "needs_knowledge": True, "is_scheduling": False, "is_status_question": False,
     "expected_intent": "product_question"},

    # -------- scheduling / booking --------
    {"utterance": "I want to book a site visit",
     "needs_knowledge": False, "is_scheduling": True, "is_status_question": False,
     "expected_intent": "book_appointment"},
    {"utterance": "Can I come see the property tomorrow?",
     "needs_knowledge": False, "is_scheduling": True, "is_status_question": False,
     "expected_intent": "book_appointment"},
    {"utterance": "Anything available Friday afternoon?",
     "needs_knowledge": False, "is_scheduling": True, "is_status_question": False,
     "expected_intent": "book_appointment"},
    {"utterance": "I need to reschedule my visit",
     "needs_knowledge": False, "is_scheduling": True, "is_status_question": False,
     "expected_intent": "reschedule"},
    {"utterance": "Cancel my appointment please",
     "needs_knowledge": False, "is_scheduling": True, "is_status_question": False,
     "expected_intent": "cancel"},
    {"utterance": "Move my visit to Saturday morning",
     "needs_knowledge": False, "is_scheduling": True, "is_status_question": False,
     "expected_intent": "reschedule"},
    {"utterance": "What time works tomorrow?",
     "needs_knowledge": False, "is_scheduling": True, "is_status_question": False,
     "expected_intent": "book_appointment"},

    # -------- booking status (own record, not a document) --------
    {"utterance": "Is my visit confirmed?",
     "needs_knowledge": False, "is_scheduling": False, "is_status_question": True,
     "expected_intent": "booking_status"},
    {"utterance": "Did my booking go through?",
     "needs_knowledge": False, "is_scheduling": False, "is_status_question": True,
     "expected_intent": "booking_status"},
    {"utterance": "Am I all set for the visit?",
     "needs_knowledge": False, "is_scheduling": False, "is_status_question": True,
     "expected_intent": "booking_status"},

    # -------- neither: greetings / small talk / trivial --------
    {"utterance": "Hi there",
     "needs_knowledge": False, "is_scheduling": False, "is_status_question": False,
     "expected_intent": "greeting"},
    {"utterance": "Yes that works",
     "needs_knowledge": False, "is_scheduling": False, "is_status_question": False,
     "expected_intent": "affirm"},
    {"utterance": "No thanks",
     "needs_knowledge": False, "is_scheduling": False, "is_status_question": False,
     "expected_intent": "deny"},
    {"utterance": "Thank you, that's all",
     "needs_knowledge": False, "is_scheduling": False, "is_status_question": False,
     "expected_intent": "closing"},
]
