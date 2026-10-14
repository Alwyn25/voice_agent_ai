"""Deterministic offline provider — lets the whole pipeline be tested without API calls.

It reads the prompt for the signals a real model would use and returns a plausible
structured proposal. Never used in production; selected via LLM_PROVIDER=mock.
"""
import re
import time

from app.providers.llm.base import LLMResult


class MockProvider:
    name = "mock"

    async def complete_json(self, *, system: str, user: str, model: str | None = None,
                            temperature: float = 0.2, max_tokens: int = 900,
                            schema: dict | None = None,
                            schema_name: str = "response") -> LLMResult:
        # schema is accepted for interface compatibility but not enforced here — this
        # provider never talks to a real model, so there is nothing for a schema to
        # constrain. It deliberately still returns the "confirm" + tool=None shape below,
        # because that is what exercises turn_runner's corrective-retry safety net.
        started = time.perf_counter()
        text = user.lower()
        tools = re.findall(r'"name":\s*"([a-z_]+)"', system)

        def has(*words: str) -> bool:
            return any(w in text for w in words)

        # The fast understanding pre-pass (planner.understanding) — its own tiny schema, so
        # it is answered here, not with the planner-shaped proposal below. Trivial turns
        # never reach this branch (rag.is_trivial_turn skips the call), so this only has to
        # route real questions. Keyed on the LATEST message alone — the prompt also carries a
        # few history turns, whose scheduling words must not misclassify a fresh question.
        if schema_name == "understanding":
            latest = text.split("## latest customer message")[-1]

            def lhas(*words: str) -> bool:
                return any(w in latest for w in words)

            is_status = bool(re.search(
                r"\b(?:is|was|has|have|did|does)\b[^?]{0,40}"
                r"\b(?:booked|confirmed|scheduled|sorted|set|go(?:ne)? through)\b", latest))
            is_scheduling = (not is_status) and (
                lhas("slot", "appointment", "reschedul", "cancel", "tomorrow", "today",
                     "morning", "afternoon", "evening", "monday", "tuesday", "wednesday",
                     "thursday", "friday", "saturday", "sunday", "o'clock", "book a visit",
                     "see the flat", "visit")
                or bool(re.search(r"\b\d{1,2}\s*(?:am|pm)\b", latest))
                or bool(re.search(r"\bon\s+(?:the\s+)?\d{1,2}(?:st|nd|rd|th)?\b", latest)))
            needs_knowledge = not (is_scheduling or is_status)
            # Conversational-signal read (mirrors planner.conversational_signal, plus a couple of
            # Hindi/Hinglish tokens) so the hybrid's LLM second pass is exercisable under the mock
            # — the real provider judges this semantically. "none" unless a clear signal shows.
            if lhas("hear me", "are you there", "you there", "still there", "cutting out",
                    "breaking up", "सुन रहे", "sun rahe", "sun paa", "sunai", "awaaz", "आवाज"):
                _signal = "check_in"
            elif lhas("mm-hmm", "mmhmm", "uh-huh", "uh huh", "mhm", "go on", "keep going",
                      "achha", "अच्छा"):
                _signal = "backchannel"
            else:
                _signal = "none"
            data = {
                "intent": ("booking_status" if is_status else
                           "scheduling" if is_scheduling else "product_question"),
                "entities_json": "{}",
                "needs_knowledge": needs_knowledge,
                "is_scheduling": is_scheduling,
                "is_status_question": is_status,
                "conversational_signal": _signal,
            }
            return LLMResult(data=data, model="mock-fast-1", raw_text="",
                             latency_ms=int((time.perf_counter() - started) * 1000),
                             tokens={"prompt": 0, "completion": 0, "total": 0})

        if has("stop calling", "don't call", "do not call", "remove me"):
            data = {"understood": {"intent": "dnc_request", "entities": {}, "sentiment": "negative"},
                    "reasoning": "Customer asked not to be contacted again.",
                    "action": "respond", "tool": None, "params": {},
                    "response_draft": "Understood — I've removed you from our calling list. "
                                      "You won't hear from us again. Sorry for the trouble.",
                    "self_confidence": 0.95, "missing_info": []}
        elif has("speak to someone", "talk to a human", "manager", "supervisor"):
            data = {"understood": {"intent": "request_human", "entities": {}, "sentiment": "neutral"},
                    "reasoning": "Customer explicitly asked for a person.",
                    "action": "escalate", "tool": None, "params": {},
                    "response_draft": "Of course — let me connect you with a colleague.",
                    "self_confidence": 0.92, "missing_info": []}
        elif has("book", "appointment", "slot", "visit") and "search_slots" in tools:
            data = {"understood": {"intent": "book_appointment_request",
                                   "entities": {"raw": text[:80]}, "sentiment": "positive"},
                    "reasoning": "Customer wants an appointment; check real availability first.",
                    "action": "invoke_tool", "tool": "search_slots",
                    "params": {"date_from": "TOMORROW", "limit": 3},
                    "response_draft": "Let me check what's available.",
                    "self_confidence": 0.86, "missing_info": []}
        elif has("price", "cost", "how much", "charges"):
            data = {"understood": {"intent": "pricing_question", "entities": {}, "sentiment": "neutral"},
                    "reasoning": "Pricing question with no knowledge retrieved — cannot answer factually.",
                    "action": "clarify", "tool": None, "params": {},
                    "response_draft": "Could you tell me which service you're asking about?",
                    "self_confidence": 0.62, "missing_info": ["product"]}
        elif has("yes", "yeah", "sure", "confirm", "ok"):
            data = {"understood": {"intent": "affirm", "entities": {}, "sentiment": "positive"},
                    "reasoning": "Customer agreed.",
                    "action": "confirm", "tool": None, "params": {},
                    "response_draft": "Great — confirming that now.",
                    "self_confidence": 0.88, "missing_info": []}
        else:
            data = {"understood": {"intent": "unclear", "entities": {}, "sentiment": "neutral"},
                    "reasoning": "Could not determine intent from the utterance.",
                    "action": "clarify", "tool": None, "params": {},
                    "response_draft": "Sorry, could you say that again?",
                    "self_confidence": 0.35, "missing_info": ["intent"]}

        return LLMResult(data=data, model="mock-1", raw_text="",
                         latency_ms=int((time.perf_counter() - started) * 1000),
                         tokens={"prompt": 0, "completion": 0, "total": 0})
