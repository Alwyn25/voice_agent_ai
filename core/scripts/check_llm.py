#!/usr/bin/env python3
"""Call the LLM directly and print exactly what comes back.

    docker compose exec core python scripts/check_llm.py

WHY THIS EXISTS: when the planner's model call fails, turn_runner catches LLMError, sets
action=escalate, and says "I'm having trouble on my side — let me get a colleague to help
you." That is the right thing to say to a customer and completely useless for debugging —
the real exception is stashed in proposal.error and never printed. If every turn in a call
produces that sentence, the model call is failing every time and this will say why.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.providers.llm.base import LLMError  # noqa: E402
from app.providers.llm.router import get_llm  # noqa: E402


def _active_main_model(s) -> str:
    """The main-tier model for whichever provider is actually configured -- each provider
    now owns its own *_model_main setting (openai_model_main, cerebras_model_main,
    groq_model_main); there is no single shared name to print any more."""
    return {"cerebras": s.cerebras_model_main, "groq": s.groq_model_main}.get(
        s.llm_provider, s.openai_model_main)


async def main() -> int:
    s = get_settings()
    key = s.openai_api_key or ""
    print("\nCONFIG")
    print(f"  provider        : {getattr(s, 'llm_provider', '(unset)')}")
    print(f"  main model      : {_active_main_model(s)}")
    print(f"  summary model   : {getattr(s, 'llm_model_summary', '(unset)')}")
    print(f"  timeout         : {getattr(s, 'llm_timeout_seconds', '(unset)')}s")
    print(f"  max retries     : {getattr(s, 'llm_max_retries', '(unset)')}")
    if key:
        print(f"  api key         : {key[:7]}...{key[-4:]}  (length {len(key)})")
    else:
        print("  api key         : NOT SET  <- this alone explains every failed turn")

    print("\nCALLING THE MODEL")
    try:
        result = await get_llm().complete_json(
            system='Reply with JSON only: {"ok": true}',
            user="Say ok.", max_tokens=20)
        print(f"  SUCCESS — {result.model} in {result.latency_ms} ms")
        print(f"  returned: {result.data}")
        print(f"  tokens  : {result.tokens}")
    except LLMError as e:
        print(f"  FAILED — {e}\n")
        text = str(e).lower()
        if "authentication" in text or "401" in text or "api key" in text:
            print("  -> the key is rejected. Rotate it and update OPENAI_API_KEY in .env,")
            print("     then: docker compose up -d --build core")
        elif "rate" in text or "429" in text or "quota" in text or "billing" in text:
            print("  -> quota or rate limit. Check billing on the OpenAI account.")
        elif "model" in text and ("not found" in text or "does not exist" in text):
            print(f"  -> the account cannot reach '{_active_main_model(s)}'. Set its main-"
                 f"model setting (OPENAI_MODEL_MAIN, CEREBRAS_MODEL_MAIN or GROQ_MODEL_MAIN,")
            print("     matching the active LLM_PROVIDER) in .env to a model it does have.")
        elif "connect" in text or "timeout" in text or "dns" in text:
            print("  -> the container cannot reach the API. Check egress from Docker.")
        else:
            print("  -> see the message above; that is the raw provider exception.")
        return 1
    except Exception as e:                                    # noqa: BLE001
        print(f"  FAILED — {e.__class__.__name__}: {e}")
        return 1

    print("\nThe model is reachable. If turns still fail, the failure is later in the "
          "planner — open the turn in Call Logs and read debug.planner_error.\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))