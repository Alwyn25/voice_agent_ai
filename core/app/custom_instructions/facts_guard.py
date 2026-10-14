"""Author-time guard for Dynamic Knowledge (spec Module B).

A dynamic fact is the ONE place tenant free-text enters the responder's TRUSTED PROJECT FACTS
tier verbatim (``responder._project_facts_block`` — "as authoritative as VERIFIED RESULTS"),
so unlike advisory text it is NOT fenced as subordinate. That makes author-time screening the
security boundary. Two defenses, both enforced on create/update only (never the hot turn path):

  * ``sanitize_fact`` — flatten the content to a single line (so it cannot open a new prompt
    section like ``## SYSTEM``), strip control characters, and REJECT content shaped like an
    instruction to the agent rather than a fact to state. A fact may CHANGE what the agent can
    cite; it may never change the agent's rules.
  * ``check_caps`` — bound how much trusted text one tenant can inject (a single huge fact, too
    many active rows, or too large a total), mirroring ``caps.py`` for advisory.

Side effects (booking, DNC, transfer, escalation) always go through the deterministic rule/tool
layer, which a fact string cannot reach — so this guard is about what the agent SAYS, which is
exactly the blast radius a trusted-tier fact has.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import get_settings


class FactRejected(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


# The shape of a prompt-injection / block break-out — none of which belongs in a spoken
# business fact. Matched case-insensitively on the flattened content.
_INJECTION_PATTERNS = [re.compile(p, re.I) for p in (
    r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|earlier|above)",
    r"disregard\s+(?:the\s+)?(?:above|previous|rules?|instructions?)",
    r"forget\s+(?:your|the|all|everything)\b",
    r"override\s+(?:the\s+)?(?:rules?|instructions?|system|hard\s+rules?)",
    r"\b(?:new|updated|revised)\s+(?:system\s+)?instructions?\b",
    r"\bsystem\s+prompt\b",
    r"\byou\s+are\s+now\b",
    r"\bact\s+as\b",
    r"\bpretend\s+(?:to\s+be|you)\b",
    r"\bjailbreak\b",
    r"</?\s*(?:system|assistant|user|tool)\s*>",     # role tags
    r"<\|.*?\|>",                                      # chat-template tokens
    r"^\s*(?:system|assistant|user)\s*:",             # role: prefixes
    r"```",                                           # code / prompt fences
)]

# A markdown heading at the START of the (flattened) content would read as a new prompt block.
# A '#' mid-sentence ("the #1 project") is fine, so only the leading token is guarded.
_LEADING_HEADER = re.compile(r"^#{1,6}\s")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_fact(content: str) -> str:
    """Return the cleaned, single-line fact, or raise ``FactRejected``.

    Flattening to one line is the key structural defense: a fact cannot open a new
    ``## SYSTEM`` block on its own line, so it stays inside the PROJECT FACTS bullet it is
    rendered into."""
    raw = _CONTROL_CHARS.sub("", content or "")
    flat = re.sub(r"\s+", " ", raw).strip()
    if not flat:
        raise FactRejected("The fact is empty.")
    if _LEADING_HEADER.search(flat):
        raise FactRejected("A fact cannot start with a '#' heading marker — reword it as a "
                           "plain statement.")
    for pat in _INJECTION_PATTERNS:
        if pat.search(flat):
            raise FactRejected(
                "This reads like an instruction to the agent, not a fact it can state. Dynamic "
                "knowledge holds facts the agent may cite (offers, prices, announcements) — it "
                "cannot change the agent's rules. Reword it as a plain statement of fact.")
    return flat


@dataclass
class FactCaps:
    max_active_rows: int
    max_fact_chars: int
    max_total_chars: int


def current_caps() -> FactCaps:
    s = get_settings()
    return FactCaps(max_active_rows=s.dk_max_active_rows,
                    max_fact_chars=s.dk_max_fact_chars,
                    max_total_chars=s.dk_max_total_chars)


def check_caps(*, active_contents: list[str], new_content: str, is_active: bool) -> None:
    """Raise ``FactRejected`` if saving ``new_content`` would breach a cap. ``active_contents``
    are the OTHER active facts' contents (excluding the row being saved). A single fact is
    capped whether or not it is active; the row-count and total budget count ACTIVE facts only,
    since an inactive draft costs nothing on the prompt."""
    caps = current_caps()
    if len(new_content) > caps.max_fact_chars:
        raise FactRejected(
            f"This fact is {len(new_content)} characters, over the {caps.max_fact_chars}-"
            f"character limit for a single fact. Shorten it.")
    if not is_active:
        return
    if len(active_contents) + 1 > caps.max_active_rows:
        raise FactRejected(
            f"That would exceed the per-tenant limit of {caps.max_active_rows} active dynamic "
            f"facts. Turn one off first.")
    total = sum(len(c or "") for c in active_contents) + len(new_content)
    if total > caps.max_total_chars:
        raise FactRejected(
            f"That would push the total active fact text to {total} characters, over the "
            f"{caps.max_total_chars}-character budget. Shorten or turn one off.")
