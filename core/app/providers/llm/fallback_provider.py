"""Sticky, ordered N-provider LLM fallback chain, entirely config-driven (LLM_FALLBACK_ORDER
in router.py builds the chain this class walks) -- built specifically for the case where a
provider's account runs out of credit (a real HTTP 402 -- confirmed live for Cerebras, see
cerebras_provider.py) and every call needs to keep working anyway, without the app ever
hardcoding which provider comes after which.

The credit-exhaustion switch is STICKY and PROCESS-WIDE, and only ever moves FORWARD through
the chain: the first time the current entry fails with a credit-exhaustion error, this
advances to the next entry for every call from then on, without ever trying an earlier entry
again. Re-probing an exhausted provider on each new turn would mean paying its full
request/retry latency on every single turn for no benefit -- exactly the cost this is built
to avoid. There is no cheaper unit of "session" available here (get_llm() itself is cached
for the whole process -- see router.py), so the position in the chain persists until the
process restarts, which is also naturally when a real credit top-up would take effect anyway
(a redeploy).

A SEPARATE, non-sticky mechanism handles rate-limiting (429): unlike credit exhaustion, a 429
is expected to clear on its own within a short window, so permanently abandoning the primary
provider over one busy minute would mean a production deployment configured with, say,
Cerebras primary quietly running on its fallback for the rest of the process's lifetime --
directly against the point of naming a primary at all. THE BUG THIS FIXES: confirmed live,
twice, with two different providers (Cerebras, then OpenAI) -- a sustained burst of 429s
exhausted the underlying HTTP client's own retries, raised as a plain LLMError this class
did not recognise as anything special, and the caller's generic exception handler produced
the customer-facing "I'm having trouble on my side" escalation instead of ever trying another
configured provider. Each chain entry gets its own short, independent cooldown instead: a 429
skips that entry for `_RATE_LIMIT_COOLDOWN_S`, trying the next configured entry immediately
(within the SAME call, so the caller never sees the failure), and automatically retries the
original entry again once its cooldown expires -- the primary provider is never permanently
demoted, only skipped while it is actually busy.

Any OTHER failure (a timeout, a transient 5xx, a bad request) is raised through unchanged
from whichever entry is currently active -- those are the existing retry/error-handling
paths' job, not a reason to fail over at all.
"""
import logging
import re
import time

from app.providers.llm.base import LLMError, LLMProvider, LLMResult

logger = logging.getLogger(__name__)

# How long a chain entry that just 429'd is skipped before being tried again. Long enough
# that a real per-minute rate limit window has actually reset; short enough that a production
# deployment never sits on its fallback for more than about a minute past the point its
# primary recovers. Not configurable via settings (yet) -- this is a first, deliberately
# simple version; a per-provider or settings-driven value is a reasonable future refinement,
# not a correctness requirement for this fix.
_RATE_LIMIT_COOLDOWN_S = 60.0

# A bare "429" substring match also fires on an unrelated error that happens to contain those
# digits (a request id, a timeout message) -- word-bounded so only the status code itself counts.
_429_RE = re.compile(r"\b429\b")


def _is_credit_exhausted(err: LLMError) -> bool:
    """True if `err` represents the active provider being out of credit/quota, not some
    other failure. Checks the wrapped HTTP status first (each provider's `raise LLMError(...)
    from e` keeps the original exception on __cause__), then falls back to matching the error
    text -- the real Cerebras response is 402 with body "Payment required to access this
    resource. Visit your billing tab.", but other OpenAI-protocol providers may phrase the
    same condition differently (e.g. OpenAI's own "insufficient_quota")."""
    cause = err.__cause__
    if getattr(cause, "status_code", None) == 402:
        return True
    message = str(err).lower()
    return "402" in message or "payment required" in message or "insufficient" in message


def _is_rate_limited(err: LLMError) -> bool:
    """True if `err` represents the active provider being rate-limited (429) -- the real
    Cerebras/OpenAI response seen live is 429 with a "too many requests" / "high traffic"
    body. Distinct from _is_credit_exhausted (402): a 429 clears on its own, so it drives the
    short, non-sticky per-entry cooldown below, never the permanent chain advance."""
    cause = err.__cause__
    if getattr(cause, "status_code", None) == 429:
        return True
    message = str(err).lower()
    return bool(_429_RE.search(message)) or "too many requests" in message or "rate limit" in message


class FallbackLLMProvider:
    """Tries the chain's entries in order; on a credit-exhaustion failure, advances to the
    next entry for the rest of this process, and on a rate-limit failure, skips that one
    entry for a short cooldown instead (see module docstring for why the two are handled so
    differently). `chain` is an ordered list of (name, provider) pairs, built entirely from
    config by router.py -- this class has no opinion on what belongs in it or in what order."""

    def __init__(self, chain: list[tuple[str, LLMProvider]]) -> None:
        if len(chain) < 2:
            raise ValueError("FallbackLLMProvider needs at least 2 chain entries")
        self._chain = chain
        self._index = 0
        # Chain index -> the time.monotonic() timestamp its 429 cooldown ends. Entries never
        # present here (or whose cooldown has passed) are fair game; this is intentionally
        # separate from `_index`, which tracks only the sticky credit-exhaustion position.
        self._rate_limited_until: dict[int, float] = {}

    @property
    def name(self) -> str:
        # Live, not snapshotted -- reflects whichever entry is actually serving calls right
        # now, at any point in the chain's progression.
        return self._chain[self._index][0]

    def _advance(self, failed_index: int, err: LLMError) -> bool:
        """Moves the sticky position past the entry that actually just failed (`failed_index`,
        which a 429 cooldown may have sent us to ahead of `self._index`) if one exists beyond
        it. Returns False (and leaves the index untouched) when that would run off the end of
        the chain, so the caller knows to let the error propagate instead of looping forever.

        THE BUG THIS FIXES: using `self._index + 1` here instead of `failed_index + 1` moved
        the WRONG entry's position. Chain A -> B -> C, A on a 429 cooldown so this call is
        serving from B; B then returns 402. The old code advanced `self._index` (0, A) to 1
        (B) -- but B is the one that just ran out of credit, not A, so B stayed "current" and
        every later call retried it first, wasting a 402 round trip before ever reaching C.
        A -- only ever busy for a minute -- was permanently dropped for nothing."""
        nxt = max(self._index, failed_index) + 1
        if nxt >= len(self._chain):
            return False
        old_name = self._chain[self._index][0]
        self._index = nxt
        new_name = self._chain[self._index][0]
        logger.warning(
            "llm-fallback: %r out of credit (%s) -- switching to %r for the rest of this "
            "process (chain: %s)", old_name, err, new_name,
            " -> ".join(n for n, _ in self._chain))
        return True

    def _next_available_index(self, start: int) -> int:
        """The chain index at or after `start` -- NEVER wrapping to an earlier one -- that is
        not currently serving a 429 cooldown. Falls back to `start` itself if every remaining
        entry is still cooling down -- never block a real call forever on provider bookkeeping;
        let it actually try and surface whatever happens through the normal error path.

        THE BUG THIS FIXES: wrapping with `% n` could return an index BEFORE `self._index`.
        Every entry before `self._index` is known dead (that is exactly what moved `_index`
        past it), so retrying one wastes a round trip and -- worse -- a 402 from it looks like
        "already on the last entry" to `_advance`, raising the error straight to the caller
        and leaving every later call in this process repeating the same dead entry first."""
        now = time.monotonic()
        for i in range(start, len(self._chain)):
            if self._rate_limited_until.get(i, 0.0) <= now:
                return i
        return start

    def _mark_rate_limited(self, index: int, err: LLMError) -> None:
        self._rate_limited_until[index] = time.monotonic() + _RATE_LIMIT_COOLDOWN_S
        logger.warning(
            "llm-fallback: %r rate-limited (%s) -- cooling down for %ss, trying the next "
            "available entry for this call (chain: %s)",
            self._chain[index][0], err, _RATE_LIMIT_COOLDOWN_S,
            " -> ".join(n for n, _ in self._chain))

    async def complete_json(self, *, system: str, user: str, model: str | None = None,
                            temperature: float = 0.2, max_tokens: int = 900,
                            schema: dict | None = None,
                            schema_name: str = "response") -> LLMResult:
        kwargs = dict(system=system, user=user, model=model, temperature=temperature,
                     max_tokens=max_tokens, schema=schema, schema_name=schema_name)
        rate_limited_this_call: set[int] = set()
        while True:
            i = self._next_available_index(self._index)
            _, provider = self._chain[i]
            try:
                return await provider.complete_json(**kwargs)
            except LLMError as e:
                if _is_credit_exhausted(e):
                    if self._advance(i, e):
                        continue
                    raise
                # Bounded to one skip per chain entry per call -- if every entry is 429ing
                # right now, this surfaces the last error rather than spinning forever.
                if _is_rate_limited(e) and i not in rate_limited_this_call:
                    rate_limited_this_call.add(i)
                    self._mark_rate_limited(i, e)
                    if len(rate_limited_this_call) < len(self._chain):
                        continue
                raise

    async def stream_text(self, *, system: str, user: str, model: str | None = None,
                          temperature: float = 0.2, max_tokens: int = 900,
                          schema: dict | None = None, schema_name: str = "response"):
        kwargs = dict(system=system, user=user, model=model, temperature=temperature,
                     max_tokens=max_tokens, schema=schema, schema_name=schema_name)
        rate_limited_this_call: set[int] = set()
        while True:
            i = self._next_available_index(self._index)
            _, provider = self._chain[i]
            try:
                # A credit-exhaustion or rate-limit failure always fails the initial request
                # before any token is yielded (every provider raises it from the opening
                # `create()` call) -- so switching entries here never risks yielding a
                # partial stream from one entry followed by a second, overlapping stream
                # from the next.
                async for token in provider.stream_text(**kwargs):
                    yield token
                return
            except LLMError as e:
                if _is_credit_exhausted(e):
                    if self._advance(i, e):
                        continue
                    raise
                if _is_rate_limited(e) and i not in rate_limited_this_call:
                    rate_limited_this_call.add(i)
                    self._mark_rate_limited(i, e)
                    if len(rate_limited_this_call) < len(self._chain):
                        continue
                raise
