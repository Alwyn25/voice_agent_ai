"""Write side of tool performance tracking.

The only caller is the one line added in app.engine.tools.executor.execute — nothing
else in the engine needs to import or know about this module. Every ToolOutcome the
executor produces (success, failed, blocked, skipped_duplicate) is recorded here, unlike
ToolExecution which only sees success/failed.
"""
import logging

from app.db.base import get_session_factory
from app.db.models.tool_metrics import ToolMetric
from app.engine.tools.types import ToolContext, ToolOutcome

logger = logging.getLogger(__name__)


async def record(ctx: ToolContext, outcome: ToolOutcome) -> None:
    """Fire-and-forget, and deliberately on its own session — never ctx.db.

    ctx.db's transaction is the tool's own: on a blocked or failed outcome the caller
    correctly rolls it back, since nothing about a failed action should be committed.
    Writing the metric into that same transaction means it would vanish along with the
    rollback, silently under-recording every non-success outcome — the exact gap this
    module exists to close in ToolExecution. A short-lived session commits the metric
    independently of whatever ctx.db decides to do.
    """
    try:
        async with get_session_factory()() as session:
            session.add(ToolMetric(
                tenant_id=ctx.tenant_id, tool_name=outcome.tool_name, status=outcome.status,
                failed_check=outcome.failed_check, error=outcome.error,
                duration_ms=outcome.duration_ms, conversation_id=ctx.conversation_id,
                decision_trace_id=ctx.decision_trace_id))
            await session.commit()
    except Exception:
        logger.exception("tool_metrics.record failed for tool=%s", outcome.tool_name)
