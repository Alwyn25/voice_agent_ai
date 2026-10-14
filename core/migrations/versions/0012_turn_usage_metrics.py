"""Per-turn TTS/STT usage on decision_traces -- the raw facts a cost dashboard prices from.

Two nullable columns on ``decision_traces``:

  * ``tts_characters`` -- how many characters of the final (post-trim) response text were
    sent to speech synthesis this turn. NULL on a chat turn, where no TTS ever runs.
  * ``stt_seconds`` -- estimated duration of the caller's spoken utterance this turn
    (from VAD start/stop timestamps -- Deepgram doesn't return an exact billed duration per
    turn, so this is an estimate, not an invoice figure). NULL on a chat turn.

Deliberately NOT a dollar amount: pricing changes over time, and a $ figure written the day
it was billed at one rate would silently go stale the day the rate changes. Only the raw
counts are persisted; app/observability/pricing.py prices them at read time, so re-pricing
history is an edit to a rate table, not a backfill. ``tokens`` (LLM usage) already has this
same "raw counts, priced on read" shape via the existing JSONB column -- this just extends
the same idea to the two usage kinds that don't fit a token count.

Revision ID: 0012
Revises: 0011
"""
import sqlalchemy as sa
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    cols = _columns("decision_traces")
    if "tts_characters" not in cols:
        op.add_column("decision_traces", sa.Column("tts_characters", sa.Integer(),
                                                    nullable=True))
    if "stt_seconds" not in cols:
        op.add_column("decision_traces", sa.Column("stt_seconds", sa.Numeric(6, 2),
                                                    nullable=True))


def downgrade() -> None:
    cols = _columns("decision_traces")
    for col in ("stt_seconds", "tts_characters"):
        if col in cols:
            op.drop_column("decision_traces", col)
