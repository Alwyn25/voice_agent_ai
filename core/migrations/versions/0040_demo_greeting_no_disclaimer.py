"""GTM demo: drop "This is a demo running on sample data." from the spoken greeting.

Seeds only ever create, so the changed real_estate.yaml greeting never reaches a database
that already has the industry, and records pin their industry version. This removes the
sentence from every stored definition's agent greeting in place, leaving the rest of each
greeting (and any admin edits) alone. The persona rule that tells the agent the data is
sample data, if the caller asks, is unchanged (app/demo/agent.py ALWAYS_GUARDRAILS).

Revision ID: 0040
Revises: 0039
"""
from alembic import op

revision = "0040"
down_revision = "0039"
branch_labels = None
depends_on = None

SENTENCE = "This is a demo running on sample data. "
AFTER = "Hello, this is the {{project_name}} enquiry desk. "


def _set_greeting(old: str, new: str) -> None:
    op.execute(
        "UPDATE demo_industry_versions SET definition = jsonb_set(definition, '{agent,greeting}', "
        f"to_jsonb(replace(definition #>> '{{agent,greeting}}', '{old}', '{new}'))) "
        f"WHERE strpos(definition #>> '{{agent,greeting}}', '{old}') > 0")


def upgrade() -> None:
    _set_greeting(SENTENCE, "")


def downgrade() -> None:
    _set_greeting(AFTER, AFTER + SENTENCE)
