"""Single write path into activity_events (Vocal Backend Tasksheet P0).

The tasksheet's "activity_events" table is this backend's existing `audit_events`
table (see app/db/models/audit.py) — same append-only, who-changed-what-when shape.
This helper is the one function new code should call to write a row there; it does
not replace the many pre-existing `db.add(AuditEvent(...))` call sites scattered
across the rest of the codebase (custom_instructions_admin.py, rules_admin.py,
tools_admin.py, etc.) — leave those as-is, this is the path for new work going forward.
"""
import uuid

from app.db.models import AuditEvent, Notification
from app.security.deps import Principal


def append_activity_event(
    db,
    principal: Principal,
    action: str,
    *,
    resource_type: str | None = None,
    resource_id: str | None = None,
    before: dict | None = None,
    after: dict | None = None,
    notify_user_ids: list[uuid.UUID] | None = None,
) -> AuditEvent:
    """Adds an AuditEvent row to the session (caller still commits). `action` is
    namespaced like "tools.connect" or "team.invite" — GET /activity's `category`
    filter matches on the prefix before the first '.'.

    `notify_user_ids`, when given, also fans this event out into `notifications`
    rows for those users (the bell icon's read-model). `id` is generated here
    explicitly rather than left to `IdMixin`'s column default, which is a
    SQLAlchemy-side default that only fires at flush/INSERT time — `event.id`
    would otherwise still be None when the Notification rows below are built.
    Opt-in per call site, not retrofitted onto every existing `AuditEvent` write."""
    event = AuditEvent(
        id=uuid.uuid4(),
        tenant_id=principal.tenant_id,
        actor=principal.actor,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        before=before,
        after=after,
    )
    db.add(event)
    for user_id in notify_user_ids or []:
        db.add(Notification(tenant_id=principal.tenant_id, user_id=user_id, event_id=event.id))
    return event
