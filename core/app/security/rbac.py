"""Role -> permission mapping and permission checks.

Permissions are strings like "customers:read", "rules:write".
"*" grants everything (admin).
Roles are seeded into the roles table; the User.role column names one of them.
"""

DEFAULT_ROLES: dict[str, list[str]] = {
    "admin": ["*"],
    # Vocal Backend Tasksheet P0 role enum (owner/admin/ops_lead/agent_manager/viewer),
    # added alongside the pre-existing admin/supervisor/agent/analyst set rather than
    # replacing it, since the rest of this codebase (seed.py, every other router)
    # already depends on those names.
    "owner": ["*"],
    "ops_lead": [
        "customers:read", "customers:write", "customers:state_override",
        "calls:read", "calls:write", "conversations:write",
        "facts:write", "conversations:read", "recordings:listen",
        "appointments:read", "appointments:write", "tickets:read", "tickets:write",
        "escalations:read", "escalations:write", "knowledge:read", "knowledge:write",
        "campaigns:read", "campaigns:write", "analytics:read", "analytics:write", "audit:read",
        "rules:read", "tools:read", "tools:write", "scheduling:write", "dnc:read",
        "instructions:read", "instructions:write", "users:read", "users:write",
        "templates:read", "templates:write", "agents:read", "agents:write",
        # Vocal Backend Tasksheet P8 (Settings tier): team/integrations management fits
        # ops_lead's existing admin-adjacent scope; billing:read only — seat/plan
        # CHANGES (billing:write) stay owner/admin-only, same "money is more
        # restricted than team management" judgement call as recordings:listen
        # being narrower than the rest of this role's grants.
        "team:read", "team:write", "integrations:read", "integrations:write",
        "billing:read",
    ],
    "agent_manager": [
        "knowledge:read", "knowledge:write", "tools:read", "tools:write",
        "instructions:read", "instructions:write", "campaigns:read",
        "analytics:read", "analytics:write", "audit:read", "users:read",
        "templates:read", "templates:write", "agents:read", "agents:write",
    ],
    "viewer": ["analytics:read", "audit:read", "conversations:read", "calls:read",
              "campaigns:read", "customers:read", "knowledge:read", "tools:read",
              "users:read", "templates:read", "agents:read",
              "team:read", "integrations:read"],
    "supervisor": [
        "customers:read", "customers:write", "customers:state_override",
        "calls:read", "calls:write", "conversations:write",
        "facts:write", "conversations:read", "calls:read", "recordings:listen",
        "appointments:read", "appointments:write", "tickets:read", "tickets:write",
        "escalations:read", "escalations:write", "knowledge:read", "knowledge:write",
        "campaigns:read", "campaigns:write", "analytics:read", "analytics:write", "audit:read",
        "rules:read", "tools:read", "tools:write", "scheduling:write", "dnc:read",
        "instructions:read", "instructions:write", "users:read",
        "templates:read", "templates:write", "agents:read", "agents:write",
        "team:read",
    ],
    "agent": [
        "customers:read", "customers:write", "conversations:read", "calls:read",
        "campaigns:read",
        "appointments:read", "appointments:write", "tickets:read", "tickets:write",
        "escalations:read", "escalations:write", "knowledge:read", "tools:read",
        "templates:read", "agents:read",
    ],
    "analyst": ["analytics:read", "audit:read", "conversations:read", "calls:read"],
}


def permissions_for(role_permissions: list[str], needed: str) -> bool:
    if "*" in role_permissions:
        return True
    if needed in role_permissions:
        return True
    # allow "customers:*" style wildcards
    prefix = needed.split(":")[0] + ":*"
    return prefix in role_permissions


def permissions_for_role(role: str) -> list[str]:
    """Return the permission list for a given role name. Always reads from
    DEFAULT_ROLES at call time, so newly-added permissions (team:*, billing:*,
    integrations:*) are available to existing tenants without a backfill
    migration — Copilot review #10."""
    return DEFAULT_ROLES.get(role, [])
