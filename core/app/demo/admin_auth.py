"""Who may administer demos (docs/gtm-demo-links.md §5.2).

Roles in this codebase are per tenant and "admin"/"owner" hold "*", so a plain
require("demo:admin") would let ANY customer's admin read every prospect and lead. Demo
administration is therefore pinned to our own staff organisation(s), listed in
DEMO_ADMIN_TENANT_IDS, and additionally needs the demo:admin permission ("*" includes it).
While DEMO_ENABLED is false the whole surface answers 404, as if it did not exist.
"""
import uuid

from fastapi import Depends, HTTPException, status

from app.config import get_settings
from app.demo.settings import csv_list
from app.security.deps import Principal, get_principal
from app.security.rbac import permissions_for


def demo_admin_tenant_ids() -> set[uuid.UUID]:
    out = set()
    for v in csv_list(get_settings().demo_admin_tenant_ids):
        try:
            out.add(uuid.UUID(v))
        except ValueError:
            continue           # the startup check already refuses this; never widen access
    return out


async def require_demo_admin(p: Principal = Depends(get_principal)) -> Principal:
    if not get_settings().demo_enabled:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found")
    if p.tenant_id not in demo_admin_tenant_ids():
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "Demo administration is limited to the Phonn staff organisation")
    if not permissions_for(p.permissions, "demo:admin"):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Missing permission: demo:admin")
    return p
