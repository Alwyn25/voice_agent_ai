"""Rule set admin: view, edit (new version), activate/rollback, dry-run test."""
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import get_db
from app.db.models import AuditEvent, Customer, RuleSet
from app.engine.rules import engine as rule_engine
from app.engine.rules.context import build_context
from app.engine.rules.schema import RuleCompileError, compile_rules
from app.security.deps import Principal, require

router = APIRouter(prefix="/rules", tags=["rules"])


@router.get("")
async def list_rule_sets(p: Principal = Depends(require("rules:read")),
                         db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(RuleSet).where(RuleSet.tenant_id == p.tenant_id)
                             .order_by(RuleSet.name, RuleSet.version.desc()))).scalars().all()
    by_name: dict[str, dict] = {}
    for r in rows:
        entry = by_name.setdefault(r.name, {"name": r.name, "category": r.category,
                                            "versions": [], "active_version": None})
        entry["versions"].append(r.version)
        if r.is_active:
            entry["active_version"] = r.version
            entry["applies_to"] = (r.compiled or {}).get("applies_to", [])
            entry["rule_count"] = len((r.compiled or {}).get("rules", []))
            entry["description"] = (r.compiled or {}).get("description", "")
    return list(by_name.values())


@router.get("/{name}")
async def get_rule_set(name: str, version: int | None = None,
                       p: Principal = Depends(require("rules:read")),
                       db: AsyncSession = Depends(get_db)):
    row = await _find(db, p.tenant_id, name, version)
    return {"name": row.name, "category": row.category, "version": row.version,
            "is_active": row.is_active, "yaml_source": row.yaml_source,
            "compiled": row.compiled, "activated_at": row.activated_at}


@router.get("/{name}/versions")
async def versions(name: str, p: Principal = Depends(require("rules:read")),
                   db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(RuleSet).where(
        RuleSet.tenant_id == p.tenant_id, RuleSet.name == name)
        .order_by(RuleSet.version.desc()))).scalars().all()
    if not rows:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Rule set not found")
    return [{"version": r.version, "is_active": r.is_active, "created_at": r.created_at,
             "activated_at": r.activated_at,
             "rule_count": len((r.compiled or {}).get("rules", []))} for r in rows]


class RuleSetPut(BaseModel):
    yaml_source: str
    category: str | None = None


_DNC_PROTECTED_RULE_ID = "dnc_absolute"
_DNC_PROTECTED_RULE_SETS = frozenset({"calling_policy"})


def _check_dnc_invariant(name: str, compiled: dict) -> None:
    """Reject a calling_policy version that removes or demotes the dnc_absolute rule.

    This is a hard platform guarantee: a tenant cannot inadvertently deploy a rule set
    that strips the DNC firewall. Admins who need a test scenario can use /rules/.../test.
    """
    if name not in _DNC_PROTECTED_RULE_SETS:
        return
    rules = compiled.get("rules", [])
    dnc_rule = next((r for r in rules if r.get("id") == _DNC_PROTECTED_RULE_ID), None)
    if dnc_rule is None:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            {"code": "DNC_RULE_REQUIRED",
             "detail": f"The '{_DNC_PROTECTED_RULE_ID}' rule must be present in '{name}' "
                       "and cannot be removed. Add it back or test via /rules/{name}/test."})
    if not dnc_rule.get("terminal", False):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            {"code": "DNC_RULE_REQUIRED",
             "detail": f"'{_DNC_PROTECTED_RULE_ID}' must be terminal: true."})
    if dnc_rule.get("priority", 999) != 0:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            {"code": "DNC_RULE_REQUIRED",
             "detail": f"'{_DNC_PROTECTED_RULE_ID}' must have priority: 0."})


@router.put("/{name}")
async def put_rule_set(name: str, body: RuleSetPut,
                       p: Principal = Depends(require("rules:write")),
                       db: AsyncSession = Depends(get_db)):
    """Validates and stores as a NEW version. Does not activate it."""
    try:
        compiled = compile_rules(body.yaml_source)
    except RuleCompileError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            {"code": "RULE_COMPILE_ERROR", "problems": e.problems})
    if compiled["rule_set"] != name:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            f"rule_set in YAML is '{compiled['rule_set']}', expected '{name}'")
    _check_dnc_invariant(name, compiled)
    existing = (await db.execute(select(RuleSet).where(
        RuleSet.tenant_id == p.tenant_id, RuleSet.name == name)
        .order_by(RuleSet.version.desc()))).scalars().first()
    next_version = (existing.version + 1) if existing else 1
    row = RuleSet(tenant_id=p.tenant_id, name=name,
                  category=body.category or (existing.category if existing else "custom"),
                  yaml_source=body.yaml_source, compiled=compiled,
                  version=next_version, is_active=False)
    db.add(row)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="rules.new_version",
                      resource_type="rule_set", resource_id=name,
                      after={"version": next_version, "rules": len(compiled["rules"])}))
    await db.commit()
    return {"name": name, "version": next_version, "is_active": False,
            "rule_count": len(compiled["rules"]),
            "note": "Version saved. Call POST /rules/{name}/activate to make it live."}


class Activate(BaseModel):
    version: int


@router.post("/{name}/activate")
async def activate(name: str, body: Activate,
                   p: Principal = Depends(require("rules:write")),
                   db: AsyncSession = Depends(get_db)):
    target = await _find(db, p.tenant_id, name, body.version)
    rows = (await db.execute(select(RuleSet).where(
        RuleSet.tenant_id == p.tenant_id, RuleSet.name == name))).scalars().all()
    previous = next((r.version for r in rows if r.is_active), None)
    for r in rows:
        r.is_active = (r.id == target.id)
    target.activated_by = p.user_id
    target.activated_at = datetime.now(timezone.utc)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="rules.activate",
                      resource_type="rule_set", resource_id=name,
                      before={"active_version": previous},
                      after={"active_version": body.version}))
    await db.commit()
    return {"name": name, "active_version": body.version, "previous_version": previous}


class RuleTest(BaseModel):
    evaluation_point: str = "outbound_call"     # outbound_call | turn | tool:<name>
    customer_id: uuid.UUID | None = None
    proposal: dict | None = None
    session: dict | None = None
    turn: dict | None = None
    context_overrides: dict | None = None       # force values e.g. {"now": {"local_time": "22:30"}}


@router.post("/{name}/test")
async def test_rule_set(name: str, body: RuleTest,
                        p: Principal = Depends(require("rules:read")),
                        db: AsyncSession = Depends(get_db)):
    """Dry-run: shows which rules fire and the resulting verdicts. Executes nothing."""
    await _find(db, p.tenant_id, name, None)
    customer = None
    if body.customer_id:
        customer = (await db.execute(select(Customer).where(
            Customer.tenant_id == p.tenant_id,
            Customer.id == body.customer_id))).scalar_one_or_none()
        if customer is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Customer not found")

    ctx = await build_context(db, p.tenant_id, evaluation_point=body.evaluation_point,
                              customer=customer, proposal=body.proposal,
                              session=body.session, turn=body.turn)
    for ns, values in (body.context_overrides or {}).items():
        ctx.setdefault(ns, {}).update(values)

    decision = await rule_engine.evaluate_point(db, p.tenant_id,
                                                evaluation_point=body.evaluation_point, ctx=ctx)
    only = [e for e in decision.evaluated if e.get("rule_set") == name] or decision.evaluated
    return {
        "rule_set": name, "evaluation_point": body.evaluation_point,
        "allowed": decision.allowed,
        "blocks": [{"rule": v.rule_id, "reason": v.reason, "scope": v.body.get("scope")}
                   for v in decision.blocks],
        "requirements": [{"rule": v.rule_id, "condition": v.body.get("condition"),
                          "reason": v.body.get("reason")} for v in decision.requirements],
        "escalations": [{"rule": v.rule_id, "type": v.body.get("type"), "reason": v.reason}
                        for v in decision.escalations],
        "flags": decision.flags,
        "evaluated": only,
        "context_used": ctx,
    }


class RuleValidate(BaseModel):
    yaml_source: str


@router.post("/validate")
async def validate_yaml(body: RuleValidate, p: Principal = Depends(require("rules:read"))):
    """Compile-check YAML without saving — for live editor feedback."""
    try:
        compiled = compile_rules(body.yaml_source)
    except RuleCompileError as e:
        return {"valid": False, "problems": e.problems}
    return {"valid": True, "rule_set": compiled["rule_set"],
            "applies_to": compiled["applies_to"],
            "rules": [{"id": r["id"], "priority": r["priority"], "terminal": r["terminal"],
                       "verdicts": [next(iter(v)) for v in r["then"]]} for r in compiled["rules"]]}


async def _find(db: AsyncSession, tenant_id: uuid.UUID, name: str, version: int | None) -> RuleSet:
    q = select(RuleSet).where(RuleSet.tenant_id == tenant_id, RuleSet.name == name)
    q = q.where(RuleSet.version == version) if version else q.where(RuleSet.is_active == True)  # noqa: E712
    row = (await db.execute(q)).scalars().first()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            f"Rule set '{name}'{f' v{version}' if version else ' (active)'} not found")
    return row
