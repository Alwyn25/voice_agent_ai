"""Campaigns — bulk outbound with the policy gate applied per dial, not per campaign.

The preview is the important endpoint: it shows exactly who would be excluded and why
BEFORE anything dials, so a compliance mistake is caught by a human reading a screen
rather than by a customer receiving a call they opted out of.
"""
import csv
import io
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (Agent, AuditEvent, Call, CallbackRequest, Campaign, CampaignTarget,
                           Conversation, Customer)
from app.db.base import get_db
from app.security.deps import Principal, require
from app.services.events import publish
from app.services.outbound import check_can_call, place_outbound_call
from app.services.telephony.base import ACTIVE_STATUSES

router = APIRouter(prefix="/campaigns", tags=["campaigns"])

PREVIEW_LIMIT = 500


class AudienceFilter(BaseModel):
    states: list[str] = []
    tags: list[str] = []
    languages: list[str] = []
    region_codes: list[str] = []
    lead_sources: list[str] = []
    exclude_states: list[str] = ["do_not_contact", "closed_lost"]


class CampaignIn(BaseModel):
    name: str
    # Vocal Backend Tasksheet P5: an explicit contact list (contact_ids) is an alternative to
    # the pre-existing filter-based `audience` — when given, it takes priority and the campaign
    # dials exactly those contacts rather than whatever currently matches the filter. Stored
    # inside `audience_filter["contact_ids"]` so `_audience_query` stays the single place that
    # resolves a campaign to its customer set (no second resolution path to keep in sync).
    contact_ids: list[uuid.UUID] | None = None
    audience: AudienceFilter = AudienceFilter()
    agent_id: uuid.UUID | None = None
    calls_per_hour: int = 30
    purpose: str = "marketing"
    concurrency_limit: int | None = None
    calling_rules: dict = {}
    starts_on: datetime | None = None
    ends_on: datetime | None = None


def _audience_query(tenant_id: uuid.UUID, af: dict):
    q = select(Customer).where(Customer.tenant_id == tenant_id,
                               Customer.is_deleted == False)  # noqa: E712
    if af.get("contact_ids"):
        return q.where(Customer.id.in_(af["contact_ids"]))
    if af.get("states"):
        q = q.where(Customer.current_state.in_(af["states"]))
    if af.get("exclude_states"):
        q = q.where(Customer.current_state.notin_(af["exclude_states"]))
    if af.get("languages"):
        q = q.where(Customer.preferred_language.in_(af["languages"]))
    if af.get("region_codes"):
        q = q.where(Customer.region_code.in_(af["region_codes"]))
    if af.get("lead_sources"):
        q = q.where(Customer.lead_source.in_(af["lead_sources"]))
    for tag in af.get("tags", []):
        q = q.where(Customer.tags.contains([tag]))
    return q


@router.post("", status_code=201)
async def create_campaign(body: CampaignIn, p: Principal = Depends(require("campaigns:write")),
                          db: AsyncSession = Depends(get_db)):
    if body.agent_id is not None:
        agent = (await db.execute(select(Agent).where(
            Agent.tenant_id == p.tenant_id, Agent.id == body.agent_id))).scalar_one_or_none()
        if agent is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "agent_id not found")

    audience = body.audience.model_dump()
    if body.contact_ids:
        audience["contact_ids"] = [str(cid) for cid in body.contact_ids]

    c = Campaign(tenant_id=p.tenant_id, name=body.name[:200], status="draft",
                 audience_filter=audience, agent_id=body.agent_id,
                 concurrency_limit=body.concurrency_limit, calling_rules=body.calling_rules,
                 schedule={"calls_per_hour": body.calls_per_hour, "purpose": body.purpose,
                           "starts_on": body.starts_on.isoformat() if body.starts_on else None,
                           "ends_on": body.ends_on.isoformat() if body.ends_on else None},
                 stats={})
    db.add(c)
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="campaigns.create",
                      resource_type="campaign", after={"name": body.name}))
    await db.commit()
    await db.refresh(c)
    return await _campaign_json_with_tiles(db, c)


class CampaignPatch(BaseModel):
    # Only the agent assignment is editable post-creation for now — the Agent Roster
    # card's "Assigned campaign" dropdown is this endpoint's one caller. Other fields
    # (audience, schedule, etc.) still go through the campaign builder instead.
    agent_id: uuid.UUID | None = None


@router.patch("/{campaign_id}")
async def update_campaign(campaign_id: uuid.UUID, body: CampaignPatch,
                          p: Principal = Depends(require("campaigns:write")),
                          db: AsyncSession = Depends(get_db)):
    c = await _campaign(db, p.tenant_id, campaign_id)
    before = {"agent_id": str(c.agent_id) if c.agent_id else None}

    if body.agent_id is not None:
        agent = (await db.execute(select(Agent).where(
            Agent.tenant_id == p.tenant_id, Agent.id == body.agent_id))).scalar_one_or_none()
        if agent is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "agent_id not found")
        c.agent_id = body.agent_id

    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="campaigns.update",
                      resource_type="campaign", resource_id=str(campaign_id),
                      before=before, after={"agent_id": str(c.agent_id) if c.agent_id else None}))
    await db.commit()
    await db.refresh(c)
    return await _campaign_json_with_tiles(db, c)


@router.get("")
async def list_campaigns(p: Principal = Depends(require("campaigns:read")),
                         db: AsyncSession = Depends(get_db),
                         status_filter: str | None = Query(None, alias="status")):
    q = select(Campaign).where(Campaign.tenant_id == p.tenant_id)
    if status_filter:
        q = q.where(Campaign.status == status_filter)
    rows = (await db.execute(q.order_by(Campaign.created_at.desc()))).scalars().all()
    tiles = await _tile_counts_batch(db, p.tenant_id, [c.id for c in rows])
    return [_campaign_json(c, tiles.get(c.id)) for c in rows]


@router.get("/{campaign_id}")
async def get_campaign(campaign_id: uuid.UUID,
                       p: Principal = Depends(require("campaigns:read")),
                       db: AsyncSession = Depends(get_db)):
    c = await _campaign(db, p.tenant_id, campaign_id)
    counts = dict((await db.execute(
        select(CampaignTarget.status, func.count()).where(
            CampaignTarget.tenant_id == p.tenant_id,
            CampaignTarget.campaign_id == campaign_id).group_by(CampaignTarget.status))).all())
    tiles = await _tile_counts_batch(db, p.tenant_id, [campaign_id])
    return {**_campaign_json(c, tiles.get(campaign_id)),
           "target_counts": {k: int(v) for k, v in counts.items()}}


@router.post("/{campaign_id}/preview")
async def preview_campaign(campaign_id: uuid.UUID,
                           p: Principal = Depends(require("campaigns:read")),
                           db: AsyncSession = Depends(get_db),
                           limit: int = Query(50, ge=1, le=PREVIEW_LIMIT)):
    """Who would be called, who would not, and exactly which rule stops each exclusion."""
    c = await _campaign(db, p.tenant_id, campaign_id)
    customers = (await db.execute(
        _audience_query(p.tenant_id, c.audience_filter or {}).limit(PREVIEW_LIMIT))).scalars().all()

    eligible, excluded = [], []
    exclusion_counts: dict[str, int] = {}
    purpose = (c.schedule or {}).get("purpose", "marketing")

    for cust in customers:
        decision = await check_can_call(db, tenant_id=p.tenant_id, customer=cust, purpose=purpose)
        if decision.allowed:
            if len(eligible) < limit:
                eligible.append({"customer_id": str(cust.id), "name": cust.full_name,
                                 "phone": cust.phone_normalized, "state": cust.current_state})
        else:
            key = decision.blocked_by or "unknown"
            exclusion_counts[key] = exclusion_counts.get(key, 0) + 1
            if len(excluded) < limit:
                excluded.append({"customer_id": str(cust.id), "name": cust.full_name,
                                 "state": cust.current_state,
                                 "blocked_by": decision.blocked_by, "reason": decision.reason})

    return {"campaign_id": str(campaign_id), "audience_matched": len(customers),
            "eligible_now": len(customers) - sum(exclusion_counts.values()),
            "excluded_total": sum(exclusion_counts.values()),
            "exclusions_by_rule": exclusion_counts,
            "sample_eligible": eligible, "sample_excluded": excluded,
            "note": "Policy is re-checked at dial time; a customer who opts out after this "
                    "preview is still protected."}


@router.post("/{campaign_id}/start")
async def start_campaign(campaign_id: uuid.UUID,
                         p: Principal = Depends(require("campaigns:write")),
                         db: AsyncSession = Depends(get_db)):
    """Materialise the target list and mark the campaign running. The dialer worker
    (telephony stage) consumes pending targets; each dial re-checks policy."""
    c = await _campaign(db, p.tenant_id, campaign_id)
    if c.status == "running":
        raise HTTPException(status.HTTP_409_CONFLICT, "Campaign is already running")

    customers = (await db.execute(
        _audience_query(p.tenant_id, c.audience_filter or {}).limit(PREVIEW_LIMIT))).scalars().all()
    existing = set((await db.execute(select(CampaignTarget.customer_id).where(
        CampaignTarget.tenant_id == p.tenant_id,
        CampaignTarget.campaign_id == campaign_id))).scalars().all())

    added = 0
    for cust in customers:
        if cust.id in existing:
            continue
        db.add(CampaignTarget(tenant_id=p.tenant_id, campaign_id=campaign_id,
                              customer_id=cust.id, status="pending"))
        added += 1

    c.status = "running"
    c.stats = {**(c.stats or {}), "targets": len(existing) + added,
               "started_at": datetime.now(timezone.utc).isoformat()}
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="campaigns.start",
                      resource_type="campaign", resource_id=str(campaign_id),
                      after={"targets_added": added}))
    await db.commit()
    publish(p.tenant_id, "campaign.started", campaign_id=str(campaign_id),
            name=c.name, targets=len(existing) + added)
    return {"id": str(campaign_id), "status": c.status, "targets_added": added,
            "targets_total": len(existing) + added}


@router.post("/{campaign_id}/pause")
async def pause_campaign(campaign_id: uuid.UUID,
                         p: Principal = Depends(require("campaigns:write")),
                         db: AsyncSession = Depends(get_db)):
    c = await _campaign(db, p.tenant_id, campaign_id)
    c.status = "paused"
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="campaigns.pause",
                      resource_type="campaign", resource_id=str(campaign_id)))
    await db.commit()
    publish(p.tenant_id, "campaign.paused", campaign_id=str(campaign_id))
    return {"id": str(campaign_id), "status": c.status}


@router.post("/{campaign_id}/resume")
async def resume_campaign(campaign_id: uuid.UUID,
                          p: Principal = Depends(require("campaigns:write")),
                          db: AsyncSession = Depends(get_db)):
    """Resume dialing a paused campaign. Distinct from /start: start (re)materialises the
    target list from the audience filter, which would be wrong here — a paused campaign's
    targets (including whichever are mid-retry) must be left exactly as they are, only the
    dialer's willingness to pick them back up changes."""
    c = await _campaign(db, p.tenant_id, campaign_id)
    if c.status != "paused":
        raise HTTPException(status.HTTP_409_CONFLICT, f"Campaign is {c.status}, not paused")
    c.status = "running"
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="campaigns.resume",
                      resource_type="campaign", resource_id=str(campaign_id)))
    await db.commit()
    publish(p.tenant_id, "campaign.resumed", campaign_id=str(campaign_id))
    return {"id": str(campaign_id), "status": c.status}


@router.post("/{campaign_id}/stop")
async def stop_campaign(campaign_id: uuid.UUID,
                        p: Principal = Depends(require("campaigns:write")),
                        db: AsyncSession = Depends(get_db)):
    c = await _campaign(db, p.tenant_id, campaign_id)
    c.status = "completed"
    c.stats = {**(c.stats or {}), "stopped_at": datetime.now(timezone.utc).isoformat()}
    db.add(AuditEvent(tenant_id=p.tenant_id, actor=p.actor, action="campaigns.stop",
                      resource_type="campaign", resource_id=str(campaign_id)))
    await db.commit()
    publish(p.tenant_id, "campaign.stopped", campaign_id=str(campaign_id))
    return {"id": str(campaign_id), "status": c.status}


@router.get("/{campaign_id}/targets")
async def list_targets(campaign_id: uuid.UUID,
                       p: Principal = Depends(require("campaigns:read")),
                       db: AsyncSession = Depends(get_db),
                       status_filter: str | None = Query(None, alias="status"),
                       page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200)):
    await _campaign(db, p.tenant_id, campaign_id)
    q = select(CampaignTarget).where(CampaignTarget.tenant_id == p.tenant_id,
                                     CampaignTarget.campaign_id == campaign_id)
    if status_filter:
        q = q.where(CampaignTarget.status == status_filter)
    rows = (await db.execute(q.order_by(CampaignTarget.created_at)
                             .offset((page - 1) * page_size).limit(page_size))).scalars().all()
    return [{"id": str(t.id), "customer_id": str(t.customer_id), "status": t.status,
             "attempts": t.attempts, "last_attempt_at": t.last_attempt_at,
             "exclusion_reason": t.exclusion_reason} for t in rows]


def _calls_query(tenant_id: uuid.UUID, campaign_id: uuid.UUID, *, status_filter: str | None,
                 intent: str | None, sentiment: str | None, search: str | None):
    """Shared by GET /calls and GET /export so the two stay in lockstep — the export button
    downloads exactly whatever filter is currently applied to the calls table."""
    q = (select(Call, Conversation, Customer, Agent)
        .outerjoin(Conversation, Conversation.id == Call.conversation_id)
        .outerjoin(Customer, Customer.id == Call.customer_id)
        .outerjoin(Agent, Agent.id == Call.agent_id)
        .where(Call.tenant_id == tenant_id, Call.campaign_id == campaign_id))
    if status_filter:
        q = q.where(Call.status == status_filter)
    if intent:
        q = q.where(Conversation.intent_labels.contains([intent]))
    if sentiment:
        q = q.where(Conversation.sentiment == sentiment)
    if search:
        escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like = "%" + escaped + "%"
        q = q.where(or_(Customer.full_name.ilike(like, escape="\\"),
                        Customer.phone_normalized.ilike(like, escape="\\")))
    return q


def _call_row_json(call: Call, conv: Conversation | None, cust: Customer | None,
                   agent: Agent | None = None) -> dict:
    return {"call_id": str(call.id), "customer_id": str(call.customer_id) if call.customer_id else None,
           "customer_name": cust.full_name if cust else None,
           "phone": cust.phone_normalized if cust else call.to_number,
           "agent_name": agent.name if agent else None,
           "status": call.status, "attempt": call.attempt,
           "intent_labels": conv.intent_labels if conv else [],
           "sentiment": conv.sentiment if conv else None,
           "outcome": conv.outcome if conv else None,
           "duration_seconds": call.duration_seconds,
           "answered_at": call.answered_at, "ended_at": call.ended_at,
           "created_at": call.created_at}


@router.get("/{campaign_id}/calls")
async def list_campaign_calls(campaign_id: uuid.UUID,
                              p: Principal = Depends(require("campaigns:read")),
                              db: AsyncSession = Depends(get_db),
                              status_filter: str | None = Query(None, alias="status"),
                              intent: str | None = None, sentiment: str | None = None,
                              search: str | None = None,
                              page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200)):
    """The big campaign-detail table: every call this campaign has made, filterable by
    intent/status/sentiment/search — distinct from GET /targets, which lists dial TARGETS
    (one row per contact) rather than calls (one row per attempt)."""
    await _campaign(db, p.tenant_id, campaign_id)
    q = _calls_query(p.tenant_id, campaign_id, status_filter=status_filter, intent=intent,
                     sentiment=sentiment, search=search)
    rows = (await db.execute(q.order_by(Call.created_at.desc())
                             .offset((page - 1) * page_size).limit(page_size))).all()
    return [_call_row_json(call, conv, cust, agent) for call, conv, cust, agent in rows]


@router.get("/{campaign_id}/export")
async def export_campaign_calls(campaign_id: uuid.UUID,
                                p: Principal = Depends(require("campaigns:read")),
                                db: AsyncSession = Depends(get_db),
                                status_filter: str | None = Query(None, alias="status"),
                                intent: str | None = None, sentiment: str | None = None,
                                search: str | None = None):
    """CSV export matching the Export button on the calls table — same filters as
    GET /calls, unpaginated (capped at PREVIEW_LIMIT rows, matching the campaign engine's
    existing per-request cap on customer-scale queries elsewhere in this router)."""
    await _campaign(db, p.tenant_id, campaign_id)
    q = _calls_query(p.tenant_id, campaign_id, status_filter=status_filter, intent=intent,
                     sentiment=sentiment, search=search)
    rows = (await db.execute(q.order_by(Call.created_at.desc()).limit(PREVIEW_LIMIT))).all()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["call_id", "customer_name", "phone", "status", "attempt", "intent_labels",
                     "sentiment", "outcome", "duration_seconds", "answered_at", "ended_at",
                     "created_at"])
    for call, conv, cust, agent in rows:
        r = _call_row_json(call, conv, cust, agent)
        writer.writerow([r["call_id"], r["customer_name"] or "", r["phone"] or "", r["status"],
                         r["attempt"], ";".join(r["intent_labels"]), r["sentiment"] or "",
                         r["outcome"] or "", r["duration_seconds"] or "",
                         r["answered_at"] or "", r["ended_at"] or "", r["created_at"]])

    filename = f"campaign_{campaign_id}_calls.csv"
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/{campaign_id}/live")
async def live_campaign(campaign_id: uuid.UUID,
                        p: Principal = Depends(require("campaigns:read")),
                        db: AsyncSession = Depends(get_db)):
    """Real-time snapshot for the campaign-detail header's '25 lines in use' badge — cheap
    enough to poll every 10s (two count() queries, no joins), unlike GET /{id} which pulls
    the full stat-tile set."""
    await _campaign(db, p.tenant_id, campaign_id)
    active_lines = int((await db.execute(select(func.count()).select_from(Call).where(
        Call.tenant_id == p.tenant_id, Call.campaign_id == campaign_id,
        Call.status.in_(ACTIVE_STATUSES)))).scalar_one())
    queued_contacts = int((await db.execute(select(func.count()).select_from(CampaignTarget).where(
        CampaignTarget.tenant_id == p.tenant_id, CampaignTarget.campaign_id == campaign_id,
        CampaignTarget.status == "pending"))).scalar_one())
    return {"campaign_id": str(campaign_id), "active_lines": active_lines,
           "queued_contacts": queued_contacts}


@router.post("/{campaign_id}/dial-next")
async def dial_next(campaign_id: uuid.UUID, count: int = Query(1, ge=1, le=10),
                    p: Principal = Depends(require("campaigns:write")),
                    db: AsyncSession = Depends(get_db)):
    """Queue the next N calls manually. This is what the dialer worker will do on a
    schedule in the telephony stage; exposing it now makes the gate testable."""
    c = await _campaign(db, p.tenant_id, campaign_id)
    if c.status != "running":
        raise HTTPException(status.HTTP_409_CONFLICT, f"Campaign is {c.status}, not running")

    targets = (await db.execute(select(CampaignTarget).where(
        CampaignTarget.tenant_id == p.tenant_id, CampaignTarget.campaign_id == campaign_id,
        CampaignTarget.status == "pending").limit(count))).scalars().all()

    queued, excluded = [], []
    purpose = (c.schedule or {}).get("purpose", "marketing")
    for t in targets:
        cust = (await db.execute(select(Customer).where(
            Customer.id == t.customer_id))).scalar_one_or_none()
        if cust is None:
            t.status = "failed"
            continue
        decision = await place_outbound_call(db, tenant_id=p.tenant_id, customer=cust,
                                             purpose=purpose, campaign_id=campaign_id,
                                             actor=p.actor)
        t.attempts += 1
        t.last_attempt_at = datetime.now(timezone.utc)
        if decision.allowed:
            t.status = "calling"
            queued.append({"customer_id": str(cust.id), "call_id": str(decision.call_id)})
        else:
            t.status = "excluded"
            t.exclusion_reason = decision.blocked_by
            excluded.append({"customer_id": str(cust.id), "blocked_by": decision.blocked_by,
                             "reason": decision.reason})
    await db.commit()
    return {"queued": queued, "excluded": excluded,
            "note": "Every dial re-checked policy at this moment, not at campaign creation."}


_EMPTY_TILES = {"contacts": 0, "calls_made": 0, "connected": 0, "completed": 0,
               "rescheduled": 0, "follow_ups": 0}


def _campaign_json(c: Campaign, tiles: dict | None = None) -> dict:
    return {"id": str(c.id), "name": c.name, "status": c.status,
            "agent_id": str(c.agent_id) if c.agent_id else None,
            "concurrency_limit": c.concurrency_limit, "calling_rules": c.calling_rules,
            "audience": c.audience_filter, "schedule": c.schedule, "stats": c.stats,
            **(tiles or _EMPTY_TILES),
            "progress_pct": round(100 * (tiles or _EMPTY_TILES)["completed"]
                                  / (tiles or _EMPTY_TILES)["contacts"], 1)
                            if (tiles or _EMPTY_TILES)["contacts"] else 0.0,
            "created_at": c.created_at}


async def _campaign_json_with_tiles(db: AsyncSession, c: Campaign) -> dict:
    tiles = await _tile_counts_batch(db, c.tenant_id, [c.id])
    return _campaign_json(c, tiles.get(c.id))


async def _tile_counts_batch(db: AsyncSession, tenant_id: uuid.UUID,
                             campaign_ids: list[uuid.UUID]) -> dict[uuid.UUID, dict]:
    """The stat-tile set the campaign-detail header/list needs (contacts, calls_made,
    connected, completed, rescheduled, follow_ups), for any number of campaigns in three
    queries total rather than N — the same "one rollup query for the whole page" pattern
    P4's agent roster already established. Tile semantics, since the tasksheet names them
    without defining them precisely:
      - contacts: campaign_targets rows (one per contact this campaign will ever dial).
      - calls_made / connected: dial ATTEMPTS (calls rows) and how many were answered —
        distinct from `contacts`, since one contact can be dialled (and answered) more than
        once across retries.
      - completed: contacts whose target reached a terminal 'done' (successfully reached).
      - rescheduled: contacts currently sitting on a scheduled retry (next_retry_at set).
      - follow_ups: contacts who asked for a callback mid-call (a real CallbackRequest row,
        reused rather than invented — the same reconciliation pattern as everywhere else in
        this codebase) that is still pending.
    """
    out = {cid: dict(_EMPTY_TILES) for cid in campaign_ids}
    if not campaign_ids:
        return out

    for cid, contacts, completed, rescheduled in (await db.execute(
        select(CampaignTarget.campaign_id, func.count(),
              func.count().filter(CampaignTarget.status == "done"),
              func.count().filter(CampaignTarget.next_retry_at.is_not(None)))
        .where(CampaignTarget.tenant_id == tenant_id,
              CampaignTarget.campaign_id.in_(campaign_ids))
        .group_by(CampaignTarget.campaign_id))):
        out[cid].update(contacts=int(contacts), completed=int(completed),
                        rescheduled=int(rescheduled))

    for cid, calls_made, connected in (await db.execute(
        select(Call.campaign_id, func.count(), func.count().filter(Call.answered_at.is_not(None)))
        .where(Call.tenant_id == tenant_id, Call.campaign_id.in_(campaign_ids))
        .group_by(Call.campaign_id))):
        out[cid].update(calls_made=int(calls_made), connected=int(connected))

    for cid, follow_ups in (await db.execute(
        select(CampaignTarget.campaign_id, func.count(func.distinct(CallbackRequest.customer_id)))
        .join(CallbackRequest, (CallbackRequest.tenant_id == CampaignTarget.tenant_id)
             & (CallbackRequest.customer_id == CampaignTarget.customer_id)
             & (CallbackRequest.status == "pending"))
        .where(CampaignTarget.tenant_id == tenant_id, CampaignTarget.campaign_id.in_(campaign_ids))
        .group_by(CampaignTarget.campaign_id))):
        out[cid]["follow_ups"] = int(follow_ups)

    return out


async def _campaign(db: AsyncSession, tenant_id: uuid.UUID, cid: uuid.UUID) -> Campaign:
    row = (await db.execute(select(Campaign).where(
        Campaign.tenant_id == tenant_id, Campaign.id == cid))).scalar_one_or_none()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Campaign not found")
    return row
