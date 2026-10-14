"""Who did what on the company links, for the admin screens (docs/gtm-demo-links.md §9).

  link_stats      one summary row per link: people sent it, people who opened it, other
                  visitors, calls, talk time, last activity
  link_activity   one link in full: each person sent it (opened? when? how often? calls?),
                  visitors we couldn't tie to a person, the timeline, the calls, and every
                  question asked
  questions       what prospects asked the agents, across demos, with who asked

"Who" is resolved the same way everywhere (who()): a visitor who typed their name is that
name (and "via" the person whose email brought them, if different); a visitor on a
recipient's code is that recipient; otherwise "Visitor n" (numbered per link by first visit).
Only the prospect's own words are "questions": messages with role customer.
"""
import uuid
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.db.models import (DemoContact, DemoEvent, DemoLink, DemoLinkRecipient, DemoProspect,
                           DemoSession, DemoVisitor, Message)
from app.demo.links import recipients_of


@dataclass
class People:
    """Visitors and recipients needed to say who did something."""
    visitors: dict[uuid.UUID, DemoVisitor]
    numbers: dict[uuid.UUID, int]                                   # visitor id → n on its link
    recipients: dict[uuid.UUID, tuple[DemoLinkRecipient, DemoContact]]

    def who(self, visitor_id: uuid.UUID | None, recipient_id: uuid.UUID | None = None) -> dict:
        v = self.visitors.get(visitor_id) if visitor_id else None
        rid = (v.recipient_id if v else None) or recipient_id
        rc = self.recipients.get(rid) if rid else None
        contact = rc[1] if rc else None
        out = {"visitor_id": str(v.id) if v else None,
               "visitor_no": self.numbers.get(v.id) if v else None,
               "recipient_id": str(rid) if rc else None,
               "contact_id": str(contact.id) if contact else None,
               "attribution": v.attribution if v else ("recipient" if contact else "unknown"),
               "via": None}
        if v is not None and v.identified_at is not None:
            out.update(name=v.identified_name, email=v.identified_email, role=v.identified_role)
            if contact is not None and (contact.email or "").lower() != (v.identified_email or ""):
                out["via"] = contact.full_name         # came on someone else's email
                out["contact_id"] = None
        elif contact is not None:
            out.update(name=contact.full_name, email=contact.email, role=contact.role)
        else:
            n = out["visitor_no"]
            out.update(name=f"Visitor {n}" if n else "Unknown visitor", email=None, role=None)
        return out


async def load_people(db: AsyncSession, *, link_ids: set[uuid.UUID] = frozenset(),
                      visitor_ids: set[uuid.UUID] = frozenset(),
                      recipient_ids: set[uuid.UUID] = frozenset()) -> People:
    visitor_ids = {v for v in visitor_ids if v}
    if visitor_ids:
        link_ids = set(link_ids) | set((await db.execute(select(DemoVisitor.demo_link_id).where(
            DemoVisitor.id.in_(visitor_ids)))).scalars())
    visitors: dict[uuid.UUID, DemoVisitor] = {}
    numbers: dict[uuid.UUID, int] = {}
    if link_ids:
        rows = (await db.execute(select(DemoVisitor).where(DemoVisitor.demo_link_id.in_(link_ids))
                                 .order_by(DemoVisitor.demo_link_id, DemoVisitor.first_seen_at,
                                           DemoVisitor.id))).scalars().all()
        count: dict[uuid.UUID, int] = defaultdict(int)
        for v in rows:
            count[v.demo_link_id] += 1
            numbers[v.id] = count[v.demo_link_id]
            visitors[v.id] = v
    rids = {r for r in recipient_ids if r} | {v.recipient_id for v in visitors.values() if v.recipient_id}
    recipients = {}
    if rids:
        recipients = {r.id: (r, c) for r, c in (await db.execute(
            select(DemoLinkRecipient, DemoContact).join(DemoContact, DemoContact.id == DemoLinkRecipient.contact_id)
            .where(DemoLinkRecipient.id.in_(rids)))).all()}
    return People(visitors=visitors, numbers=numbers, recipients=recipients)


# ── list: one row per link ───────────────────────────────────────────────────────────────

async def link_stats(db: AsyncSession, link_ids: list[uuid.UUID]) -> dict[uuid.UUID, dict]:
    if not link_ids:
        return {}
    out = {lid: {"people": 0, "people_emailed": 0, "people_opened": 0, "visitors": 0,
                 "unknown_visitors": 0, "calls": 0, "talk_seconds": 0, "goal_completed": 0,
                 "first_opened_at": None, "last_activity_at": None} for lid in link_ids}
    for lid, people, emailed in (await db.execute(
            select(DemoLinkRecipient.demo_link_id, func.count(),
                   func.count(DemoLinkRecipient.last_exported_at))
            .where(DemoLinkRecipient.demo_link_id.in_(link_ids))
            .group_by(DemoLinkRecipient.demo_link_id))).all():
        out[lid].update(people=people, people_emailed=emailed)
    seen = DemoVisitor.first_viewed_at.is_not(None)
    for lid, visitors, unknown, people_opened, first, last in (await db.execute(
            select(DemoVisitor.demo_link_id, func.count(),
                   func.count().filter(DemoVisitor.recipient_id.is_(None)),
                   func.count(func.distinct(DemoVisitor.recipient_id)).filter(seen),
                   func.min(DemoVisitor.first_viewed_at), func.max(DemoVisitor.last_seen_at))
            .where(DemoVisitor.demo_link_id.in_(link_ids))
            .group_by(DemoVisitor.demo_link_id))).all():
        out[lid].update(visitors=visitors, unknown_visitors=unknown, people_opened=people_opened,
                        first_opened_at=first, last_activity_at=last)
    for lid, calls, talk, goals in (await db.execute(
            select(DemoSession.demo_link_id, func.count(),
                   func.coalesce(func.sum(DemoSession.duration_seconds), 0),
                   func.count().filter(DemoSession.outcome == "goal_completed"))
            .where(DemoSession.demo_link_id.in_(link_ids), DemoSession.started_at.is_not(None))
            .group_by(DemoSession.demo_link_id))).all():
        out[lid].update(calls=calls, talk_seconds=int(talk), goal_completed=goals)
    return out


# ── one link in full ─────────────────────────────────────────────────────────────────────

def _questions_query(where, *, limit: int, offset: int = 0):
    reply = aliased(Message)
    next_reply = (select(reply.content).where(
        reply.conversation_id == Message.conversation_id, reply.role == "agent",
        reply.created_at > Message.created_at).order_by(reply.created_at).limit(1)
        .correlate(Message).scalar_subquery())
    return (select(Message.id, Message.content, Message.created_at, next_reply.label("reply"),
                   DemoSession.id.label("session_id"), DemoSession.demo_visitor_id,
                   DemoSession.recipient_id, DemoLink.id.label("link_id"),
                   DemoProspect.id.label("record_id"), DemoProspect.company_name)
            .join(DemoSession, DemoSession.conversation_id == Message.conversation_id)
            .join(DemoLink, DemoLink.id == DemoSession.demo_link_id)
            .join(DemoProspect, DemoProspect.id == DemoLink.prospect_id)
            .where(Message.role == "customer", *where)
            .order_by(Message.created_at.desc()).limit(limit).offset(offset))


def _question_out(row, people: People) -> dict:
    return {"id": str(row.id), "text": row.content, "at": row.created_at, "reply": row.reply,
            "session_id": str(row.session_id), "link_id": str(row.link_id),
            "record_id": str(row.record_id), "company": row.company_name,
            "who": people.who(row.demo_visitor_id, row.recipient_id)}


async def link_activity(db: AsyncSession, link: DemoLink, *, event_limit: int = 300) -> dict:
    recipients = await recipients_of(db, link.id)
    sessions = (await db.execute(select(DemoSession).where(DemoSession.demo_link_id == link.id)
                                 .order_by(DemoSession.created_at.desc()))).scalars().all()
    events = (await db.execute(select(DemoEvent).where(DemoEvent.demo_link_id == link.id)
                               .order_by(DemoEvent.occurred_at.desc()).limit(event_limit))).scalars().all()
    people = await load_people(db, link_ids={link.id}, recipient_ids={r.id for r, _ in recipients}
                               | {s.recipient_id for s in sessions} | {e.recipient_id for e in events})
    q_rows = (await db.execute(_questions_query([DemoLink.id == link.id], limit=500))).all()

    views = defaultdict(int)
    for e in events:
        if e.type == "page_viewed" and e.recipient_id:
            views[e.recipient_id] += 1
    started = [s for s in sessions if s.started_at is not None]
    asked = defaultdict(int)
    for row in q_rows:
        v = people.visitors.get(row.demo_visitor_id)
        rid = (v.recipient_id if v else None) or row.recipient_id
        if rid:
            asked[rid] += 1

    person_rows = []
    for r, c in recipients:
        mine = [v for v in people.visitors.values() if v.recipient_id == r.id]
        opened = [v.first_viewed_at for v in mine if v.first_viewed_at]
        calls = [s for s in started if s.recipient_id == r.id
                 or (s.demo_visitor_id and people.visitors.get(s.demo_visitor_id)
                     and people.visitors[s.demo_visitor_id].recipient_id == r.id)]
        person_rows.append({
            "recipient_id": str(r.id), "ref_code": r.ref_code, "last_exported_at": r.last_exported_at,
            "contact": {"id": str(c.id), "name": c.full_name, "email": c.email, "role": c.role,
                        "other_company": c.prospect_id != link.prospect_id},
            "opened": bool(opened), "first_opened_at": min(opened) if opened else None,
            "last_seen_at": max((v.last_seen_at for v in mine), default=None),
            "views": views[r.id], "browsers": len(mine),
            "forwarded_browsers": sum(1 for v in mine if v.attribution == "shared"),
            "calls": len(calls), "talk_seconds": sum(s.duration_seconds or 0 for s in calls),
            "goal_completed": any(s.outcome == "goal_completed" for s in calls),
            "questions": asked[r.id]})
    person_rows.sort(key=lambda p: (not p["opened"], p["first_opened_at"] or link.created_at))

    # Visitors not credited to a listed person: no code, or a stated identity of their own.
    others = []
    for v in people.visitors.values():
        w = people.who(v.id)
        if v.recipient_id is None or w["via"]:
            others.append({**w, "first_seen_at": v.first_seen_at, "last_seen_at": v.last_seen_at,
                           "opened": v.first_viewed_at is not None, "source_step": v.source_step,
                           "utm": v.utm,
                           "calls": sum(1 for s in started if s.demo_visitor_id == v.id)})
    others.sort(key=lambda o: o["first_seen_at"])

    return {
        "link_id": str(link.id),
        "people": person_rows,
        "other_visitors": others,
        "calls": [{"id": str(s.id), "who": people.who(s.demo_visitor_id, s.recipient_id),
                   "created_at": s.created_at, "started_at": s.started_at, "ended_at": s.ended_at,
                   "duration_seconds": s.duration_seconds, "outcome": s.outcome,
                   "end_reason": s.end_reason, "has_transcript": s.conversation_id is not None}
                  for s in sessions],
        "timeline": [{"at": e.occurred_at, "type": e.type, "source_step": e.source_step,
                      "session_id": str(e.session_id) if e.session_id else None,
                      "utm": (e.meta or {}).get("utm"),
                      "who": people.who(e.demo_visitor_id, e.recipient_id)} for e in events],
        "questions": [_question_out(row, people) for row in q_rows],
    }


# ── questions across demos ───────────────────────────────────────────────────────────────

async def questions(db: AsyncSession, *, record_id: uuid.UUID | None = None,
                    link_id: uuid.UUID | None = None, industry_key: str | None = None,
                    contains: str | None = None, limit: int = 100, offset: int = 0) -> list[dict]:
    where = []
    if record_id:
        where.append(DemoProspect.id == record_id)
    if link_id:
        where.append(DemoLink.id == link_id)
    if industry_key:
        where.append(DemoProspect.industry_key == industry_key)
    if contains:
        like = "%" + contains.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        where.append(Message.content.ilike(like, escape="\\"))
    rows = (await db.execute(_questions_query(where, limit=limit, offset=offset))).all()
    people = await load_people(db, visitor_ids={r.demo_visitor_id for r in rows},
                               recipient_ids={r.recipient_id for r in rows})
    return [_question_out(r, people) for r in rows]


async def session_people(db: AsyncSession, sessions: list[DemoSession]) -> People:
    return await load_people(db, visitor_ids={s.demo_visitor_id for s in sessions},
                             recipient_ids={s.recipient_id for s in sessions})

