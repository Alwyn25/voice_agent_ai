"""Leads from the demo page: intake, notification and the sheet sync (docs/gtm-demo-links.md
§10.3). Outbox pattern:

  POST /contact ─► demo_leads row committed ─► 201 to the browser ─► wake the worker
  worker (on wake, and every DEMO_NOTIFY_RETRY_MINUTES)
     ├─► append new leads to the sheet's Leads tab       → sheet_synced_at
     └─► email them to DEMO_LEAD_NOTIFY_EMAILS           → email_notified_at
  sheet sync (every DEMO_SHEET_SYNC_MINUTES)
     ├─► read Status / Called by / Notes back into demo_leads
     ├─► refresh our columns on existing rows (demo calls, questions…)
     └─► rewrite the Demo activity tab

A lead is never lost: it is in the database before the browser hears "thanks", and a
failure only bumps notify_attempts for a retry. After DEMO_NOTIFY_MAX_ATTEMPTS failures it
stops retrying, an alert goes to DEMO_ALERT_EMAILS, and the admin screen shows it with a
Resend button. Each run takes a Postgres advisory lock, so with several server instances
only one sends, and nobody gets a lead twice.
"""
import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import (DemoBatch, DemoContact, DemoEvent, DemoIndustry, DemoLead, DemoLink,
                           DemoLinkRecipient, DemoProspect, DemoSession, DemoVisitor, Message)
from app.demo.leads import sheets
from app.demo.leads.mailer import MailError, alert_mail, email_on, get_mailer, lead_mail

log = logging.getLogger(__name__)

KINDS = ("contact", "schedule_call", "real_calls_pilot", "rebuild_request")
KIND_LABEL = {"contact": "Contact us", "schedule_call": "Schedule a call",
              "real_calls_pilot": "Run on real calls (14-day pilot)", "rebuild_request": "Rebuild demo"}
PHONE_REQUIRED = {"schedule_call", "real_calls_pilot"}
_LOCK_KEY = 0x50484F4E4E01           # "PHONN" + 1: the leads worker's advisory lock

# What the last runs did, for the admin screen (per process).
STATUS: dict = {"last_push_at": None, "last_push_error": None,
                "last_sync_at": None, "last_sync_error": None}


def new_lead_id() -> str:
    return "L-" + uuid.uuid4().hex[:8].upper()


def sheet_on() -> bool:
    return bool(get_settings().demo_leads_gsheet_id.strip())


# ── intake ───────────────────────────────────────────────────────────────────────────────

async def create_lead(db: AsyncSession, *, link: DemoLink | None, visitor: DemoVisitor | None,
                      kind: str, name: str, email: str, phone: str | None, company: str | None,
                      role: str | None, preferred_time: str | None, message: str | None,
                      now: datetime) -> tuple[DemoLead, bool]:
    """Save a lead. Returns (lead, created). The same email + kind again on the same (UTC)
    day returns the existing lead instead of a duplicate (double-click, refresh)."""
    email = email.strip().lower()
    day_start = now.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    existing = (await db.execute(select(DemoLead).where(
        func.lower(DemoLead.email) == email, DemoLead.kind == kind,
        DemoLead.created_at >= day_start))).scalars().first()
    if existing is not None:
        return existing, False

    prospect = await db.get(DemoProspect, link.prospect_id) if link else None
    contact_id = None
    if visitor is not None and visitor.recipient_id is not None:
        r = await db.get(DemoLinkRecipient, visitor.recipient_id)
        contact_id = r.contact_id if r else None
    if contact_id is None and prospect is not None:
        contact_id = (await db.execute(select(DemoContact.id).where(
            DemoContact.prospect_id == prospect.id, func.lower(DemoContact.email) == email))).scalars().first()
    last_session = None
    if visitor is not None:
        last_session = (await db.execute(select(DemoSession.id).where(
            DemoSession.demo_visitor_id == visitor.id).order_by(DemoSession.created_at.desc()))).scalars().first()

    lead = DemoLead(demo_link_id=link.id if link else None, prospect_id=prospect.id if prospect else None,
                    contact_id=contact_id, kind=kind, name=name.strip()[:120], email=email[:254],
                    phone=phone, company=(company or (prospect.company_name if prospect else None) or None),
                    role=(role or "").strip()[:120] or None,
                    preferred_time=(preferred_time or "").strip()[:120] or None,
                    message=(message or "").strip()[:2000] or None, last_session_id=last_session,
                    sheet_row_key=new_lead_id(), call_status="New")
    db.add(lead)
    await db.flush()
    if link is not None:
        db.add(DemoEvent(demo_link_id=link.id, visitor_id=visitor.visitor_key if visitor else None,
                         demo_visitor_id=visitor.id if visitor else None,
                         recipient_id=visitor.recipient_id if visitor else None,
                         type="contact_submitted", occurred_at=now, meta={"kind": kind, "lead": lead.sheet_row_key}))
    return lead, True


# ── rows for the sheet and the email ─────────────────────────────────────────────────────

def _fmt_seconds(s: int | None) -> str:
    return f"{s // 60}:{s % 60:02d}" if s else ""


def _local(dt: datetime | None) -> str:
    if dt is None:
        return ""
    return dt.astimezone(ZoneInfo(get_settings().demo_default_timezone)).strftime("%Y-%m-%d %H:%M")


async def _person_sessions(db: AsyncSession, lead: DemoLead) -> list[DemoSession]:
    """The calls this lead's person made on their demo link."""
    if lead.demo_link_id is None:
        return []
    cond = []
    if lead.contact_id is not None:
        cond.append(DemoSession.recipient_id.in_(select(DemoLinkRecipient.id).where(
            DemoLinkRecipient.contact_id == lead.contact_id)))
    if lead.last_session_id is not None:
        visitor = (await db.execute(select(DemoSession.demo_visitor_id).where(
            DemoSession.id == lead.last_session_id))).scalar_one_or_none()
        if visitor:
            cond.append(DemoSession.demo_visitor_id == visitor)
    if not cond:
        return []
    return list((await db.execute(select(DemoSession).where(
        DemoSession.demo_link_id == lead.demo_link_id, DemoSession.started_at.is_not(None), or_(*cond))
        .order_by(DemoSession.created_at))).scalars())


async def lead_row(db: AsyncSession, lead: DemoLead) -> dict:
    s = get_settings()
    industry, batch = "", ""
    if lead.prospect_id:
        industry = (await db.execute(select(DemoIndustry.label).join(
            DemoProspect, DemoProspect.industry_key == DemoIndustry.key).where(
            DemoProspect.id == lead.prospect_id))).scalar_one_or_none() or ""
    if lead.contact_id:
        batch = (await db.execute(select(DemoBatch.name).join(
            DemoContact, DemoContact.batch_id == DemoBatch.id).where(
            DemoContact.id == lead.contact_id))).scalar_one_or_none() or ""
    calls = await _person_sessions(db, lead)
    questions = []
    convs = [c.conversation_id for c in calls if c.conversation_id]
    if convs:
        questions = list((await db.execute(select(Message.content).where(
            Message.conversation_id.in_(convs), Message.role == "customer")
            .order_by(Message.created_at).limit(5))).scalars())
    return {"Lead ID": lead.sheet_row_key, "Received at": _local(lead.created_at),
            "Kind": KIND_LABEL.get(lead.kind, lead.kind), "Name": lead.name, "Phone": lead.phone or "",
            "Email": lead.email, "Company": lead.company or "", "Role": lead.role or "",
            "Industry": industry, "Preferred time": lead.preferred_time or "", "Message": lead.message or "",
            "Demo calls": str(len(calls)) if calls else "0",
            "Longest call": _fmt_seconds(max((c.duration_seconds or 0 for c in calls), default=0)),
            "Goal completed": "yes" if any(c.outcome == "goal_completed" for c in calls) else "no",
            "Questions asked": " | ".join(q.strip()[:150] for q in questions),
            "Batch": batch, "Admin link": s.dashboard_url or "",
            "Status": lead.call_status or "New", "Called by": lead.called_by or "", "Notes": lead.notes or ""}


async def activity_rows(db: AsyncSession) -> list[dict]:
    """One row per company link that is live or expired (replaced/revoked ones are left out)."""
    from app.demo.activity import link_stats
    links = (await db.execute(select(DemoLink, DemoProspect, DemoIndustry.label)
                              .join(DemoProspect, DemoProspect.id == DemoLink.prospect_id)
                              .join(DemoIndustry, DemoIndustry.key == DemoProspect.industry_key)
                              .where(DemoLink.status != "revoked")
                              .order_by(DemoProspect.company_name))).all()
    ids = [link.id for link, _, _ in links]
    stats = await link_stats(db, ids)
    opened_by: dict[uuid.UUID, list[str]] = {}
    if ids:
        for lid, name in (await db.execute(
                select(DemoVisitor.demo_link_id, DemoContact.full_name).distinct()
                .join(DemoLinkRecipient, DemoLinkRecipient.id == DemoVisitor.recipient_id)
                .join(DemoContact, DemoContact.id == DemoLinkRecipient.contact_id)
                .where(DemoVisitor.demo_link_id.in_(ids), DemoVisitor.first_viewed_at.is_not(None))
                .order_by(DemoContact.full_name))).all():
            opened_by.setdefault(lid, []).append(name)
    asked = dict((await db.execute(
        select(DemoSession.demo_link_id, func.count(Message.id))
        .join(Message, Message.conversation_id == DemoSession.conversation_id)
        .where(DemoSession.demo_link_id.in_(ids), Message.role == "customer")
        .group_by(DemoSession.demo_link_id))).all()) if ids else {}
    leads = dict((await db.execute(
        select(DemoLead.demo_link_id, func.string_agg(DemoLead.kind, ", "))
        .where(DemoLead.demo_link_id.in_(ids)).group_by(DemoLead.demo_link_id))).all()) if ids else {}
    from app.demo.links import effective_status
    out = []
    for link, prospect, label in links:
        st = stats[link.id]
        out.append({"Company": prospect.company_name, "Industry": label,
                    "Link status": effective_status(link), "Expires on": _local(link.expires_at)[:10],
                    "People sent": st["people"], "People opened": st["people_opened"],
                    "Opened by": ", ".join(opened_by.get(link.id, [])),
                    "Other visitors": st["unknown_visitors"], "Calls": st["calls"],
                    "Talk time": _fmt_seconds(st["talk_seconds"]), "Goal reached": st["goal_completed"],
                    "Questions asked": asked.get(link.id, 0),
                    "First opened": _local(st["first_opened_at"]), "Last activity": _local(st["last_activity_at"]),
                    "Lead submitted": ", ".join(KIND_LABEL.get(k.strip(), k) for k in (leads.get(link.id) or "").split(",") if k.strip())})
    return out


# ── the worker ───────────────────────────────────────────────────────────────────────────

async def _locked(db: AsyncSession) -> bool:
    """Take this run's advisory lock (released at commit). False: another instance is on it."""
    return bool((await db.execute(text("SELECT pg_try_advisory_xact_lock(:k)"), {"k": _LOCK_KEY})).scalar())


async def push_pending(db: AsyncSession) -> dict:
    """Email and/or append every lead still owed it. The caller commits."""
    s = get_settings()
    want_sheet, want_mail = sheet_on(), email_on()
    result = {"appended": 0, "emailed": 0, "failed": 0, "skipped_locked": False}
    if not (want_sheet or want_mail):
        return result
    if not await _locked(db):
        result["skipped_locked"] = True
        return result
    owed = []
    if want_sheet:
        owed.append(DemoLead.sheet_synced_at.is_(None))
    if want_mail:
        owed.append(DemoLead.email_notified_at.is_(None))
    leads = list((await db.execute(select(DemoLead).where(
        or_(*owed), DemoLead.notify_attempts < s.demo_notify_max_attempts)
        .order_by(DemoLead.created_at).limit(200).with_for_update(skip_locked=True))).scalars())
    if not leads:
        return result
    rows = {lead.id: await lead_row(db, lead) for lead in leads}
    errors: dict[uuid.UUID, str] = {}
    now = datetime.now(timezone.utc)

    to_sheet = [lead for lead in leads if want_sheet and lead.sheet_synced_at is None]
    if to_sheet:
        try:
            backend = await sheets.run(sheets.open_backend)
            await sheets.run(sheets.ensure_tabs, backend)
            await sheets.run(sheets.append_leads, backend, [rows[x.id] for x in to_sheet])
            for lead in to_sheet:
                lead.sheet_synced_at = now
            result["appended"] = len(to_sheet)
            STATUS.update(last_push_at=now, last_push_error=None)
        except Exception as e:  # noqa: BLE001 — retried next sweep
            msg = f"sheet: {type(e).__name__}: {str(e)[:300]}"
            log.warning("demo leads: %s", msg)
            STATUS.update(last_push_error=msg)
            for lead in to_sheet:
                errors[lead.id] = msg

    if want_mail:
        mailer = get_mailer()
        for lead in leads:
            if lead.email_notified_at is not None:
                continue
            row = rows[lead.id]
            company = f"{row['Company']} ({row['Industry']})" if row["Company"] and row["Industry"] else row["Company"]
            bits = " · ".join(x for x in (row["Kind"], company, row["Name"]) if x)
            try:
                await mailer.send(lead_mail(row, subject_bits=bits))
                lead.email_notified_at = now
                result["emailed"] += 1
            except MailError as e:
                errors[lead.id] = (errors.get(lead.id, "") + f" email: {e}").strip()

    for lead in leads:
        if lead.id in errors:
            lead.notify_attempts += 1
            lead.last_notify_error = errors[lead.id][:1000]
            result["failed"] += 1
            if lead.notify_attempts >= s.demo_notify_max_attempts:
                log.error("demo lead %s not delivered after %s attempts: %s", lead.sheet_row_key,
                          lead.notify_attempts, lead.last_notify_error)
                await _alert(f"lead {lead.sheet_row_key} not delivered",
                             f"Lead {lead.sheet_row_key} ({lead.name}, {lead.email}) could not be "
                             f"delivered after {lead.notify_attempts} attempts.\n\n{lead.last_notify_error}\n\n"
                             "It is safe in the database; use Resend on the admin Leads tab once fixed.")
        elif lead.last_notify_error:
            lead.last_notify_error = None
    await db.flush()
    return result


async def sync_sheet(db: AsyncSession) -> dict:
    """Read the team's columns back, refresh ours, rewrite Demo activity. The caller commits."""
    result = {"status_updates": 0, "cells_refreshed": 0, "activity_rows": 0, "skipped_locked": False}
    if not sheet_on():
        return result
    if not await _locked(db):
        result["skipped_locked"] = True
        return result
    try:
        backend = await sheets.run(sheets.open_backend)
        await sheets.run(sheets.ensure_tabs, backend)
        team = await sheets.run(sheets.read_team_columns, backend)
        leads = list((await db.execute(select(DemoLead).where(
            DemoLead.sheet_row_key.in_(list(team))))).scalars()) if team else []
        for lead in leads:
            t = team[lead.sheet_row_key]
            status = t["Status"][:30] or "New"
            called_by = t["Called by"][:120] or None
            notes = t["Notes"] or None
            if (lead.call_status, lead.called_by, lead.notes) != (status, called_by, notes):
                lead.call_status, lead.called_by, lead.notes = status, called_by, notes
                result["status_updates"] += 1
        recent = [lead for lead in leads if lead.demo_link_id is not None]
        if recent:
            fresh = [await lead_row(db, lead) for lead in recent]
            result["cells_refreshed"] = await sheets.run(sheets.refresh_system_columns, backend, fresh)
        rows = await activity_rows(db)
        await sheets.run(sheets.write_activity, backend, rows)
        result["activity_rows"] = len(rows)
        STATUS.update(last_sync_at=datetime.now(timezone.utc), last_sync_error=None)
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {str(e)[:300]}"
        log.warning("demo sheet sync failed: %s", msg)
        if STATUS.get("last_sync_error") != msg:
            await _alert("leads sheet sync failing", f"The leads Google Sheet sync failed:\n\n{msg}")
        STATUS.update(last_sync_error=msg)
        result["error"] = msg
    await db.flush()
    return result


async def _alert(subject: str, body: str) -> None:
    if not email_on():
        return
    mail = alert_mail(subject, body)
    if mail is None:
        return
    try:
        await get_mailer().send(mail)
    except MailError:
        log.exception("demo alert email failed")


_wake: asyncio.Event | None = None


def wake() -> None:
    """Ask the worker to push new leads now instead of at its next sweep."""
    if _wake is not None:
        _wake.set()


async def worker_loop() -> None:
    """Runs for the life of the process when DEMO_ENABLED (started in main.py)."""
    global _wake
    from app.db.base import get_session_factory
    _wake = asyncio.Event()
    s = get_settings()
    retry_s = max(30, s.demo_notify_retry_minutes * 60)
    sync_s = max(60, s.demo_sheet_sync_minutes * 60)
    last_sync = 0.0
    log.info("demo leads worker starting (sheet=%s, email=%s, retry=%ss, sync=%ss)",
             "on" if sheet_on() else "off", s.demo_email_provider, retry_s, sync_s)
    while True:
        try:
            async with get_session_factory()() as db:
                r = await push_pending(db)
                await db.commit()
                if r["appended"] or r["emailed"] or r["failed"]:
                    log.info("demo leads pushed: %s", r)
            if time.monotonic() - last_sync >= sync_s:
                async with get_session_factory()() as db:
                    await sync_sheet(db)
                    await db.commit()
                last_sync = time.monotonic()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("demo leads worker tick failed — continuing")
        try:
            await asyncio.wait_for(_wake.wait(), timeout=min(retry_s, max(5.0, sync_s - (time.monotonic() - last_sync))))
        except asyncio.TimeoutError:
            pass
        _wake.clear()
