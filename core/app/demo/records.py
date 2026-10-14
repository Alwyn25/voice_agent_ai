"""Persisting demo data (docs/gtm-demo-links.md §5, §6, §16.4): workbook imports, company
records with their own tenant and knowledge base, facts, contacts, knowledge documents,
dummy data, and industry versions.

No FastAPI here: the API layer (app/api/demo_admin.py) translates DemoError into HTTP and
writes the audit trail; tests call these functions directly. Functions add to the session
and flush; the caller commits. Work that must happen after the commit (embedding a
knowledge document) is returned as IngestJob values for the caller to run.

Re-import rules (production data, edited by people between imports):
  * a record is matched on (industry, record_key) — the same row imported twice updates it
  * facts follow the sheet: the sheet is where research happens
  * contacts are never overwritten: a re-import only fills blanks and adds new people, so
    an admin's review of a contact survives; people missing from the sheet are reported,
    never deleted
"""
import hashlib
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import (DemoContact, DemoImport, DemoIndustry, DemoIndustryVersion,
                           DemoProspect, DemoProspectField, DemoSampleDataset, KnowledgeBase,
                           KnowledgeChunk, KnowledgeDocument, Tenant)
from app.db.models.platform import TENANT_KIND_DEMO
from app.demo.company_facts import ROLE as COMPANY_FACTS_ROLE
from app.demo.company_facts import content_hash, render_company_facts
from app.demo.facts import FactValue, check_minimum, normalize_fact
from app.demo.importer import WorkbookError, apply_column_proposals, parse_workbook
from app.demo.industries.definition import DefinitionError, IndustryDefinition, compile_definition
from app.demo.sample.resolve import ResolvedContext, resolve_dataset
from app.demo.sample.validate import validate_dataset

log = logging.getLogger(__name__)

KB_NAME = "Company facts and documents"


# ── errors ────────────────────────────────────────────────────────────────────────────────

class DemoError(Exception):
    status = 400

    def __init__(self, message: str, *, errors: list[str] | None = None):
        super().__init__(message)
        self.message = message
        self.errors = errors or []


class NotFound(DemoError):
    status = 404


class Conflict(DemoError):
    status = 409


class Invalid(DemoError):
    status = 422


@dataclass(frozen=True)
class IngestJob:
    tenant_id: uuid.UUID
    document_id: uuid.UUID
    text: str


async def run_ingest_jobs(jobs: list[IngestJob]) -> None:
    """Embed documents one after another (never in parallel: an import of 30 records must
    not fire 30 concurrent embedding calls). Each job has its own session; a failure marks
    that document failed (ingest_document does) and the rest continue."""
    from app.engine.rag.ingestion import ingest_task
    for job in jobs:
        try:
            await ingest_task(job.tenant_id, job.document_id, job.text)
        except Exception:  # noqa: BLE001 — one bad document must not stop the others
            log.exception("demo ingest failed for document %s", job.document_id)


# ── industry definitions ─────────────────────────────────────────────────────────────────

# Versions are immutable, so a compiled version can be cached forever.
_DEF_CACHE: dict[tuple[str, int], IndustryDefinition] = {}


async def load_definition(db: AsyncSession, key: str, version: int | None = None
                          ) -> IndustryDefinition:
    if version is None:
        industry = await db.get(DemoIndustry, key)
        if industry is None:
            raise NotFound(f"industry {key!r} not found")
        version = industry.current_version
    cached = _DEF_CACHE.get((key, version))
    if cached is not None:
        return cached
    row = await db.get(DemoIndustryVersion, (key, version))
    if row is None:
        raise NotFound(f"industry {key!r} has no version {version}")
    definition = compile_definition(row.definition)
    _DEF_CACHE[(key, version)] = definition
    return definition


def _without_version(raw: dict) -> dict:
    return {k: v for k, v in raw.items() if k != "version"}


async def publish_industry_version(db: AsyncSession, raw: Any, *, actor: str,
                                   activate: bool) -> tuple[IndustryDefinition, bool]:
    """Validate and store a definition as the industry's next version. Returns
    (definition, created_industry). activate=True makes it the version new records use and
    publishes its sample_template as the industry's active dummy data."""
    if not isinstance(raw, dict):
        raise Invalid("the definition must be a YAML/JSON mapping")
    key = raw.get("key")
    industry = await db.get(DemoIndustry, key) if isinstance(key, str) else None
    latest = 0
    if industry is not None:
        latest = (await db.execute(select(func.max(DemoIndustryVersion.version)).where(
            DemoIndustryVersion.industry_key == key))).scalar() or 0
        prev = await db.get(DemoIndustryVersion, (key, latest))
        if prev is not None and _without_version(prev.definition) == _without_version(raw):
            raise Conflict(f"no changes since version {latest}")
    raw = {**raw, "version": latest + 1}
    try:
        definition = compile_definition(raw)
    except DefinitionError as e:
        raise Invalid("the industry definition is invalid", errors=e.errors) from None

    created = industry is None
    if created:
        industry = DemoIndustry(key=definition.key, label=definition.label,
                                status="active" if activate else "draft",
                                current_version=definition.version)
        db.add(industry)
        await db.flush()
    db.add(DemoIndustryVersion(industry_key=definition.key, version=definition.version,
                               definition=definition.raw, created_by=None))
    await db.flush()
    if activate:
        industry.current_version = definition.version
        industry.label = definition.label
        if industry.status == "draft":
            industry.status = "active"
        if definition.has_agent:
            await _store_dataset(db, prospect_id=None, definition=definition,
                                 data=definition.sample_template)
    await db.flush()
    return definition, created


async def add_columns_to_industry(db: AsyncSession, definition: IndustryDefinition,
                                  proposals: list[dict]) -> IndustryDefinition:
    """Publish `definition` + the workbook columns an import proposed (importer.
    propose_columns) as the industry's next version, and make it current. The dummy data is
    left as it is: new facts are optional text, so the dummy-data schema is unchanged."""
    industry = await db.get(DemoIndustry, definition.key, with_for_update=True)
    if industry is None:
        raise NotFound(f"industry {definition.key!r} not found")
    if industry.current_version != definition.version:
        raise Conflict("the industry definition changed since this preview; upload the "
                       "workbook again to preview it against the current version")
    latest = (await db.execute(select(func.max(DemoIndustryVersion.version)).where(
        DemoIndustryVersion.industry_key == definition.key))).scalar() or 0
    raw = {**apply_column_proposals(definition.raw, proposals), "version": latest + 1}
    try:
        evolved = compile_definition(raw)
    except DefinitionError as e:
        raise Invalid("the new columns can't be added to the industry definition",
                      errors=e.errors) from None
    db.add(DemoIndustryVersion(industry_key=evolved.key, version=evolved.version,
                               definition=evolved.raw, created_by=None))
    await db.flush()
    industry.current_version = evolved.version
    await db.flush()
    _DEF_CACHE[(evolved.key, evolved.version)] = evolved
    return evolved


async def industry_readiness(db: AsyncSession, industry: DemoIndustry) -> dict:
    template = (await db.execute(select(DemoSampleDataset).where(
        DemoSampleDataset.industry_key == industry.key, DemoSampleDataset.prospect_id.is_(None),
        DemoSampleDataset.is_active.is_(True)))).scalar_one_or_none()
    records = (await db.execute(select(func.count()).select_from(DemoProspect).where(
        DemoProspect.industry_key == industry.key))).scalar_one()
    versions = (await db.execute(select(DemoIndustryVersion.version).where(
        DemoIndustryVersion.industry_key == industry.key).order_by(
        DemoIndustryVersion.version))).scalars().all()
    definition = await load_definition(db, industry.key, industry.current_version)
    problems = []
    if industry.status != "active":
        problems.append(f"industry is {industry.status}")
    if not definition.has_agent:
        problems.append("no demo agent yet (data only): records import, but links need a "
                        "version with sample_data, sample_template, agent and prompts")
    elif template is None:
        problems.append("no dummy data template")
    sheet = definition.sheet or {}
    return {"key": industry.key, "label": industry.label, "status": industry.status,
            "current_version": industry.current_version, "versions": list(versions),
            "records": int(records), "has_template": template is not None,
            "has_agent": definition.has_agent,
            "use_case": (definition.raw.get("use_case") or {}).get("label"),
            "tab_names": list(sheet.get("tab_names") or []),
            "facts": len(definition.facts_fields),
            "can_import": bool(sheet), "can_go_live": not problems, "problems": problems}


# ── imports ──────────────────────────────────────────────────────────────────────────────

async def all_current_definitions(db: AsyncSession) -> list[IndustryDefinition]:
    """The current version of every industry that isn't archived."""
    rows = (await db.execute(select(DemoIndustry).where(DemoIndustry.status != "archived")
                             .order_by(DemoIndustry.key))).scalars().all()
    return [await load_definition(db, r.key, r.current_version) for r in rows]


async def preview_import(db: AsyncSession, *, data: bytes, filename: str, industry_key: str,
                         actor: str, tab: str | None = None,
                         skip_columns: list[str] | None = None) -> DemoImport:
    """Dry run of one tab for one industry. tab = the tab the admin picked (default: the
    definition's tab_names); skip_columns = new columns the admin chose to leave out."""
    s = get_settings()
    if len(data) > s.demo_upload_max_file_mb * 1024 * 1024:
        raise Invalid(f"the file is larger than {s.demo_upload_max_file_mb} MB")
    definition = await load_definition(db, industry_key)
    industry = await db.get(DemoIndustry, industry_key)
    if industry is not None and industry.status == "archived":
        raise Conflict(f"{definition.label} is archived")
    try:
        report = parse_workbook(data, definition, source_name=filename,
                                timezone=s.demo_default_timezone, tab=tab,
                                skip_columns=skip_columns)
    except WorkbookError as e:
        raise Invalid(str(e)) from None
    out = report.as_json()

    # Say, per record, whether this import creates it or updates an existing one, and which
    # facts would change — so the admin sees the effect before applying.
    keys = [r["record_key"] for r in out["records"]]
    existing = {p.record_key: p for p in (await db.execute(select(DemoProspect).where(
        DemoProspect.industry_key == industry_key, DemoProspect.record_key.in_(keys)))).scalars()}
    current_facts: dict[uuid.UUID, dict[str, str | None]] = {}
    if existing:
        for f in (await db.execute(select(DemoProspectField).where(
                DemoProspectField.prospect_id.in_([p.id for p in existing.values()])))).scalars():
            current_facts.setdefault(f.prospect_id, {})[f.field_key] = f.value
    for r in out["records"]:
        p = existing.get(r["record_key"])
        if p is None:
            r["effect"] = {"action": "create"}
        else:
            before = current_facts.get(p.id, {})
            changed = sorted(k for k, v in r["facts"].items() if (before.get(k) or None) != v["value"])
            r["effect"] = {"action": "update", "record_id": str(p.id), "changed_facts": changed}
    out["summary"]["creates"] = sum(r["effect"]["action"] == "create" for r in out["records"])
    out["summary"]["updates"] = len(out["records"]) - out["summary"]["creates"]

    row = DemoImport(industry_key=industry_key, industry_version=definition.version,
                     source_name=filename[:255], file_sha256=hashlib.sha256(data).hexdigest(),
                     file_size=len(data), report=out, status="previewed", created_by=actor)
    db.add(row)
    await db.flush()
    return row


async def apply_import(db: AsyncSession, import_id: uuid.UUID, *, actor: str,
                       batch_id: uuid.UUID | None = None,
                       rows: list[int] | None = None) -> tuple[dict, list[IngestJob]]:
    """Save the previewed report. `rows` = the sheet rows the admin kept selected in the
    preview (None = all of them); the rest are left out and listed in the result."""
    row = (await db.execute(select(DemoImport).where(DemoImport.id == import_id)
                            .with_for_update())).scalar_one_or_none()
    if row is None:
        raise NotFound("import not found")
    if row.status != "previewed":
        raise Conflict(f"this import is already {row.status}")
    definition = await load_definition(db, row.industry_key)
    if definition.version != row.industry_version:
        raise Conflict("the industry definition changed since this preview; upload the "
                       "workbook again to preview it against the current version")
    if row.report.get("errors"):
        raise Invalid("the workbook has sheet-level errors; fix them and upload again",
                      errors=row.report["errors"])

    # Columns the preview proposed (new facts, or a renamed column re-connected to its
    # fact): their values are in the records already, so they become part of the industry
    # definition first, as its next version, and the records are saved against that.
    proposals = row.report.get("new_columns") or []
    if proposals:
        definition = await add_columns_to_industry(db, definition, proposals)

    records = row.report["records"]
    skipped: list[int] = []
    if rows is not None:
        wanted, known = set(rows), {r["row"] for r in records}
        if not wanted:
            raise Invalid("select at least one row to import")
        unknown = sorted(wanted - known)
        if unknown:
            raise Invalid(f"rows not in this preview: {', '.join(map(str, unknown[:10]))}")
        skipped = sorted(known - wanted)
        records = [r for r in records if r["row"] in wanted]

    result = {"created": 0, "updated": 0, "contacts_added": 0, "contacts_filled": 0,
              "documents_queued": 0, "agent_errors": [], "records": [],
              "selected": len(records), "skipped_rows": skipped,
              "industry_version": definition.version, "has_agent": definition.has_agent,
              "columns_added": [{"header": p["header"], "key": p["key"], "action": p["action"]}
                                for p in proposals]}
    jobs: list[IngestJob] = []
    for rec in records:
        prospect, created, contact_stats, job = await upsert_record(
            db, definition, rec, source_name=row.source_name, batch_id=batch_id)
        await provision_record(db, prospect, result)
        result["created" if created else "updated"] += 1
        result["contacts_added"] += contact_stats["added"]
        result["contacts_filled"] += contact_stats["filled"]
        if job:
            jobs.append(job)
        result["records"].append({"record_id": str(prospect.id), "record_key": prospect.record_key,
                                  "action": "create" if created else "update",
                                  "not_in_sheet": contact_stats["not_in_sheet"]})
    result["documents_queued"] = len(jobs)
    row.status = "applied"
    row.batch_id = batch_id
    row.applied_by = actor
    row.applied_at = datetime.now(timezone.utc)
    row.apply_result = result
    await db.flush()
    return result, jobs


async def provision_record(db: AsyncSession, prospect: DemoProspect, result: dict | None = None) -> None:
    """(Re)build the record's demo agent. A record that can't be provisioned yet (e.g. its
    dummy data doesn't fit its facts) is reported, not fatal: it is provisioned again at
    every later edit and at the start of every call. A data-only industry has no agent to
    build, so its records are skipped quietly."""
    from app.demo.agent import provision_demo_agent
    definition = await load_definition(db, prospect.industry_key, prospect.industry_version)
    if not definition.has_agent:
        return
    try:
        async with db.begin_nested():
            await provision_demo_agent(db, prospect)
    except DemoError as e:
        if result is not None:
            result["agent_errors"].append({"record_key": prospect.record_key, "error": e.message})
        log.info("demo agent for %s not provisioned: %s", prospect.record_key, e.message)


# ── company records ──────────────────────────────────────────────────────────────────────

async def create_record_tenant(db: AsyncSession, prospect: DemoProspect) -> None:
    """The record's own tenant and knowledge base (§2a): one tenant per record, so every
    tenant-scoped part of the engine isolates this demo from every other."""
    s = get_settings()
    tenant = Tenant(name=prospect.company_name[:200],
                    slug=f"{s.demo_tenant_slug_prefix}{prospect.id.hex[:12]}",
                    kind=TENANT_KIND_DEMO, timezone=prospect.timezone, plan="demo",
                    settings={"demo": {"prospect_id": str(prospect.id),
                                       "industry": prospect.industry_key}})
    db.add(tenant)
    await db.flush()
    kb = KnowledgeBase(tenant_id=tenant.id, name=KB_NAME,
                       scope_description=f"Public facts and documents for {prospect.company_name}",
                       reindex_policy="manual")
    db.add(kb)
    await db.flush()
    prospect.tenant_id = tenant.id
    prospect.knowledge_base_id = kb.id


def _fact_from_json(d: dict, fact_type: str) -> tuple[str | None, str | None, datetime | None]:
    value = d.get("value")
    if value is not None:
        value = normalize_fact(value, fact_type).value
    fetched = d.get("fetched_at")
    if isinstance(fetched, str):
        fetched = datetime.fromisoformat(fetched)
    return value, (d.get("source_url") or None) if value else None, fetched if value else None


async def upsert_facts(db: AsyncSession, prospect: DemoProspect, definition: IndustryDefinition,
                       facts: dict[str, dict], *, replace_missing: bool) -> list[str]:
    """Store facts; returns the keys whose value changed. replace_missing=True (a sheet
    import) blanks a fact the sheet no longer states; False (an admin edit of some facts)
    leaves the others alone."""
    existing = {f.field_key: f for f in (await db.execute(select(DemoProspectField).where(
        DemoProspectField.prospect_id == prospect.id))).scalars()}
    changed = []
    for f in definition.facts_fields:
        key = f["key"]
        if key not in facts and not replace_missing:
            continue
        value, source, fetched = _fact_from_json(facts.get(key) or {}, f.get("type", "text"))
        row = existing.get(key)
        if row is None:
            if value is None:
                continue
            db.add(DemoProspectField(prospect_id=prospect.id, field_key=key, value=value,
                                     source_url=source, fetched_at=fetched))
            changed.append(key)
        elif (row.value, row.source_url) != (value, source):
            if row.value != value:
                changed.append(key)
            row.value, row.source_url, row.fetched_at = value, source, fetched
    unknown = sorted(set(facts) - definition.fact_keys)
    if unknown:
        raise Invalid(f"unknown fact(s) for {definition.label}: {', '.join(unknown)}")
    await db.flush()
    return changed


async def load_facts(db: AsyncSession, prospect: DemoProspect,
                     definition: IndustryDefinition) -> dict[str, FactValue]:
    rows = {f.field_key: f for f in (await db.execute(select(DemoProspectField).where(
        DemoProspectField.prospect_id == prospect.id))).scalars()}
    out = {}
    for f in definition.facts_fields:
        row = rows.get(f["key"])
        fv = normalize_fact(row.value if row else None, f.get("type", "text"))
        if row is not None and not fv.is_blank:
            fv.source_url, fv.fetched_at = row.source_url, row.fetched_at
        out[f["key"]] = fv
    return out


def _facts_for_render(facts: dict[str, FactValue]) -> dict[str, dict]:
    return {k: {"value": v.value, "items": v.items, "source_url": v.source_url} for k, v in facts.items()}


async def refresh_company_facts_document(db: AsyncSession, prospect: DemoProspect,
                                         definition: IndustryDefinition) -> IngestJob | None:
    """Re-render the record's "Company facts" document; returns an ingest job when its text
    changed (or the last ingest didn't finish), None when nothing needs embedding."""
    if prospect.tenant_id is None or prospect.knowledge_base_id is None:
        return None
    facts = await load_facts(db, prospect, definition)
    text = render_company_facts(definition, company_name=prospect.company_name,
                                facts=_facts_for_render(facts))
    digest = content_hash(text)
    doc = (await db.execute(select(KnowledgeDocument).where(
        KnowledgeDocument.tenant_id == prospect.tenant_id,
        KnowledgeDocument.knowledge_base_id == prospect.knowledge_base_id,
        KnowledgeDocument.meta["demo_role"].astext == COMPANY_FACTS_ROLE))).scalar_one_or_none()
    if doc is not None and (doc.meta or {}).get("content_sha256") == digest and doc.status == "ready":
        return None
    meta = {"demo_role": COMPANY_FACTS_ROLE, "content_sha256": digest, "chars": len(text),
            "demo_prospect_id": str(prospect.id)}
    if doc is None:
        doc = KnowledgeDocument(tenant_id=prospect.tenant_id,
                                knowledge_base_id=prospect.knowledge_base_id,
                                title=f"{prospect.company_name}: company facts"[:300],
                                doc_type="faq", source_type="doc",
                                source_uri=prospect.website_url, status="processing", meta=meta)
        db.add(doc)
    else:
        doc.status, doc.meta = "processing", meta
    await db.flush()
    return IngestJob(prospect.tenant_id, doc.id, text)


async def upsert_record(db: AsyncSession, definition: IndustryDefinition, rec: dict, *,
                        source_name: str, batch_id: uuid.UUID | None
                        ) -> tuple[DemoProspect, bool, dict, IngestJob | None]:
    prospect = (await db.execute(select(DemoProspect).where(
        DemoProspect.industry_key == definition.key,
        DemoProspect.record_key == rec["record_key"]))).scalar_one_or_none()
    created = prospect is None
    if created:
        prospect = DemoProspect(industry_key=definition.key, industry_version=definition.version,
                                record_key=rec["record_key"], company_name=rec["company_name"],
                                timezone=get_settings().demo_default_timezone, status="draft",
                                created_by=None)
        db.add(prospect)
        await db.flush()
        await create_record_tenant(db, prospect)
    prospect.company_name = rec["company_name"]
    prospect.industry_version = definition.version
    prospect.city = rec.get("city")
    prospect.website_url = rec.get("website_url")
    prospect.trigger_note = rec.get("trigger_note")
    prospect.research_verified = bool(rec.get("research_verified"))
    prospect.source_file, prospect.source_row = source_name[:255], rec.get("row")
    await upsert_facts(db, prospect, definition, rec["facts"], replace_missing=True)
    stats = await upsert_contacts(db, prospect, rec.get("contacts") or [], batch_id=batch_id)
    job = await refresh_company_facts_document(db, prospect, definition)
    return prospect, created, stats, job


async def get_record(db: AsyncSession, record_id: uuid.UUID) -> DemoProspect:
    p = await db.get(DemoProspect, record_id)
    if p is None:
        raise NotFound("record not found")
    return p


async def record_summary(db: AsyncSession, prospect: DemoProspect, *, detail: bool) -> dict:
    definition = await load_definition(db, prospect.industry_key, prospect.industry_version)
    facts = await load_facts(db, prospect, definition)
    missing = check_minimum(definition.facts_fields, definition.minimum, facts)
    contacts = (await db.execute(select(DemoContact).where(
        DemoContact.prospect_id == prospect.id).order_by(DemoContact.full_name))).scalars().all()
    record_name_field = (definition.sheet or {}).get("record_name_field")
    out = {
        "id": str(prospect.id), "industry_key": prospect.industry_key,
        "industry_version": prospect.industry_version, "record_key": prospect.record_key,
        "company_name": prospect.company_name,
        "record_name": facts[record_name_field].value if record_name_field else None,
        "city": prospect.city, "status": prospect.status, "minimum_met": not missing,
        "minimum_missing": missing, "research_verified": prospect.research_verified,
        "contacts": len(contacts),
        "contacts_needing_review": sum(c.needs_review for c in contacts if c.status == "active"),
        "has_agent": definition.has_agent,
        "tenant_id": str(prospect.tenant_id) if prospect.tenant_id else None,
        "updated_at": prospect.updated_at,
    }
    if detail:
        docs = await list_kb_documents(db, prospect)
        _, data, source, version = await effective_dataset(db, prospect, definition)
        out.update({
            "website_url": prospect.website_url, "trigger_note": prospect.trigger_note,
            "timezone": prospect.timezone, "source_file": prospect.source_file,
            "source_row": prospect.source_row,
            "facts": {f["key"]: {"label": f["label"], "type": f.get("type", "text"),
                                 "required": bool(f.get("required")),
                                 "value": facts[f["key"]].value, "items": facts[f["key"]].items,
                                 "source_url": facts[f["key"]].source_url}
                      for f in definition.facts_fields},
            "contact_list": [contact_out(c) for c in contacts],
            "documents": docs,
            "dummy_data": {"source": source, "version": version},
        })
    return out


# ── contacts ─────────────────────────────────────────────────────────────────────────────

def contact_out(c: DemoContact) -> dict:
    return {"id": str(c.id), "record_id": str(c.prospect_id), "full_name": c.full_name,
            "first_name": c.first_name, "role": c.role, "email": c.email,
            "linkedin_url": c.linkedin_url, "needs_review": c.needs_review,
            "review_notes": c.review_notes, "status": c.status,
            "batch_id": str(c.batch_id) if c.batch_id else None}


async def upsert_contacts(db: AsyncSession, prospect: DemoProspect, people: list[dict], *,
                          batch_id: uuid.UUID | None) -> dict:
    existing = {c.full_name.lower(): c for c in (await db.execute(select(DemoContact).where(
        DemoContact.prospect_id == prospect.id))).scalars()}
    used_emails = {c.email.lower() for c in existing.values() if c.email}
    added = filled = 0
    seen = set()
    for person in people:
        name = person["full_name"].strip()
        seen.add(name.lower())
        email = (person.get("email") or None)
        if email and email.lower() in used_emails and (name.lower() not in existing or
                                                       (existing[name.lower()].email or "").lower() != email.lower()):
            email = None    # already someone else's at this company; leave it for review
        row = existing.get(name.lower())
        if row is None:
            notes = list(person.get("review_notes") or [])
            if person.get("email") and email is None:
                notes.append(f"email {person['email']} already belongs to another contact")
            db.add(DemoContact(prospect_id=prospect.id, batch_id=batch_id, full_name=name,
                               first_name=person.get("first_name"), role=person.get("role"),
                               email=email, linkedin_url=person.get("linkedin_url"),
                               needs_review=bool(notes) or not email,
                               review_notes="; ".join(notes) or None))
            if email:
                used_emails.add(email.lower())
            added += 1
            continue
        changed = False
        for attr, value in (("role", person.get("role")), ("email", email),
                            ("linkedin_url", person.get("linkedin_url"))):
            if getattr(row, attr) is None and value:
                setattr(row, attr, value)
                changed = True
                if attr == "email":
                    used_emails.add(value.lower())
        if row.batch_id is None and batch_id is not None:
            row.batch_id = batch_id
        if changed:
            filled += 1
    await db.flush()
    not_in_sheet = sorted(c.full_name for k, c in existing.items() if k not in seen)
    return {"added": added, "filled": filled, "not_in_sheet": not_in_sheet}


async def get_contact(db: AsyncSession, contact_id: uuid.UUID) -> DemoContact:
    c = await db.get(DemoContact, contact_id)
    if c is None:
        raise NotFound("contact not found")
    return c


async def ensure_contact_unique(db: AsyncSession, prospect_id: uuid.UUID, *, full_name: str | None,
                                email: str | None, exclude: uuid.UUID | None = None) -> None:
    q = select(DemoContact).where(DemoContact.prospect_id == prospect_id)
    if exclude is not None:
        q = q.where(DemoContact.id != exclude)
    others = (await db.execute(q)).scalars().all()
    if full_name and any(o.full_name.lower() == full_name.lower() for o in others):
        raise Conflict(f"{full_name} is already a contact at this company")
    if email and any((o.email or "").lower() == email.lower() for o in others):
        raise Conflict(f"{email} already belongs to another contact at this company")


# ── knowledge documents ──────────────────────────────────────────────────────────────────

async def list_kb_documents(db: AsyncSession, prospect: DemoProspect) -> list[dict]:
    if prospect.tenant_id is None:
        return []
    docs = (await db.execute(select(KnowledgeDocument).where(
        KnowledgeDocument.tenant_id == prospect.tenant_id,
        KnowledgeDocument.knowledge_base_id == prospect.knowledge_base_id).order_by(
        KnowledgeDocument.created_at))).scalars().all()
    counts = dict((await db.execute(select(KnowledgeChunk.document_id, func.count()).where(
        KnowledgeChunk.tenant_id == prospect.tenant_id).group_by(KnowledgeChunk.document_id))).all())
    return [{"id": str(d.id), "title": d.title, "doc_type": d.doc_type, "source_uri": d.source_uri,
             "status": d.status, "chunks": int(counts.get(d.id, 0)),
             "generated": (d.meta or {}).get("demo_role") == COMPANY_FACTS_ROLE,
             "kind": (d.meta or {}).get("kind"), "chars": (d.meta or {}).get("chars"),
             "last_indexed_at": d.last_indexed_at, "created_at": d.created_at} for d in docs]


async def add_kb_document(db: AsyncSession, prospect: DemoProspect, *, title: str, text: str,
                          doc_type: str, kind: str, source_uri: str | None) -> IngestJob:
    from app.api.knowledge import DOC_TYPES
    if prospect.tenant_id is None:
        raise Conflict("this record has no knowledge base yet")
    if doc_type not in DOC_TYPES:
        raise Invalid(f"doc_type must be one of: {', '.join(DOC_TYPES)}")
    s = get_settings()
    docs = (await db.execute(select(KnowledgeDocument).where(
        KnowledgeDocument.tenant_id == prospect.tenant_id,
        KnowledgeDocument.knowledge_base_id == prospect.knowledge_base_id))).scalars().all()
    if len(docs) >= s.demo_upload_max_files_per_prospect:
        raise Conflict(f"this record already has {len(docs)} documents "
                       f"(limit {s.demo_upload_max_files_per_prospect})")
    digest = content_hash(text)
    dupe = next((d for d in docs if (d.meta or {}).get("content_sha256") == digest), None)
    if dupe is not None:
        raise Conflict(f"the same content is already uploaded as {dupe.title!r}")
    doc = KnowledgeDocument(tenant_id=prospect.tenant_id, knowledge_base_id=prospect.knowledge_base_id,
                            title=title[:300] or "untitled", doc_type=doc_type,
                            source_type="url" if source_uri and source_uri.startswith("http") else "file",
                            source_uri=source_uri, status="processing",
                            meta={"content_sha256": digest, "chars": len(text), "kind": kind,
                                  "demo_prospect_id": str(prospect.id)})
    db.add(doc)
    await db.flush()
    return IngestJob(prospect.tenant_id, doc.id, text)


async def delete_kb_document(db: AsyncSession, prospect: DemoProspect, document_id: uuid.UUID) -> str:
    doc = (await db.execute(select(KnowledgeDocument).where(
        KnowledgeDocument.id == document_id, KnowledgeDocument.tenant_id == prospect.tenant_id))
    ).scalar_one_or_none()
    if doc is None:
        raise NotFound("document not found for this record")
    if (doc.meta or {}).get("demo_role") == COMPANY_FACTS_ROLE:
        raise Conflict("the company facts document is generated from the facts; edit the facts instead")
    await db.execute(delete(KnowledgeChunk).where(KnowledgeChunk.document_id == doc.id))
    await db.delete(doc)
    await db.flush()
    return doc.title


# ── dummy data ───────────────────────────────────────────────────────────────────────────

async def _active_dataset(db: AsyncSession, *, prospect_id: uuid.UUID | None,
                          industry_key: str) -> DemoSampleDataset | None:
    q = select(DemoSampleDataset).where(DemoSampleDataset.industry_key == industry_key,
                                        DemoSampleDataset.is_active.is_(True))
    q = q.where(DemoSampleDataset.prospect_id == prospect_id) if prospect_id else \
        q.where(DemoSampleDataset.prospect_id.is_(None))
    return (await db.execute(q)).scalar_one_or_none()


async def effective_dataset(db: AsyncSession, prospect: DemoProspect, definition: IndustryDefinition
                            ) -> tuple[DemoSampleDataset | None, dict | None, str, int | None]:
    """(row, data, source, version): the record's own dummy data if it has one, else the
    industry template. source is "record", "template" or "none"."""
    own = await _active_dataset(db, prospect_id=prospect.id, industry_key=prospect.industry_key)
    if own is not None:
        return own, own.data, "record", own.version
    tpl = await _active_dataset(db, prospect_id=None, industry_key=prospect.industry_key)
    if tpl is not None:
        return tpl, tpl.data, "template", tpl.version
    return None, None, "none", None


async def _store_dataset(db: AsyncSession, *, prospect_id: uuid.UUID | None,
                         definition: IndustryDefinition, data: dict) -> DemoSampleDataset:
    q = select(func.max(DemoSampleDataset.version)).where(DemoSampleDataset.industry_key == definition.key)
    q = q.where(DemoSampleDataset.prospect_id == prospect_id) if prospect_id else \
        q.where(DemoSampleDataset.prospect_id.is_(None))
    latest = (await db.execute(q)).scalar() or 0
    current = await _active_dataset(db, prospect_id=prospect_id, industry_key=definition.key)
    if current is not None:
        current.is_active = False
        await db.flush()        # the partial unique index allows one active row per owner
    row = DemoSampleDataset(prospect_id=prospect_id, industry_key=definition.key,
                            industry_version=definition.version, data=data,
                            version=latest + 1, is_active=True)
    db.add(row)
    await db.flush()
    return row


def _require_agent(definition: IndustryDefinition) -> None:
    if not definition.has_agent:
        raise Conflict(f"{definition.label} is a data-only industry: it has no dummy data "
                       "until a version with a demo agent is published")


def check_dataset(definition: IndustryDefinition, data: Any,
                  facts: dict[str, FactValue] | None) -> list[str]:
    if not isinstance(data, dict):
        return ["(root): dummy data must be a JSON object"]
    values = {k: v.value for k, v in facts.items()} if facts is not None else None
    return [str(e) for e in validate_dataset(data, definition.sample_schema,
                                             allowed_placeholders=definition.fact_keys,
                                             facts=values)]


async def save_record_dataset(db: AsyncSession, prospect: DemoProspect, data: Any, *,
                              dry_run: bool) -> tuple[list[str], DemoSampleDataset | None]:
    definition = await load_definition(db, prospect.industry_key, prospect.industry_version)
    _require_agent(definition)
    errors = check_dataset(definition, data, await load_facts(db, prospect, definition))
    if errors or dry_run:
        return errors, None
    return [], await _store_dataset(db, prospect_id=prospect.id, definition=definition, data=data)


async def save_template_dataset(db: AsyncSession, industry_key: str, data: Any, *,
                                dry_run: bool) -> tuple[list[str], DemoSampleDataset | None]:
    definition = await load_definition(db, industry_key)
    _require_agent(definition)
    errors = check_dataset(definition, data, None)
    if errors or dry_run:
        return errors, None
    return [], await _store_dataset(db, prospect_id=None, definition=definition, data=data)


async def clear_record_dataset(db: AsyncSession, prospect: DemoProspect) -> bool:
    own = await _active_dataset(db, prospect_id=prospect.id, industry_key=prospect.industry_key)
    if own is None:
        return False
    own.is_active = False
    await db.flush()
    return True


async def rollback_dataset(db: AsyncSession, *, industry_key: str, prospect_id: uuid.UUID | None,
                           version: int) -> DemoSampleDataset:
    q = select(DemoSampleDataset).where(DemoSampleDataset.industry_key == industry_key,
                                        DemoSampleDataset.version == version)
    q = q.where(DemoSampleDataset.prospect_id == prospect_id) if prospect_id else \
        q.where(DemoSampleDataset.prospect_id.is_(None))
    target = (await db.execute(q)).scalar_one_or_none()
    if target is None:
        raise NotFound(f"no dummy data version {version}")
    if target.is_active:
        return target
    current = await _active_dataset(db, prospect_id=prospect_id, industry_key=industry_key)
    if current is not None:
        current.is_active = False
        await db.flush()
    target.is_active = True
    await db.flush()
    return target


async def resolve_for_call(db: AsyncSession, prospect: DemoProspect, at: datetime
                           ) -> tuple["ResolvedContext", str, int | None]:
    """(context, source, version): the record's effective dummy data, checked against its
    facts and resolved at `at` — exactly what a call starting at that moment works from.
    Raises Conflict/Invalid when there is no usable dummy data."""
    definition = await load_definition(db, prospect.industry_key, prospect.industry_version)
    _require_agent(definition)
    facts = await load_facts(db, prospect, definition)
    _, data, source, version = await effective_dataset(db, prospect, definition)
    if data is None:
        raise Conflict("no dummy data: upload one for this record or an industry template")
    errors = check_dataset(definition, data, facts)
    if errors:
        raise Invalid("the dummy data doesn't fit this record's facts", errors=errors)
    ctx = resolve_dataset(data, definition.sample_schema, {k: v.value for k, v in facts.items()},
                          now=at, tz=prospect.timezone, skip_weekdays=definition.skip_weekdays)
    return ctx, source, version


async def preview_dataset(db: AsyncSession, prospect: DemoProspect, at: datetime) -> dict:
    ctx, source, version = await resolve_for_call(db, prospect, at)
    return {"source": source, "version": version, **ctx.as_json()}
