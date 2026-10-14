"""Launch industry definitions: load the YAML seeds, and put them in the database.

Seeds only ever CREATE: an industry that already exists in demo_industries belongs to the
admins from then on, so a later deploy with a changed seed never overwrites their edits.
A seed whose version is newer than anything in the database is logged, so a developer
knows to publish it as a new version through the admin screen (or a migration) on purpose.

Every seed is compiled (validated) at load time, and the test suite compiles all of them,
so a broken seed fails CI rather than a deploy.
"""
import logging
from pathlib import Path

import yaml
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import DemoIndustry, DemoIndustryVersion, DemoSampleDataset
from app.demo.industries.definition import (DefinitionError, IndustryDefinition,
                                            compile_definition)

log = logging.getLogger(__name__)
_CORE_DIR = Path(__file__).resolve().parents[3]          # .../core


def seed_dir() -> Path:
    p = Path(get_settings().demo_industries_seed_dir)
    return p if p.is_absolute() else _CORE_DIR / p


def load_seed_definitions(directory: Path | None = None, *,
                          errors: list[str] | None = None) -> list[IndustryDefinition]:
    """Every *.yaml seed, compiled. Raises on the first invalid seed — unless `errors` is
    given: then an invalid seed is described there and skipped, and the rest still load."""
    directory = directory or seed_dir()
    defs = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            definition = compile_definition(raw)
            if path.stem != definition.key:
                raise ValueError(f"{path.name}: file name must match its key {definition.key!r}")
        except (DefinitionError, ValueError, yaml.YAMLError) as e:
            if errors is None:
                raise
            errors.append(f"{path.name}: {e}")
            continue
        defs.append(definition)
    return defs


async def sync_seed_industries(db: AsyncSession, definitions: list[IndustryDefinition] | None = None
                               ) -> list[str]:
    """Create any seed industry missing from the database (status active, its version, and
    its industry-wide dummy-data template when it has a demo agent). Returns the keys
    created. Caller commits."""
    if definitions is None:
        # One broken seed must not keep the others out of the database.
        problems: list[str] = []
        definitions = load_seed_definitions(errors=problems)
        for p in problems:
            log.error("demo industry seed skipped: %s", p)
    created = []
    for d in definitions:
        existing = await db.get(DemoIndustry, d.key)
        if existing is not None:
            latest = (await db.execute(select(func.max(DemoIndustryVersion.version)).where(
                DemoIndustryVersion.industry_key == d.key))).scalar() or 0
            if d.version > latest:
                log.info("demo industry %s: seed v%s is newer than the database's v%s; not "
                         "applied automatically (publish it from the admin screen)",
                         d.key, d.version, latest)
            continue
        db.add(DemoIndustry(key=d.key, label=d.label, status="active", current_version=d.version))
        await db.flush()
        db.add(DemoIndustryVersion(industry_key=d.key, version=d.version, definition=d.raw))
        await db.flush()
        if d.has_agent:                  # a data-only industry has no dummy data yet
            db.add(DemoSampleDataset(prospect_id=None, industry_key=d.key,
                                     industry_version=d.version, data=d.sample_template,
                                     version=1, is_active=True))
        created.append(d.key)
    if created:
        await db.flush()
        log.info("demo industries created from seeds: %s", ", ".join(created))
    return created
