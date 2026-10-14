"""PackRegistry — the one place the engine asks "which industry pack is this
tenant on?" Concrete packs (app.packs.real_estate_india, app.packs.generic) are
loaded and self-registered the first time this module is imported anywhere (see
_load_builtin_packs() at the bottom); callers never import a concrete pack module
directly, only `from app.packs import PackRegistry`.
"""
from __future__ import annotations

import uuid

from app.packs.base import IndustryPack

DEFAULT_SLUG = "real_estate_india"
GENERIC_SLUG = "generic"

_REGISTRY: dict[str, IndustryPack] = {}

# A tenant's assigned pack essentially never changes at runtime (there is no
# reassignment endpoint yet), so caching the slug lookup avoids re-running the
# industry_packs/tenant_pack_config join on every single planner turn of every
# call — the same tradeoff engine/planner/prompts.py's _rendered_base_cache
# already makes for a pack's rendered prompt sections. LRU-bounded (unlike that
# cache, which is bounded by there only ever being a handful of packs) since
# this one has one entry per tenant ever seen — an unbounded version would grow
# for the life of the process; revisit invalidation if/when a reassignment
# endpoint is added in a later wave.
_tenant_slug_cache: dict[uuid.UUID, str | None] = {}
_tenant_slug_cache_order: list[uuid.UUID] = []
_TENANT_SLUG_CACHE_MAX = 4096


class PackRegistry:
    @classmethod
    def register(cls, pack: IndustryPack) -> None:
        _REGISTRY[pack.slug] = pack

    @classmethod
    def get(cls, slug: str | None) -> IndustryPack:
        """Falls back to the generic pack for an unknown/blank slug rather than
        raising — a tenant with no pack assigned yet, or a bad slug, degrades to
        the 5-state generic journey, never a 500."""
        if slug and slug in _REGISTRY:
            return _REGISTRY[slug]
        if GENERIC_SLUG in _REGISTRY:
            return _REGISTRY[GENERIC_SLUG]
        raise LookupError("No industry packs are registered")

    @classmethod
    def get_default(cls) -> IndustryPack:
        """Used wherever there is no tenant context at all to resolve a pack
        from (e.g. a one-off script). Real request/call paths should always
        prefer for_tenant()."""
        return cls.get(DEFAULT_SLUG)

    @classmethod
    async def slug_for_tenant(cls, db, tenant_id: uuid.UUID) -> str | None:
        from sqlalchemy import select

        from app.db.models import IndustryPackDef, TenantPackConfig
        return (await db.execute(
            select(IndustryPackDef.slug)
            .join(TenantPackConfig, TenantPackConfig.pack_id == IndustryPackDef.id)
            .where(TenantPackConfig.tenant_id == tenant_id))).scalar_one_or_none()

    @classmethod
    async def for_tenant(cls, db, tenant_id: uuid.UUID) -> IndustryPack:
        if tenant_id in _tenant_slug_cache:
            _tenant_slug_cache_order.remove(tenant_id)
            _tenant_slug_cache_order.append(tenant_id)
            return cls.get(_tenant_slug_cache[tenant_id])

        slug = await cls.slug_for_tenant(db, tenant_id)
        _tenant_slug_cache[tenant_id] = slug
        _tenant_slug_cache_order.append(tenant_id)
        while len(_tenant_slug_cache_order) > _TENANT_SLUG_CACHE_MAX:
            _tenant_slug_cache.pop(_tenant_slug_cache_order.pop(0), None)
        return cls.get(slug)

    @classmethod
    async def vocabulary_for_tenant(cls, db, tenant_id: uuid.UUID):
        """Vocal Backend Tasksheet P6 Generic Sprint 3's "tenant.vocabulary config table" —
        reconciled onto the pre-existing `tenant_pack_config.overrides` JSONB column rather
        than a new table, since that column's own docstring already anticipated exactly this
        ("e.g. a tenant-specific vocabulary tweak"). A tenant's vocabulary is therefore never
        a materialised snapshot to seed/backfill — it is always the active pack's own
        `vocabulary` (the "seed from pack defaults" the tasksheet asks for, satisfied for
        free since an empty `overrides` dict means "use the pack's defaults exactly") with
        any `overrides["vocabulary"]` fields layered on top, resolved fresh on every call
        the same way `for_tenant()` already resolves the pack itself."""
        from dataclasses import replace

        from sqlalchemy import select

        from app.db.models import TenantPackConfig

        pack = await cls.for_tenant(db, tenant_id)
        cfg = (await db.execute(select(TenantPackConfig).where(
            TenantPackConfig.tenant_id == tenant_id))).scalar_one_or_none()
        voc_overrides = (cfg.overrides or {}).get("vocabulary") if cfg else None
        if not voc_overrides:
            return pack.vocabulary
        return replace(pack.vocabulary, **voc_overrides)


def _load_builtin_packs() -> None:
    from app.packs.generic.pack import GenericPack
    from app.packs.healthcare.pack import HealthcarePack
    from app.packs.real_estate_india.pack import RealEstateIndiaPack
    from app.packs.saas_sales.pack import SaasSalesPack
    PackRegistry.register(RealEstateIndiaPack())
    PackRegistry.register(GenericPack())
    PackRegistry.register(HealthcarePack())
    PackRegistry.register(SaasSalesPack())


_load_builtin_packs()
