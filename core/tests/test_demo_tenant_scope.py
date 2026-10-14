"""Demo tenants must never be walked by the background workers (docs/gtm-demo-links.md
§2a): the dialer would try to dial for them, and the aggregator would roll demo calls
into the dashboards. Both now select their tenants through customer_tenant_ids().

Pure unit tests (the statement is compiled, not run):

    cd core && python -m pytest tests/test_demo_tenant_scope.py -q
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy.dialects import postgresql  # noqa: E402

from app.db.models import DemoLead, DemoProspect, Tenant  # noqa: E402
from app.db.models.platform import (TENANT_KIND_CUSTOMER, TENANT_KIND_DEMO,  # noqa: E402
                                    customer_tenant_ids)


def _sql(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect(),
                            compile_kwargs={"literal_binds": True}))


def test_customer_tenant_ids_filters_on_kind():
    sql = _sql(customer_tenant_ids())
    assert "tenants.kind = 'customer'" in sql
    assert sql.strip().startswith("SELECT tenants.id")


def test_kind_constants():
    assert (TENANT_KIND_CUSTOMER, TENANT_KIND_DEMO) == ("customer", "demo")


def test_tenant_kind_defaults_to_customer():
    col = Tenant.__table__.c.kind
    assert col.default.arg == "customer"
    assert col.server_default.arg == "customer"


def test_dialer_and_aggregator_use_customer_tenants_only():
    from app.services import aggregator
    from app.services.telephony import dialer
    for fn in (aggregator.run_aggregator_tick, dialer.run_dialer_tick):
        src = inspect.getsource(fn)
        assert "customer_tenant_ids()" in src, fn.__qualname__
        assert "select(Tenant.id)" not in src, fn.__qualname__


def test_no_other_unfiltered_tenant_sweep():
    """Guard for future code: any new `select(Tenant.id)` sweep over all tenants in app/
    must use customer_tenant_ids() instead, or demos leak into it."""
    app_dir = Path(__file__).resolve().parents[1] / "app"
    helper = app_dir / "db" / "models" / "platform.py"   # customer_tenant_ids() itself
    offenders = [str(p.relative_to(app_dir)) for p in app_dir.rglob("*.py")
                 if p != helper and "select(Tenant.id)" in p.read_text(encoding="utf-8")]
    assert offenders == [], offenders


def _unique_constraints(model) -> dict[str, list[str]]:
    from sqlalchemy import UniqueConstraint
    return {c.name: [col.name for col in c.columns]
            for c in model.__table__.constraints if isinstance(c, UniqueConstraint)}


def test_one_tenant_per_prospect_is_enforced_by_the_model():
    assert _unique_constraints(DemoProspect)["uq_demo_prospects_tenant"] == ["tenant_id"]


def test_lead_row_key_is_unique():
    assert _unique_constraints(DemoLead)["uq_demo_leads_sheet_row_key"] == ["sheet_row_key"]
