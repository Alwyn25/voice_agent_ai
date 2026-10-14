"""Tests for the demo startup check (app/demo/settings.py, docs/gtm-demo-links.md §13).

Pure unit tests, no DB or network:

    cd core && python -m pytest tests/test_demo_settings.py -q
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.demo.settings import assert_demo_settings_valid, csv_list, validate_demo_settings  # noqa: E402


FAKE_GOOGLE_KEY = "eyJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsICJjbGllbnRfZW1haWwiOiAiZGVtb0B0ZXN0LmlhbS5nc2VydmljZWFjY291bnQuY29tIiwgInByaXZhdGVfa2V5IjogIi0tLS0tQkVHSU4gUFJJVkFURSBLRVktLS0tLVxubm90LWEtcmVhbC1rZXlcbi0tLS0tRU5EIFBSSVZBVEUgS0VZLS0tLS1cbiJ9"     # well-formed, not a real key
KEY = "uR5bYcG2Hk0yq0Qm3m4aQ1dO8JtV3bq7wP9xXo2nZzQ="     # a well-formed Fernet key, test only


def _settings(**overrides) -> Settings:
    """A valid production-style demo config; each test breaks one thing. _env_file=None
    so the developer's own .env never leaks into the test."""
    base = dict(
        env="production",
        demo_enabled=True,
        demo_admin_tenant_ids="7c6e2c1e-3f43-4c0e-9d7a-1f2b3c4d5e6f",
        demo_public_base_url="https://demo.phonn.ai",
        demo_allowed_origins="https://demo.phonn.ai",
        demo_ip_hash_secret="a-real-secret",
        demo_turnstile_enabled=True,
        demo_turnstile_secret="0x-turnstile-secret",
        demo_email_provider="smtp",
        demo_lead_notify_emails="founder@phonn.ai, sales@phonn.ai",
        demo_lead_from_email="demo-leads@phonn.ai",
        smtp_host="smtp.gmail.com",
        smtp_username="demo-leads@phonn.ai",
        smtp_password="app-password",
        demo_leads_gsheet_id="sheet-id",
        google_service_account_json=FAKE_GOOGLE_KEY,
        demo_link_token_keys=KEY,
    )
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _assert_error(settings: Settings, fragment: str) -> None:
    errors = validate_demo_settings(settings)
    assert any(fragment in e for e in errors), f"expected {fragment!r} in {errors}"


def test_valid_config_passes():
    assert validate_demo_settings(_settings()) == []
    assert_demo_settings_valid(_settings())   # does not raise


def test_nothing_checked_while_disabled():
    s = Settings(_env_file=None, demo_enabled=False, demo_public_base_url="",
                 demo_email_provider="none")
    assert validate_demo_settings(s) == []


def test_defaults_are_off():
    s = Settings(_env_file=None)
    assert s.demo_enabled is False
    assert validate_demo_settings(s) == []


def test_raise_lists_every_problem_at_once():
    s = _settings(demo_public_base_url="", demo_lead_notify_emails="", smtp_host="")
    with pytest.raises(RuntimeError) as exc:
        assert_demo_settings_valid(s)
    msg = str(exc.value)
    assert "DEMO_PUBLIC_BASE_URL is required" in msg
    assert "DEMO_LEAD_NOTIFY_EMAILS is required" in msg
    assert "SMTP_HOST is required" in msg


# ── Domain ──

def test_base_url_required():
    _assert_error(_settings(demo_public_base_url=""), "DEMO_PUBLIC_BASE_URL is required")


def test_https_required_outside_local():
    _assert_error(_settings(demo_public_base_url="http://demo.phonn.ai",
                            demo_allowed_origins="http://demo.phonn.ai"), "must use https")


def test_http_localhost_allowed_locally():
    s = _settings(env="local", demo_public_base_url="http://localhost:5174",
                  demo_allowed_origins="http://localhost:5174")
    assert validate_demo_settings(s) == []


def test_http_non_localhost_refused_even_locally():
    _assert_error(_settings(env="local", demo_public_base_url="http://demo.example.com",
                            demo_allowed_origins="http://demo.example.com"), "must use https")


def test_origin_must_be_allowed():
    _assert_error(_settings(demo_allowed_origins="https://app.phonn.ai"),
                  "DEMO_ALLOWED_ORIGINS must include https://demo.phonn.ai")


def test_origin_match_ignores_trailing_slash_and_case():
    s = _settings(demo_public_base_url="https://Demo.Phonn.ai/",
                  demo_allowed_origins="https://app.phonn.ai, https://demo.phonn.ai/")
    assert validate_demo_settings(s) == []


def test_base_url_with_query_refused():
    _assert_error(_settings(demo_public_base_url="https://demo.phonn.ai?x=1"),
                  "query string or fragment")


def test_garbage_base_url_refused():
    _assert_error(_settings(demo_public_base_url="demo.phonn.ai"), "not a valid URL")


# ── Limits ──

@pytest.mark.parametrize("name", ["demo_link_ttl_days", "demo_max_call_seconds",
                                  "demo_max_concurrent_calls_global", "demo_sheet_sync_minutes"])
def test_limits_must_be_positive(name):
    _assert_error(_settings(**{name: 0}), f"{name.upper()} must be greater than 0")


def test_per_link_concurrency_cannot_exceed_global():
    _assert_error(_settings(demo_max_concurrent_calls_per_link=5,
                            demo_max_concurrent_calls_global=2), "cannot exceed")


def test_lifetime_minutes_shorter_than_one_call():
    _assert_error(_settings(demo_max_minutes_per_link=1, demo_max_call_seconds=300),
                  "shorter than one full call")


def test_budget_zero_means_no_cap_and_is_valid():
    assert validate_demo_settings(_settings(demo_daily_budget_usd=0)) == []


def test_negative_budget_refused():
    _assert_error(_settings(demo_daily_budget_usd=-1), "cannot be negative")


def test_alert_pct_range():
    _assert_error(_settings(demo_budget_alert_pct=150), "between 1 and 100")


def test_bad_timezone():
    _assert_error(_settings(demo_budget_timezone="Mars/Olympus"), "not a known timezone")


# ── Abuse protection ──

def test_turnstile_needs_secret():
    _assert_error(_settings(demo_turnstile_enabled=True, demo_turnstile_secret=""),
                  "DEMO_TURNSTILE_SECRET is required")


def test_default_ip_hash_secret_refused_in_production():
    _assert_error(_settings(demo_ip_hash_secret="change-me"), "DEMO_IP_HASH_SECRET")


def test_turnstile_required_outside_local():
    _assert_error(_settings(demo_turnstile_enabled=False), "DEMO_TURNSTILE_ENABLED must be true")
    s = _settings(env="local", demo_public_base_url="http://localhost:5174",
                  demo_allowed_origins="http://localhost:5174", demo_turnstile_enabled=False)
    assert validate_demo_settings(s) == []


def test_turnstile_explicit_opt_out():
    s = _settings(demo_turnstile_enabled=False, demo_turnstile_required=False)
    assert validate_demo_settings(s) == []


def test_proxy_hops():
    _assert_error(_settings(demo_trusted_proxy_hops=0), "every caller shares the proxy's IP")
    _assert_error(_settings(demo_trusted_proxy_hops=9), "between 0 and 5")
    _assert_error(_settings(demo_client_ip_header="x forwarded"), "DEMO_CLIENT_IP_HEADER")
    assert validate_demo_settings(_settings(demo_trusted_proxy_hops=2)) == []
    assert validate_demo_settings(_settings(demo_client_ip_header="cf-connecting-ip")) == []


# ── Where leads go ──

def test_leads_must_reach_someone():
    _assert_error(_settings(demo_email_provider="none", demo_leads_gsheet_id=""),
                  "Leads would reach nobody")


def test_sheet_only_is_valid():
    s = _settings(demo_email_provider="none", demo_lead_notify_emails="",
                  demo_lead_from_email="", smtp_host="")
    assert validate_demo_settings(s) == []


def test_email_only_is_valid():
    assert validate_demo_settings(_settings(demo_leads_gsheet_id="",
                                            google_service_account_json="")) == []


def test_sheet_needs_service_account():
    _assert_error(_settings(google_service_account_json=""), "GOOGLE_SERVICE_ACCOUNT_JSON")


def test_smtp_needs_host():
    _assert_error(_settings(smtp_host=""), "SMTP_HOST is required")


def test_smtp_user_and_password_together():
    _assert_error(_settings(smtp_password=""), "must be set together")


@pytest.mark.parametrize("provider", ["sendgrid", "resend"])
def test_api_providers_need_key(provider):
    _assert_error(_settings(demo_email_provider=provider, demo_email_api_key=""),
                  "DEMO_EMAIL_API_KEY is required")


def test_api_provider_with_key_is_valid():
    assert validate_demo_settings(_settings(demo_email_provider="resend",
                                            demo_email_api_key="re_123")) == []


def test_unknown_email_provider_rejected_by_settings():
    with pytest.raises(Exception):
        _settings(demo_email_provider="carrier-pigeon")


def test_bad_recipient_address():
    _assert_error(_settings(demo_lead_notify_emails="founder@phonn.ai, sales team"),
                  "Not valid email addresses: sales team")


def test_alert_emails_need_an_email_provider():
    _assert_error(_settings(demo_email_provider="none", demo_lead_notify_emails="",
                            demo_alert_emails="founder@phonn.ai"), "alerts could never be sent")


def test_statuses_cannot_be_empty():
    _assert_error(_settings(demo_lead_statuses=" , "), "DEMO_LEAD_STATUSES")


def test_csv_list():
    assert csv_list(" a@x.com, ,b@x.com ") == ["a@x.com", "b@x.com"]
    assert csv_list("") == []


def test_admin_tenants_required_and_must_be_ids():
    _assert_error(_settings(demo_admin_tenant_ids=""), "DEMO_ADMIN_TENANT_IDS is required")
    _assert_error(_settings(demo_admin_tenant_ids="phonn"), "not a tenant id: 'phonn'")


# ── company links ────────────────────────────────────────────────────────────────────────

def test_link_token_key_required_and_well_formed():
    _assert_error(_settings(demo_link_token_keys=""), "DEMO_LINK_TOKEN_KEYS is required")
    _assert_error(_settings(demo_link_token_keys="not-a-key"), "entry 1 is not a valid Fernet key")
    # Rotation: a second, older key is fine.
    assert validate_demo_settings(_settings(demo_link_token_keys=f"{KEY}, {KEY}")) == []


def test_export_steps_and_utm_values():
    _assert_error(_settings(demo_export_steps="e1,email2"), "DEMO_EXPORT_STEPS")
    _assert_error(_settings(demo_export_steps=""), "DEMO_EXPORT_STEPS")
    _assert_error(_settings(demo_utm_source="cold email!"), "DEMO_UTM_SOURCE")
    _assert_error(_settings(demo_utm_campaign="oct batch"), "DEMO_UTM_CAMPAIGN")
    # UTM values aren't checked when UTM tagging is off.
    assert validate_demo_settings(_settings(demo_utm_enabled=False, demo_utm_source="x y")) == []


def test_google_key_must_be_a_service_account_key():
    _assert_error(_settings(google_service_account_json="eyJ0eXBlIjoic2VydmljZV9hY2NvdW50In0="),
                  "not a service-account key")
    _assert_error(_settings(google_service_account_json="not base64 or json"), "not a service-account key")
