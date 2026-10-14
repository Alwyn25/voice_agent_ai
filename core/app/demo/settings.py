"""Startup check for the demo settings (docs/gtm-demo-links.md §13).

The demo surface is public and its whole job is to turn visitors into leads, so a
misconfigured deploy must fail loudly at boot rather than run and silently drop leads
(no recipients, no sheet), leak links on plain http, or reject every browser call on an
Origin mismatch. validate_demo_settings() returns every problem at once, so one restart
shows the full list instead of one error per attempt.
"""
import logging
import re
import uuid
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from cryptography.fernet import Fernet

from app.config import Settings

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}
STEP_RE = re.compile(r"^(e[1-9]|li)$")              # email step 1–9, or a LinkedIn note
UTM_VALUE_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")
IP_HEADER_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")


def csv_list(value: str) -> list[str]:
    """A comma-separated setting as a clean list (blank entries dropped)."""
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def _origin(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}".lower()


def _bad_emails(values: list[str]) -> list[str]:
    # A shape check only — deliverability is the mail provider's business. Catches the
    # common .env slips: a missing @, a stray space, a name instead of an address.
    return [v for v in values if "@" not in v or " " in v or v.startswith("@") or v.endswith("@")]


def validate_demo_settings(s: Settings) -> list[str]:
    """Every problem with the demo settings, as human-readable lines. Empty = OK.
    Nothing is checked while the feature is off."""
    if not s.demo_enabled:
        return []
    errors: list[str] = []
    local = s.env == "local"

    # ── Domain / URLs ──
    base = s.demo_public_base_url.strip().rstrip("/")
    if not base:
        errors.append("DEMO_PUBLIC_BASE_URL is required (every demo link is built from it).")
    else:
        parts = urlsplit(base)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            errors.append(f"DEMO_PUBLIC_BASE_URL is not a valid URL: {base!r}.")
        elif parts.scheme != "https" and not (local and parts.hostname in _LOCAL_HOSTS):
            errors.append("DEMO_PUBLIC_BASE_URL must use https outside local development "
                          "(the token in the link must not travel in clear text).")
        else:
            origins = {o.rstrip("/").lower() for o in csv_list(s.demo_allowed_origins)}
            if _origin(base) not in origins:
                errors.append(f"DEMO_ALLOWED_ORIGINS must include {_origin(base)} "
                              "(the demo page's own origin), or every browser call is refused.")
        if parts.query or parts.fragment:
            errors.append("DEMO_PUBLIC_BASE_URL must not contain a query string or fragment.")

    # ── Who may administer demos ──
    admin_ids = csv_list(s.demo_admin_tenant_ids)
    if not admin_ids:
        errors.append("DEMO_ADMIN_TENANT_IDS is required: the tenant id(s) of our own staff "
                      "organisation, whose admins may use the demo admin API.")
    for v in admin_ids:
        try:
            uuid.UUID(v)
        except ValueError:
            errors.append(f"DEMO_ADMIN_TENANT_IDS has a value that is not a tenant id: {v!r}.")

    # ── Company links ──
    keys = csv_list(s.demo_link_token_keys)
    if not keys:
        errors.append("DEMO_LINK_TOKEN_KEYS is required: a Fernet key that encrypts each company "
                      "link's token so the link can be exported again. Generate one with "
                      "python -c \"from cryptography.fernet import Fernet; "
                      "print(Fernet.generate_key().decode())\"")
    for i, key in enumerate(keys, 1):
        try:
            Fernet(key.encode())
        except (ValueError, TypeError):
            errors.append(f"DEMO_LINK_TOKEN_KEYS entry {i} is not a valid Fernet key.")
    steps = csv_list(s.demo_export_steps)
    bad_steps = [x for x in steps if not STEP_RE.match(x)]
    if not steps or bad_steps:
        errors.append("DEMO_EXPORT_STEPS must list email steps e1…e9 and/or li "
                      f"(got {s.demo_export_steps!r}).")
    if s.demo_utm_enabled:
        for name in ("demo_utm_source", "demo_utm_medium"):
            if not UTM_VALUE_RE.match(getattr(s, name)):
                errors.append(f"{name.upper()} must be 1–100 letters, digits, '-', '_' or '.'.")
        if s.demo_utm_campaign and not UTM_VALUE_RE.match(s.demo_utm_campaign):
            errors.append("DEMO_UTM_CAMPAIGN must be letters, digits, '-', '_' or '.'.")

    # ── Limits ──
    for name in ("demo_link_ttl_days", "demo_max_call_seconds", "demo_max_calls_per_link_per_day",
                 "demo_max_concurrent_calls_per_link", "demo_max_minutes_per_link",
                 "demo_max_concurrent_calls_global", "demo_rate_limit_per_min",
                 "demo_sessions_per_ip_per_hour", "demo_upload_max_file_mb",
                 "demo_upload_max_files_per_prospect", "demo_notify_retry_minutes",
                 "demo_notify_max_attempts", "demo_sheet_sync_minutes",
                 "demo_transcript_retention_days"):
        if getattr(s, name) <= 0:
            errors.append(f"{name.upper()} must be greater than 0.")
    if s.demo_max_concurrent_calls_per_link > s.demo_max_concurrent_calls_global:
        errors.append("DEMO_MAX_CONCURRENT_CALLS_PER_LINK cannot exceed "
                      "DEMO_MAX_CONCURRENT_CALLS_GLOBAL.")
    if s.demo_max_minutes_per_link * 60 < s.demo_max_call_seconds:
        errors.append("DEMO_MAX_MINUTES_PER_LINK is shorter than one full call "
                      "(DEMO_MAX_CALL_SECONDS); no call could ever finish.")
    if s.demo_daily_budget_usd < 0:
        errors.append("DEMO_DAILY_BUDGET_USD cannot be negative (use 0 for no cap).")
    if not 1 <= s.demo_budget_alert_pct <= 100:
        errors.append("DEMO_BUDGET_ALERT_PCT must be between 1 and 100.")

    for name in ("demo_default_timezone", "demo_budget_timezone"):
        try:
            ZoneInfo(getattr(s, name))
        except (ZoneInfoNotFoundError, ValueError):
            errors.append(f"{name.upper()} is not a known timezone: {getattr(s, name)!r}.")

    # ── Abuse protection ──
    if not local and not s.demo_turnstile_enabled and s.demo_turnstile_required:
        errors.append("DEMO_TURNSTILE_ENABLED must be true outside local development: without the "
                      "bot check a script can start demo calls (and spend the budget) unattended. "
                      "To run without it on purpose, set DEMO_TURNSTILE_REQUIRED=false.")
    if s.demo_turnstile_enabled and not s.demo_turnstile_secret:
        errors.append("DEMO_TURNSTILE_SECRET is required when DEMO_TURNSTILE_ENABLED=true.")
    if not local and s.demo_ip_hash_secret in ("", "change-me"):
        errors.append("DEMO_IP_HASH_SECRET must be set to a real secret outside local development.")
    if not 0 <= s.demo_trusted_proxy_hops <= 5:
        errors.append("DEMO_TRUSTED_PROXY_HOPS must be between 0 and 5 (the number of proxies "
                      "in front of the API: 1 for Railway alone).")
    if s.demo_trusted_proxy_hops and not IP_HEADER_RE.match(s.demo_client_ip_header.strip()):
        errors.append("DEMO_CLIENT_IP_HEADER must be a header name, e.g. x-forwarded-for.")
    if not local and s.demo_trusted_proxy_hops == 0:
        errors.append("DEMO_TRUSTED_PROXY_HOPS=0 outside local development means every caller "
                      "shares the proxy's IP, so one visitor's per-IP limits block everyone.")

    # ── Where leads go ──
    sheet_on = bool(s.demo_leads_gsheet_id.strip())
    email_on = s.demo_email_provider != "none"
    if not sheet_on and not email_on:
        errors.append("Leads would reach nobody: set DEMO_LEADS_GSHEET_ID, or a "
                      "DEMO_EMAIL_PROVIDER other than none.")
    if email_on:
        recipients = csv_list(s.demo_lead_notify_emails)
        if not recipients:
            errors.append("DEMO_LEAD_NOTIFY_EMAILS is required when DEMO_EMAIL_PROVIDER is "
                          f"{s.demo_email_provider}.")
        if not s.demo_lead_from_email:
            errors.append("DEMO_LEAD_FROM_EMAIL is required when lead email is on.")
        bad = _bad_emails(recipients + csv_list(s.demo_lead_from_email))
        if bad:
            errors.append(f"Not valid email addresses: {', '.join(bad)}.")
        if s.demo_email_provider == "smtp" and not s.smtp_host:
            errors.append("SMTP_HOST is required when DEMO_EMAIL_PROVIDER=smtp.")
        if s.demo_email_provider == "smtp" and bool(s.smtp_username) != bool(s.smtp_password):
            errors.append("SMTP_USERNAME and SMTP_PASSWORD must be set together.")
        if s.demo_email_provider in ("sendgrid", "resend") and not s.demo_email_api_key:
            errors.append(f"DEMO_EMAIL_API_KEY is required when DEMO_EMAIL_PROVIDER="
                          f"{s.demo_email_provider}.")
    bad_alerts = _bad_emails(csv_list(s.demo_alert_emails))
    if bad_alerts:
        errors.append(f"DEMO_ALERT_EMAILS has invalid addresses: {', '.join(bad_alerts)}.")
    if csv_list(s.demo_alert_emails) and not email_on:
        errors.append("DEMO_ALERT_EMAILS is set but DEMO_EMAIL_PROVIDER=none, so alerts "
                      "could never be sent.")
    if sheet_on:
        if not s.google_service_account_json.strip():
            errors.append("GOOGLE_SERVICE_ACCOUNT_JSON is required when DEMO_LEADS_GSHEET_ID is set.")
        else:
            from app.demo.leads.sheets import SheetError, service_account_info
            try:
                service_account_info(s.google_service_account_json)
            except SheetError as e:
                errors.append(f"{e}.")
        if not s.demo_sheet_leads_tab.strip():
            errors.append("DEMO_SHEET_LEADS_TAB cannot be empty when the leads sheet is on.")
    if not csv_list(s.demo_lead_statuses):
        errors.append("DEMO_LEAD_STATUSES needs at least one status.")
    return errors


def assert_demo_settings_valid(s: Settings) -> None:
    """Raise at startup with every problem listed, so the deploy fails loudly."""
    errors = validate_demo_settings(s)
    if (s.demo_enabled and s.env != "local" and not s.demo_turnstile_enabled
            and not s.demo_turnstile_required):
        logging.getLogger(__name__).warning(
            "demo bot check is OFF (DEMO_TURNSTILE_REQUIRED=false): demo calls and lead forms "
            "are protected only by the rate, call and budget limits")
    if errors:
        raise RuntimeError("Demo settings are invalid (DEMO_ENABLED=true):\n  - "
                           + "\n  - ".join(errors))
