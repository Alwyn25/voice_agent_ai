"""Lead and alert emails to our own inbox (docs/gtm-demo-links.md §10.3).

One small interface, LeadMailer.send(), with a class per provider chosen by
DEMO_EMAIL_PROVIDER: smtp (Google Workspace / Gmail app password, Zoho, SES, Microsoft 365,
any mail server), sendgrid, resend, or none. Switching provider is an .env change.

Everything the prospect typed is HTML-escaped and length-limited before it goes into a
message, and the demo link's token never appears in one.
"""
import html
import logging
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr

import httpx

from app.config import get_settings
from app.demo.settings import csv_list

log = logging.getLogger(__name__)


class MailError(RuntimeError):
    pass


@dataclass
class Mail:
    to: list[str]
    subject: str
    text: str
    html: str
    reply_to: str | None = None


class LeadMailer:
    async def send(self, mail: Mail) -> None:
        raise NotImplementedError


class NoMailer(LeadMailer):
    async def send(self, mail: Mail) -> None:
        return None


def _sender() -> tuple[str, str]:
    s = get_settings()
    return s.demo_lead_from_name, s.demo_lead_from_email


class SmtpMailer(LeadMailer):
    async def send(self, mail: Mail) -> None:
        import aiosmtplib
        s = get_settings()
        msg = EmailMessage()
        msg["From"] = formataddr(_sender())
        msg["To"] = ", ".join(mail.to)
        msg["Subject"] = mail.subject
        if mail.reply_to:
            msg["Reply-To"] = mail.reply_to
        msg.set_content(mail.text)
        msg.add_alternative(mail.html, subtype="html")
        try:
            await aiosmtplib.send(msg, hostname=s.smtp_host, port=s.smtp_port,
                                  username=s.smtp_username or None, password=s.smtp_password or None,
                                  start_tls=s.smtp_starttls and s.smtp_port != 465,
                                  use_tls=s.smtp_port == 465, timeout=20)
        except Exception as e:  # noqa: BLE001
            raise MailError(f"SMTP {s.smtp_host}:{s.smtp_port}: {type(e).__name__}: {str(e)[:200]}") from None


class SendGridMailer(LeadMailer):
    async def send(self, mail: Mail) -> None:
        name, email = _sender()
        body = {"personalizations": [{"to": [{"email": t} for t in mail.to]}],
                "from": {"email": email, "name": name}, "subject": mail.subject,
                "content": [{"type": "text/plain", "value": mail.text},
                            {"type": "text/html", "value": mail.html}]}
        if mail.reply_to:
            body["reply_to"] = {"email": mail.reply_to}
        await _post("https://api.sendgrid.com/v3/mail/send", body)


class ResendMailer(LeadMailer):
    async def send(self, mail: Mail) -> None:
        body = {"from": formataddr(_sender()), "to": mail.to, "subject": mail.subject,
                "text": mail.text, "html": mail.html}
        if mail.reply_to:
            body["reply_to"] = mail.reply_to
        await _post("https://api.resend.com/emails", body)


async def _post(url: str, body: dict) -> None:
    key = get_settings().demo_email_api_key
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(url, json=body, headers={"Authorization": f"Bearer {key}"})
    except httpx.HTTPError as e:
        raise MailError(f"{url}: {type(e).__name__}") from None
    if r.status_code >= 300:
        raise MailError(f"{url}: HTTP {r.status_code} {r.text[:200]}")


def get_mailer() -> LeadMailer:
    return {"smtp": SmtpMailer, "sendgrid": SendGridMailer, "resend": ResendMailer}.get(
        get_settings().demo_email_provider, NoMailer)()


def email_on() -> bool:
    return get_settings().demo_email_provider != "none"


# ── the messages ─────────────────────────────────────────────────────────────────────────

def _clip(value, limit: int = 500) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


def lead_mail(row: dict, *, subject_bits: str) -> Mail:
    """The lead alert. `row` is the lead as it goes into the sheet (LEAD_COLUMNS keys)."""
    s = get_settings()
    lines = [("Name", row["Name"]), ("Phone", row["Phone"]), ("Email", row["Email"]),
             ("Company / role", " · ".join(x for x in (row["Company"], row["Role"]) if x)),
             ("Industry", row["Industry"]), ("Preferred time", row["Preferred time"]),
             ("Message", row["Message"]),
             ("Demo activity", f"{row['Demo calls'] or 0} calls, longest {row['Longest call'] or '—'}, "
                               f"goal completed: {row['Goal completed'] or 'no'}"),
             ("What they asked", row["Questions asked"]), ("Lead ID", row["Lead ID"])]
    if s.dashboard_url:
        lines.append(("Open in admin", s.dashboard_url))
    if s.demo_leads_gsheet_id:
        from app.demo.leads.sheets import sheet_url
        lines.append(("Google Sheet", sheet_url(s.demo_leads_gsheet_id)))
    text = "\n".join(f"{k + ':':<17}{_clip(v, 2000)}" for k, v in lines if v)
    rows = "".join(f"<tr><td style='padding:4px 12px 4px 0;color:#666;vertical-align:top'>{html.escape(k)}</td>"
                   f"<td style='padding:4px 0'>{html.escape(_clip(v, 2000)).replace(chr(10), '<br>')}</td></tr>"
                   for k, v in lines if v)
    return Mail(to=csv_list(s.demo_lead_notify_emails),
                subject=_clip(f"{s.demo_lead_email_subject_prefix} {subject_bits}", 200).replace("\n", " "),
                text=text, html=f"<table style='font-family:sans-serif;font-size:14px'>{rows}</table>",
                reply_to=row["Email"] or None)


def alert_mail(subject: str, body: str) -> Mail | None:
    s = get_settings()
    to = csv_list(s.demo_alert_emails) or csv_list(s.demo_lead_notify_emails)
    if not to:
        return None
    return Mail(to=to, subject=f"{s.demo_lead_email_subject_prefix} ALERT: {subject}", text=body,
                html=f"<pre style='font-family:sans-serif'>{html.escape(body)}</pre>")
