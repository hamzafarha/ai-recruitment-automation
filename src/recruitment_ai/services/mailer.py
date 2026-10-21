"""E-mails to the candidates (diagram part 7: notifications).

The application pipeline writes two e-mails per application: the acknowledgement (step
"acknowledge") and the answer after the recruiter's decision (step "notify_candidate"), with the
interview's video meeting and calendar invite when the recruiter scheduled one. No answer is ever
written before a human decided (KB-RR-001). Every e-mail is stored (table ``emails``, Outbox page):
- EMAIL_MODE=outbox (default): stored and shown in the app, not sent;
- EMAIL_MODE=smtp: sent through SMTP (e.g. Gmail with an App password), then stored with its status.
EMAIL_REDIRECT_TO sends every e-mail to one address instead of the candidate's (demos: the evaluation
CVs belong to real people). The reserved example domains (example.com/.org/.net, the synthetic
candidates) are never e-mailed.
"""

import html
import re
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid
from typing import Any

from sqlalchemy.orm import Session

from recruitment_ai.database.models import Email
from recruitment_ai.utils.logging import get_logger

logger = get_logger(__name__)

RESERVED = re.compile(r"@(?:[\w.-]+\.)?example\.(?:com|org|net)$|\.(?:test|example|invalid|localhost)$",
                      re.IGNORECASE)
SMTP_TIMEOUT = 30


def sender(settings: Any) -> str:
    return settings.email_from or settings.smtp_user or "recruiting@localhost"


def config(settings: Any) -> dict[str, Any]:
    """What the app shows about the e-mail and meeting settings (never the password)."""
    from recruitment_ai.services.meetings import google_ready, resolved_provider

    return {"mode": settings.email_mode, "smtp_host": settings.smtp_host, "smtp_port": settings.smtp_port,
            "smtp_user": settings.smtp_user, "smtp_password_set": bool(settings.smtp_password),
            "from": formataddr((settings.email_from_name, sender(settings))),
            "redirect_to": settings.email_redirect_to or None, "acknowledge": settings.email_acknowledge,
            "meeting_provider": settings.meeting_provider, "meeting_provider_used": resolved_provider(settings),
            "google_ready": google_ready(settings), "interview_minutes": settings.interview_minutes,
            "interview_timezone": settings.interview_timezone}


def html_body(text: str, meeting: dict[str, Any] | None = None) -> str:
    """The HTML version of the e-mail: the same text, plus a button to join the interview."""
    paragraphs = "".join(f"<p style='margin:0 0 14px'>{html.escape(p).replace(chr(10), '<br>')}</p>"
                         for p in text.split("\n\n") if p.strip())
    button = ""
    if meeting and meeting.get("url"):
        label = "Join with Google Meet" if meeting.get("provider") == "google" else "Join the video interview"
        button = (f"<p style='margin:22px 0'><a href='{html.escape(meeting['url'])}' style='background:#4f46e5;"
                  f"color:#fff;padding:12px 22px;border-radius:8px;text-decoration:none;font-weight:600'>"
                  f"{label}</a></p>")
    return ("<div style='font-family:Segoe UI,Helvetica,Arial,sans-serif;font-size:15px;line-height:1.55;"
            f"color:#1f2937;max-width:620px'>{paragraphs}{button}</div>")


def deliver(settings: Any, to: str, subject: str, text: str, html_text: str | None,
            attachments: list[tuple[str, bytes, str, dict[str, str] | None]]) -> None:
    """Send one e-mail through SMTP (STARTTLS on 587, SSL on 465). Raises on failure."""
    message = EmailMessage()
    message["From"] = formataddr((settings.email_from_name, sender(settings)))
    message["To"] = to
    message["Subject"] = subject
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid(domain=sender(settings).split("@")[-1] or "localhost")
    message.set_content(text)
    if html_text:
        message.add_alternative(html_text, subtype="html")
    for filename, data, content_type, params in attachments:
        maintype, subtype = content_type.split("/", 1)
        message.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename, params=params or {})
    context = ssl.create_default_context()
    if settings.smtp_port == 465:
        with smtplib.SMTP_SSL(settings.smtp_host, settings.smtp_port, timeout=SMTP_TIMEOUT, context=context) as smtp:
            if settings.smtp_user:
                smtp.login(settings.smtp_user, settings.smtp_password or "")
            smtp.send_message(message)
        return
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=SMTP_TIMEOUT) as smtp:
        smtp.ehlo()
        if settings.smtp_starttls:
            smtp.starttls(context=context)
            smtp.ehlo()
        if settings.smtp_user:
            smtp.login(settings.smtp_user, settings.smtp_password or "")
        smtp.send_message(message)


def _attachments(email: Email) -> list[tuple[str, bytes, str, dict[str, str] | None]]:
    ics = (email.meeting or {}).get("ics")
    return [("invite.ics", ics.encode("utf-8"), "text/calendar", {"method": "REQUEST"})] if ics else []


def _attempt(settings: Any, email: Email) -> None:
    """Send (or keep in the outbox) one stored e-mail, and record what happened."""
    to = email.to_address
    email.error = None
    if not to:
        email.status, email.error = "skipped", "no e-mail address in the CV"
    elif settings.email_mode != "smtp":
        email.status = "outbox"
    elif RESERVED.search(to):
        email.status, email.error = "skipped", "reserved example address (synthetic candidate): never e-mailed"
    elif not settings.smtp_host:
        email.status, email.error = "failed", "SMTP_HOST is empty: set the SMTP settings in .env"
    else:
        try:
            deliver(settings, to, email.subject, email.text, email.html, _attachments(email))
            email.status, email.sent_at = "sent", datetime.utcnow()
        except (smtplib.SMTPException, OSError) as exc:
            logger.warning(f"E-mail {email.id} to {to} not sent: {exc}")
            email.status, email.error = "failed", f"{type(exc).__name__}: {str(exc)[:400]}"
    email.mode = settings.email_mode


def send(db: Session, settings: Any, kind: str, subject: str, text: str, candidate_address: str | None,
         application_id: str | None = None, candidate_id: str | None = None, decision: str | None = None,
         meeting: dict[str, Any] | None = None) -> Email:
    """Write one e-mail, send it (or keep it in the outbox) and store it. Never raises for a delivery
    problem: the returned e-mail says what happened (status, error)."""
    email = Email(application_id=application_id, candidate_id=candidate_id, kind=kind, decision=decision,
                  candidate_address=candidate_address, to_address=settings.email_redirect_to or candidate_address,
                  subject=subject[:255], text=text, html=html_body(text, meeting), meeting=meeting,
                  attachments=[{"filename": "invite.ics", "content_type": "text/calendar"}]
                  if (meeting or {}).get("ics") else None, status="outbox", mode=settings.email_mode)
    db.add(email)
    db.flush()  # an id for the logs
    _attempt(settings, email)
    db.commit()
    db.refresh(email)
    return email


def resend(db: Session, settings: Any, email: Email) -> Email:
    """Send again an e-mail of the outbox (or a failed one) with the current settings."""
    email.to_address = settings.email_redirect_to or email.candidate_address
    _attempt(settings, email)
    db.commit()
    db.refresh(email)
    return email


def email_dict(e: Email, full: bool = False) -> dict[str, Any]:
    meeting = {k: v for k, v in (e.meeting or {}).items() if k != "ics"} or None
    out = {"id": e.id, "application_id": e.application_id, "candidate_id": e.candidate_id, "kind": e.kind,
           "decision": e.decision, "to": e.to_address, "candidate_address": e.candidate_address,
           "redirected": bool(e.to_address and e.candidate_address and e.to_address != e.candidate_address),
           "subject": e.subject, "status": e.status, "mode": e.mode, "error": e.error, "meeting": meeting,
           "attachments": e.attachments or [], "sent_at": e.sent_at.isoformat() if e.sent_at else None,
           "created_at": e.created_at.isoformat()}
    if full:
        out.update(text=e.text, html=e.html, ics=(e.meeting or {}).get("ics"))
    return out
