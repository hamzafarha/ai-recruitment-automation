"""E-mails to the candidates and interview meetings (diagram part 7): the Outbox page.

The application pipeline writes them (services/mailer.py, services/meetings.py); here they are listed,
read, and sent again (an e-mail of the outbox once SMTP is set up, or a failed one).
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.dependencies import Container, get_container, get_db
from recruitment_ai.database.repositories import EmailRepository
from recruitment_ai.services import mailer

router = APIRouter(prefix="/emails", tags=["E-mails & meetings"])


@router.get("/settings")
def email_settings(container: Container = Depends(get_container)) -> dict[str, Any]:
    """How e-mails are sent (SMTP or outbox, redirection) and how interview meetings are created."""
    return mailer.config(container.settings)


@router.get("")
def list_emails(application_id: str | None = None, status: str | None = None, kind: str | None = None,
                limit: int = 200, db: Session = Depends(get_db)) -> dict[str, Any]:
    emails = EmailRepository(db).filter(application_id=application_id, status=status, kind=kind, limit=limit)
    counts: dict[str, int] = {}
    for e in EmailRepository(db).filter(limit=10000):
        counts[e.status] = counts.get(e.status, 0) + 1
    return {"count": len(emails), "by_status": counts, "data": [mailer.email_dict(e) for e in emails]}


@router.get("/{email_id}")
def get_email(email_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    email = EmailRepository(db).get(email_id)
    if email is None:
        raise HTTPException(status_code=404, detail=f"E-mail {email_id} not found")
    return mailer.email_dict(email, full=True)


@router.post("/{email_id}/send")
def send_email(email_id: str, db: Session = Depends(get_db),
               container: Container = Depends(get_container)) -> dict[str, Any]:
    """Send again with the current settings (EMAIL_MODE=smtp to really send)."""
    email = EmailRepository(db).get(email_id)
    if email is None:
        raise HTTPException(status_code=404, detail=f"E-mail {email_id} not found")
    if email.status == "sent":
        raise HTTPException(status_code=409, detail="This e-mail was already sent")
    return mailer.email_dict(mailer.resend(db, container.settings, email), full=True)
