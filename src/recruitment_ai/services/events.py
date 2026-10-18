"""Notifications and events (diagram part 7: notifications; part 2: n8n).

Every important event (new candidate, new job, analysis done, recruiter decision) becomes a
notification shown in the app, and is forwarded to the n8n "recruitment-events" workflow when
N8N_EVENTS_ENABLED=true. Forwarding runs in the background and never blocks or breaks the app
(n8n may be stopped). Notifications created by n8n itself are not sent back (no loop).
"""

import logging
import threading
from typing import Any

import httpx
from sqlalchemy.orm import Session

from recruitment_ai.database.models import Notification

logger = logging.getLogger(__name__)
EVENTS_WEBHOOK = "/webhook/recruitment-events"


def _post(url: str, body: dict[str, Any]) -> None:
    try:
        httpx.post(url, json=body, timeout=5)
    except httpx.HTTPError as exc:
        logger.info(f"n8n not reachable for event {body.get('event')}: {exc}")


def notification_dict(n: Notification) -> dict[str, Any]:
    return {"id": n.id, "event": n.event, "title": n.title, "message": n.message, "payload": n.payload or {},
            "source": n.source, "read": n.read, "created_at": n.created_at.isoformat()}


def notify(db: Session, container: Any, event: str, title: str, message: str = "",
           payload: dict[str, Any] | None = None, source: str = "app", forward: bool = True) -> Notification:
    """``forward=False``: shown in the app only (the application pipeline's own workflow in n8n already
    handles the messages of its applications)."""
    notification = Notification(event=event, title=title[:255], message=message, payload=payload or {},
                                source=source)
    db.add(notification)
    db.commit()
    db.refresh(notification)
    settings = container.settings
    if forward and source == "app" and settings.n8n_events_enabled:
        body = notification_dict(notification)
        threading.Thread(target=_post, args=(settings.n8n_url.rstrip("/") + EVENTS_WEBHOOK, body),
                         daemon=True).start()
    return notification
