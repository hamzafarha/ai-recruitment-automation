"""Notifications (diagram part 7): events of the app and messages prepared by the n8n workflows."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.dependencies import Container, get_container, get_db
from recruitment_ai.database.models import Notification
from recruitment_ai.database.repositories import NotificationRepository
from recruitment_ai.services.events import notification_dict, notify

router = APIRouter(prefix="/notifications", tags=["Notifications"])


class NotificationIn(BaseModel):
    event: str = Field(min_length=2, max_length=60)
    title: str = Field(min_length=1, max_length=255)
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    source: str = Field(default="n8n", max_length=30)


@router.get("")
def list_notifications(unread_only: bool = False, limit: int = 100, db: Session = Depends(get_db)) -> dict[str, Any]:
    repo = NotificationRepository(db)
    items = repo.list(limit=limit)
    if unread_only:
        items = [n for n in items if not n.read]
    return {"unread": repo.unread_count(), "count": len(items), "data": [notification_dict(n) for n in items]}


@router.post("", status_code=201)
def create_notification(body: NotificationIn, db: Session = Depends(get_db),
                        container: Container = Depends(get_container)) -> dict[str, Any]:
    """Used by n8n to show a message in the app (not sent back to n8n: no loop)."""
    notification = notify(db, container, body.event, body.title, body.message, body.payload, source=body.source)
    return notification_dict(notification)


@router.post("/read-all")
def mark_all_read(db: Session = Depends(get_db)) -> dict[str, Any]:
    updated = db.query(Notification).filter(Notification.read.is_(False)).update({"read": True})
    db.commit()
    return {"marked_read": updated}


@router.post("/{notification_id}/read")
def mark_read(notification_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    notification = NotificationRepository(db).get(notification_id)
    if notification is None:
        raise HTTPException(status_code=404, detail=f"Notification {notification_id} not found")
    notification.read = True
    db.commit()
    return notification_dict(notification)
