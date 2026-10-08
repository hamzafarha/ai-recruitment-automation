"""n8n Orchestrator webhook trigger endpoints (Architecture Box 2)."""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter(prefix="/webhooks/n8n", tags=["n8n Orchestrator"])


class CandidateTriggerPayload(BaseModel):
    candidate_id: str
    document_url: str | None = None
    source: str = "n8n_workflow"


class JobTriggerPayload(BaseModel):
    job_id: str
    department: str | None = None
    source: str = "n8n_workflow"


class NotificationPayload(BaseModel):
    recipient_email: str
    subject: str
    message: str
    event_type: str = Field(default="candidate_match_completed")


@router.post("/trigger/cv-ingested")
def trigger_cv_workflow(payload: CandidateTriggerPayload) -> dict[str, Any]:
    """Webhook triggered by n8n or external services upon new CV upload."""
    return {
        "status": "triggered",
        "orchestrator": "n8n",
        "event": "cv_ingested",
        "candidate_id": payload.candidate_id,
        "action": "Triggering Document Processing & Structured JSON Extraction",
    }


@router.post("/trigger/job-created")
def trigger_job_workflow(payload: JobTriggerPayload) -> dict[str, Any]:
    """Webhook triggered by n8n when a new job vacancy is published."""
    return {
        "status": "triggered",
        "orchestrator": "n8n",
        "event": "job_created",
        "job_id": payload.job_id,
        "action": "Triggering Job Analysis & Vector Knowledge Indexing",
    }


@router.post("/dispatch-notification")
def dispatch_notification(payload: NotificationPayload) -> dict[str, Any]:
    """Webhook handler for dispatching recruiter/candidate notification emails."""
    return {
        "status": "dispatched",
        "orchestrator": "n8n",
        "recipient": payload.recipient_email,
        "event_type": payload.event_type,
    }
