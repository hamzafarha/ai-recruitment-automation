"""Recruiter feedback and validation endpoints."""

from typing import Any

from fastapi import APIRouter

from recruitment_ai.extraction.schemas import RecruiterFeedback

router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post("")
def submit_recruiter_feedback(feedback: RecruiterFeedback) -> dict[str, Any]:
    """Submit human validation decision on candidate match."""
    return {
        "status": "success",
        "message": f"Feedback for match '{feedback.match_id}' recorded with decision '{feedback.decision}'.",
        "data": feedback.model_dump(),
    }
