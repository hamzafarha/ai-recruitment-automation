"""Candidate-job matching endpoints."""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/matching", tags=["Matching"])


class MatchRequest(BaseModel):
    candidate_id: str
    job_id: str


@router.post("")
def calculate_match(request: MatchRequest) -> dict[str, Any]:
    """Calculate match score between candidate and job."""
    return {
        "status": "placeholder",
        "message": "Candidate ↔ Job matching endpoint initialized. Multidimensional scoring pipeline will run in Phase 5.",
        "candidate_id": request.candidate_id,
        "job_id": request.job_id,
        "schema_target": "CandidateMatch (v1.0.0)",
    }
