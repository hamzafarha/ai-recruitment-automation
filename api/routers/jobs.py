"""Job vacancy endpoints."""

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from recruitment_ai.database.connection import get_db
from recruitment_ai.services.candidate_service import CandidateService

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("")
def list_jobs(db: Session = Depends(get_db)) -> dict[str, Any]:
    """List open job vacancies."""
    service = CandidateService(db)
    jobs = service.list_jobs()
    return {
        "status": "success",
        "count": len(jobs),
        "data": jobs,
    }


@router.get("/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Get single job description."""
    return {
        "status": "placeholder",
        "message": f"Job vacancy '{job_id}' retrieval endpoint active. Detailed job schema ready.",
        "data": None,
    }
