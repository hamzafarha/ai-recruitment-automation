"""Candidate management endpoints."""

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from recruitment_ai.database.connection import get_db
from recruitment_ai.services.candidate_service import CandidateService

router = APIRouter(prefix="/candidates", tags=["Candidates"])


@router.get("")
def list_candidates(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Retrieve list of candidates."""
    service = CandidateService(db)
    candidates = service.list_candidates()
    return {
        "status": "success",
        "count": len(candidates),
        "data": candidates,
    }


@router.get("/{candidate_id}")
def get_candidate(candidate_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Get single candidate by ID."""
    service = CandidateService(db)
    candidate = service.candidate_repo.get(candidate_id) if service.db else None
    if not candidate:
        return {
            "status": "placeholder",
            "message": f"Candidate '{candidate_id}' retrieval endpoint active. Detailed profile will be provided in extraction phase.",
            "data": None,
        }
    return {"status": "success", "data": candidate}
