"""Candidates (diagram parts 1, 3, 8): CV upload -> structured profile -> PostgreSQL + Qdrant."""

from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from api.dependencies import Container, get_container, get_db
from api.errors import service_errors
from recruitment_ai.config.settings import get_settings
from recruitment_ai.database.repositories import CandidateRepository
from recruitment_ai.services import candidates as candidate_service
from recruitment_ai.services.matching import analysis_dict
from recruitment_ai.utils.security import sanitize_filename, validate_file_upload

router = APIRouter(prefix="/candidates", tags=["Candidates"])


def read_upload(file: UploadFile) -> tuple[str, bytes]:
    """A validated CV upload (type and size); the file itself is never kept (personal data)."""
    settings = get_settings()
    filename = sanitize_filename(file.filename or "cv")
    content = file.file.read()
    is_valid, reason = validate_file_upload(filename=filename, file_size_bytes=len(content),
                                            allowed_extensions=settings.allowed_extensions,
                                            max_size_mb=settings.max_upload_size_mb)
    if not is_valid:
        raise HTTPException(status_code=400, detail=reason)
    return filename, content


@router.get("")
def list_candidates(q: str | None = None, limit: int = 500, db: Session = Depends(get_db)) -> dict[str, Any]:
    candidates = CandidateRepository(db).list(limit=limit)
    if q:
        needle = q.strip().lower()
        candidates = [c for c in candidates if needle in (c.full_name or "").lower()
                      or needle in (c.location or "").lower()]
    return {"count": len(candidates), "data": [candidate_service.candidate_dict(c) for c in candidates]}


@router.post("", status_code=201)
def upload_candidate(file: UploadFile = File(...), db: Session = Depends(get_db),
                     container: Container = Depends(get_container)) -> dict[str, Any]:
    """Upload a CV (PDF, Word or image; OCR for scans): extraction by Qwen (fine-tuned adapter),
    then the candidate is stored and indexed for search. About a minute on the Kaggle GPU."""
    filename, content = read_upload(file)
    with service_errors(value_status=422):
        candidate = candidate_service.create_from_file(db, container, filename, content)
    return candidate_service.candidate_dict(candidate, full=True)


@router.post("/extract")
def extract_candidate(file: UploadFile = File(...), container: Container = Depends(get_container)) -> dict[str, Any]:
    """Extraction only (nothing stored): the structured profile of a CV, to test DSO1."""
    import tempfile
    from pathlib import Path

    filename, content = read_upload(file)
    with service_errors(value_status=422), tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / filename
        path.write_bytes(content)
        profile, stats, _ = container.extractor.extract_file(path)
    return {"filename": filename, "llm_provider": container.settings.llm_provider,
            "profile": profile.model_dump(mode="json", exclude={"created_at"}),
            "stats": {k: v for k, v in stats.items() if k != "raw_answer"}}


@router.get("/{candidate_id}")
def get_candidate(candidate_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    candidate = CandidateRepository(db).get(candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found")
    data = candidate_service.candidate_dict(candidate, full=True)
    data["analyses"] = [analysis_dict(a) for a in candidate_service.analyses_of(db, candidate.id)]
    return data


@router.get("/{candidate_id}/cv-text")
def get_cv_text(candidate_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    candidate = CandidateRepository(db).get(candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found")
    return {"id": candidate.id, "cv_filename": candidate.cv_filename, "cv_text": candidate.cv_text or ""}


@router.delete("/{candidate_id}")
def delete_candidate(candidate_id: str, db: Session = Depends(get_db),
                     container: Container = Depends(get_container)) -> dict[str, Any]:
    """Delete a candidate, their analyses and their search entries (right to erasure, KB-HR-004)."""
    if not candidate_service.delete(db, container, candidate_id):
        raise HTTPException(status_code=404, detail=f"Candidate {candidate_id} not found")
    return {"deleted": candidate_id}
