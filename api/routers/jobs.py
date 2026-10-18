"""Jobs (diagram parts 1, 6): job descriptions and their structured requirements, ranking of candidates."""

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.dependencies import Container, get_container, get_db
from api.errors import service_errors
from recruitment_ai.database.repositories import JobRepository
from recruitment_ai.services import jobs as job_service
from recruitment_ai.services import matching as matching_service

router = APIRouter(prefix="/jobs", tags=["Jobs"])


class RequirementIn(BaseModel):
    id: str | None = None
    requirement_text: str = Field(min_length=2, max_length=400)
    requirement_type: Literal["required", "preferred"] = "required"
    importance: Literal["high", "medium", "low"] = "medium"
    min_years_experience: float | None = Field(default=None, ge=0, le=40)
    hard: bool = False


class JobIn(BaseModel):
    title: str = Field(min_length=2, max_length=255)
    company: str | None = None
    location: str | None = None
    employment_type: str | None = None
    role: str | None = Field(default=None, description="knowledge-base role key, e.g. ml_engineer")
    description: str = ""
    requirements: list[RequirementIn] = Field(min_length=1, max_length=30)


class JobText(BaseModel):
    text: str = Field(min_length=80, max_length=20000, description="The full job advertisement")


class JobStatus(BaseModel):
    status: Literal["open", "closed"]


@router.get("")
def list_jobs(db: Session = Depends(get_db)) -> dict[str, Any]:
    jobs = JobRepository(db).list(limit=1000)
    return {"count": len(jobs), "data": [job_service.job_dict(j, full=False) for j in jobs]}


@router.post("", status_code=201)
def create_job(body: JobIn, db: Session = Depends(get_db),
               container: Container = Depends(get_container)) -> dict[str, Any]:
    with service_errors():
        job = job_service.create(db, container, body.model_dump(exclude={"role"}), role=body.role)
    return job_service.job_dict(job)


@router.post("/extract")
def extract_job(body: JobText, container: Container = Depends(get_container)) -> dict[str, Any]:
    """A job advertisement -> title and structured requirements (Qwen). Not stored: review, then POST /jobs."""
    with service_errors(value_status=422):
        return job_service.extract_from_text(container, body.text)


@router.get("/{job_id}")
def get_job(job_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    job = JobRepository(db).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job_service.job_dict(job)


@router.patch("/{job_id}")
def set_job_status(job_id: str, body: JobStatus, db: Session = Depends(get_db)) -> dict[str, Any]:
    job = JobRepository(db).get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    job.status = body.status
    db.commit()
    return job_service.job_dict(job)


@router.delete("/{job_id}")
def delete_job(job_id: str, db: Session = Depends(get_db),
               container: Container = Depends(get_container)) -> dict[str, Any]:
    if not job_service.delete(db, container, job_id):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"deleted": job_id}


@router.get("/{job_id}/ranking")
def rank_candidates(job_id: str, db: Session = Depends(get_db),
                    container: Container = Depends(get_container)) -> dict[str, Any]:
    """All candidates graded for this job in seconds (BGE-M3 + BGE reranker + rubric, no LLM)."""
    with service_errors():
        rows = matching_service.rank(db, container, job_id)
    return {"job_id": job_id, "count": len(rows), "method": "semantic", "data": rows}
