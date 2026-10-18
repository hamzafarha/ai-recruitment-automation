"""Endpoints called by the n8n workflows (diagram part 2, n8n orchestrator).

- POST /webhooks/n8n/job: a job advertisement text, structured by Qwen and stored;
- GET  /webhooks/n8n/digest: the daily recruiting digest (pending reviews, service levels).
n8n reaches the API at http://host.docker.internal:8000 (start the API with --host 0.0.0.0).
A CV received by n8n goes into the application pipeline (api/routers/applications.py).
"""

from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.dependencies import Container, get_container, get_db
from api.errors import service_errors
from recruitment_ai.database.repositories import (
    AnalysisRepository,
    ApplicationRepository,
    CandidateRepository,
)
from recruitment_ai.services import jobs as job_service
from recruitment_ai.services.dashboard import summary

router = APIRouter(prefix="/webhooks/n8n", tags=["n8n orchestrator"])

ANSWER_DAYS = 10    # every candidate receives an answer within ten working days (KB-RR-001)
ANALYSIS_DAYS = 1   # a first analysis within one working day of reception


class JobAdIn(BaseModel):
    text: str = Field(min_length=80, max_length=20000)
    title: str | None = None
    company: str | None = None


@router.post("/job", status_code=201)
def receive_job(body: JobAdIn, db: Session = Depends(get_db), container: Container = Depends(get_container)) -> dict[str, Any]:
    with service_errors(value_status=422):
        data = job_service.extract_from_text(container, body.text)
        if body.title:
            data["title"] = body.title
        if body.company:
            data["company"] = body.company
        if not data.get("title"):
            raise ValueError("No job title found in the advertisement: send it with a title")
        job = job_service.create(db, container, data, source="n8n")
    return job_service.job_dict(job)


@router.get("/digest")
def digest(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Daily digest for recruiters: what waits for a decision, and service levels at risk."""
    now = datetime.utcnow()
    figures = summary(db)
    pending = AnalysisRepository(db).filter(status="pending_review", limit=1000)
    late_reviews = [a for a in pending if now - a.created_at > timedelta(days=ANALYSIS_DAYS)]
    analysed = {a.candidate_id for a in AnalysisRepository(db).list(limit=10000)}
    waiting = [c for c in CandidateRepository(db).list(limit=10000) if c.id not in analysed]
    late_answers = [c for c in waiting if now - c.created_at > timedelta(days=ANSWER_DAYS)]
    failed = ApplicationRepository(db).filter(status="failed", limit=1000)
    lines = [f"- {figures['candidates']} candidates, {figures['open_jobs']} open jobs, "
             f"{figures['analyses']} analyses ({figures['shortlisted']} shortlisted or in interview)",
             f"- {len(pending)} analyses wait for a recruiter decision"
             + (f" ({len(late_reviews)} for more than {ANALYSIS_DAYS} day)" if late_reviews else ""),
             f"- {len(waiting)} candidates not analysed yet"
             + (f", {len(late_answers)} received more than {ANSWER_DAYS} days ago: answer them" if late_answers else "")]
    if failed:
        lines.append(f"- {len(failed)} applications stopped at a failed step: continue them (Pipeline page)")
    for a in pending[:5]:
        report = a.report or {}
        lines.append(f"  - review: {report.get('candidate_name') or 'candidate'} for {report.get('job_title') or 'job'} "
                     f"(grade {a.grade}, {a.score:g}/100)")
    return {"title": f"Recruiting digest — {now:%Y-%m-%d}", "message": "\n".join(lines),
            "pending_reviews": len(pending), "late_reviews": len(late_reviews), "not_analysed": len(waiting),
            "late_answers": len(late_answers), "failed_applications": len(failed),
            "figures": {k: v for k, v in figures.items() if not k.startswith("recent_")}}
