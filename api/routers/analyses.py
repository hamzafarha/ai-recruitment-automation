"""Analyses (diagram parts 5, 6, 9): evidence-backed candidate-job reports, their export and the
recruiter's decision on them."""

import unicodedata
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.dependencies import Container, get_container, get_db
from api.errors import service_errors
from recruitment_ai.database.repositories import AnalysisRepository
from recruitment_ai.reports.markdown import report_markdown
from recruitment_ai.services import matching as matching_service
from recruitment_ai.services import reviews as review_service

router = APIRouter(prefix="/analyses", tags=["Analyses & human validation"])


class AnalysisIn(BaseModel):
    candidate_id: str
    job_id: str
    method: Literal["llm", "semantic"] = Field(
        default="llm", description="llm: full AI analysis with Qwen (about a minute); semantic: fast estimate")


class ReviewIn(BaseModel):
    decision: Literal["shortlist", "interview", "hold", "reject"]
    reviewer: str = Field(default="recruiter", max_length=100)
    adjusted_grade: int | None = Field(default=None, ge=0, le=3)
    interview_at: str | None = Field(default=None, max_length=50)
    notes: str | None = Field(default=None, max_length=4000)


def _get(db: Session, analysis_id: str) -> Any:
    analysis = AnalysisRepository(db).get(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    return analysis


def _file_name(analysis: Any, extension: str) -> str:
    """ASCII only: an HTTP header cannot carry every letter ("Tomás" -> "Tomas")."""
    report = analysis.report or {}
    name = "_".join((report.get("candidate_name") or "candidate").split())
    job = "_".join((report.get("job_title") or "job").split())
    plain = unicodedata.normalize("NFKD", f"report_{name}_{job}")
    safe = "".join(ch for ch in plain if ch.isascii() and (ch.isalnum() or ch in "_-"))
    return f"{safe[:80] or 'report'}.{extension}"


@router.post("", status_code=201)
def create_analysis(body: AnalysisIn, db: Session = Depends(get_db),
                    container: Container = Depends(get_container)) -> dict[str, Any]:
    """Analyse one candidate for one job; the analysis waits for a recruiter's review."""
    with service_errors():
        analysis = matching_service.analyze(db, container, body.candidate_id, body.job_id, body.method)
    return matching_service.analysis_dict(analysis, full=True)


@router.get("")
def list_analyses(job_id: str | None = None, candidate_id: str | None = None, status: str | None = None,
                  limit: int = 500, db: Session = Depends(get_db)) -> dict[str, Any]:
    analyses = AnalysisRepository(db).filter(job_id=job_id, candidate_id=candidate_id, status=status, limit=limit)
    return {"count": len(analyses), "data": [matching_service.analysis_dict(a) for a in analyses]}


@router.get("/{analysis_id}")
def get_analysis(analysis_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    return matching_service.analysis_dict(_get(db, analysis_id), full=True)


@router.delete("/{analysis_id}")
def delete_analysis(analysis_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    if not AnalysisRepository(db).delete(analysis_id):
        raise HTTPException(status_code=404, detail=f"Analysis {analysis_id} not found")
    return {"deleted": analysis_id}


@router.get("/{analysis_id}/report.md")
def report_md(analysis_id: str, db: Session = Depends(get_db)) -> Response:
    analysis = _get(db, analysis_id)
    data = matching_service.analysis_dict(analysis)
    text = report_markdown(matching_service.report_of(analysis), data["last_review"])
    return Response(text, media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{_file_name(analysis, "md")}"'})


@router.get("/{analysis_id}/report.pdf")
def report_pdf(analysis_id: str, db: Session = Depends(get_db)) -> Response:
    from recruitment_ai.reports.pdf import markdown_to_pdf

    analysis = _get(db, analysis_id)
    report = matching_service.report_of(analysis)
    data = matching_service.analysis_dict(analysis)
    pdf = markdown_to_pdf(report_markdown(report, data["last_review"]),
                          title=f"Screening report — {report.candidate_name or 'Candidate'}",
                          footer_text="AI Recruitment Automation — decision support, a recruiter decides")
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{_file_name(analysis, "pdf")}"'})


@router.post("/{analysis_id}/review")
def review_analysis(analysis_id: str, body: ReviewIn, db: Session = Depends(get_db),
                    container: Container = Depends(get_container)) -> dict[str, Any]:
    """Human validation: shortlist, schedule an interview, put on hold or reject (with a reason)."""
    with service_errors():
        review_service.review(db, container, analysis_id, body.decision, body.reviewer, body.adjusted_grade,
                              body.interview_at, body.notes)
    db.expire_all()
    return matching_service.analysis_dict(_get(db, analysis_id), full=True)
