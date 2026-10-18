"""Matching (DSO3): fast ranking of all candidates for a job, and stored evidence-backed analyses.

Analyses asked from the Matching page use the fixed analysis (``llm``) or the fast estimate; the
application pipeline's analyses come from its screening agent (services/pipeline.py, method ``agent``).
"""

from typing import Any

from sqlalchemy.orm import Session

from recruitment_ai.database.models import Analysis, Candidate
from recruitment_ai.database.repositories import (
    AnalysisRepository,
    CandidateRepository,
    JobRepository,
)
from recruitment_ai.matching.schemas import MatchReport
from recruitment_ai.services.errors import NotFoundError
from recruitment_ai.services.events import notify
from recruitment_ai.services.jobs import job_payload

__all__ = ["NotFoundError", "analysis_dict", "analyze", "candidate_payload", "rank", "report_of"]


def candidate_payload(c: Candidate) -> dict[str, Any]:
    return {"id": c.id, "name": c.full_name, "profile": c.profile or {}, "cv_text": c.cv_text}


def analysis_dict(a: Analysis, full: bool = False) -> dict[str, Any]:
    report = a.report or {}
    out = {
        "id": a.id, "candidate_id": a.candidate_id, "job_id": a.job_id,
        "candidate_name": report.get("candidate_name") or (a.candidate.full_name if a.candidate else None),
        "job_title": report.get("job_title") or (a.job.title if a.job else None),
        "method": a.method, "grade": a.grade, "grade_label": report.get("grade_label"),
        "final_grade": a.final_grade, "score": a.score, "must_coverage": a.must_coverage, "action": a.action,
        "status": a.status, "summary": report.get("summary"), "created_at": a.created_at.isoformat(),
        "last_review": _review_dict(a.reviews[-1]) if a.reviews else None,
        "application_id": a.application_id,
        # the pipeline application this analysis belongs to: a decision resumes its workflow
        "application": ({"id": a.application.id, "status": a.application.status, "stage": a.application.stage,
                         "decision": a.application.decision, "orchestrator": a.application.orchestrator}
                        if a.application else None),
    }
    if full:
        out["report"] = report
        out["reviews"] = [_review_dict(r) for r in a.reviews]
    return out


def _review_dict(r: Any) -> dict[str, Any]:
    return {"id": r.id, "reviewer": r.reviewer, "decision": r.decision, "adjusted_grade": r.adjusted_grade,
            "interview_at": r.interview_at, "notes": r.notes, "created_at": r.created_at.isoformat()}


def rank(db: Session, container: Any, job_id: str, candidate_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Fast semantic ranking (no LLM) of all (or the given) candidates for a job, best first."""
    job = JobRepository(db).get(job_id)
    if job is None:
        raise NotFoundError(f"Job {job_id} not found")
    candidates = CandidateRepository(db).list(limit=10000)
    if candidate_ids:
        candidates = [c for c in candidates if c.id in set(candidate_ids)]
    if not candidates:
        return []
    reports = container.semantic.match_many(job_payload(job), [candidate_payload(c) for c in candidates])
    latest = {}
    for a in AnalysisRepository(db).filter(job_id=job_id):
        latest.setdefault(a.candidate_id, a)  # newest first
    rows = []
    for report in reports:
        must = [r for r in report.requirements if r.kind == "must"] or report.requirements
        existing = latest.get(report.candidate_id)
        rows.append({
            "candidate_id": report.candidate_id, "name": report.candidate_name, "grade": report.grade,
            "grade_label": report.grade_label, "score": report.score, "must_coverage": report.must_coverage,
            "met": sum(r.status == "met" for r in must), "partial": sum(r.status == "partial" for r in must),
            "missing": sum(r.status == "missing" for r in must), "must_total": len(must),
            "total_years": report.total_years, "gaps": report.gaps[:3],
            "analysis_id": existing.id if existing else None,
            "analysis_status": existing.status if existing else None,
            "analysis_grade": (existing.final_grade if existing and existing.final_grade is not None
                               else existing.grade if existing else None),
        })
    return sorted(rows, key=lambda r: (r["grade"], r["score"]), reverse=True)


def analyze(db: Session, container: Any, candidate_id: str, job_id: str, method: str = "llm") -> Analysis:
    """Evidence-backed analysis of one candidate for one job, stored for recruiter review."""
    candidate = CandidateRepository(db).get(candidate_id)
    job = JobRepository(db).get(job_id)
    if candidate is None or job is None:
        raise NotFoundError("Candidate or job not found")
    payload, job_data = candidate_payload(candidate), job_payload(job)
    report: MatchReport = (container.analyzer.analyze(job_data, payload) if method == "llm"
                           else container.semantic.match(job_data, payload))
    analysis = Analysis(candidate_id=candidate.id, job_id=job.id, method=report.method, grade=report.grade,
                        score=report.score, must_coverage=report.must_coverage, nice_coverage=report.nice_coverage,
                        action=report.action, report=report.model_dump(mode="json"))
    AnalysisRepository(db).create(analysis)
    name = candidate.full_name or "Candidate"
    notify(db, container, "analysis.completed", f"Analysis ready: {name} for {job.title}",
           f"Grade {report.grade} ({report.grade_label}), score {report.score:g}/100. Recruiter review needed.",
           {"analysis_id": analysis.id, "candidate_id": candidate.id, "job_id": job.id, "candidate_name": name,
            "job_title": job.title, "grade": report.grade, "score": report.score, "action": report.action,
            "method": report.method})
    return analysis


def report_of(analysis: Analysis) -> MatchReport:
    return MatchReport.model_validate(analysis.report)
