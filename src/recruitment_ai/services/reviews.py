"""Human validation (diagram part 9): the recruiter validates, adjusts or rejects an analysis.

The AI never makes a hiring decision (KB-RR-001): an analysis stays "pending_review" until a
recruiter decides. Every rejection needs a written reason, and a recruiter may adjust the grade.
When the analysis belongs to an application of the pipeline, the decision resumes that application's
workflow (services/pipeline.py: final decision -> message to the candidate -> closed).
"""

import logging
from typing import Any

from sqlalchemy.orm import Session

from recruitment_ai.database.models import Review
from recruitment_ai.database.repositories import AnalysisRepository
from recruitment_ai.services import pipeline
from recruitment_ai.services.errors import NotFoundError
from recruitment_ai.services.events import notify

logger = logging.getLogger(__name__)
DECISIONS = {"shortlist": "shortlisted", "interview": "interview", "hold": "on_hold", "reject": "rejected"}
DECISION_LABELS = {"shortlist": "Shortlisted", "interview": "Interview scheduled", "hold": "On hold",
                   "reject": "Rejected"}


def review(db: Session, container: Any, analysis_id: str, decision: str, reviewer: str = "recruiter",
           adjusted_grade: int | None = None, interview_at: str | None = None, notes: str | None = None) -> Review:
    analysis = AnalysisRepository(db).get(analysis_id)
    if analysis is None:
        raise NotFoundError(f"Analysis {analysis_id} not found")
    if decision not in DECISIONS:
        raise ValueError(f"Unknown decision {decision!r}: use one of {', '.join(DECISIONS)}")
    if decision == "reject" and not (notes or "").strip():
        raise ValueError("A rejection needs a written reason (a recruiter confirms every rejection, KB-RR-002)")
    if adjusted_grade is not None and not 0 <= adjusted_grade <= 3:
        raise ValueError("The grade must be between 0 and 3")
    entry = Review(analysis_id=analysis.id, reviewer=reviewer or "recruiter", decision=decision,
                   adjusted_grade=adjusted_grade, interview_at=interview_at or None, notes=(notes or "").strip() or None)
    db.add(entry)
    analysis.status = DECISIONS[decision]
    if adjusted_grade is not None:
        analysis.final_grade = adjusted_grade
    db.commit()
    db.refresh(entry)
    report = analysis.report or {}
    name, job = report.get("candidate_name") or "Candidate", report.get("job_title") or "the job"
    # an application of the pipeline: its own workflow writes the candidate's message, not the events workflow
    notify(db, container, "review.decided", f"{DECISION_LABELS[decision]}: {name} for {job}",
           f"Decision by {entry.reviewer}" + (f", interview on {interview_at}" if interview_at else "")
           + (f", grade adjusted to {adjusted_grade}" if adjusted_grade is not None else "") + ".",
           {"analysis_id": analysis.id, "candidate_id": analysis.candidate_id, "job_id": analysis.job_id,
            "application_id": analysis.application_id, "candidate_name": name, "job_title": job,
            "decision": decision, "adjusted_grade": adjusted_grade, "interview_at": interview_at,
            "notes": entry.notes, "reviewer": entry.reviewer}, forward=analysis.application_id is None)
    try:
        pipeline.on_review(db, container, analysis)
    except Exception as exc:  # the decision is saved; the application can be continued from the Pipeline page
        db.rollback()
        logger.warning(f"Analysis {analysis_id}: the application's workflow did not continue ({exc})")
    return entry
