"""Dashboard figures: real counts from the database (no placeholder numbers)."""

from collections import Counter
from typing import Any

from sqlalchemy.orm import Session

from recruitment_ai.database.models import Analysis, Job
from recruitment_ai.database.repositories import (
    AgentRunRepository,
    AnalysisRepository,
    ApplicationRepository,
    CandidateRepository,
    JobRepository,
    NotificationRepository,
)
from recruitment_ai.services.events import notification_dict
from recruitment_ai.services.matching import analysis_dict


def summary(db: Session) -> dict[str, Any]:
    analyses: list[Analysis] = AnalysisRepository(db).list(limit=10000)
    statuses = Counter(a.status for a in analyses)
    grades = Counter(a.final_grade if a.final_grade is not None else a.grade for a in analyses)
    jobs: list[Job] = JobRepository(db).list(limit=10000)
    applications = Counter(a.status for a in ApplicationRepository(db).list(limit=10000))
    return {
        "candidates": CandidateRepository(db).count(),
        "jobs": len(jobs), "open_jobs": sum(j.status == "open" for j in jobs),
        "analyses": len(analyses),
        "pending_reviews": statuses.get("pending_review", 0),
        "shortlisted": statuses.get("shortlisted", 0) + statuses.get("interview", 0),
        "rejected": statuses.get("rejected", 0),
        "average_score": round(sum(a.score for a in analyses) / len(analyses), 1) if analyses else None,
        "grades": {str(g): grades.get(g, 0) for g in (3, 2, 1, 0)},
        "statuses": dict(statuses),
        "agent_questions": AgentRunRepository(db).count(),
        # the application pipeline: running | waiting_review | closed | failed
        "applications": {status: applications.get(status, 0) for status in ("running", "waiting_review", "closed",
                                                                            "failed")},
        "unread_notifications": NotificationRepository(db).unread_count(),
        "recent_analyses": [analysis_dict(a) for a in analyses[:6]],
        "recent_notifications": [notification_dict(n) for n in NotificationRepository(db).list(limit=8)],
    }
