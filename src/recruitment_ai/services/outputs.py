"""What every step of an application produced, in full (the step panels of the Pipeline page).

The step log (``applications.steps``) keeps a summary of each step; the full outputs live where the
pipeline stored them and are gathered here, read-only:
- document: the CV text that was read (PostgreSQL ``candidates.cv_text``);
- extract: the structured profile written by the fine-tuned Qwen (identity, positions, education,
  projects, skills, certifications, languages) and, once compared, the base model's profile;
- store: the candidate's row and its Qdrant points (the CV snippets embedded with BGE-M3);
- acknowledge / notify_candidate: the e-mails (text, delivery status, meeting, calendar invite);
- screen: per job, the screening agent's trace: plans, every tool call with its results, the CV
  evidence and the HR knowledge chunks retrieved (RAG), the prompt, Qwen's raw answer, checks by code;
- report: the evidence-backed reports; notify_recruiters: the notifications;
- validation / decision: the recruiters' reviews.
"""

from typing import Any

from sqlalchemy.orm import Session

from recruitment_ai.database.models import Application, Notification
from recruitment_ai.rag.search_index import CANDIDATES
from recruitment_ai.services.mailer import email_dict

MAX_POINTS = 200


def _points(container: Any, candidate_id: str) -> dict[str, Any]:
    """The candidate's snippets in Qdrant (what the CV search tool searches)."""
    try:
        items, _ = container.store.scroll(CANDIDATES, limit=MAX_POINTS, filter_criteria={"candidate_id": candidate_id})
    except Exception as exc:  # Qdrant down: the rest of the outputs still shows
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}", "points": []}
    points = [{"id": p["id"], "section": p["payload"].get("section"), "label": p["payload"].get("label"),
               "text": p["payload"].get("text", "")} for p in items]
    return {"collection": CANDIDATES, "count": len(points), "points": points}


def _screen_job(analysis: Any) -> dict[str, Any]:
    report = analysis.report or {}
    trace = report.get("trace") or {}
    return {"analysis_id": analysis.id, "job_id": analysis.job_id, "job_title": report.get("job_title"),
            "grade": report.get("grade"), "grade_label": report.get("grade_label"), "score": report.get("score"),
            "method": report.get("method"), "agent": trace.get("agent"), "prompt": trace.get("prompt"),
            "prompt_chars": trace.get("prompt_chars"), "raw_answer": trace.get("raw_answer"),
            "llm_seconds": trace.get("llm_seconds"), "prompt_tokens": trace.get("prompt_tokens"),
            "output_tokens": trace.get("output_tokens"), "checks": trace.get("checks"), "error": trace.get("error"),
            "knowledge_sources": report.get("knowledge_sources") or [],
            "requirements": report.get("requirements") or [], "warnings": report.get("warnings") or []}


def _system_prompts() -> dict[str, str]:
    from recruitment_ai.agents.screening import (
        AGENT_ANALYSIS_SYSTEM_PROMPT,
        MAX_CALLS,
        PLAN_SYSTEM_PROMPT,
        TOOLS,
    )

    return {"plan": PLAN_SYSTEM_PROMPT.format(tools="\n".join(f"- {t}" for t in TOOLS.values()), max_calls=MAX_CALLS),
            "analysis": AGENT_ANALYSIS_SYSTEM_PROMPT}


def outputs(db: Session, container: Any, app: Application) -> dict[str, Any]:
    from recruitment_ai.services.candidates import candidate_dict
    from recruitment_ai.services.events import notification_dict
    from recruitment_ai.services.matching import analysis_dict

    candidate, work = app.candidate, app.work or {}
    profile = (candidate.profile if candidate else None) or work.get("profile")
    extraction = (candidate.extraction if candidate else None) or work.get("extraction") or {}
    cv_text = (candidate.cv_text if candidate else None) or work.get("cv_text")
    analyses = list(app.analyses or [])
    emails = list(app.emails or [])
    notifications = [n for n in db.query(Notification).order_by(Notification.created_at).limit(2000).all()
                     if (n.payload or {}).get("application_id") == app.id]
    reviews = [{"analysis_id": a.id, "job_title": (a.report or {}).get("job_title"), "reviewer": r.reviewer,
                "decision": r.decision, "adjusted_grade": r.adjusted_grade, "interview_at": r.interview_at,
                "notes": r.notes, "created_at": r.created_at.isoformat()} for a in analyses for r in a.reviews]
    return {
        "application_id": app.id,
        "document": {"cv_text": cv_text, "characters": len(cv_text or "")},
        "extract": {"profile": profile,
                    "extraction": {k: v for k, v in extraction.items() if k not in ("raw_answer", "model_comparison")},
                    "model_comparison": extraction.get("model_comparison"),
                    "candidate_id": candidate.id if candidate else None},
        "store": {"candidate": candidate_dict(candidate) if candidate else None,
                  "qdrant": _points(container, candidate.id) if candidate else None},
        "acknowledge": {"emails": [email_dict(e, full=True) for e in emails if e.kind == "acknowledgement"]},
        "screen": {"jobs": [_screen_job(a) for a in analyses], "system_prompts": _system_prompts()},
        "report": {"analyses": [analysis_dict(a, full=True) for a in analyses]},
        "notify_recruiters": {"notifications": [notification_dict(n) for n in notifications
                                                if n.event in ("application.review_needed", "application.reminder")]},
        "validation": {"reviews": reviews},
        "decision": {"reviews": reviews, "decision": app.decision},
        "notify_candidate": {"emails": [email_dict(e, full=True) for e in emails if e.kind == "decision"],
                             "notifications": [notification_dict(n) for n in notifications
                                               if n.event == "application.answer"]},
        "notifications": [notification_dict(n) for n in notifications],
    }
