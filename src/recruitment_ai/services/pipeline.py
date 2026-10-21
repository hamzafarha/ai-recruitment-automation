"""The application pipeline: ONE workflow per application, from the CV received to the final decision.

    1 intake -> 3 document -> 3-4 extract -> 8 store -> 7 acknowledge -> 5 screen -> 6 report
      -> 7 notify_recruiters -> 9 validation (waits for the recruiter) -> 9 decision
      -> 7 notify_candidate -> 8 close
    (numbers: the parts of the architecture diagram)

Each step is a function of this module, logged in the application's ``steps`` (status, timings,
result, error). Two ways to run them, with the same code:
- n8n (workflow "Recruitment — Application pipeline"): one node per step, each calling
  POST /applications/{id}/steps/{step}. At "validation" an n8n Wait node pauses the execution; the
  recruiter's decision resumes it (the Wait node's resume URL is given to the notify_recruiters step);
- the API itself (``run``): the steps in a row, in the background (recruiter upload, candidate
  portal's direct route, n8n stopped), then the steps after the decision as soon as a recruiter decides.
A failed step is recorded (and notified); the workflow continues from that step once fixed.
Fine-tuning is not a step: it ran once on Kaggle (QLoRA); the "extract" step uses its adapter.
The CV file is never kept: it is read at the "document" step only (personal data, KB-HR-004).
The candidate's e-mails (acknowledgement, answer after the decision, with the interview's video meeting
and calendar invite) are sent by the API itself (services/mailer.py, services/meetings.py): SMTP, or the
app's outbox when EMAIL_MODE=outbox. What every step produced, in full: services/outputs.py.
"""

import logging
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from sqlalchemy.orm import Session

from recruitment_ai.database.models import Analysis, Application, Review
from recruitment_ai.database.repositories import ApplicationRepository, JobRepository
from recruitment_ai.matching.schemas import MatchReport
from recruitment_ai.services import mailer, meetings
from recruitment_ai.services.errors import ConflictError, NotFoundError
from recruitment_ai.services.events import notify
from recruitment_ai.utils.security import sanitize_filename, validate_file_upload

logger = logging.getLogger(__name__)

ANSWER_DAYS = 10     # every candidate receives an answer within ten working days (KB-RR-001)
CONTACT_DAYS = 5     # a shortlisted candidate is contacted within five working days (KB-RR-001)
MAX_OPEN_JOBS = 20   # a spontaneous application is screened against the newest open jobs
RESUME_ATTEMPTS = 3  # the recruiter's decision resumes the n8n execution (then the API takes over)
RESUME_RETRY_SECONDS = 2.0
SOURCES = ("upload", "portal", "email", "n8n")
POSITIVE_DECISIONS = {"interview": 2, "shortlist": 1}


@dataclass(frozen=True)
class Step:
    name: str
    label: str
    part: str   # part(s) of the architecture diagram
    what: str   # what runs


STEPS = [
    Step("intake", "Intake checks", "1-2", "consent (KB-HR-004), file type and size; the application is registered"),
    Step("document", "Document processing", "3", "PDF text layer (PyMuPDF), Word (python-docx), OCR of scans "
         "(PaddleOCR models); the file is then deleted"),
    Step("extract", "Candidate JSON", "3-4", "Qwen2.5-7B with our QLoRA adapter (fine-tuned on 873 CVs) structures "
         "the CV: identity, positions, education, projects, skills"),
    Step("store", "Store and index", "8, 5", "PostgreSQL (candidates) and Qdrant (CV snippets as BGE-M3 vectors)"),
    Step("acknowledge", "Acknowledge the candidate", "7", "the candidate is e-mailed that the application was "
         "received (SMTP, or the app's outbox)"),
    Step("screen", "Screening agent", "5", "LangGraph agent: understands the task, chooses its tools, retrieves and "
         "reranks CV evidence and HR rules, iterates, then Qwen analyses and code checks every claim"),
    Step("report", "Evidence-backed report", "6", "the analysis is stored with each requirement's CV evidence "
         "(Markdown and PDF exports)"),
    Step("notify_recruiters", "Notify the recruiters", "7", "review needed (priority for grades 2-3)"),
    Step("validation", "Human validation", "9", "the workflow waits until a recruiter decides (KB-RR-001)"),
    Step("decision", "Final decision", "9", "interview, shortlist or rejection; on hold keeps waiting"),
    Step("notify_candidate", "Notify the candidate", "7", "the answer is e-mailed; an interview gets a video "
         "meeting (Google Meet or Jitsi) and a calendar invite"),
    Step("close", "Close the application", "8", "final status and timings stored"),
]
STEP_NAMES = [s.name for s in STEPS]
BY_NAME = {s.name: s for s in STEPS}


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


# --------------------------------------------------------------------------- views


def _entry(app: Application, name: str) -> dict[str, Any] | None:
    return next((s for s in app.steps or [] if s.get("step") == name), None)


def status_of(app: Application, name: str) -> str:
    return (_entry(app, name) or {}).get("status", "pending")


def next_step(app: Application) -> str | None:
    """The first step not done yet (the failed one, or "validation" while the recruiter decides)."""
    return next((name for name in STEP_NAMES if status_of(app, name) != "done"), None)


def _best(analyses: list[Analysis]) -> Analysis | None:
    return max(analyses, key=lambda a: (a.final_grade if a.final_grade is not None else a.grade, a.score),
               default=None)


def application_dict(app: Application, full: bool = False) -> dict[str, Any]:
    analyses = list(app.analyses or [])
    best = _best(analyses)
    intake = (_entry(app, "intake") or {}).get("result") or {}
    out = {
        "id": app.id, "candidate_id": app.candidate_id,
        "candidate_name": app.candidate.full_name if app.candidate else None,
        "job_id": app.job_id, "job_title": app.job.title if app.job else None,
        "applied_for": intake.get("applied_for"), "source": app.source, "filename": app.filename,
        "orchestrator": app.orchestrator, "status": app.status, "stage": app.stage,
        "stage_label": BY_NAME[app.stage].label if app.stage in BY_NAME else app.stage,
        "decision": app.decision, "error": app.error,
        "steps_done": sum(status_of(app, name) == "done" for name in STEP_NAMES), "steps_total": len(STEPS),
        "grade": (best.final_grade if best.final_grade is not None else best.grade) if best else None,
        "score": best.score if best else None,
        "analyses": [{"id": a.id, "job_id": a.job_id, "job_title": a.job.title if a.job else None,
                      "grade": a.grade, "final_grade": a.final_grade, "score": a.score, "method": a.method,
                      "status": a.status} for a in analyses],
        "created_at": app.created_at.isoformat(), "updated_at": app.updated_at.isoformat(),
    }
    if full:
        out["steps"] = [{"step": s.name, "label": s.label, "part": s.part, "what": s.what, "status": "pending",
                         **(_entry(app, s.name) or {})} for s in STEPS]
    return out


def steps_catalogue() -> list[dict[str, str]]:
    return [{"step": s.name, "label": s.label, "part": s.part, "what": s.what} for s in STEPS]


def get(db: Session, application_id: str) -> Application:
    app = ApplicationRepository(db).get(application_id)
    if app is None:
        raise NotFoundError(f"Application {application_id} not found")
    return app


# --------------------------------------------------------------------------- step log


def _save_entry(app: Application, name: str, **fields: Any) -> None:
    steps = [dict(s) for s in app.steps or []]
    for s in steps:
        if s.get("step") == name:
            s.update(fields)
            break
    else:
        steps.append({"step": name, "label": BY_NAME[name].label, **fields})
    app.steps = steps  # a new list: SQLAlchemy sees the change


def _set_work(app: Application, **values: Any) -> None:
    app.work = {**(app.work or {}), **values}


def _drop_work(app: Application, *keys: str) -> None:
    app.work = {k: v for k, v in (app.work or {}).items() if k not in keys} or None


def _who(app: Application) -> str:
    return (app.candidate.full_name if app.candidate else None) or app.filename or "application"


def _fail(db: Session, container: Any, app: Application, name: str, exc: Exception, seconds: float) -> None:
    error = f"{type(exc).__name__}: {str(exc)[:500]}"
    _save_entry(app, name, status="failed", seconds=round(seconds, 2), finished_at=_now(), error=error)
    app.status, app.stage, app.error = "failed", name, error
    db.commit()
    notify(db, container, "application.failed", f"Application failed at «{BY_NAME[name].label}»: {_who(app)}",
           f"{error}\nFix the cause, then continue the workflow (Pipeline page).",
           {"application_id": app.id, "step": name, "error": error}, forward=False)


# --------------------------------------------------------------------------- 1. intake


def submit(db: Session, container: Any, filename: str, content: bytes, source: str = "upload",
           consent: bool = False, job_id: str | None = None, orchestrator: str = "api") -> Application:
    """Intake checks (consent, file type and size, the job) and registration of the application.
    The file itself is not stored: the caller passes it to the document step."""
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}: use one of {', '.join(SOURCES)}")
    if source == "portal" and not consent:
        raise ValueError("The candidate must accept the data protection policy (KB-HR-004)")
    settings = container.settings
    name = sanitize_filename(filename)
    ok, reason = validate_file_upload(filename=name, file_size_bytes=len(content),
                                      allowed_extensions=settings.allowed_extensions,
                                      max_size_mb=settings.max_upload_size_mb)
    if not ok:
        raise ValueError(reason)
    job = None
    if job_id:
        job = JobRepository(db).get(job_id)
        if job is None:
            raise ValueError(f"Unknown job {job_id}: choose an open job, or none (spontaneous application)")
        if job.status != "open":
            raise ValueError(f"The job {job.title!r} is closed: choose an open job")
    now = _now()
    app = Application(source=source, filename=name, job_id=job.id if job else None, orchestrator=orchestrator,
                      status="running", stage="intake")
    app.steps = [{"step": "intake", "label": BY_NAME["intake"].label, "status": "done", "started_at": now,
                  "finished_at": now, "seconds": 0.0,
                  "result": {"filename": name, "size_kb": round(len(content) / 1024, 1), "source": source,
                             "consent": consent, "orchestrator": orchestrator,
                             "applied_for": job.title if job else "spontaneous application (the screening agent "
                                                                  "looks for the open jobs that fit)"}}]
    db.add(app)
    db.commit()
    db.refresh(app)
    return app


# --------------------------------------------------------------------------- 3. document processing


def _document(db: Session, container: Any, app: Application, content: bytes | None = None, **_: Any) -> dict[str, Any]:
    if content is None:
        raise ValueError("The CV file is needed for this step and it is never kept after reading (personal data, "
                         "KB-HR-004): submit the application again")
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as folder:  # deleted with the file at the end of the block
        path = Path(folder) / Path(app.filename or "cv").name
        path.write_bytes(content)
        parsed = container.extractor.parser.parse(path)
    seconds = round(time.perf_counter() - start, 2)
    text = (parsed.raw_text or "").strip()
    if not text:
        raise ValueError("No text could be read from this file (an empty page, or a scan the OCR could not read): "
                         "send the CV as a PDF or Word file")
    pages = [{"page": p.page_number, "method": p.layout_info.get("method", parsed.file_type),
              "characters": len(p.text), "ocr_confidence": p.layout_info.get("ocr_confidence")} for p in parsed.pages]
    _set_work(app, cv_text=text, document={"input_type": parsed.metadata.get("input_type"),
                                           "parser": parsed.metadata.get("engine"), "parse_seconds": seconds})
    return {"file_type": parsed.file_type, "engine": parsed.metadata.get("engine"),
            "input_type": parsed.metadata.get("input_type"), "pages": len(pages),
            "ocr_pages": sum(p["method"] == "ocr" for p in pages), "characters": len(text), "seconds": seconds,
            "page_details": pages[:20], "preview": text[:300]}


# --------------------------------------------------------------------------- 3-4. candidate JSON (fine-tuned Qwen)


def _extract(db: Session, container: Any, app: Application, **_: Any) -> dict[str, Any]:
    work = app.work or {}
    if not work.get("cv_text"):
        raise ValueError("No CV text: the document processing step must run first")
    profile, stats = container.extractor.extract_text(work["cv_text"])
    data = profile.model_dump(mode="json", exclude={"created_at", "id"})
    _set_work(app, profile=data,
              extraction={**{k: v for k, v in stats.items() if k != "raw_answer"}, **(work.get("document") or {})})
    return {"full_name": data.get("full_name"), "location": data.get("location"),
            **{section: len(data.get(section) or []) for section in
               ("experience", "education", "projects", "skills", "certifications", "languages")},
            "model": stats.get("model"), "adapter": stats.get("adapter"), "seconds": stats.get("seconds"),
            "prompt_tokens": stats.get("prompt_tokens"), "output_tokens": stats.get("output_tokens"),
            "input_truncated": stats.get("input_truncated"),
            "retried_after_out_of_memory": stats.get("retried_after_out_of_memory"),
            "contact_fixes": stats.get("contact_fixes")}


# --------------------------------------------------------------------------- 8. store and index


def _store(db: Session, container: Any, app: Application, **_: Any) -> dict[str, Any]:
    from recruitment_ai.services.candidates import save_profile

    work = app.work or {}
    if not work.get("profile"):
        raise ValueError("No structured profile: the extraction step must run first")
    candidate = save_profile(db, container, work["profile"], cv_text=work.get("cv_text"), source=app.source,
                             cv_filename=app.filename, extraction=work.get("extraction"), announce=False)
    app.candidate_id = candidate.id
    _drop_work(app, "cv_text", "profile", "extraction", "document")  # the candidate holds them now
    extraction = candidate.extraction or {}
    return {"candidate_id": candidate.id, "full_name": candidate.full_name, "indexed": extraction.get("indexed", True),
            "index_error": extraction.get("index_error"), "tables": ["candidates"],
            "collection": "candidates (Qdrant, one BGE-M3 vector per position, project, degree, skills group)"}


# --------------------------------------------------------------------------- 7. messages


def _email_state(email: Any) -> str:
    return {"sent": "e-mailed", "outbox": "in the outbox", "failed": "NOT sent", "skipped": "not sent"}.get(
        email.status if email else "", "prepared")


def _email_summary(email: Any) -> dict[str, Any] | None:
    if email is None:
        return None
    return {"id": email.id, "status": email.status, "mode": email.mode, "to": email.to_address,
            "redirected": bool(email.candidate_address and email.to_address != email.candidate_address),
            "error": email.error, "attachments": [a["filename"] for a in email.attachments or []]}


def _first_name(name: str | None) -> str:
    return (name or "").split(" ")[0] or "candidate"


def acknowledgement(name: str | None, job_title: str | None) -> dict[str, str]:
    """What the candidate is told when the application is received."""
    position = f" for the {job_title} position" if job_title else ""
    return {"message": f"Thank you{' ' + name if name else ''}: your application was received. A recruiter will "
                       f"answer within {ANSWER_DAYS} working days.",
            "subject": "We received your application",
            "text": f"Dear {_first_name(name)},\n\nThank you for your application{position}: we received your CV. "
                    f"A recruiter will answer you within ten working days. Your data is used only for this "
                    "application and kept at most 24 months (candidate data policy KB-HR-004).\n\n"
                    "The recruiting team"}


def _meeting_lines(meeting: dict[str, Any] | None) -> str:
    """How to join the interview, for the e-mail."""
    if not meeting:
        return ""
    start = datetime.fromisoformat(meeting["start"])
    when = f"{start:%A %d %B %Y at %H:%M} ({meeting.get('timezone')}), {meeting.get('minutes')} minutes"
    lines = [f"When: {when}"]
    if meeting.get("url"):
        lines.append(f"Where: video meeting — {meeting['url']}")
    if meeting.get("invitation_by") == "google":
        lines.append("Google Calendar also sent you an invitation: accept it to add the interview to your agenda.")
    elif meeting.get("url"):
        lines.append("The calendar invitation (invite.ics) is attached: open it to add the interview to your agenda.")
    return "\n\n" + "\n".join(lines)


def answer(decision: str, name: str | None, job_title: str | None, interview_at: str | None = None,
           meeting: dict[str, Any] | None = None) -> dict[str, str]:
    """The candidate's answer after the recruiter's decision (never sent before a human decided)."""
    job = job_title or "the"
    dear = f"Dear {_first_name(name)},\n\n"
    if decision == "interview":
        return {"title": f"Interview invitation prepared for {name or 'the candidate'}",
                "subject": f"Interview invitation — {job_title or 'your application'}",
                "text": f"{dear}Following your application for the {job} position, we would be glad to meet you "
                        f"for an interview" + (f" on {interview_at}" if interview_at else "") + ". The interview "
                        "follows our structured format (KB-RR-004). Please confirm your availability."
                        + _meeting_lines(meeting) + "\n\nThe recruiting team"}
    if decision == "shortlist":
        return {"title": f"Shortlist message prepared for {name or 'the candidate'}",
                "subject": f"Your application — {job_title or 'next step'}",
                "text": f"{dear}Good news: after reviewing your application for the {job} position, our recruiters "
                        f"selected it for the next step. We will contact you within {CONTACT_DAYS} working days to "
                        "arrange an interview.\n\nThe recruiting team"}
    return {"title": f"Answer prepared for {name or 'the candidate'}",
            "subject": f"Your application — {job_title or 'answer'}",
            "text": f"{dear}Thank you for your interest in the {job} position. After a careful review by our "
                    "recruiters, we will not continue with your application this time. You may ask for a human "
                    "re-examination of your application, and for access to or deletion of your data, at any time "
                    "(KB-HR-004).\n\nThe recruiting team"}


def _acknowledge(db: Session, container: Any, app: Application, **_: Any) -> dict[str, Any]:
    candidate = app.candidate
    if candidate is None:
        raise ValueError("No candidate: the store step must run first")
    job_title = app.job.title if app.job else None
    message = acknowledgement(candidate.full_name, job_title)
    email = None
    if container.settings.email_acknowledge:  # a delivery problem is recorded on the e-mail, the workflow goes on
        email = mailer.send(db, container.settings, "acknowledgement", message["subject"], message["text"],
                            candidate.email, application_id=app.id, candidate_id=candidate.id)
    notify(db, container, "application.received", f"Acknowledgement {_email_state(email)} for {_who(app)}",
           f"To: {candidate.email or '(no e-mail in the CV)'}\nSubject: {message['subject']}\n\n{message['text']}",
           {"application_id": app.id, "candidate_id": candidate.id, "email": candidate.email,
            "email_id": email.id if email else None}, forward=False)
    return {**message, "to": candidate.email, "application_id": app.id, "candidate_id": candidate.id,
            "full_name": candidate.full_name, "skills_count": len((candidate.profile or {}).get("skills") or []),
            "email": _email_summary(email)}


# --------------------------------------------------------------------------- 5. screening agent


def _screen(db: Session, container: Any, app: Application, **_: Any) -> dict[str, Any]:
    from recruitment_ai.agents.screening import ScreeningAgent
    from recruitment_ai.services.jobs import job_payload
    from recruitment_ai.services.matching import candidate_payload

    if app.candidate is None:
        raise ValueError("No candidate: the store step must run first")
    if app.job_id:
        job = JobRepository(db).get(app.job_id)
        if job is None:
            raise ValueError("The job of this application was deleted")
        jobs = [job_payload(job)]
    else:
        open_jobs = [j for j in JobRepository(db).list(limit=10000) if j.status == "open"][:MAX_OPEN_JOBS]
        if not open_jobs:
            raise ValueError("No open job to screen this spontaneous application against: create (or reopen) a job, "
                             "then continue the workflow")
        jobs = [job_payload(j) for j in open_jobs]
    result = ScreeningAgent.from_container(container).run(candidate_payload(app.candidate), jobs,
                                                          chosen_job_id=app.job_id)
    _set_work(app, reports=[r.model_dump(mode="json") for r in result["reports"]])
    return {"task": "the job applied to" if app.job_id else "find the open jobs that fit",
            "jobs_considered": result["considered"],
            "jobs_analysed": [{"job_id": r.job_id, "title": r.job_title, "grade": r.grade, "grade_label": r.grade_label,
                               "score": r.score, "method": r.method,
                               "planning_rounds": ((r.trace or {}).get("agent") or {}).get("planning_rounds"),
                               "checks_changed": len(((r.trace or {}).get("checks") or {}).get("verification") or [])}
                              for r in result["reports"]],
            "llm_calls": result["llm_calls"], "tool_calls": len(result["tool_calls"]), "tools": result["tool_calls"],
            "timeline": result["timeline"], "warnings": result["warnings"], "passages": result["passages"],
            "passages_from": result["passages_from"], "seconds": result["seconds"], "model": result["model"],
            "graph": result["graph"]}


# --------------------------------------------------------------------------- 6. evidence-backed report


def _report(db: Session, container: Any, app: Application, **_: Any) -> dict[str, Any]:
    from recruitment_ai.reports.markdown import report_markdown

    reports = [MatchReport.model_validate(r) for r in (app.work or {}).get("reports") or []]
    if not reports:
        raise ValueError("No report to store: the screening agent must run first")
    created = []
    for report in reports:
        report_markdown(report)  # the export works (the PDF is rendered when downloaded)
        analysis = Analysis(candidate_id=app.candidate_id, job_id=report.job_id, application_id=app.id,
                            method=report.method, grade=report.grade, score=report.score,
                            must_coverage=report.must_coverage, nice_coverage=report.nice_coverage,
                            action=report.action, report=report.model_dump(mode="json"))
        db.add(analysis)
        created.append((analysis, report))
    _drop_work(app, "reports")
    db.commit()
    prefix = container.settings.api_v1_str
    return {"analyses": [{"id": a.id, "job_id": r.job_id, "job_title": r.job_title, "grade": r.grade,
                          "grade_label": r.grade_label, "score": r.score, "action_label": r.action_label,
                          "method": r.method, "markdown": f"{prefix}/analyses/{a.id}/report.md",
                          "pdf": f"{prefix}/analyses/{a.id}/report.pdf"} for a, r in created]}


# --------------------------------------------------------------------------- 7. notify the recruiters, 9. wait


def _notify_recruiters(db: Session, container: Any, app: Application, resume_url: str | None = None,
                       **_: Any) -> dict[str, Any]:
    analyses = list(app.analyses or [])
    best = _best(analyses)
    if best is None:
        raise ValueError("No analysis to review: the report step must run first")
    report = best.report or {}
    priority = best.grade >= 2
    title = (f"{'Priority review' if priority else 'Review needed'}: {_who(app)} for "
             f"{report.get('job_title') or 'the job'}")
    others = [a for a in analyses if a.id != best.id]
    message = (f"Grade {best.grade} ({report.get('grade_label')}), score {best.score:g}/100. "
               f"{report.get('action_label') or ''} The application's workflow waits for your decision "
               "(Human validation)."
               + (" Also analysed: " + ", ".join(f"{(a.report or {}).get('job_title')} (grade {a.grade})"
                                                 for a in others) + "." if others else ""))
    notify(db, container, "application.review_needed", title, message,
           {"application_id": app.id, "analysis_ids": [a.id for a in analyses], "grade": best.grade,
            "score": best.score, "priority": priority}, forward=False)
    if resume_url:
        app.resume_url, app.orchestrator = resume_url, "n8n"
    _save_entry(app, "validation", status="waiting", started_at=_now())
    app.status, app.stage = "waiting_review", "validation"
    return {"title": title, "priority": priority, "analyses": [a.id for a in analyses],
            "waiting_for": "a recruiter's decision (Human validation page)",
            "resumed_by": "n8n (Wait node)" if resume_url else "the API"}


def application_decision(analyses: list[Analysis]) -> tuple[Analysis, Review] | None:
    """The application's final decision: an interview or a shortlist on any of its analyses decides at once;
    a rejection decides when every analysis is rejected; otherwise (no decision yet, on hold) it waits."""
    latest = [(a, a.reviews[-1]) for a in analyses if a.reviews]
    positive = [(a, r) for a, r in latest if r.decision in POSITIVE_DECISIONS]
    if positive:
        return max(positive, key=lambda ar: (POSITIVE_DECISIONS[ar[1].decision], ar[1].created_at))
    if analyses and len(latest) == len(analyses) and all(r.decision == "reject" for _, r in latest):
        return max(latest, key=lambda ar: ar[1].created_at)
    return None


def _decision(db: Session, container: Any, app: Application, **_: Any) -> dict[str, Any]:
    decided = application_decision(list(app.analyses or []))
    if decided is None:
        return {"decided": False, "waiting_for": [{"analysis_id": a.id, "job_title": (a.report or {}).get("job_title"),
                                                  "status": a.status} for a in app.analyses or []]}
    analysis, review = decided
    app.decision = review.decision
    started = (_entry(app, "validation") or {}).get("started_at")
    waited = (datetime.utcnow() - datetime.fromisoformat(started)).total_seconds() if started else None
    _save_entry(app, "validation", status="done", finished_at=_now(), seconds=round(waited, 1) if waited else None,
                result={"decision": review.decision, "reviewer": review.reviewer, "analysis_id": analysis.id})
    app.status = "running"
    return {"decided": True, "decision": review.decision, "analysis_id": analysis.id,
            "job_title": (analysis.report or {}).get("job_title"), "reviewer": review.reviewer,
            "interview_at": review.interview_at, "adjusted_grade": review.adjusted_grade}


def _notify_candidate(db: Session, container: Any, app: Application, **_: Any) -> dict[str, Any]:
    decided = application_decision(list(app.analyses or []))
    if decided is None:
        raise ValueError("No final decision yet: a recruiter decides first (Human validation)")
    analysis, review = decided
    settings = container.settings
    candidate = app.candidate
    name = candidate.full_name if candidate else None
    job_title = (analysis.report or {}).get("job_title")
    to = candidate.email if candidate else None
    meeting = None
    if review.decision == "interview" and review.interview_at and settings.meeting_provider != "none":
        meeting = (app.work or {}).get("meeting")  # made by an earlier try of this step: never two meetings
        if meeting is None:
            attendee = settings.email_redirect_to or to
            title = f"Interview — {job_title or 'your application'}" + (f" — {name}" if name else "")
            meeting = meetings.create_meeting(settings, title, f"Structured interview (KB-RR-004) for the "
                                              f"{job_title or ''} position.", review.interview_at, attendee, job_title)
            if meeting["invitation_by"] != "google":  # Google already sends its own invitation
                meeting["ics"] = meetings.ics_invite(meeting, f"Interview for the {job_title or ''} position. "
                                                     f"Join: {meeting.get('url') or 'link to follow'}",
                                                     mailer.sender(settings), settings.email_from_name, attendee, name)
            _set_work(app, meeting=meeting)
            db.commit()
    message = answer(review.decision, name, job_title, review.interview_at, meeting)
    email = mailer.send(db, settings, "decision", message["subject"], message["text"], to, application_id=app.id,
                        candidate_id=candidate.id if candidate else None, decision=review.decision, meeting=meeting)
    notify(db, container, "application.answer", f"{message['title']} ({_email_state(email)})",
           f"To: {to or '(no e-mail in the CV)'}\nSubject: {message['subject']}\n\n{message['text']}",
           {"application_id": app.id, "decision": review.decision, "email": to, "email_id": email.id,
            "meeting_url": (meeting or {}).get("url")}, forward=False)
    if email.status == "failed":  # the candidate must get an answer (KB-RR-001): the step fails, visibly
        raise RuntimeError(f"The answer was not e-mailed to the candidate ({email.error}). Fix the SMTP settings "
                           "in .env (or set EMAIL_MODE=outbox), restart the API, then continue the workflow")
    return {"decision": review.decision, "to": to, "subject": message["subject"], "text": message["text"],
            "email": _email_summary(email),
            "meeting": {k: v for k, v in meeting.items() if k != "ics"} if meeting else None}


def _close(db: Session, container: Any, app: Application, **_: Any) -> dict[str, Any]:
    app.status, app.work = "closed", None
    return {"decision": app.decision, "total_seconds": round((datetime.utcnow() - app.created_at).total_seconds())}


STEP_FUNCTIONS = {"document": _document, "extract": _extract, "store": _store, "acknowledge": _acknowledge,
                  "screen": _screen, "report": _report, "notify_recruiters": _notify_recruiters,
                  "decision": _decision, "notify_candidate": _notify_candidate, "close": _close}


# --------------------------------------------------------------------------- running the steps


def run_step(db: Session, container: Any, app: Application, name: str, content: bytes | None = None,
             resume_url: str | None = None) -> dict[str, Any]:
    """Run one step (n8n calls them one by one). A step already done answers its stored result
    (a retried n8n node never runs it twice); a failed step is recorded, notified, and raised."""
    if name not in BY_NAME:
        raise NotFoundError(f"Unknown step {name!r}: use one of {', '.join(STEP_FUNCTIONS)}")
    if name not in STEP_FUNCTIONS:
        raise ConflictError(f"«{BY_NAME[name].label}» is not run by a call: "
                            + ("POST /applications registers the application" if name == "intake"
                               else "the workflow waits for a recruiter's decision (Human validation)"))
    entry = _entry(app, name) or {}
    if entry.get("status") == "done":
        return {**(entry.get("result") or {}), "already_done": True}
    expected = next_step(app)
    if name != expected and not (name == "decision" and expected == "validation"):
        raise ConflictError(f"«{BY_NAME[name].label}» cannot run yet: the next step of this application is "
                            f"«{BY_NAME[expected].label if expected else 'none (closed)'}»")
    if name == "decision" and status_of(app, "validation") != "waiting":
        raise ConflictError("The final decision comes after the recruiters were notified")
    waiting = name == "decision"
    _save_entry(app, name, status="running", started_at=_now(), error=None)
    app.stage, app.error = ("validation", None) if waiting else (name, None)
    if not waiting:
        app.status = "running"
    db.commit()
    start = time.perf_counter()
    try:
        result = STEP_FUNCTIONS[name](db, container, app, content=content, resume_url=resume_url)
    except Exception as exc:
        db.rollback()
        _fail(db, container, app, name, exc, time.perf_counter() - start)
        raise
    seconds = round(time.perf_counter() - start, 2)
    if name == "decision" and not result.get("decided"):
        _save_entry(app, name, status="pending", seconds=seconds, result=result)
        app.status = "waiting_review"
    else:
        _save_entry(app, name, status="done", seconds=seconds, finished_at=_now(), result=result)
    db.commit()
    return result


def run(db: Session, container: Any, application_id: str, content: bytes | None = None) -> Application:
    """The next steps in a row, by the API: until the recruiter's decision is awaited, or the end.
    A failed step stops the run (it is recorded on the application)."""
    app = get(db, application_id)
    while (name := next_step(app)) is not None:
        if name == "validation":
            if status_of(app, "validation") != "waiting" or application_decision(list(app.analyses or [])) is None:
                break  # waits for a recruiter's decision
            name = "decision"
        try:
            result = run_step(db, container, app, name, content=content if name == "document" else None)
        except Exception as exc:
            logger.warning(f"Application {app.id}: step {name} failed ({type(exc).__name__}: {exc})")
            break
        if name == "decision" and not result.get("decided"):
            break
    db.refresh(app)
    return app


def run_in_background(session_factory: Any, container: Any, application_id: str, content: bytes) -> None:
    """``run`` with its own database session (FastAPI background task)."""
    db = session_factory()
    try:
        run(db, container, application_id, content=content)
    except Exception:
        logger.exception(f"Application {application_id}: the pipeline stopped")
    finally:
        db.close()


def continue_workflow(db: Session, container: Any, application_id: str) -> Application:
    """Continue an application from its next step with the API (after a fix, or without n8n)."""
    app = get(db, application_id)
    if app.status == "closed":
        raise ConflictError("This application is closed")
    if app.orchestrator != "api":
        app.orchestrator = "api"
        db.commit()
    return run(db, container, app.id)


# --------------------------------------------------------------------------- 9. the recruiter's decision


def resume_address(url: str, n8n_url: str) -> str:
    """n8n builds the resume URL with its public address (WEBHOOK_URL: localhost); an API that reaches n8n
    elsewhere (N8N_URL, e.g. http://n8n:5678 inside Docker) calls that address instead."""
    resume, base = urlsplit(url), urlsplit(n8n_url)
    local = ("localhost", "127.0.0.1")
    if resume.hostname in local and base.hostname and base.hostname not in local:
        return urlunsplit((base.scheme, base.netloc, resume.path, resume.query, resume.fragment))
    return url


def resume_n8n(url: str, body: dict[str, Any]) -> tuple[bool, str | None]:
    """Resume the waiting n8n execution (its Wait node's resume URL)."""
    error = None
    for attempt in range(RESUME_ATTEMPTS):
        try:
            response = httpx.post(url, json=body, timeout=10)
            if response.status_code < 400:
                return True, None
            error = f"HTTP {response.status_code}: {response.text[:200]}"
        except httpx.HTTPError as exc:
            error = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < RESUME_ATTEMPTS:
            time.sleep(RESUME_RETRY_SECONDS)
    return False, error


def on_review(db: Session, container: Any, analysis: Analysis) -> dict[str, Any] | None:
    """Called after every recruiter decision: when it decides the application, its workflow continues
    (n8n resumed, or the API runs the last steps)."""
    app = analysis.application
    if app is None or app.status != "waiting_review":
        return None
    decided = application_decision(list(app.analyses or []))
    if decided is None:
        return {"application_id": app.id, "continued": False,
                "why": "the application waits for another decision (an analysis on hold or not decided yet)"}
    if app.orchestrator == "n8n" and app.resume_url:
        ok, error = resume_n8n(resume_address(app.resume_url, container.settings.n8n_url),
                               {"application_id": app.id, "decision": decided[1].decision})
        if ok:
            return {"application_id": app.id, "continued": True, "by": "n8n"}
        logger.warning(f"Application {app.id}: n8n did not resume ({error}); the API finishes the workflow")
        _save_entry(app, "validation", note=f"n8n did not answer ({error}): the API finished the workflow")
        app.orchestrator = "api"
        db.commit()
    app = run(db, container, app.id)
    return {"application_id": app.id, "continued": True, "by": "api", "status": app.status}
