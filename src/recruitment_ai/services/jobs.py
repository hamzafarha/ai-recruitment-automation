"""Jobs: free text -> structured requirements (LLM) -> storage -> search index -> event."""

import logging
from typing import Any

from sqlalchemy.orm import Session

from recruitment_ai.database.models import Job
from recruitment_ai.database.repositories import JobRepository
from recruitment_ai.services.events import notify

logger = logging.getLogger(__name__)


def normalize_requirements(requirements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Requirements with ids (M1.. must-haves, N1.. nice-to-haves) and valid fields."""
    out, counters = [], {"M": 0, "N": 0}
    used = {str(r.get("id")) for r in requirements or [] if r.get("id")}
    for r in requirements or []:
        text = (r.get("requirement_text") or r.get("text") or "").strip()
        if not text:
            continue
        kind = "required" if (r.get("requirement_type") or "required") == "required" else "preferred"
        prefix = "M" if kind == "required" else "N"
        requirement_id = r.get("id")
        while not requirement_id or (requirement_id in used and not r.get("id")):
            counters[prefix] += 1
            requirement_id = f"{prefix}{counters[prefix]}"
        used.add(requirement_id)
        importance = r.get("importance") if r.get("importance") in ("high", "medium", "low") else "medium"
        years = r.get("min_years_experience")
        out.append({"id": requirement_id, "requirement_text": text,
                    "requirement_type": kind, "importance": importance,
                    "min_years_experience": float(years) if years else None, "hard": bool(r.get("hard"))})
    return out


def job_dict(job: Job, full: bool = True) -> dict[str, Any]:
    requirements = job.requirements or []
    out = {"id": job.id, "external_id": job.external_id, "title": job.title, "company": job.company,
           "location": job.location, "employment_type": job.employment_type, "role": job.role,
           "status": job.status, "source": job.source,
           "must_haves": sum(r.get("requirement_type") == "required" for r in requirements),
           "nice_to_haves": sum(r.get("requirement_type") != "required" for r in requirements),
           "created_at": job.created_at.isoformat()}
    if full:
        out.update(description=job.description, requirements=requirements)
    return out


def job_payload(job: Job) -> dict[str, Any]:
    """What the matching works with."""
    return {"id": job.id, "title": job.title, "company": job.company, "role": job.role,
            "description": job.description, "requirements": job.requirements or []}


def create(db: Session, container: Any, data: dict[str, Any], source: str = "manual",
           external_id: str | None = None, role: str | None = None, announce: bool = True) -> Job:
    if not (data.get("title") or "").strip():
        raise ValueError("A job needs a title")
    requirements = normalize_requirements(data.get("requirements") or [])
    if not requirements:
        raise ValueError("A job needs at least one requirement")
    job = Job(external_id=external_id, title=data["title"].strip(), company=data.get("company"),
              location=data.get("location"), employment_type=data.get("employment_type"),
              role=role or data.get("role"), description=data.get("description") or "",
              requirements=requirements, source=source)
    JobRepository(db).create(job)
    try:
        container.index.index_job(job.id, job_payload(job))
    except Exception as exc:
        logger.warning(f"Indexing job {job.id} failed: {exc}")
    if announce:
        notify(db, container, "job.created", f"New job: {job.title}",
               f"{job.title}" + (f" at {job.company}" if job.company else "")
               + f" with {len(requirements)} requirements.",
               {"job_id": job.id, "title": job.title, "source": source})
    return job


def extract_from_text(container: Any, text: str) -> dict[str, Any]:
    """Structured job from a job advertisement (Qwen), not stored: the recruiter reviews it first."""
    from recruitment_ai.extraction.job_extractor import extract_job

    if len(text.strip()) < 80:
        raise ValueError("Paste the full job advertisement (at least a few lines)")
    data = extract_job(text, container.llm)
    data["requirements"] = normalize_requirements(data["requirements"])
    return data


def delete(db: Session, container: Any, job_id: str) -> bool:
    try:
        container.index.remove_job(job_id)
    except Exception as exc:
        logger.warning(f"Removing job {job_id} from the index failed: {exc}")
    return JobRepository(db).delete(job_id)


def reindex_all(db: Session, container: Any) -> int:
    jobs = JobRepository(db).list(limit=10000)
    for job in jobs:
        container.index.index_job(job.id, job_payload(job))
    return len(jobs)
