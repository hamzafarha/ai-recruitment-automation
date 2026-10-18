"""Candidates: CV upload -> extraction (DSO1) -> storage (PostgreSQL) -> search index (Qdrant) -> event."""

import logging
import tempfile
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from recruitment_ai.database.models import Candidate
from recruitment_ai.database.repositories import AnalysisRepository, CandidateRepository
from recruitment_ai.services.events import notify

logger = logging.getLogger(__name__)


def candidate_dict(c: Candidate, full: bool = False) -> dict[str, Any]:
    profile = c.profile or {}
    out = {
        "id": c.id, "external_id": c.external_id, "full_name": c.full_name, "email": c.email, "phone": c.phone,
        "location": c.location, "source": c.source, "cv_filename": c.cv_filename,
        "headline": next((f"{e.get('title')} at {e.get('company')}" for e in profile.get("experience") or []
                          if e.get("title")), None),
        "skills_count": len(profile.get("skills") or []),
        "experience_count": len(profile.get("experience") or []),
        "indexed": (c.extraction or {}).get("indexed", True),
        "created_at": c.created_at.isoformat(),
    }
    if full:
        out.update(summary=c.summary, profile=profile, extraction=c.extraction or {},
                   has_cv_text=bool(c.cv_text))
    return out


def save_profile(db: Session, container: Any, profile: dict[str, Any], cv_text: str | None = None,
                 source: str = "upload", cv_filename: str | None = None, extraction: dict[str, Any] | None = None,
                 external_id: str | None = None, announce: bool = True) -> Candidate:
    """Store a structured profile, index it for search and announce it."""
    extraction = {k: v for k, v in (extraction or {}).items() if k != "raw_answer"}
    candidate = Candidate(
        external_id=external_id, full_name=profile.get("full_name"), email=profile.get("email"),
        phone=profile.get("phone"), location=profile.get("location"), summary=profile.get("summary"),
        profile=profile, cv_text=cv_text, cv_filename=cv_filename, source=source, extraction=extraction,
    )
    repo = CandidateRepository(db)
    repo.create(candidate)
    try:
        container.index.index_candidate(candidate.id, candidate.full_name, profile)
        indexed, error = True, None
    except Exception as exc:  # the candidate is kept; search will miss it until re-indexed
        logger.warning(f"Indexing candidate {candidate.id} failed: {exc}")
        indexed, error = False, str(exc)[:300]
    candidate.extraction = {**extraction, "indexed": indexed, **({"index_error": error} if error else {})}
    db.commit()
    db.refresh(candidate)
    if announce:
        name = candidate.full_name or "Unnamed candidate"
        notify(db, container, "candidate.created", f"New candidate: {name}",
               f"{name} was added ({source})" + (f" from {cv_filename}" if cv_filename else "") + ".",
               {"candidate_id": candidate.id, "name": name, "source": source, "email": candidate.email})
    return candidate


def create_from_file(db: Session, container: Any, filename: str, content: bytes,
                     source: str = "upload") -> Candidate:
    """Extract a CV file (PDF, Word, image; OCR for scans) and store the candidate."""
    with tempfile.TemporaryDirectory() as folder:  # CVs are personal data: the file is not kept
        path = Path(folder) / Path(filename).name
        path.write_bytes(content)
        profile, stats, text = container.extractor.extract_file(path)
    data = profile.model_dump(mode="json", exclude={"created_at", "id"})
    return save_profile(db, container, data, cv_text=text, source=source, cv_filename=filename, extraction=stats)


def delete(db: Session, container: Any, candidate_id: str) -> bool:
    try:
        container.index.remove_candidate(candidate_id)
    except Exception as exc:
        logger.warning(f"Removing candidate {candidate_id} from the index failed: {exc}")
    return CandidateRepository(db).delete(candidate_id)


def reindex_all(db: Session, container: Any) -> int:
    count = 0
    for candidate in CandidateRepository(db).list(limit=10000):
        container.index.index_candidate(candidate.id, candidate.full_name, candidate.profile or {})
        candidate.extraction = {**(candidate.extraction or {}), "indexed": True}
        count += 1
    db.commit()
    return count


def analyses_of(db: Session, candidate_id: str) -> list[Any]:
    return AnalysisRepository(db).filter(candidate_id=candidate_id)
