"""Setup and demo data: index the HR knowledge base, load the evaluation jobs and synthetic candidates.

Demo candidates are the 12 synthetic CVs of the evaluation set (invented people, no personal
data), stored with their reference profiles and the text of their PDF/Word file. Demo jobs are
the 5 evaluation job descriptions (real 2026 postings with structured requirements). Running the
setup again adds nothing twice; it re-indexes everything (useful after a Qdrant reset).
"""

import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from recruitment_ai.database.repositories import CandidateRepository, JobRepository
from recruitment_ai.services import candidates as candidate_service
from recruitment_ai.services import jobs as job_service
from recruitment_ai.services.events import notify

logger = logging.getLogger(__name__)
EVAL_DIR = Path("data/processed/eval_v1")


def _cv_text(files: Path, cv_id: str) -> str | None:
    from recruitment_ai.ingestion.parser import UnifiedDocumentParser

    for suffix in (".pdf", ".docx"):
        path = files / f"{cv_id}{suffix}"
        if path.exists():
            try:
                return UnifiedDocumentParser().parse(path).raw_text
            except Exception as exc:
                logger.warning(f"Could not read {path}: {exc}")
    return None


def setup(db: Session, container: Any, root: Path = Path("."), demo_data: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {"knowledge_chunks": container.index.build_knowledge(root / "data" / "knowledge_base"),
                              "jobs_created": 0, "candidates_created": 0}
    if demo_data:
        eval_dir = root / EVAL_DIR
        for path in sorted((eval_dir / "jobs").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if JobRepository(db).get_by_external_id(data["id"]):
                continue
            job_service.create(db, container, {
                "title": data["title"], "company": (data.get("source") or {}).get("company"),
                "location": data.get("location"), "employment_type": data.get("employment_type"),
                "description": data.get("description") or "", "requirements": data["requirements"],
            }, source="demo", external_id=data["id"], role=data.get("role"), announce=False)
            result["jobs_created"] += 1
        for path in sorted((eval_dir / "synthetic" / "profiles").glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            cv_id = data["meta"]["id"]
            if CandidateRepository(db).get_by_external_id(cv_id):
                continue
            candidate_service.save_profile(
                db, container, data["profile"], cv_text=_cv_text(eval_dir / "synthetic" / "files", cv_id),
                source="demo", cv_filename=f"{cv_id} (synthetic)", external_id=cv_id, announce=False,
                extraction={"origin": "eval_v1 synthetic CV (invented person) with its reference profile",
                            "case": data["meta"].get("case"), "language": data["meta"].get("language")})
            result["candidates_created"] += 1
    result["jobs_indexed"] = job_service.reindex_all(db, container)
    result["candidates_indexed"] = candidate_service.reindex_all(db, container)
    notify(db, container, "setup.completed", "Knowledge base and search index ready",
           f"{result['knowledge_chunks']} HR knowledge chunks, {result['jobs_indexed']} jobs and "
           f"{result['candidates_indexed']} candidates indexed ({result['jobs_created']} demo jobs and "
           f"{result['candidates_created']} demo candidates added).", result)
    return result
