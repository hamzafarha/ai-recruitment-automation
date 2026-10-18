"""Setup, system status and dashboard."""

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.dependencies import Container, get_container, get_db
from api.errors import service_errors
from recruitment_ai.services import candidates as candidate_service
from recruitment_ai.services import demo, observe
from recruitment_ai.services import jobs as job_service
from recruitment_ai.services.dashboard import summary

router = APIRouter(tags=["Admin & dashboard"])


class SetupIn(BaseModel):
    demo_data: bool = True


def ai_server_status(container: Container) -> dict[str, Any]:
    """Models served by the Kaggle AI server (only when the app is configured to use it)."""
    return observe.ai_server(container.settings)


@router.post("/admin/setup")
def setup(body: SetupIn | None = None, db: Session = Depends(get_db),
          container: Container = Depends(get_container)) -> dict[str, Any]:
    """Index the HR knowledge base (RAG) and, by default, load the demo jobs and synthetic candidates.
    Safe to run again: nothing is added twice, everything is re-indexed."""
    with service_errors():
        return demo.setup(db, container, demo_data=(body or SetupIn()).demo_data)


@router.post("/admin/reindex")
def reindex(db: Session = Depends(get_db), container: Container = Depends(get_container)) -> dict[str, Any]:
    """Rebuild the search entries of every job and candidate (e.g. after a Qdrant reset)."""
    with service_errors():
        return {"jobs": job_service.reindex_all(db, container),
                "candidates": candidate_service.reindex_all(db, container)}


@router.get("/admin/status")
def status(db: Session = Depends(get_db), container: Container = Depends(get_container)) -> dict[str, Any]:
    settings = container.settings
    try:
        collections = container.index.counts()
        qdrant_error = None
    except Exception as exc:
        collections, qdrant_error = {}, str(exc)[:300]
    figures = summary(db)
    return {
        "llm_provider": settings.llm_provider, "embedding_provider": settings.embedding_provider,
        "llm_model": settings.huggingface_model if settings.llm_provider in ("remote", "huggingface")
        else getattr(container.llm, "model", settings.llm_provider),
        "embedding_model": settings.embedding_model, "reranking_model": settings.reranking_model,
        "ai_server": ai_server_status(container),
        "collections": collections, "qdrant_error": qdrant_error,
        "knowledge_ready": collections.get("hr_knowledge", 0) > 0,
        "database": {"candidates": figures["candidates"], "jobs": figures["jobs"], "analyses": figures["analyses"]},
        "n8n": {"url": settings.n8n_url, "events_enabled": settings.n8n_events_enabled},
    }


@router.get("/dashboard")
def dashboard(db: Session = Depends(get_db)) -> dict[str, Any]:
    return summary(db)
