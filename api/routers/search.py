"""Semantic search (diagram parts 5, 6): BGE-M3 in Qdrant, reordered by the BGE reranker."""

from typing import Any

from fastapi import APIRouter, Depends, Query

from api.dependencies import Container, get_container
from api.errors import service_errors

router = APIRouter(prefix="/search", tags=["Search"])


@router.get("/knowledge")
def search_knowledge(q: str = Query(min_length=2), role: str | None = None, top_k: int = Query(default=4, ge=1, le=10),
                     container: Container = Depends(get_container)) -> dict[str, Any]:
    """HR knowledge base: policies, scoring rubric, experience rules, job profiles, skills."""
    with service_errors():
        hits = container.index.search_knowledge(q, role=role or None, top_k=top_k)
    return {"query": q, "data": [{"doc_id": (h.metadata or {}).get("document_id"), "title": (h.metadata or {}).get("title"),
                                  "section": (h.metadata or {}).get("section"),
                                  "category": (h.metadata or {}).get("category"),
                                  "language": (h.metadata or {}).get("language"), "text": h.text,
                                  "score": round(h.relevance_score, 4)} for h in hits]}


@router.get("/candidates")
def search_candidates(q: str = Query(min_length=2), top_k: int = Query(default=5, ge=1, le=20),
                      container: Container = Depends(get_container)) -> dict[str, Any]:
    """Candidates whose CV evidence (positions, projects, skills...) best matches the query."""
    with service_errors():
        return {"query": q, "data": container.index.search_candidates(q, top_k=top_k)}


@router.get("/jobs")
def search_jobs(q: str = Query(min_length=2), top_k: int = Query(default=5, ge=1, le=20),
                container: Container = Depends(get_container)) -> dict[str, Any]:
    with service_errors():
        return {"query": q, "data": container.index.search_jobs(q, top_k=top_k)}
