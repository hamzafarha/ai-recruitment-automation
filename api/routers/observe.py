"""Observe each part of the architecture while it works (the interface's "Under the hood" pages).

Read-only, except /observe/extraction, which runs the extraction model on a CV without storing it.
"""

from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from api.dependencies import Container, get_container, get_db
from api.errors import service_errors
from api.routers.candidates import read_upload
from recruitment_ai.services import observe

router = APIRouter(prefix="/observe", tags=["Observe (under the hood)"])


class PairIn(BaseModel):
    candidate_id: str
    job_id: str


@router.get("/overview")
def overview(db: Session = Depends(get_db), container: Container = Depends(get_container)) -> dict[str, Any]:
    """Live state of the 9 parts of the architecture."""
    return observe.overview(db, container)


@router.get("/tables")
def tables(db: Session = Depends(get_db)) -> dict[str, Any]:
    """PostgreSQL: every table of the application with its number of rows."""
    return observe.tables(db)


@router.get("/tables/{name}")
def table_rows(name: str, limit: int = Query(default=20, ge=1, le=200), offset: int = Query(default=0, ge=0),
               db: Session = Depends(get_db)) -> dict[str, Any]:
    """PostgreSQL: the rows of one table, newest first (long texts cut, JSON columns whole)."""
    with service_errors():
        return observe.table_rows(db, name, limit, offset)


@router.get("/collections")
def collections(container: Container = Depends(get_container)) -> dict[str, Any]:
    """Qdrant: the three collections, their size, vector dimension and distance."""
    with service_errors():
        return {"data": observe.collections(container)}


@router.get("/collections/{name}/points")
def points(name: str, limit: int = Query(default=20, ge=1, le=200), offset: str | None = None,
           container: Container = Depends(get_container)) -> dict[str, Any]:
    with service_errors():
        return observe.points(container, name, limit, offset)


@router.get("/collections/{name}/points/{point_id}")
def point(name: str, point_id: str, container: Container = Depends(get_container)) -> dict[str, Any]:
    """One point with its whole vector."""
    with service_errors():
        return observe.point(container, name, point_id)


@router.get("/collections/{name}/map")
def vector_map(name: str, limit: int = Query(default=400, ge=3, le=2000), color_by: str | None = None,
               container: Container = Depends(get_container)) -> dict[str, Any]:
    """The collection's vectors projected on a plane (PCA): close points are similar texts."""
    with service_errors():
        return observe.vector_map(container, name, limit, color_by)


@router.get("/retrieval")
def retrieval(q: str = Query(min_length=2), collection: str = "hr_knowledge", role: str | None = None,
              top_k: int = Query(default=5, ge=1, le=20),
              container: Container = Depends(get_container)) -> dict[str, Any]:
    """One search step by step: query vector, Qdrant hits (cosine), BGE reranker order."""
    with service_errors():
        return observe.explain_retrieval(container, q, collection, role or None, top_k)


@router.post("/matching")
def matching(body: PairIn, db: Session = Depends(get_db),
             container: Container = Depends(get_container)) -> dict[str, Any]:
    """What feeds an analysis (no LLM call): evidence per requirement, HR rules, the prompt for Qwen."""
    with service_errors():
        return observe.explain_matching(db, container, body.candidate_id, body.job_id)


@router.post("/extraction")
def extraction(file: UploadFile = File(...), compare_base: bool = Form(default=False),
               container: Container = Depends(get_container)) -> dict[str, Any]:
    """A CV through document processing, step by step; with compare_base, also the base model without the
    fine-tuned adapter (two generations: about 2-3 minutes). Nothing is stored."""
    filename, content = read_upload(file)
    with service_errors(value_status=422):
        return observe.explain_extraction(container, filename, content, compare_base)


@router.get("/n8n")
def n8n(container: Container = Depends(get_container)) -> dict[str, Any]:
    """Which n8n workflows are live (their webhooks answer), and their executions with N8N_API_KEY."""
    return observe.n8n_status(container.settings)


@router.get("/ai-server")
def ai_server(container: Container = Depends(get_container)) -> dict[str, Any]:
    """The models on the Kaggle AI server and its activity."""
    return observe.ai_server(container.settings)
