"""Health checks: is every service the app needs reachable?"""

from typing import Any

import httpx
from fastapi import APIRouter

from recruitment_ai.config.settings import get_settings
from recruitment_ai.database.connection import check_db_connection
from recruitment_ai.vectorstore.qdrant import QdrantVectorStore

router = APIRouter(tags=["Health"])


def check_ai_server() -> tuple[bool, str]:
    from recruitment_ai.llm.remote_client import AIServerClient

    try:
        info = AIServerClient(get_settings(), timeout=8).request("GET", "/info", timeout=8)
    except RuntimeError as exc:
        return False, str(exc)
    return True, f"{info.get('model')} (adapter: {info.get('adapter') or 'none'}), embeddings: " \
                 f"{info.get('embedding_model') or 'none'}, reranker: {info.get('reranking_model') or 'none'}"


def check_n8n() -> tuple[bool, str]:
    url = get_settings().n8n_url.rstrip("/")
    try:
        response = httpx.get(f"{url}/healthz", timeout=3)
        return response.status_code == 200, f"n8n answered {response.status_code} at {url}"
    except httpx.HTTPError as exc:
        return False, f"n8n not reachable at {url} ({type(exc).__name__}): docker compose up -d n8n"


@router.get("/health")
def root_health() -> dict[str, Any]:
    """Lightweight readiness check."""
    return {"status": "ok", "app": "AI Recruitment Automation"}


def detailed_health() -> dict[str, Any]:
    settings = get_settings()
    db_ok, db_msg = check_db_connection()
    qdrant_ok, qdrant_msg = QdrantVectorStore(settings).health_check()
    components: dict[str, Any] = {
        "postgresql": {"healthy": db_ok, "details": db_msg},
        "qdrant": {"healthy": qdrant_ok, "details": qdrant_msg},
        "llm_provider": {"configured": settings.llm_provider, "embedding_provider": settings.embedding_provider},
    }
    healthy = db_ok and qdrant_ok
    if "remote" in (settings.llm_provider, settings.embedding_provider):
        ai_ok, ai_msg = check_ai_server()
        components["ai_server"] = {"healthy": ai_ok, "details": ai_msg}
        healthy = healthy and ai_ok
    if settings.n8n_events_enabled:  # optional: the app works without n8n
        n8n_ok, n8n_msg = check_n8n()
        components["n8n"] = {"healthy": n8n_ok, "details": n8n_msg}
    return {"status": "healthy" if healthy else "degraded", "app_name": settings.app_name,
            "environment": settings.app_env, "components": components}


@router.get("/api/v1/health")
def api_v1_health() -> dict[str, Any]:
    """Detailed health check of the backend services."""
    return detailed_health()
