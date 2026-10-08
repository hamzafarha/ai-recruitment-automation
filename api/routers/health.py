"""Health check endpoints."""

from typing import Any

from fastapi import APIRouter

from recruitment_ai.config.settings import get_settings
from recruitment_ai.database.connection import check_db_connection
from recruitment_ai.vectorstore.qdrant import QdrantVectorStore

router = APIRouter(tags=["Health"])


@router.get("/health")
def root_health() -> dict[str, Any]:
    """Lightweight readiness health check."""
    return {"status": "ok", "app": "AI Recruitment Automation"}


@router.get("/api/v1/health")
def api_v1_health() -> dict[str, Any]:
    """Detailed health check for backend services."""
    settings = get_settings()
    db_ok, db_msg = check_db_connection()
    qdrant = QdrantVectorStore(settings)
    qdrant_ok, qdrant_msg = qdrant.health_check()

    return {
        "status": "healthy" if (db_ok and qdrant_ok) else "degraded",
        "app_name": settings.app_name,
        "environment": settings.app_env,
        "components": {
            "postgresql": {"healthy": db_ok, "details": db_msg},
            "qdrant": {"healthy": qdrant_ok, "details": qdrant_msg},
            "llm_provider": {"configured": settings.llm_provider},
        },
    }
