"""FastAPI application: the backend of the interface, of the n8n workflows (the application pipeline)
and of the AI assistant.

Start it with ``uvicorn api.main:app --host 0.0.0.0 --port 8000`` (``0.0.0.0`` lets n8n, in
Docker, call it at http://host.docker.internal:8000). Interactive documentation: /docs.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import (
    admin,
    agent,
    analyses,
    applications,
    candidates,
    health,
    jobs,
    notifications,
    observe,
    search,
    webhooks,
)
from recruitment_ai.config.settings import get_settings
from recruitment_ai.utils.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(level="DEBUG" if settings.debug else "INFO")
logger = get_logger("api.main")
ROOT = Path(__file__).resolve().parents[1]


def migrate_database() -> None:
    """Create or update the PostgreSQL tables (``alembic upgrade head``) when the API starts."""
    from alembic import command
    from alembic.config import Config

    config = Config()  # no alembic.ini: its logging setup would mute the API's logs
    config.set_main_option("script_location", str(ROOT / "alembic"))
    try:
        command.upgrade(config, "head")
        logger.info("Database schema is up to date")
    except Exception as exc:  # PostgreSQL down: the API still starts, /api/v1/health says what is wrong
        logger.warning(f"Database migration skipped ({type(exc).__name__}: {exc}). "
                       "Start PostgreSQL (docker compose up -d postgres), then restart the API.")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info(f"Starting {settings.app_name} (environment: {settings.app_env}, LLM: {settings.llm_provider}, "
                f"embeddings: {settings.embedding_provider})")
    if settings.app_env != "testing":
        migrate_database()
    yield
    logger.info(f"Shutting down {settings.app_name}")


app = FastAPI(
    title="AI Recruitment Automation API",
    version="1.0.0",
    description="One application pipeline from the CV to the final decision: CV extraction (Qwen2.5 + QLoRA), "
                "a LangGraph screening agent (RAG over the HR knowledge base and the CV: BGE-M3, Qdrant, BGE "
                "reranker; evidence-checked Qwen analysis), human validation, notifications, orchestrated by n8n. "
                "Plus a LangGraph assistant with tools.",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Unexpected errors: logged with their traceback, answered without it."""
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"status": "error", "detail": f"Internal error: {type(exc).__name__}",
                                                  "path": request.url.path})


app.include_router(health.router)  # /health and /api/v1/health

api_v1 = APIRouter(prefix=settings.api_v1_str)
for module in (applications, candidates, jobs, analyses, agent, search, notifications, webhooks, admin,
               observe):
    api_v1.include_router(module.router)
app.include_router(api_v1)
