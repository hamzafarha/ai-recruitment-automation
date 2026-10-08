"""FastAPI main application entrypoint."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import (
    analysis,
    candidates,
    documents,
    extraction,
    feedback,
    health,
    jobs,
    matching,
    reports,
    search,
)
from recruitment_ai.config.settings import get_settings
from recruitment_ai.utils.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(level="DEBUG" if settings.debug else "INFO")
logger = get_logger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown events."""
    logger.info(f"Starting {settings.app_name} (Environment: {settings.app_env})")
    yield
    logger.info(f"Shutting down {settings.app_name}")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Production-grade AI Recruitment Automation API with multi-agent architecture and RAG pipelines.",
    lifespan=lifespan,
)

# Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Safe global error handling to prevent leaking sensitive tracebacks."""
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An internal server error occurred.",
            "path": request.url.path,
        },
    )


# Root health router
app.include_router(health.router)

# Versioned API routes under /api/v1
api_v1_router = health.APIRouter(prefix=settings.api_v1_str)
api_v1_router.include_router(health.router)
api_v1_router.include_router(candidates.router)
api_v1_router.include_router(jobs.router)
api_v1_router.include_router(documents.router)
api_v1_router.include_router(extraction.router)
api_v1_router.include_router(matching.router)
api_v1_router.include_router(search.router)
api_v1_router.include_router(analysis.router)
api_v1_router.include_router(reports.router)
api_v1_router.include_router(feedback.router)

app.include_router(api_v1_router)
