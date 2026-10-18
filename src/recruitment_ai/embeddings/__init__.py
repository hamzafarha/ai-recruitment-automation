"""Embeddings abstraction and service package."""

from recruitment_ai.config.settings import Settings, get_settings
from recruitment_ai.embeddings.service import (
    BaseEmbeddingService,
    SentenceTransformerEmbeddingService,
)


def get_embedding_service(settings: Settings | None = None) -> BaseEmbeddingService:
    """BGE-M3 on this machine (EMBEDDING_PROVIDER=local) or on the Kaggle AI server (remote)."""
    settings = settings or get_settings()
    if settings.embedding_provider == "remote":
        from recruitment_ai.embeddings.remote import RemoteEmbeddingService

        return RemoteEmbeddingService(settings)
    return SentenceTransformerEmbeddingService(settings)


__all__ = ["BaseEmbeddingService", "SentenceTransformerEmbeddingService", "get_embedding_service"]
