"""Reranking abstractions and implementations."""

from recruitment_ai.config.settings import Settings, get_settings
from recruitment_ai.reranking.reranker import (
    BaseReranker,
    OptionalReranker,
    RerankResult,
)


def get_reranker(settings: Settings | None = None) -> BaseReranker:
    """BGE reranker on this machine, or on the Kaggle AI server (follows EMBEDDING_PROVIDER)."""
    settings = settings or get_settings()
    if settings.embedding_provider == "remote":
        from recruitment_ai.reranking.remote import RemoteReranker

        return RemoteReranker(settings)
    return OptionalReranker(settings)


__all__ = ["BaseReranker", "OptionalReranker", "RerankResult", "get_reranker"]
