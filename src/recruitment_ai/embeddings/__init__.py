"""Embeddings abstraction and service package."""

from recruitment_ai.embeddings.service import (
    BaseEmbeddingService,
    SentenceTransformerEmbeddingService,
)

__all__ = ["BaseEmbeddingService", "SentenceTransformerEmbeddingService"]
