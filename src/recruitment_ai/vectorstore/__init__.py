"""Vector store abstraction and implementations."""

from recruitment_ai.vectorstore.base import (
    VectorDocument,
    VectorSearchResult,
    VectorStore,
)
from recruitment_ai.vectorstore.qdrant import QdrantVectorStore

__all__ = ["VectorStore", "VectorDocument", "VectorSearchResult", "QdrantVectorStore"]
