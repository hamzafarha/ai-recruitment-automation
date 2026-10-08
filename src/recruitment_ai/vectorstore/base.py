"""Abstract base class and models for vector store backends."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class VectorDocument(BaseModel):
    """Document representation stored in vector stores."""

    id: str
    text: str
    vector: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorSearchResult(BaseModel):
    """Result returned from vector similarity search."""

    id: str
    score: float
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorStore(ABC):
    """Abstract vector store interface."""

    @abstractmethod
    def create_collection(
        self, collection_name: str, dimension: int, distance: str = "Cosine"
    ) -> bool:
        """Create vector collection if it does not already exist."""
        pass

    @abstractmethod
    def upsert(self, collection_name: str, documents: list[VectorDocument]) -> bool:
        """Upsert embedded documents into the collection."""
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Perform semantic search with query vector and optional metadata filters."""
        pass

    @abstractmethod
    def delete(self, collection_name: str, document_ids: list[str]) -> bool:
        """Delete documents from the collection by IDs."""
        pass

    @abstractmethod
    def health_check(self) -> tuple[bool, str]:
        """Check vector store connectivity and readiness."""
        pass
