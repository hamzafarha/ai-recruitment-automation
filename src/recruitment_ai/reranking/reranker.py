"""Reranker abstraction allowing optional neural reranking (e.g., Cohere) or passthrough."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from recruitment_ai.config.settings import Settings, get_settings
from recruitment_ai.vectorstore.base import VectorSearchResult

logger = logging.getLogger(__name__)


class RerankResult(BaseModel):
    """Result item after reranking step."""

    id: str
    text: str
    relevance_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseReranker(ABC):
    """Abstract interface for document reranking."""

    @abstractmethod
    def rerank(
        self, query: str, documents: list[VectorSearchResult], top_n: int = 3
    ) -> list[RerankResult]:
        """Rerank candidates based on semantic cross-attention or model score."""
        pass


class OptionalReranker(BaseReranker):
    """Optional reranker that uses Cohere if API key is provided, or passes through safely."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def rerank(
        self, query: str, documents: list[VectorSearchResult], top_n: int = 3
    ) -> list[RerankResult]:
        if not documents:
            return []

        # If BGE reranker is configured (Architecture: BAAI/bge-reranker-v2-m3)
        if "bge-reranker" in self.settings.reranking_model.lower():
            try:
                from sentence_transformers import CrossEncoder

                model = CrossEncoder(self.settings.reranking_model)
                pairs = [[query, d.text] for d in documents]
                scores = model.predict(pairs)
                ranked_pairs = sorted(
                    zip(documents, scores, strict=True),
                    key=lambda x: x[1],
                    reverse=True,
                )[:top_n]
                return [
                    RerankResult(
                        id=d.id,
                        text=d.text,
                        relevance_score=float(s),
                        metadata=d.metadata,
                    )
                    for d, s in ranked_pairs
                ]
            except Exception as exc:
                logger.info(
                    f"BGE reranker not loaded in offline mode ({exc}). Using vector similarity ranking."
                )

        # If Cohere API key is present, attempt neural reranking
        if self.settings.cohere_api_key:
            try:
                import cohere

                co = cohere.Client(self.settings.cohere_api_key)
                doc_texts = [d.text for d in documents]
                response = co.rerank(
                    model=self.settings.reranking_model,
                    query=query,
                    documents=doc_texts,
                    top_n=min(top_n, len(documents)),
                )

                reranked: list[RerankResult] = []
                for item in response.results:
                    original_doc = documents[item.index]
                    reranked.append(
                        RerankResult(
                            id=original_doc.id,
                            text=original_doc.text,
                            relevance_score=float(item.relevance_score),
                            metadata=original_doc.metadata,
                        )
                    )
                return reranked
            except Exception as exc:
                logger.warning(
                    f"Cohere reranking failed ({exc}). Falling back to vector similarity ranking."
                )

        # Fallback passthrough: slice top_n directly
        sorted_docs = sorted(documents, key=lambda x: x.score, reverse=True)[:top_n]
        return [
            RerankResult(
                id=d.id,
                text=d.text,
                relevance_score=d.score,
                metadata=d.metadata,
            )
            for d in sorted_docs
        ]
