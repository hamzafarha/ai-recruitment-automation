"""RAG retriever orchestrating vector search and reranking."""

from typing import Any

from recruitment_ai.embeddings.service import (
    BaseEmbeddingService,
    SentenceTransformerEmbeddingService,
)
from recruitment_ai.reranking.reranker import (
    BaseReranker,
    OptionalReranker,
    RerankResult,
)
from recruitment_ai.vectorstore.base import VectorStore
from recruitment_ai.vectorstore.qdrant import QdrantVectorStore


class RAGRetriever:
    """Retrieves relevant chunks via vector search and optional reranking."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_service: BaseEmbeddingService | None = None,
        reranker: BaseReranker | None = None,
    ) -> None:
        self.vector_store = vector_store or QdrantVectorStore()
        self.embedding_service = (
            embedding_service or SentenceTransformerEmbeddingService()
        )
        self.reranker = reranker or OptionalReranker()

    def retrieve(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5,
        rerank_top_n: int = 3,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[RerankResult]:
        """Generate query embedding, execute vector search, and apply reranking."""
        query_vector = self.embedding_service.embed_query(query)
        raw_results = self.vector_store.search(
            collection_name=collection_name,
            query_vector=query_vector,
            top_k=top_k,
            filter_criteria=filter_criteria,
        )
        return self.reranker.rerank(
            query=query, documents=raw_results, top_n=rerank_top_n
        )
