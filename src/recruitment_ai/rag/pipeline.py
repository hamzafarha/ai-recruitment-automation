"""End-to-end RAG indexing and query pipeline."""

from pathlib import Path
from typing import Any

from recruitment_ai.config.settings import Settings, get_settings
from recruitment_ai.embeddings.service import (
    BaseEmbeddingService,
    SentenceTransformerEmbeddingService,
)
from recruitment_ai.rag.chunker import BaseChunker, RecursiveChunker
from recruitment_ai.rag.context_builder import ContextBuilder
from recruitment_ai.rag.loader import DocumentLoader
from recruitment_ai.rag.retriever import RAGRetriever
from recruitment_ai.vectorstore.base import VectorDocument, VectorStore
from recruitment_ai.vectorstore.qdrant import QdrantVectorStore


class RAGPipeline:
    """Orchestrates document loading, chunking, embedding, indexing, and retrieval."""

    def __init__(
        self,
        settings: Settings | None = None,
        vector_store: VectorStore | None = None,
        embedding_service: BaseEmbeddingService | None = None,
        chunker: BaseChunker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vector_store = vector_store or QdrantVectorStore(self.settings)
        self.embedding_service = (
            embedding_service or SentenceTransformerEmbeddingService(self.settings)
        )
        self.chunker = chunker or RecursiveChunker()
        self.loader = DocumentLoader()
        self.retriever = RAGRetriever(
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
        )
        self.context_builder = ContextBuilder()

    def index_document(
        self,
        file_path: Path,
        document_type: str,
        document_id: str,
        candidate_id: str | None = None,
        job_id: str | None = None,
    ) -> int:
        """Parse, chunk, embed, and index a document into the vector store."""
        parsed = self.loader.load_file(file_path)

        base_metadata: dict[str, Any] = {
            "document_id": document_id,
            "document_type": document_type,
            "candidate_id": candidate_id,
            "job_id": job_id,
            "source": parsed.filename,
            "page": 1,
        }

        chunks = self.chunker.chunk(parsed.raw_text, base_metadata)
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_service.embed_documents(texts)

        vector_docs: list[VectorDocument] = []
        for chunk, emb in zip(chunks, embeddings, strict=True):
            vector_docs.append(
                VectorDocument(
                    id=chunk.id,
                    text=chunk.text,
                    vector=emb,
                    metadata=chunk.metadata.model_dump(),
                )
            )

        self.vector_store.create_collection(
            collection_name=self.settings.qdrant_collection,
            dimension=self.settings.embedding_dimension,
        )
        self.vector_store.upsert(
            collection_name=self.settings.qdrant_collection,
            documents=vector_docs,
        )

        return len(vector_docs)

    def query_context(
        self,
        query: str,
        top_k: int = 5,
        rerank_top_n: int = 3,
        filter_criteria: dict[str, Any] | None = None,
    ) -> tuple[str, list[Any]]:
        """Retrieve relevant chunks and return formatted prompt context."""
        retrieved = self.retriever.retrieve(
            collection_name=self.settings.qdrant_collection,
            query=query,
            top_k=top_k,
            rerank_top_n=rerank_top_n,
            filter_criteria=filter_criteria,
        )
        context = self.context_builder.build_context(retrieved)
        return context, retrieved
