"""Unit tests for embedding service abstraction."""

from recruitment_ai.embeddings.service import SentenceTransformerEmbeddingService


def test_embedding_service_dimensions():
    service = SentenceTransformerEmbeddingService()
    # Uses deterministic fallback or model without network requirement
    vec = service.embed_text("Machine Learning Engineer")
    assert isinstance(vec, list)
    assert len(vec) == service.settings.embedding_dimension
    assert all(isinstance(x, float) for x in vec)


def test_batch_embeddings():
    service = SentenceTransformerEmbeddingService()
    docs = ["Python developer", "Docker containerization"]
    matrix = service.embed_documents(docs)
    assert len(matrix) == 2
    assert len(matrix[0]) == service.settings.embedding_dimension
