"""Unit tests for retriever and reranking interfaces."""

from recruitment_ai.rag.context_builder import ContextBuilder
from recruitment_ai.reranking.reranker import OptionalReranker
from recruitment_ai.vectorstore.base import VectorSearchResult


def test_optional_reranker_fallback():
    reranker = OptionalReranker()
    mock_results = [
        VectorSearchResult(
            id="1", score=0.75, text="Python expert", metadata={"source": "cv1.pdf"}
        ),
        VectorSearchResult(
            id="2",
            score=0.92,
            text="Deep learning lead",
            metadata={"source": "cv2.pdf"},
        ),
    ]
    reranked = reranker.rerank(query="Deep learning", documents=mock_results, top_n=2)
    assert len(reranked) == 2
    # Verify sorted by relevance score
    assert reranked[0].id == "2"
    assert reranked[0].relevance_score == 0.92


def test_context_builder():
    builder = ContextBuilder()
    reranker = OptionalReranker()
    mock_results = [
        VectorSearchResult(
            id="1",
            score=0.88,
            text="Experience with PyTorch and Transformers.",
            metadata={"source": "cv.pdf", "page": 2, "document_type": "cv"},
        )
    ]
    reranked = reranker.rerank("PyTorch", mock_results, top_n=1)
    context = builder.build_context(reranked)
    assert "[Document 1 | Type: cv | Source: cv.pdf (Page 2)]" in context
    assert "PyTorch and Transformers" in context
