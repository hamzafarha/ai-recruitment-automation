"""Unit tests for RAG chunking."""

from recruitment_ai.rag.chunker import RecursiveChunker


def test_recursive_chunker_basic():
    chunker = RecursiveChunker(chunk_size=10, chunk_overlap=2)
    sample_text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
    base_meta = {
        "document_id": "doc_01",
        "document_type": "cv",
        "source": "cv.pdf",
    }
    chunks = chunker.chunk(sample_text, base_meta)
    assert len(chunks) >= 2
    assert chunks[0].metadata.document_id == "doc_01"
    assert chunks[0].metadata.chunk_id.startswith("doc_01_chunk_")
    assert "one" in chunks[0].text


def test_recursive_chunker_empty():
    chunker = RecursiveChunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker.chunk(
        "", {"document_id": "doc_01", "document_type": "cv", "source": "cv.pdf"}
    )
    assert chunks == []
