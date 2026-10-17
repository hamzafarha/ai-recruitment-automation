"""Tests for Step 3 (RAG): section chunking, embedded Qdrant with filters, retrieval metrics.

Qdrant runs embedded in memory and embeddings are fakes: no Docker, GPU or model download.
"""

import json
from pathlib import Path

import pytest

from recruitment_ai.config.settings import Settings
from recruitment_ai.data.knowledge import KnowledgeBaseLoader
from recruitment_ai.evaluation.rag_eval import (
    load_queries,
    query_metrics,
    ranked_documents,
    summarize,
)
from recruitment_ai.rag.knowledge_index import (
    MAX_CHUNK_WORDS,
    KnowledgeIndex,
    job_chunks,
    kb_chunks,
    split_paragraphs,
    split_sections,
)
from recruitment_ai.reranking.reranker import RerankResult
from recruitment_ai.vectorstore.qdrant import QdrantVectorStore, build_filter

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "data" / "knowledge_base"
JOBS = ROOT / "data" / "processed" / "eval_v1" / "jobs"
QUERIES = ROOT / "data" / "processed" / "eval_v1" / "rag" / "queries.json"


class FakeEmbedder:
    """Bag-of-keywords vectors: texts sharing these words end up close."""

    WORDS = ["intern", "salary", "remote", "python", "aws", "grade"]

    def require_model(self) -> None:
        pass

    def _vec(self, text: str) -> list[float]:
        lower = text.lower()
        return [float(w in lower) for w in self.WORDS] + [0.01]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


# --------------------------------------------------------------------------- chunking


def test_split_sections():
    md = "# Title\n\nIntro line.\n\n## First\nBody one.\n\n## Second\nBody two.\n"
    assert split_sections(md) == [("Introduction", "Intro line."), ("First", "Body one."), ("Second", "Body two.")]


def test_split_paragraphs_never_cuts_a_paragraph():
    text = "\n\n".join(["word " * 120, "word " * 120, "word " * 50])
    pieces = split_paragraphs(text, 200)
    assert len(pieces) == 2 and all(len(p.split()) <= 240 for p in pieces)


def test_kb_chunks_are_sections_with_stable_ids_and_metadata():
    chunks = kb_chunks(KnowledgeBaseLoader(base_path=KB).load_all())
    ids = [c.metadata.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)) and "KB-RR-002#0" in ids
    rubric = next(c for c in chunks if c.metadata.chunk_id == "KB-RR-002#0")
    assert rubric.text.startswith("Candidate–Job Fit Scoring Rubric — ")  # title prefix
    assert rubric.metadata.roles == ["all"] and rubric.metadata.language == "en"
    assert max(len(c.text.split()) for c in chunks) <= MAX_CHUNK_WORDS + 30
    assert kb_chunks(KnowledgeBaseLoader(base_path=KB).load_all())[0].id == chunks[0].id  # deterministic


def test_job_chunks_start_with_the_requirements():
    job = json.loads((JOBS / "ml_engineer.json").read_text(encoding="utf-8"))
    chunks = job_chunks(job)
    first = chunks[0]
    assert first.metadata.document_id == "JD-ml_engineer" and first.metadata.section == "Requirements"
    assert "M1 (required, high importance, 1+ years)" in first.text
    assert all(c.metadata.roles == ["ml_engineer"] for c in chunks) and len(chunks) > 1


# --------------------------------------------------------------------------- Qdrant (embedded)


def test_build_filter():
    assert build_filter(None) is None and build_filter({}) is None
    f = build_filter({"roles": ["ml_engineer", "all"], "language": "en", "skip": None})
    assert len(f.must) == 2


@pytest.fixture
def index(tmp_path) -> KnowledgeIndex:
    settings = Settings(QDRANT_LOCATION=":memory:")
    store = QdrantVectorStore(settings)
    idx = KnowledgeIndex(store, FakeEmbedder(), reranker=None, collection="test_kb")
    chunks = kb_chunks(KnowledgeBaseLoader(base_path=KB).load_all())
    chunks += [c for p in sorted(JOBS.glob("*.json")) for c in job_chunks(json.loads(p.read_text(encoding="utf-8")))]
    assert idx.build(chunks, dimension=len(FakeEmbedder.WORDS) + 1) == len(chunks)
    return idx


def test_rebuild_does_not_duplicate(index: KnowledgeIndex):
    count = index.store.client.count("test_kb").count
    chunks = kb_chunks(KnowledgeBaseLoader(base_path=KB).load_all())
    index.build(chunks, dimension=len(FakeEmbedder.WORDS) + 1)
    assert index.store.client.count("test_kb").count == len(chunks) < count  # rebuilt, not appended


def test_search_role_filter_keeps_role_and_shared_documents(index: KnowledgeIndex):
    results = index.search("python requirements", role="ml_engineer", top_k=50, candidates=200, rerank=False)
    roles = {tuple(r.metadata["roles"]) for r in results}
    assert roles <= {("ml_engineer",), ("all",)} and ("ml_engineer",) in roles


def test_search_language_filter(index: KnowledgeIndex):
    results = index.search("salaire", language="fr", top_k=50, candidates=200, rerank=False)
    assert results and {r.metadata["language"] for r in results} == {"fr"}


# --------------------------------------------------------------------------- metrics


def _r(doc_id: str) -> RerankResult:
    return RerankResult(id=doc_id, text="", relevance_score=0.0, metadata={"document_id": doc_id})


def test_ranked_documents_collapse_chunks():
    assert ranked_documents([_r("A"), _r("B"), _r("A"), _r("C")]) == ["A", "B", "C"]


def test_query_metrics():
    m = query_metrics(["X", "A", "Y", "B"], ["A", "B"])
    assert m["rr"] == 0.5 and m["hit@1"] == 0 and m["hit@3"] == 1
    assert m["recall@3"] == 0.5 and m["recall@5"] == 1.0
    assert query_metrics(["X"], ["A"])["rr"] == 0.0


def test_summary_means():
    rows = [query_metrics(["A"], ["A"]), query_metrics(["X", "A"], ["A"])]
    s = summarize(rows)
    assert s["queries"] == 2 and s["mrr"] == 0.75 and s["hit@1"] == 0.5


def test_queries_point_to_existing_documents():
    known = {d.doc_id for d in KnowledgeBaseLoader(base_path=KB).load_all()}
    known |= {f"JD-{p.stem}" for p in JOBS.glob("*.json")}
    queries = load_queries(QUERIES)
    assert len(queries) == 40 and len({q["id"] for q in queries}) == 40
    assert sum(q["language"] == "fr" for q in queries) >= 8
    for q in queries:
        assert set(q["relevant"]) <= known, q["id"]
