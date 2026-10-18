"""Searchable knowledge sources (diagram part 6 "Knowledge Sources", in Qdrant):

- ``hr_knowledge``: the 15 HR knowledge-base documents (policies, rubric, job profiles, skills);
- ``candidates``: every stored candidate as evidence snippets (positions, projects, skills...);
- ``jobs``: every open job (its requirements and description paragraphs).
All embedded with BGE-M3 and reranked with the BGE cross-encoder. Used by the agent's tools (CV
search, job search, HR knowledge search) and by the matching analysis (company rules).
"""

import uuid
from typing import Any

from recruitment_ai.data.knowledge import KnowledgeBaseLoader
from recruitment_ai.matching.evidence import candidate_snippets
from recruitment_ai.rag.knowledge_index import COLLECTION as KNOWLEDGE
from recruitment_ai.rag.knowledge_index import (
    KnowledgeIndex,
    kb_chunks,
    split_paragraphs,
)
from recruitment_ai.reranking.reranker import RerankResult
from recruitment_ai.vectorstore.base import VectorDocument, VectorSearchResult

CANDIDATES = "candidates"
JOBS = "jobs"


def _point_id(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def job_texts(job: dict[str, Any]) -> list[tuple[str, str]]:
    """(section, text) chunks of a job: its requirements, then its description by paragraphs."""
    title = job["title"] + (f" ({job['company']})" if job.get("company") else "")
    lines = [f"{r.get('id', '')} ({'must-have' if r.get('requirement_type', 'required') == 'required' else 'nice-to-have'}"
             f", {r.get('importance', 'medium')}): {r.get('requirement_text', '')}"
             for r in job.get("requirements") or []]
    chunks = [("requirements", f"{title} — Requirements\n" + "\n".join(lines))]
    for i, piece in enumerate(split_paragraphs(job.get("description") or "", 200), 1):
        chunks.append((f"description {i}", f"{title} — Description (part {i})\n{piece}"))
    return chunks


class SearchIndex:
    """Index and search the three collections."""

    def __init__(self, store: Any, embedder: Any, reranker: Any | None = None, dimension: int = 1024) -> None:
        self.store, self.embedder, self.reranker, self.dimension = store, embedder, reranker, dimension
        self.knowledge = KnowledgeIndex(store, embedder, reranker)

    # ------------------------------------------------------------------ indexing

    def _upsert(self, collection: str, items: list[tuple[str, str, dict[str, Any]]]) -> int:
        """items: (point key, text, payload)."""
        if not items:
            return 0
        if not self.store.create_collection(collection, self.dimension):
            raise RuntimeError("Qdrant is not reachable: start it with `docker compose up -d qdrant`")
        vectors = self.embedder.embed_documents([text for _, text, _ in items])
        documents = [VectorDocument(id=_point_id(key), text=text, vector=vector, metadata=payload)
                     for (key, text, payload), vector in zip(items, vectors, strict=True)]
        if not self.store.upsert(collection, documents):
            raise RuntimeError(f"Indexing into the Qdrant collection '{collection}' failed")
        return len(documents)

    def index_candidate(self, candidate_id: str, name: str | None, profile: dict[str, Any]) -> int:
        self.store.delete_by_filter(CANDIDATES, {"candidate_id": candidate_id})
        items = [(f"{candidate_id}#{s.ref}", s.text,
                  {"candidate_id": candidate_id, "name": name or "", "section": s.section, "label": s.label})
                 for s in candidate_snippets(profile)]
        return self._upsert(CANDIDATES, items)

    def remove_candidate(self, candidate_id: str) -> None:
        self.store.delete_by_filter(CANDIDATES, {"candidate_id": candidate_id})

    def index_job(self, job_id: str, job: dict[str, Any]) -> int:
        self.store.delete_by_filter(JOBS, {"job_id": job_id})
        items = [(f"{job_id}#{section}", text, {"job_id": job_id, "title": job["title"], "section": section})
                 for section, text in job_texts(job)]
        return self._upsert(JOBS, items)

    def remove_job(self, job_id: str) -> None:
        self.store.delete_by_filter(JOBS, {"job_id": job_id})

    def build_knowledge(self, base_path: Any = None) -> int:
        """(Re)build the HR knowledge collection from data/knowledge_base/."""
        return self.knowledge.build(kb_chunks(KnowledgeBaseLoader(base_path).load_all()), self.dimension)

    def counts(self) -> dict[str, int]:
        return {name: self.store.count(name) for name in (KNOWLEDGE, CANDIDATES, JOBS)}

    # ------------------------------------------------------------------ search

    def _grouped(self, collection: str, key: str, query: str, top_k: int, rerank: bool) -> list[dict[str, Any]]:
        """Search chunks, keep the best chunk per candidate/job, rerank those, return the top ``top_k``."""
        hits: list[VectorSearchResult] = self.store.search(collection, self.embedder.embed_query(query),
                                                           top_k=max(top_k * 8, 30))
        best: dict[str, VectorSearchResult] = {}
        for hit in hits:
            owner = hit.metadata.get(key)
            if owner and (owner not in best or hit.score > best[owner].score):
                best[owner] = hit
        ranked = list(best.values())
        if rerank and self.reranker is not None and ranked:
            reranked = self.reranker.rerank(query, ranked, top_n=top_k)
        else:
            reranked = [RerankResult(id=h.id, text=h.text, relevance_score=h.score, metadata=h.metadata)
                        for h in sorted(ranked, key=lambda h: h.score, reverse=True)[:top_k]]
        return [{key: r.metadata.get(key), "name": r.metadata.get("name") or r.metadata.get("title"),
                 "section": r.metadata.get("section"), "label": r.metadata.get("label"),
                 "evidence": r.text, "score": round(r.relevance_score, 4)} for r in reranked]

    def search_candidates(self, query: str, top_k: int = 5, rerank: bool = True) -> list[dict[str, Any]]:
        return self._grouped(CANDIDATES, "candidate_id", query, top_k, rerank)

    def search_jobs(self, query: str, top_k: int = 3, rerank: bool = True) -> list[dict[str, Any]]:
        return self._grouped(JOBS, "job_id", query, top_k, rerank)

    def search_knowledge(self, query: str, role: str | None = None, top_k: int = 4) -> list[RerankResult]:
        return self.knowledge.search(query, role=role, top_k=top_k)
