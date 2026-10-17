"""HR knowledge index for RAG (Step 3, DSO2): chunk, embed (BGE-M3), store (Qdrant), retrieve.

Indexed: the 15 knowledge-base documents (data/knowledge_base/) and the evaluation job postings.
- Knowledge-base documents are split by ``## section``: one section = one rule = one chunk, prefixed
  with the document title ("Fit Scoring Rubric — Evidence levels"), since a section heading alone is
  ambiguous. Sections longer than MAX_CHUNK_WORDS are split on paragraph boundaries.
- A job posting gives one "Requirements" chunk (its structured M1../N1.. list) plus paragraph
  chunks of the posting text.
Every chunk has a stable id ("KB-RR-002#3"), so rebuilding replaces chunks instead of duplicating them.
Retrieval: BGE-M3 dense search (optionally filtered by role / language), then optional
re-ordering by the BGE cross-encoder reranker.
"""

import re
import uuid
from typing import Any

from recruitment_ai.data.knowledge import KnowledgeDocument
from recruitment_ai.rag.chunker import DocumentChunk
from recruitment_ai.rag.metadata import ChunkMetadata
from recruitment_ai.reranking.reranker import RerankResult
from recruitment_ai.vectorstore.base import VectorDocument

COLLECTION = "hr_knowledge"
MAX_CHUNK_WORDS = 300
JOB_PARAGRAPH_WORDS = 200
DOC_TYPES = {
    "hr_policies": "policy",
    "recruitment_rules": "recruitment_document",
    "skill_definitions": "skill_definition",
    "job_profiles": "job_profile",
}


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """(section title, body) pairs of a markdown document; text before the first ## is "Introduction"."""
    body = re.sub(r"^\s*# .*\n", "", markdown, count=1)  # drop the H1 title (kept in metadata)
    parts = re.split(r"\n(?=## )", "\n" + body)
    sections = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("## "):
            heading, _, text = part.partition("\n")
            sections.append((heading[3:].strip(), text.strip()))
        else:
            sections.append(("Introduction", part))
    return [(title, text) for title, text in sections if text]


def split_paragraphs(text: str, max_words: int) -> list[str]:
    """Group paragraphs into pieces of at most ``max_words`` (a paragraph is never cut)."""
    pieces, current, count = [], [], 0
    for paragraph in [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]:
        words = len(paragraph.split())
        if current and count + words > max_words:
            pieces.append("\n\n".join(current))
            current, count = [], 0
        current.append(paragraph)
        count += words
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def _chunk(text: str, chunk_id: str, **meta: Any) -> DocumentChunk:
    metadata = ChunkMetadata(chunk_id=chunk_id, **meta)
    return DocumentChunk(id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)), text=text, metadata=metadata)


def kb_chunks(docs: list[KnowledgeDocument]) -> list[DocumentChunk]:
    chunks = []
    for doc in docs:
        index = 0
        for section, body in split_sections(doc.content):
            for piece in split_paragraphs(body, MAX_CHUNK_WORDS):
                chunks.append(_chunk(
                    f"{doc.title} — {section}\n\n{piece}", f"{doc.doc_id}#{index}",
                    document_id=doc.doc_id, document_type=DOC_TYPES[doc.category], source=doc.file_path,
                    section=section, title=doc.title, category=doc.category, language=doc.language,
                    roles=doc.roles,
                ))
                index += 1
    return chunks


def job_chunks(job: dict[str, Any]) -> list[DocumentChunk]:
    """A job posting: one structured "Requirements" chunk, then the posting text by paragraphs."""
    doc_id, role = f"JD-{job['role']}", job["role"]
    company = job.get("source", {}).get("company") or ""
    title = f"{job['title']} ({company})" if company else job["title"]
    common = {"document_id": doc_id, "document_type": "job_description", "source": f"eval_v1/jobs/{role}.json",
              "title": title, "category": "job_postings", "language": "en", "roles": [role], "job_id": job["id"]}
    requirement_lines = [
        f"{r['id']} ({r['requirement_type']}, {r['importance']} importance"
        f"{', ' + str(r['min_years_experience']) + '+ years' if r.get('min_years_experience') else ''}): "
        f"{r['requirement_text']}"
        for r in job["requirements"]
    ]
    chunks = [_chunk(f"{title} — Requirements\n\n" + "\n".join(requirement_lines), f"{doc_id}#0",
                     section="Requirements", **common)]
    for i, piece in enumerate(split_paragraphs(job["description"], JOB_PARAGRAPH_WORDS), 1):
        chunks.append(_chunk(f"{title} — Posting (part {i})\n\n{piece}", f"{doc_id}#{i}",
                             section=f"Posting part {i}", **common))
    return chunks


class KnowledgeIndex:
    """Build and query the HR knowledge collection."""

    def __init__(self, store: Any, embedder: Any, reranker: Any | None = None, collection: str = COLLECTION) -> None:
        self.store, self.embedder, self.reranker, self.collection = store, embedder, reranker, collection

    def build(self, chunks: list[DocumentChunk], dimension: int) -> int:
        """Embed all chunks with the real model and (re)create the collection with them."""
        self.embedder.require_model()
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        if not self.store.reset_collection(self.collection, dimension):
            raise RuntimeError("Could not (re)create the Qdrant collection: is Qdrant running?")
        documents = [VectorDocument(id=c.id, text=c.text, vector=v, metadata=c.metadata.model_dump())
                     for c, v in zip(chunks, vectors, strict=True)]
        if not self.store.upsert(self.collection, documents):
            raise RuntimeError("Upserting chunks into Qdrant failed.")
        return len(documents)

    def search(self, query: str, role: str | None = None, language: str | None = None, top_k: int = 5,
               candidates: int = 20, rerank: bool = True) -> list[RerankResult]:
        """Dense search over ``candidates`` chunks, optionally reranked; returns the best ``top_k``.

        ``role`` keeps chunks for that role or for every role; ``language`` keeps one language.
        """
        criteria: dict[str, Any] = {}
        if role:
            criteria["roles"] = [role, "all"]
        if language:
            criteria["language"] = language
        hits = self.store.search(self.collection, self.embedder.embed_query(query), top_k=candidates,
                                 filter_criteria=criteria or None)
        if rerank and self.reranker is not None and hits:
            return self.reranker.rerank(query, hits, top_n=top_k)
        return [RerankResult(id=h.id, text=h.text, relevance_score=h.score, metadata=h.metadata)
                for h in hits[:top_k]]
