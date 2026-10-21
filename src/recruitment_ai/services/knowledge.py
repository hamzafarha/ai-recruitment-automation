"""The HR knowledge base as documents (diagram part 6, "Knowledge Sources"): what the RAG retrieves.

The screening agent, the fixed analysis and the assistant cite chunks of these documents (e.g. chunk
"KB-RR-002#3", section "Evidence levels" of the fit scoring rubric). The app opens the exact document
the agent used, on the cited section. Read-only: the documents are the markdown files of
data/knowledge_base/ (indexed in Qdrant by POST /admin/setup).
"""

from pathlib import Path
from typing import Any

from recruitment_ai.data.knowledge import KnowledgeBaseLoader, KnowledgeDocument
from recruitment_ai.rag.knowledge_index import kb_chunks
from recruitment_ai.services.errors import NotFoundError

ROOT = Path(__file__).resolve().parents[3]
CATEGORY_LABELS = {"hr_policies": "HR policies", "recruitment_rules": "Recruitment rules",
                   "job_profiles": "Job profiles", "skill_definitions": "Skill definitions"}


def knowledge_dir() -> Path:
    folder = ROOT / "data" / "knowledge_base"
    return folder if folder.is_dir() else Path("data/knowledge_base")


def _load() -> list[KnowledgeDocument]:
    return [d for d in KnowledgeBaseLoader(knowledge_dir()).load_all() if d.doc_id]


def _summary(doc: KnowledgeDocument, chunks: list[Any]) -> dict[str, Any]:
    return {"doc_id": doc.doc_id, "title": doc.title, "category": doc.category,
            "category_label": CATEGORY_LABELS.get(doc.category, doc.category), "language": doc.language,
            "roles": doc.roles, "version": doc.version, "file": Path(doc.file_path).name,
            "sections": list(dict.fromkeys(c.metadata.section for c in chunks)), "chunks": len(chunks),
            "words": len(doc.content.split())}


def documents() -> list[dict[str, Any]]:
    """Every knowledge-base document with its sections and number of chunks (as indexed)."""
    docs = _load()
    by_doc: dict[str, list[Any]] = {}
    for chunk in kb_chunks(docs):
        by_doc.setdefault(chunk.metadata.document_id, []).append(chunk)
    return [_summary(d, by_doc.get(d.doc_id or "", [])) for d in docs]


def document(doc_id: str) -> dict[str, Any]:
    """One document: its markdown and the chunks the RAG indexes (id, section, text)."""
    doc = next((d for d in _load() if (d.doc_id or "").lower() == doc_id.strip().lower()), None)
    if doc is None:
        raise NotFoundError(f"No knowledge-base document {doc_id!r}")
    chunks = kb_chunks([doc])
    return {**_summary(doc, chunks), "content": doc.content,
            "chunk_list": [{"chunk_id": c.metadata.chunk_id, "section": c.metadata.section, "text": c.text}
                           for c in chunks]}
