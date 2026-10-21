"""The HR knowledge base as documents (diagram part 6): the exact documents the RAG retrieves from."""

from typing import Any

from fastapi import APIRouter

from api.errors import service_errors
from recruitment_ai.services import knowledge

router = APIRouter(prefix="/knowledge", tags=["HR knowledge base"])


@router.get("/documents")
def list_documents() -> dict[str, Any]:
    """The 15 documents (policies, rules, job profiles, skills) with their sections and chunks."""
    docs = knowledge.documents()
    return {"count": len(docs), "data": docs}


@router.get("/documents/{doc_id}")
def get_document(doc_id: str) -> dict[str, Any]:
    """One document (e.g. KB-RR-002): its markdown and the chunks indexed in Qdrant."""
    with service_errors():
        return knowledge.document(doc_id)
