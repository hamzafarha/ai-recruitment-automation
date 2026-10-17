"""Metadata schema for RAG chunks and documents."""

from typing import Literal

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk indexed in the vector store."""

    document_id: str
    document_type: Literal[
        "cv", "job_description", "job_profile", "policy", "skill_definition", "recruitment_document"
    ]
    candidate_id: str | None = None
    job_id: str | None = None
    source: str = Field(..., description="Original filename or document URI")
    page: int | None = Field(default=1)
    section: str | None = Field(default=None)
    chunk_id: str
    # Knowledge-base fields: used to filter retrieval and to cite sources ("per KB-RR-002 ...")
    title: str | None = None
    category: str | None = None
    language: str | None = None
    roles: list[str] = Field(default_factory=list)  # target role keys, or ["all"]
