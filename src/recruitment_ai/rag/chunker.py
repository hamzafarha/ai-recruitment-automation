"""Chunking strategies for document text."""

import uuid
from abc import ABC, abstractmethod

from pydantic import BaseModel

from recruitment_ai.rag.metadata import ChunkMetadata


class DocumentChunk(BaseModel):
    """Chunk of text with attached provenance metadata."""

    id: str
    text: str
    metadata: ChunkMetadata


class BaseChunker(ABC):
    """Abstract chunker interface."""

    @abstractmethod
    def chunk(self, text: str, base_metadata: dict) -> list[DocumentChunk]:
        """Split text into chunks with metadata."""
        pass


class RecursiveChunker(BaseChunker):
    """Recursive token- or character-based chunker with boundary preservation."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, base_metadata: dict) -> list[DocumentChunk]:
        """Split text by paragraphs, then sentences, then words with overlap."""
        if not text.strip():
            return []

        chunks: list[DocumentChunk] = []
        words = text.split()
        if not words:
            return []

        start = 0
        chunk_index = 0
        step = max(1, self.chunk_size - self.chunk_overlap)

        while start < len(words):
            end = min(len(words), start + self.chunk_size)
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            meta_dict = dict(base_metadata)
            meta_dict["chunk_id"] = (
                f"{meta_dict.get('document_id', 'doc')}_chunk_{chunk_index}"
            )

            chunk_meta = ChunkMetadata.model_validate(meta_dict)
            chunks.append(
                DocumentChunk(
                    # deterministic: re-indexing the same document replaces its chunks
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_meta.chunk_id)),
                    text=chunk_text,
                    metadata=chunk_meta,
                )
            )
            chunk_index += 1
            if end == len(words):
                break
            start += step

        return chunks
