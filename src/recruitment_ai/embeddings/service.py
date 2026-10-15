"""Embedding service abstraction with lazy-loaded SentenceTransformers."""

import logging
from abc import ABC, abstractmethod
from typing import Any

from recruitment_ai.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class BaseEmbeddingService(ABC):
    """Abstract embedding service interface."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Generate embedding vector for a single string."""
        pass

    @abstractmethod
    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Generate embedding vectors for a batch of documents."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Generate embedding vector for a search query."""
        pass


class SentenceTransformerEmbeddingService(BaseEmbeddingService):
    """Concrete embedding service backed by SentenceTransformers."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: Any = None

    @property
    def model(self) -> Any:
        """Lazy load sentence transformers model only when required."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(
                    f"Loading embedding model: {self.settings.embedding_model} on {self.settings.embedding_device}"
                )
                device = self.settings.embedding_device
                # fp16 halves VRAM (bge-m3: ~2.2 GB -> ~1.1 GB on a 4 GB GPU)
                model_kwargs = {"torch_dtype": "float16"} if device == "cuda" else {}
                self._model = SentenceTransformer(
                    self.settings.embedding_model,
                    device=device,
                    model_kwargs=model_kwargs,
                )
            except Exception as exc:
                logger.warning(
                    f"SentenceTransformer could not be loaded ({exc}). Using deterministic fallback."
                )
                self._model = None
        return self._model

    def _fallback_vector(self, text: str) -> list[float]:
        """Generate deterministic pseudo-embedding when offline or model is not downloaded."""
        import hashlib
        import math

        dim = self.settings.embedding_dimension
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        vec = [math.sin(seed + i) for i in range(dim)]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_text(self, text: str) -> list[float]:
        """Embed single text."""
        if self.model is not None:
            emb = self.model.encode(text, convert_to_numpy=True)
            return [float(x) for x in emb.tolist()]
        return self._fallback_vector(text)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Batch embed multiple documents."""
        if not documents:
            return []
        if self.model is not None:
            embeddings = self.model.encode(documents, convert_to_numpy=True)
            return [[float(x) for x in row] for row in embeddings.tolist()]
        return [self._fallback_vector(doc) for doc in documents]

    def embed_query(self, query: str) -> list[float]:
        """Embed search query."""
        return self.embed_text(query)
