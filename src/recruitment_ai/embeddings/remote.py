"""BGE-M3 embeddings computed by the Kaggle AI server (EMBEDDING_PROVIDER=remote).

Same interface as the local ``SentenceTransformerEmbeddingService``: the app indexes and searches
exactly as on a GPU machine, while no model runs on our PCs.
"""

from typing import Any

from recruitment_ai.config.settings import Settings, get_settings
from recruitment_ai.embeddings.service import BaseEmbeddingService
from recruitment_ai.llm.remote_client import AIServerClient


class RemoteEmbeddingService(BaseEmbeddingService):
    """Embeddings from the AI server's ``/embed`` endpoint, in batches."""

    batch_size = 64

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or AIServerClient(self.settings, timeout=120)

    def require_model(self) -> None:
        """Fail loudly when the AI server has no embedding model (never index meaningless vectors)."""
        if not self.client.info().get("embedding_model"):
            raise RuntimeError("The Kaggle AI server has no embedding model loaded: restart its notebook")

    def _embed(self, texts: list[str], kind: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[start:start + self.batch_size]]
            vectors += self.client.request("POST", "/embed", json={"texts": batch, "kind": kind})["vectors"]
        return vectors

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return self._embed(documents, "document") if documents else []

    def embed_query(self, query: str) -> list[float]:
        return self._embed([query], "query")[0]

    def embed_text(self, text: str) -> list[float]:
        return self._embed([text], "document")[0]
