"""BGE reranker scores computed by the Kaggle AI server (EMBEDDING_PROVIDER=remote)."""

from typing import Any

from recruitment_ai.config.settings import Settings, get_settings
from recruitment_ai.llm.remote_client import AIServerClient
from recruitment_ai.reranking.reranker import BaseReranker, RerankResult
from recruitment_ai.vectorstore.base import VectorSearchResult


class RemoteReranker(BaseReranker):
    """Cross-encoder reranking through the AI server's ``/rerank`` endpoint."""

    batch_size = 512

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = client or AIServerClient(self.settings, timeout=120)

    def score_pairs(self, pairs: list[list[str]]) -> list[float] | None:
        if not pairs:
            return []
        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            scores += self.client.request("POST", "/rerank", json={"pairs": batch})["scores"]
        return scores

    def rerank(self, query: str, documents: list[VectorSearchResult], top_n: int = 3) -> list[RerankResult]:
        if not documents:
            return []
        scores = self.score_pairs([[query, d.text] for d in documents]) or [d.score for d in documents]
        ranked = sorted(zip(documents, scores, strict=True), key=lambda pair: pair[1], reverse=True)[:top_n]
        return [RerankResult(id=d.id, text=d.text, relevance_score=float(s), metadata=d.metadata) for d, s in ranked]
