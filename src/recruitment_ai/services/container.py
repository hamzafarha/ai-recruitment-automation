"""Wiring of the application: the configured models and stores, created once and only when needed.

With the settings of .env (LLM_PROVIDER=remote, EMBEDDING_PROVIDER=remote) every model call goes
to the Kaggle AI server; Qdrant and PostgreSQL run in Docker. The API depends on ``get_container``,
which unit tests replace with fakes.
"""

from functools import cached_property, lru_cache
from typing import Any

from recruitment_ai.config.settings import Settings, get_settings


class Container:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @cached_property
    def llm(self) -> Any:
        from recruitment_ai.llm import get_llm_provider

        return get_llm_provider()

    @cached_property
    def embedder(self) -> Any:
        from recruitment_ai.embeddings import get_embedding_service

        return get_embedding_service(self.settings)

    @cached_property
    def reranker(self) -> Any:
        from recruitment_ai.reranking import get_reranker

        return get_reranker(self.settings)

    @cached_property
    def store(self) -> Any:
        from recruitment_ai.vectorstore.qdrant import QdrantVectorStore

        return QdrantVectorStore(self.settings)

    @cached_property
    def index(self) -> Any:
        from recruitment_ai.rag.search_index import SearchIndex

        return SearchIndex(self.store, self.embedder, self.reranker, self.settings.embedding_dimension)

    @cached_property
    def normalizer(self) -> Any:
        from recruitment_ai.extraction.extractor import default_normalizer

        return default_normalizer()

    @cached_property
    def extractor(self) -> Any:
        from recruitment_ai.extraction.extractor import CVExtractor

        return CVExtractor(llm=self.llm, normalizer=self.normalizer)

    @cached_property
    def semantic(self) -> Any:
        from recruitment_ai.matching.semantic import SemanticMatcher, load_thresholds

        return SemanticMatcher(self.embedder, self.reranker, self.normalizer, load_thresholds())

    def knowledge_search(self, query: str, role: str | None = None) -> list[Any]:
        return self.index.search_knowledge(query, role=role, top_k=3)

    @cached_property
    def analyzer(self) -> Any:
        from recruitment_ai.matching.analyzer import EvidenceAnalyzer

        return EvidenceAnalyzer(self.llm, knowledge_search=self.knowledge_search, semantic=self.semantic)


@lru_cache(maxsize=1)
def get_container() -> Container:
    """The application's container (FastAPI dependency)."""
    return Container()
