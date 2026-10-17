"""Qdrant implementation of VectorStore interface."""

import logging
from typing import Any

from recruitment_ai.config.settings import Settings, get_settings
from recruitment_ai.vectorstore.base import (
    VectorDocument,
    VectorSearchResult,
    VectorStore,
)

logger = logging.getLogger(__name__)


def build_filter(criteria: dict[str, Any] | None) -> Any:
    """Payload filter from ``{"field": value}``: a list value means "any of these values".

    Example: ``{"roles": ["ml_engineer", "all"], "language": "en"}`` keeps chunks for the ML
    role or for every role, written in English. None/empty means no filter.
    """
    if not criteria:
        return None
    from qdrant_client.http import models

    conditions = [
        models.FieldCondition(
            key=key,
            match=models.MatchAny(any=list(value)) if isinstance(value, list | tuple | set)
            else models.MatchValue(value=value),
        )
        for key, value in criteria.items()
        if value is not None
    ]
    return models.Filter(must=conditions) if conditions else None


class QdrantVectorStore(VectorStore):
    """Vector store adapter for Qdrant."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy initialization of Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient

                location = self.settings.qdrant_location
                if location:  # embedded mode: same API, no server
                    self._client = (QdrantClient(location=":memory:") if location == ":memory:"
                                    else QdrantClient(path=location))
                    return self._client

                # An empty QDRANT_API_KEY= must mean "no key": qdrant-client
                # switches to HTTPS whenever api_key is not None.
                api_key = self.settings.qdrant_api_key or None
                self._client = QdrantClient(
                    host=self.settings.qdrant_host,
                    port=self.settings.qdrant_port,
                    api_key=api_key,
                    https=api_key is not None,
                    timeout=5,
                )
            except ImportError:
                logger.warning(
                    "qdrant-client package is not installed. Run pip install qdrant-client."
                )
                self._client = None
            except Exception as e:
                logger.warning(f"Failed to connect to Qdrant: {e}")
                self._client = None
        return self._client

    def create_collection(
        self, collection_name: str, dimension: int, distance: str = "Cosine"
    ) -> bool:
        """Create a collection in Qdrant if it does not already exist."""
        if not self.client:
            logger.error("Qdrant client unavailable.")
            return False
        try:
            from qdrant_client.http import models

            distance_enum = getattr(
                models.Distance, distance.upper(), models.Distance.COSINE
            )
            collections = [c.name for c in self.client.get_collections().collections]
            if collection_name not in collections:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=dimension,
                        distance=distance_enum,
                    ),
                )
                logger.info(f"Created Qdrant collection '{collection_name}'.")
            return True
        except Exception as exc:
            logger.error(f"Error creating Qdrant collection: {exc}")
            return False

    def reset_collection(self, collection_name: str, dimension: int) -> bool:
        """Drop and recreate a collection (a full rebuild leaves no stale chunks behind)."""
        if not self.client:
            logger.error("Qdrant client unavailable.")
            return False
        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
        return self.create_collection(collection_name, dimension)

    def upsert(self, collection_name: str, documents: list[VectorDocument]) -> bool:
        """Upsert documents into Qdrant."""
        if not self.client:
            logger.error("Qdrant client unavailable.")
            return False
        try:
            from qdrant_client.http import models

            points = []
            for doc in documents:
                if not doc.vector:
                    continue
                payload = dict(doc.metadata)
                payload["text"] = doc.text
                points.append(
                    models.PointStruct(
                        id=doc.id,
                        vector=doc.vector,
                        payload=payload,
                    )
                )

            if points:
                self.client.upsert(
                    collection_name=collection_name,
                    points=points,
                )
            return True
        except Exception as exc:
            logger.error(f"Error upserting into Qdrant: {exc}")
            return False

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        filter_criteria: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search similar vectors in Qdrant."""
        if not self.client:
            logger.error("Qdrant client unavailable.")
            return []
        try:
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=build_filter(filter_criteria),
                with_payload=True,
            )
            search_results: list[VectorSearchResult] = []
            for hit in response.points:
                payload = hit.payload or {}
                text = payload.get("text", "")
                metadata = {k: v for k, v in payload.items() if k != "text"}
                search_results.append(
                    VectorSearchResult(
                        id=str(hit.id),
                        score=float(hit.score),
                        text=text,
                        metadata=metadata,
                    )
                )
            return search_results
        except Exception as exc:
            logger.error(f"Error searching in Qdrant: {exc}")
            return []

    def delete(self, collection_name: str, document_ids: list[str]) -> bool:
        """Delete points by IDs."""
        if not self.client:
            return False
        try:
            from typing import Any, cast

            from qdrant_client.http import models

            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=cast(Any, document_ids)),
            )
            return True
        except Exception as exc:
            logger.error(f"Error deleting from Qdrant: {exc}")
            return False

    def health_check(self) -> tuple[bool, str]:
        """Check Qdrant health."""
        try:
            if not self.client:
                return False, "Qdrant client is not initialized or unreachable."
            self.client.get_collections()
            return True, f"Qdrant connected successfully at {self.settings.qdrant_url}."
        except Exception as exc:
            return False, f"Qdrant health check failed: {exc}"
