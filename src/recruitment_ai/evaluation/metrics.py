"""Evaluation interfaces and quantitative evaluation metric schemas."""

from pydantic import BaseModel, Field


class ExtractionMetrics(BaseModel):
    """Metrics for structured entity extraction from CVs."""

    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    f1_score: float = Field(..., ge=0.0, le=1.0)
    entity_type: str = Field(default="all")


class RetrievalMetrics(BaseModel):
    """Information retrieval quality metrics."""

    recall_at_k: float = Field(..., ge=0.0, le=1.0)
    k: int = 5
    mrr: float = Field(..., ge=0.0, le=1.0, description="Mean Reciprocal Rank")
    ndcg: float = Field(
        ..., ge=0.0, le=1.0, description="Normalized Discounted Cumulative Gain"
    )


class RAGQualityMetrics(BaseModel):
    """Triad RAG quality metrics."""

    context_relevance: float = Field(..., ge=0.0, le=1.0)
    groundedness: float = Field(..., ge=0.0, le=1.0)
    answer_relevance: float = Field(..., ge=0.0, le=1.0)


class SystemMetrics(BaseModel):
    """Operational efficiency and production telemetry."""

    latency_ms: float = Field(..., ge=0.0)
    error_rate: float = Field(..., ge=0.0, le=1.0)
    token_usage: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)


def compute_f1(precision: float, recall: float) -> float:
    """Helper to compute harmonic mean F1."""
    if precision + recall == 0:
        return 0.0
    return round(2 * (precision * recall) / (precision + recall), 4)
