"""Matching results (DSO3): one assessment per job requirement, and the evidence-backed report.

The report follows "What every analysis must contain" of the fit scoring rubric (KB-RR-002):
the grade (0-3) and score (0-100); for each requirement its status (met, partial, missing) with the
CV excerpt that supports it; gaps to explore in interview; a summary and a proposed action.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

EvidenceLevel = Literal["demonstrated", "equivalent", "listed", "not_evidenced"]
Status = Literal["met", "partial", "missing"]
Action = Literal["shortlist", "review", "not_recommended"]


class RequirementAssessment(BaseModel):
    id: str
    text: str
    kind: Literal["must", "nice"] = "must"
    importance: Literal["high", "medium", "low"] = "high"
    min_years: float | None = None
    hard: bool = False  # legally/contractually mandatory (rubric "hard requirement")
    level: EvidenceLevel = "not_evidenced"
    status: Status = "missing"
    evidence: str | None = Field(default=None, description="CV excerpt supporting the level")
    evidence_source: str | None = Field(default=None, description="Where in the CV, e.g. 'Experience: ...'")
    grounded: bool = Field(default=True, description="The excerpt was found in the CV text")
    years_found: float | None = None
    years_short: bool = False
    confidence: float | None = Field(default=None, description="Semantic/reranker score of the evidence")
    note: str | None = None


class KnowledgeSource(BaseModel):
    doc_id: str
    title: str
    section: str | None = None
    excerpt: str = ""


class MatchReport(BaseModel):
    """Evidence-backed candidate-job analysis (stored in PostgreSQL, shown and exported by the app)."""

    candidate_id: str | None = None
    candidate_name: str | None = None
    job_id: str | None = None
    job_title: str = ""
    # llm: fixed analysis (Matching page); agent: the pipeline's screening agent; semantic: fast estimate
    method: Literal["llm", "agent", "semantic"] = "llm"
    grade: int = Field(ge=0, le=3)
    grade_label: str
    score: float = Field(ge=0, le=100)
    must_coverage: float
    nice_coverage: float
    total_years: float | None = None
    experience_ok: bool | None = None
    requirements: list[RequirementAssessment] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    interview_questions: list[str] = Field(default_factory=list)
    summary: str = ""
    action: Action = "review"
    action_label: str = ""
    knowledge_sources: list[KnowledgeSource] = Field(default_factory=list)
    model: str | None = None
    seconds: float | None = None
    warnings: list[str] = Field(default_factory=list)
    trace: dict[str, Any] | None = Field(
        default=None, description="AI analysis internals: HR knowledge searches, prompt, raw model answer, checks "
                                  "(and, for the screening agent, its steps, tool calls and evidence)")
    created_at: datetime = Field(default_factory=datetime.utcnow)
