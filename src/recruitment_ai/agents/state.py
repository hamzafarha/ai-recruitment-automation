"""Typed state model for LangGraph multi-agent recruitment workflow."""

from typing import Literal

from pydantic import BaseModel, Field

from recruitment_ai.extraction.schemas import (
    AnalysisReport,
    CandidateMatch,
    CandidateProfile,
    JobDescription,
    RecruiterFeedback,
)


class AgentWorkflowState(BaseModel):
    """Explicit workflow state passed across LangGraph nodes."""

    # Workflow Identifiers
    trace_id: str
    candidate_id: str
    job_id: str

    # Workflow Status
    current_step: str = "initialized"
    status: Literal[
        "in_progress", "awaiting_human_validation", "completed", "error"
    ] = "in_progress"
    error: str | None = None
    iteration_count: int = 0
    max_iterations: int = 10

    # Domain State Artifacts
    raw_cv_text: str | None = None
    raw_job_text: str | None = None
    candidate_profile: CandidateProfile | None = None
    job_description: JobDescription | None = None
    match_result: CandidateMatch | None = None
    analysis_report: AnalysisReport | None = None
    recruiter_feedback: RecruiterFeedback | None = None

    # Audit Trail
    logs: list[str] = Field(default_factory=list)
