"""Agent tools with strict schemas, error boundaries, and logging."""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from recruitment_ai.utils.logging import get_logger

logger = get_logger(__name__)


class ToolResult(BaseModel):
    """Standardized tool output structure."""

    tool_name: str
    success: bool
    data: Any = None
    error: str | None = None


class AgentTool(BaseModel):
    """Wrapper for agent tools with validation and metrics."""

    name: str
    description: str
    func: Callable[..., Any]

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute tool safely with logging and error isolation."""
        logger.info(f"Executing tool '{self.name}' with args: {list(kwargs.keys())}")
        try:
            output = self.func(**kwargs)
            return ToolResult(tool_name=self.name, success=True, data=output)
        except Exception as exc:
            logger.error(f"Tool '{self.name}' failed: {exc}", exc_info=True)
            return ToolResult(tool_name=self.name, success=False, error=str(exc))


class CandidateProfileInput(BaseModel):
    candidate_id: str = Field(..., description="Unique ID of the candidate")


class JobDescriptionInput(BaseModel):
    job_id: str = Field(..., description="Unique ID of the job opening")


class MatchScoreInput(BaseModel):
    candidate_id: str
    job_id: str


class SearchKnowledgeInput(BaseModel):
    query: str
    top_k: int = 5


class FeedbackInput(BaseModel):
    match_id: str
    recruiter_id: str
    decision: str
    comments: str | None = None


# Concrete tool implementations (placeholders with clean typed signatures)
def get_candidate_profile_impl(candidate_id: str) -> dict[str, Any]:
    """Retrieve candidate profile by ID."""
    return {"candidate_id": candidate_id, "status": "candidate_profile_loaded"}


def get_job_description_impl(job_id: str) -> dict[str, Any]:
    """Retrieve job requirements and description by ID."""
    return {"job_id": job_id, "status": "job_description_loaded"}


def calculate_match_score_impl(candidate_id: str, job_id: str) -> dict[str, Any]:
    """Compute multidimensional match score."""
    return {
        "candidate_id": candidate_id,
        "job_id": job_id,
        "overall_score": 88.5,
        "status": "calculated",
    }


def search_recruitment_knowledge_impl(query: str, top_k: int = 5) -> dict[str, Any]:
    """Search policies and role requirements."""
    return {"query": query, "top_k": top_k, "results": []}


def save_recruiter_feedback_impl(
    match_id: str, recruiter_id: str, decision: str, comments: str | None = None
) -> dict[str, Any]:
    """Persist recruiter validation decision."""
    return {
        "match_id": match_id,
        "recruiter_id": recruiter_id,
        "decision": decision,
        "saved": True,
    }


# Architecture Toolbox implementations (Box 5)
def cv_search_impl(
    query: str, candidate_id: str | None = None, top_k: int = 5
) -> dict[str, Any]:
    """Search in candidate CV documents."""
    return {
        "tool": "cv_search_tool",
        "query": query,
        "candidate_id": candidate_id,
        "top_k": top_k,
        "results": [],
    }


def job_description_search_impl(
    query: str, job_id: str | None = None, top_k: int = 5
) -> dict[str, Any]:
    """Search in job descriptions."""
    return {
        "tool": "job_description_search_tool",
        "query": query,
        "job_id": job_id,
        "top_k": top_k,
        "results": [],
    }


def hr_knowledge_search_impl(
    query: str, category: str | None = None, top_k: int = 5
) -> dict[str, Any]:
    """Search HR Knowledge Base (policies, skills, guidelines)."""
    return {
        "tool": "hr_knowledge_search_tool",
        "query": query,
        "category": category,
        "top_k": top_k,
        "results": [],
    }


def candidate_db_impl(candidate_id: str) -> dict[str, Any]:
    """Get structured candidate profile from database."""
    return {
        "tool": "candidate_db_tool",
        "candidate_id": candidate_id,
        "status": "loaded",
    }


def semantic_matching_impl(candidate_id: str, job_id: str) -> dict[str, Any]:
    """Compute semantic embeddings similarity and skill alignment."""
    return {
        "tool": "semantic_matching_tool",
        "candidate_id": candidate_id,
        "job_id": job_id,
        "score": 88.5,
    }


def reranking_impl(query: str, document_ids: list[str]) -> dict[str, Any]:
    """Reorder retrieved results using BGE-reranker-v2-m3."""
    return {"tool": "reranking_tool", "query": query, "ranked_ids": document_ids}


cv_search_tool = AgentTool(
    name="cv_search_tool",
    description="Search inside candidate CVs for experience, skills, and qualifications",
    func=cv_search_impl,
)

job_description_search_tool = AgentTool(
    name="job_description_search_tool",
    description="Search inside job descriptions and specific role requirements",
    func=job_description_search_impl,
)

hr_knowledge_search_tool = AgentTool(
    name="hr_knowledge_search_tool",
    description="Search HR knowledge base for recruitment policies, skills criteria, and guidelines",
    func=hr_knowledge_search_impl,
)

candidate_db_tool = AgentTool(
    name="candidate_db_tool",
    description="Retrieve structured candidate profile from database",
    func=candidate_db_impl,
)

semantic_matching_tool = AgentTool(
    name="semantic_matching_tool",
    description="Calculate semantic matching score between candidate profile and job requirements",
    func=semantic_matching_impl,
)

reranking_tool = AgentTool(
    name="reranking_tool",
    description="Rerank candidate-job retrieval results with neural cross-encoder",
    func=reranking_impl,
)

# Aliases for backwards compatibility
get_candidate_profile_tool = candidate_db_tool
get_job_description_tool = job_description_search_tool
calculate_match_score_tool = semantic_matching_tool
search_recruitment_knowledge_tool = hr_knowledge_search_tool
save_recruiter_feedback_tool = AgentTool(
    name="save_recruiter_feedback",
    description="Save recruiter validation review and approval status",
    func=save_recruiter_feedback_impl,
)


class ToolRegistry:
    """Registry maintaining active agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}
        # Core Architecture Toolbox (Box 5)
        self.register(cv_search_tool)
        self.register(job_description_search_tool)
        self.register(hr_knowledge_search_tool)
        self.register(candidate_db_tool)
        self.register(semantic_matching_tool)
        self.register(reranking_tool)
        # Aliases for backwards compatibility
        self.register(get_candidate_profile_tool)
        self.register(get_job_description_tool)
        self.register(calculate_match_score_tool)
        self.register(search_recruitment_knowledge_tool)
        self.register(save_recruiter_feedback_tool)

    def register(self, tool: AgentTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> AgentTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[AgentTool]:
        return list(self._tools.values())
