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


get_candidate_profile_tool = AgentTool(
    name="get_candidate_profile",
    description="Retrieve structured candidate profile from database",
    func=get_candidate_profile_impl,
)

get_job_description_tool = AgentTool(
    name="get_job_description",
    description="Retrieve job vacancy details and requirements from database",
    func=get_job_description_impl,
)

calculate_match_score_tool = AgentTool(
    name="calculate_match_score",
    description="Compute match score between candidate and job profile",
    func=calculate_match_score_impl,
)

search_recruitment_knowledge_tool = AgentTool(
    name="search_recruitment_knowledge",
    description="Search recruitment guidelines, job competencies, and knowledge base",
    func=search_recruitment_knowledge_impl,
)

save_recruiter_feedback_tool = AgentTool(
    name="save_recruiter_feedback",
    description="Save recruiter validation review and approval status",
    func=save_recruiter_feedback_impl,
)


class ToolRegistry:
    """Registry maintaining active agent tools."""

    def __init__(self) -> None:
        self._tools: dict[str, AgentTool] = {}
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
