"""AI assistant (DSO4, diagram part 5): questions answered by a LangGraph agent with tools."""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.dependencies import Container, get_container, get_db
from api.errors import service_errors
from recruitment_ai.database.repositories import AgentRunRepository
from recruitment_ai.services import assistant

router = APIRouter(prefix="/agent", tags=["AI assistant"])


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=assistant.MAX_QUESTION_CHARS)


@router.post("/ask")
def ask(body: Question, db: Session = Depends(get_db), container: Container = Depends(get_container)) -> dict[str, Any]:
    """Ask the assistant (e.g. "Who are the best candidates for the ML Engineer job?"). It chooses its
    tools, runs them, and answers from their results; the run is stored with its trace."""
    with service_errors():
        run = assistant.ask(db, container, body.question)
    return assistant.run_dict(run)


@router.get("/runs")
def list_runs(limit: int = 50, db: Session = Depends(get_db)) -> dict[str, Any]:
    runs = AgentRunRepository(db).list(limit=limit)
    return {"count": len(runs), "data": [assistant.run_dict(r) for r in runs]}


@router.get("/runs/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    run = AgentRunRepository(db).get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return assistant.run_dict(run)


@router.get("/tools")
def list_tools() -> dict[str, Any]:
    """The tools and the two system prompts (decide, respond) the assistant works with."""
    return {"data": assistant.tools_description(), "system_prompts": assistant.system_prompts()}
