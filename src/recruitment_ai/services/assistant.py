"""AI assistant (DSO4): a recruiter's question -> LangGraph agent with tools -> answer, stored with its trace."""

from typing import Any

from sqlalchemy.orm import Session

from recruitment_ai.agents.agent import RecruiterAgent, workspace_context
from recruitment_ai.database.models import AgentRun
from recruitment_ai.database.repositories import AgentRunRepository
from recruitment_ai.tools.registry import ToolContext, ToolRegistry

MAX_QUESTION_CHARS = 1000


def tools_description() -> list[dict[str, str]]:
    return [{"name": t.name, "signature": t.signature(), "description": t.description}
            for t in ToolRegistry().list_tools()]


def run_dict(run: AgentRun) -> dict[str, Any]:
    trace = run.steps or []
    steps = [s for s in trace if s.get("tool") != "final_answer"]
    final = next((s for s in trace if s.get("tool") == "final_answer"), {})
    return {"id": run.id, "question": run.question, "answer": run.answer, "steps": steps, "sources": run.sources or [],
            "seconds": run.seconds, "final_thought": final.get("thought"), "warnings": final.get("warnings") or [],
            "llm_calls": final.get("llm_calls"), "model": final.get("model"), "graph": final.get("graph"),
            "timeline": final.get("timeline") or [], "created_at": run.created_at.isoformat()}


def system_prompts() -> dict[str, str]:
    """What the assistant is told before each decision and before its answer (with the tool list)."""
    from recruitment_ai.agents.agent import DECIDE_SYSTEM_PROMPT, RESPOND_SYSTEM_PROMPT

    return {"decide": DECIDE_SYSTEM_PROMPT.format(tools=ToolRegistry().describe()), "respond": RESPOND_SYSTEM_PROMPT}


def ask(db: Session, container: Any, question: str) -> AgentRun:
    question = (question or "").strip()
    if len(question) < 3:
        raise ValueError("Ask a question (at least a few words)")
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"The question is too long (at most {MAX_QUESTION_CHARS} characters)")
    agent = RecruiterAgent(container.llm, ToolContext(db, container))
    result = agent.run(question, workspace_context(db))
    trace = [{k: v for k, v in step.items() if k not in ("observation", "requested")} for step in result["steps"]]
    trace.append({"tool": "final_answer", "thought": result["final_thought"], "warnings": result["warnings"],
                  "llm_calls": result["llm_calls"], "model": result["model"], "graph": result["graph"],
                  "timeline": result["timeline"]})
    run = AgentRun(question=question, answer=result["answer"], steps=trace, sources=result["sources"],
                   seconds=result["seconds"])
    return AgentRunRepository(db).create(run)
