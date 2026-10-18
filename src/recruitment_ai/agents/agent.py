"""AI recruiter assistant (DSO4, diagram part 5): a LangGraph agent that answers with tools.

Graph::

    START -> decide --(tool chosen)--> act --> decide ...
                    \\--(enough to answer)--> respond -> END

- decide: Qwen2.5-7B (base model) reads the question, the workspace (jobs, number of candidates)
  and the results so far, and picks ONE tool with its arguments, or decides it can answer;
- act: the tool runs (Qdrant search + BGE reranker, PostgreSQL, semantic matching), its result
  becomes an observation;
- respond: Qwen writes the answer from the observations only, naming the candidates, jobs and HR
  documents it used.
Guards (by code, not by the model): at most MAX_STEPS tool calls, the same call never twice, a
failed decision or answer degrades gracefully. Every run (question, steps, sources, timings) is
stored by the caller for traceability (KB-RR-001).
"""

import inspect
import json
import time
from typing import Any, TypedDict

from pydantic import Field, field_validator
from sqlalchemy.orm import Session

from recruitment_ai.database.repositories import CandidateRepository, JobRepository
from recruitment_ai.extraction.extractor import _Lenient
from recruitment_ai.tools.registry import ToolContext, ToolRegistry, ToolResult
from recruitment_ai.utils.logging import get_logger

logger = get_logger(__name__)

MAX_STEPS = 4
FINAL = "final_answer"
FINAL_ALIASES = {"final_answer", "final", "answer", "respond", "finish", "done", "none", ""}
DECISION_TOKENS = 300
ANSWER_TOKENS = 700
OBSERVATION_CHARS = 2500

DECIDE_SYSTEM_PROMPT = """You are the AI assistant of a recruitment team. You answer recruiters' questions about candidates, jobs and the company's HR rules. You work step by step: at each step you call ONE tool, or you decide that you can answer.

TOOLS:
{tools}

Rules:
1. Facts about candidates, jobs and HR rules come only from tools, never from memory.
2. Pick the most direct tool: best candidates for a job -> rank_candidates; one candidate against one job -> match_candidate; people with a skill or an experience -> search_candidates; a candidate's background -> get_candidate; a job's requirements -> get_job; policies, scoring rubric, salary levels, interview or experience rules -> search_hr_knowledge; ordering a few candidates on one precise criterion -> rerank_candidates.
3. Name jobs and candidates exactly as in the workspace or in previous results (title, name or id).
4. Choose "final_answer" as soon as the results are enough, when no tool is needed (e.g. a greeting) or when tools cannot help. Never repeat a call already made.
5. Never search, rank or judge on age, gender, origin, nationality, religion, health or family situation: answer such requests directly with "final_answer".

Answer with one JSON object: {{"thought": "<one short sentence>", "action": "<tool name or final_answer>", "arguments": {{...}}}}."""

RESPOND_SYSTEM_PROMPT = """You are the AI assistant of a recruitment team. Answer the recruiter's question using only the tool results given.

Rules:
1. Be concise and concrete: name the candidates, jobs and HR documents (with their ids, e.g. KB-RR-002) you rely on, with the evidence or figures from the results (grade, score, years, requirements met).
2. If the results do not answer the question, say what is missing and what the recruiter could check.
3. Never invent candidates, facts or figures. Never make a hiring decision: you recommend, a recruiter decides. "Not evidenced" means the CV does not mention it, not that the candidate lacks it.
4. Never use or mention age, gender, origin, nationality, religion, health or family situation; if asked to, explain that the company's fair evaluation policy (KB-HR-003) forbids it.
5. Answer in the language of the question, in Markdown (short paragraphs or bullet points), never in JSON."""


class AgentDecision(_Lenient):
    thought: str = Field(default="", description="One short sentence: what is needed next and why")
    action: str = Field(default=FINAL, description="A tool name, or final_answer")
    arguments: dict[str, Any] = Field(default_factory=dict, description="The tool's arguments")

    @field_validator("thought", "action", mode="before")
    @classmethod
    def _text(cls, value: Any) -> Any:
        return "" if value is None else value

    @field_validator("arguments", mode="before")
    @classmethod
    def _arguments(cls, value: Any) -> Any:
        if value is None:
            return {}
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return {"query": value} if value.strip() else {}
        return value if isinstance(value, dict) else {}


class AgentState(TypedDict, total=False):
    question: str
    context: str
    steps: list[dict[str, Any]]
    pending: dict[str, Any] | None
    final_thought: str
    answer: str
    warnings: list[str]
    llm_calls: int
    timeline: list[dict[str, Any]]  # one event per node run: decide / act / respond


def _summary(result: ToolResult) -> str:
    """One line about a tool's result, for the timeline."""
    if not result.ok:
        return f"error: {(result.error or '')[:120]}"
    data = result.data
    if isinstance(data, list):
        return f"{len(data)} result(s)"
    if isinstance(data, dict):
        for key in ("top", "ranking", "requirements", "experience"):
            if isinstance(data.get(key), list):
                return f"{len(data[key])} {key}"
        return ", ".join(list(data)[:5])
    return str(data)[:120]


def workspace_context(db: Session, max_jobs: int = 25) -> str:
    """What the agent knows before any tool: the jobs (titles, ids) and how many candidates exist."""
    jobs = JobRepository(db).list(limit=max_jobs)
    lines = [f"Candidates in the database: {CandidateRepository(db).count()}",
             f"Jobs ({len(jobs)}):" if jobs else "Jobs: none yet"]
    for j in jobs:
        requirements = j.requirements or []
        must = sum(r.get("requirement_type") == "required" for r in requirements)
        lines.append(f"- {j.title}" + (f" ({j.company})" if j.company else "")
                     + f" | id {j.id} | {must} must-haves, {len(requirements) - must} nice-to-haves | {j.status}")
    return "\n".join(lines)


def _complete(llm: Any, prompt: str, system_prompt: str, max_tokens: int) -> str:
    """Free-text answer with the base model (the fine-tuned adapter is for CV extraction only)."""
    kwargs = {"use_adapter": False} if "use_adapter" in inspect.signature(llm.generate).parameters else {}
    return llm.generate(prompt, system_prompt=system_prompt, max_tokens=max_tokens, **kwargs).content


class RecruiterAgent:
    """Question in, answer + tool trace out."""

    def __init__(self, llm: Any, ctx: ToolContext, registry: ToolRegistry | None = None,
                 max_steps: int = MAX_STEPS) -> None:
        self.llm, self.ctx = llm, ctx
        self.registry = registry or ToolRegistry()
        self.max_steps = max_steps
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ graph

    def _build_graph(self) -> Any:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:  # same nodes, run by a plain loop
            logger.info("langgraph is not installed: the assistant runs its nodes with a simple loop")
            return None
        graph = StateGraph(AgentState)
        graph.add_node("decide", self.decide)
        graph.add_node("act", self.act)
        graph.add_node("respond", self.respond)
        graph.add_edge(START, "decide")
        graph.add_conditional_edges("decide", self.route, {"act": "act", "respond": "respond"})
        graph.add_edge("act", "decide")
        graph.add_edge("respond", END)
        return graph.compile()

    @staticmethod
    def route(state: AgentState) -> str:
        return "act" if state.get("pending") else "respond"

    # ------------------------------------------------------------------ prompts

    def _observations(self, steps: list[dict[str, Any]]) -> str:
        if not steps:
            return "none yet"
        parts = []
        for i, s in enumerate(steps, 1):
            args = json.dumps(s["arguments"], ensure_ascii=False)
            parts.append(f"[{i}] {s['tool']}({args}) ->\n{s['observation']}")
        return "\n\n".join(parts)

    def decision_prompt(self, state: AgentState) -> str:
        steps = state.get("steps") or []
        return (f"QUESTION: {state['question']}\n\nWORKSPACE:\n{state.get('context') or '-'}\n\n"
                f"RESULTS SO FAR:\n{self._observations(steps)}\n\n"
                f"Tool calls left: {self.max_steps - len(steps)}. What is the next step?")

    # ------------------------------------------------------------------ nodes

    def _usage(self) -> dict[str, Any]:
        """Tokens of the last model call (shown in the agent's timeline)."""
        stats = getattr(self.llm, "last_stats", None) or {}
        return {k: stats[k] for k in ("prompt_tokens", "output_tokens") if stats.get(k) is not None}

    @staticmethod
    def _logged(state: AgentState, update: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
        """The node's state update, plus one event in the timeline (node, decision, timing, tokens)."""
        return {**update, "timeline": (state.get("timeline") or []) + [event]}

    def decide(self, state: AgentState) -> dict[str, Any]:
        steps = state.get("steps") or []
        calls = state.get("llm_calls", 0)
        event: dict[str, Any] = {"node": "decide"}
        if len(steps) >= self.max_steps:
            return self._logged(state, {"pending": None},
                                {**event, "seconds": 0.0, "next": "respond",
                                 "why": f"limit of {self.max_steps} tool calls reached"})
        system = DECIDE_SYSTEM_PROMPT.format(tools=self.registry.describe())
        start = time.perf_counter()
        try:
            decision = self.llm.generate_structured(self.decision_prompt(state), AgentDecision, system_prompt=system,
                                                    max_tokens=DECISION_TOKENS, use_adapter=False)
        except Exception as exc:
            event.update(seconds=round(time.perf_counter() - start, 2), error=f"{type(exc).__name__}: {str(exc)[:150]}")
            warnings = (state.get("warnings") or []) + [f"Planning step failed ({event['error']})"]
            if not steps:  # nothing found yet: a safe default plan instead of an empty answer
                pending = {"tool": "search_hr_knowledge", "arguments": {"query": state["question"]},
                           "thought": "Planning failed: searching the HR knowledge base by default"}
                return self._logged(state, {"pending": pending, "warnings": warnings, "llm_calls": calls + 1},
                                    {**event, "next": "act", "why": "planning failed: default HR knowledge search"})
            return self._logged(state, {"pending": None, "warnings": warnings, "llm_calls": calls + 1},
                                {**event, "next": "respond", "why": "planning failed"})
        event.update(seconds=round(time.perf_counter() - start, 2), **self._usage(),
                     decision={"thought": decision.thought, "action": decision.action,
                               "arguments": decision.arguments})
        action = decision.action.strip()
        if action.lower() in FINAL_ALIASES:
            return self._logged(state, {"pending": None, "final_thought": decision.thought, "llm_calls": calls + 1},
                                {**event, "next": "respond", "why": "enough to answer"})
        if action.lower() in self.registry.names():  # "Rank_Candidates" is still rank_candidates
            action = action.lower()
        call = {"tool": action, "arguments": decision.arguments, "thought": decision.thought}
        # compare what was asked (the tools add their defaults, e.g. top_k, to the arguments they report)
        if any(s["tool"] == action and s.get("requested", s["arguments"]) == decision.arguments for s in steps):
            warnings = (state.get("warnings") or []) + [f"Repeated call to {action} skipped"]
            return self._logged(state, {"pending": None, "llm_calls": calls + 1, "warnings": warnings},
                                {**event, "next": "respond", "why": f"repeated call to {action} skipped"})
        return self._logged(state, {"pending": call, "llm_calls": calls + 1},
                            {**event, "next": "act", "why": f"tool chosen: {action}"})

    def act(self, state: AgentState) -> dict[str, Any]:
        call = state["pending"] or {}
        result: ToolResult = self.registry.run(call.get("tool", ""), self.ctx, call.get("arguments"))
        step = {"tool": result.tool, "arguments": result.arguments or call.get("arguments") or {},
                "requested": call.get("arguments") or {},
                "thought": call.get("thought", ""), "ok": result.ok, "error": result.error, "data": result.data,
                "sources": result.sources, "seconds": result.seconds,
                "observation": result.observation(OBSERVATION_CHARS)}
        event = {"node": "act", "tool": result.tool, "arguments": step["arguments"], "ok": result.ok,
                 "seconds": result.seconds, "result": _summary(result), "next": "decide"}
        return self._logged(state, {"steps": (state.get("steps") or []) + [step], "pending": None}, event)

    def respond(self, state: AgentState) -> dict[str, Any]:
        steps = state.get("steps") or []
        prompt = (f"QUESTION: {state['question']}\n\nWORKSPACE:\n{state.get('context') or '-'}\n\n"
                  f"TOOL RESULTS:\n{self._observations(steps)}\n\nWrite the answer for the recruiter.")
        calls = state.get("llm_calls", 0) + 1
        start = time.perf_counter()
        try:
            answer = _complete(self.llm, prompt, RESPOND_SYSTEM_PROMPT, ANSWER_TOKENS).strip()
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:150]}"
            warnings = (state.get("warnings") or []) + [f"Answer step failed ({error})"]
            return self._logged(state, {"answer": self.fallback_answer(steps), "warnings": warnings,
                                        "llm_calls": calls},
                                {"node": "respond", "seconds": round(time.perf_counter() - start, 2),
                                 "error": error, "fallback": True, "next": "end"})
        event = {"node": "respond", "seconds": round(time.perf_counter() - start, 2), **self._usage(),
                 "answer_chars": len(answer), "fallback": not answer, "next": "end"}
        return self._logged(state, {"answer": answer or self.fallback_answer(steps), "llm_calls": calls}, event)

    @staticmethod
    def fallback_answer(steps: list[dict[str, Any]]) -> str:
        """When the model cannot write the answer: what the tools found, as is."""
        found = [s for s in steps if s.get("ok")]
        if not found:
            return ("I could not answer: the AI model is not reachable or no tool returned a result. "
                    "Check that the Kaggle model server is running (Dashboard > System status).")
        lines = ["The AI model could not write the answer; here is what the tools found:"]
        for s in found:
            lines.append(f"- **{s['tool']}**: {s['observation'][:500]}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ run

    def run(self, question: str, context: str = "") -> dict[str, Any]:
        start = time.perf_counter()
        state: AgentState = {"question": question.strip(), "context": context, "steps": [], "pending": None,
                             "final_thought": "", "answer": "", "warnings": [], "llm_calls": 0, "timeline": []}
        if self.graph is not None:
            state = self.graph.invoke(state, {"recursion_limit": 2 * self.max_steps + 6})
        else:
            while True:
                state.update(self.decide(state))
                if self.route(state) == "respond":
                    break
                state.update(self.act(state))
            state.update(self.respond(state))
        steps = state.get("steps") or []
        sources, seen = [], set()
        for s in steps:
            for src in s.get("sources") or []:
                key = (src.get("type"), src.get("id"))
                if key not in seen:
                    seen.add(key)
                    sources.append(src)
        return {"question": state["question"], "answer": state.get("answer") or "", "steps": steps,
                "final_thought": state.get("final_thought") or "", "sources": sources,
                "warnings": state.get("warnings") or [], "llm_calls": state.get("llm_calls", 0),
                "timeline": state.get("timeline") or [], "seconds": round(time.perf_counter() - start, 1),
                "model": getattr(self.llm, "model", None), "graph": "langgraph" if self.graph is not None else "loop"}
