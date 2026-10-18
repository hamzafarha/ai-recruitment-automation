"""Screening agent (DSO4, diagram part 5): the LangGraph agent inside the application pipeline.

The pipeline (services/pipeline.py) gives it one task: screen this candidate for the job they applied
to, or for the open jobs that fit them. It returns the evidence-backed report of each job. Its graph
follows the seven steps of the architecture's agent box:

    START -> understand -> plan -> act -> evaluate --(weak must-haves, rounds left)--> plan
                                            \\--(evidence enough)--> analyse -> verify -> next_job
                                                                    next_job --(another job)--> plan
                                                                    next_job --(done)--> END

1. understand the task: the job the candidate applied to, or, for a spontaneous application, every
   open job graded by the fast semantic matching ("Semantic Matching Tool"): the jobs with a grade of
   2 or more are analysed (at most MAX_JOBS), else the best one;
2. decide which tools to use (plan): the first round is systematic (by code): the CV is searched for
   every requirement and the HR knowledge base for the job's rules. In later rounds Qwen reads the
   requirements whose evidence is still weak, with the passages found so far, and chooses further
   searches: the CV again with other words ("CV Search Tool"), or the HR rules, e.g. the skills the job
   profile accepts as equivalent ("HR Knowledge Search Tool");
3-4. call the tools, retrieve (act): BGE-M3 + Qdrant / the CV passages;
5. evaluate and rerank: the BGE cross-encoder ("Reranking Tool") orders each requirement's passages,
   the best EVIDENCE_PER_REQUIREMENT are kept;
6. iterate if needed: must-haves still without strong evidence -> another planning round (at most
   MAX_ROUNDS, and only while the model asks for searches);
7. generate the final analysis: Qwen judges every requirement from its passages and the CV (level,
   quote, years), then code checks every claim (verify): the quote must be in the CV (as in the fixed
   analysis) and support the requirement (cross-encoder); a skill only found in a skills section or the
   summary is "listed"; strong evidence the model left out is used. The rubric (KB-RR-002) gives the
   grade, score and proposed action. The recruiter decides (human validation).
The checks of step 7 address the error types of the fixed analysis on the dev pairs of the matching
evaluation (requirements claimed "met" while the reference says partial or missing, or the opposite).
Guards by code: the systematic searches always run, at most MAX_CALLS tool calls per round, no repeated
search, no search on protected attributes, a failed model call degrades (default searches, or the fast
semantic estimate) instead of stopping the pipeline.
"""

import re
import time
from typing import Any, TypedDict

from pydantic import Field, field_validator

from recruitment_ai.extraction.extractor import _Lenient
from recruitment_ai.matching import rubric
from recruitment_ai.matching.analyzer import ANALYSIS_SYSTEM_PROMPT, MatchJudgement
from recruitment_ai.matching.cv_index import (
    LISTING_SECTIONS,
    USAGE_SECTIONS,
    CVEvidenceIndex,
)
from recruitment_ai.matching.evidence import (
    is_grounded,
    positions,
    profile_text,
    requirement_templates,
    total_years,
)
from recruitment_ai.matching.schemas import (
    KnowledgeSource,
    MatchReport,
    RequirementAssessment,
)
from recruitment_ai.utils.logging import get_logger

logger = get_logger(__name__)

MAX_JOBS = 2                 # a spontaneous application is analysed for at most this many open jobs
FIT_GRADE = 2                # ... those graded at least this by the fast matching
MAX_ROUNDS = 2               # planning rounds by the model, after the systematic first round
MAX_CALLS = 4                # tool calls per planning round
EVIDENCE_PER_REQUIREMENT = 3
PLAN_TOKENS = 400
WEAK_SHOWN = 6               # weak requirements shown to the planner
STRONG_SECTIONS = USAGE_SECTIONS | {"education", "certification", "languages"}  # context for "demonstrated"
LEVEL_ORDER = ["not_evidenced", "listed", "equivalent", "demonstrated"]
# searches on the candidate's protected attributes (KB-HR-003); a job's domain ("health data") stays allowed
PROTECTED = re.compile(r"\b(age|âge|ages|gender|genre|sexe|nationalit\w*|religio\w*|ethni\w*|origin|origine|origins|"
                       r"marital|married|pregnan\w*|enceinte|handicap\w*|disabilit\w*)\b", re.IGNORECASE)

STEPS = {"understand": "1. Understand the task", "plan": "2. Decide which tools to use",
         "act": "3-4. Call the tools, retrieve", "evaluate": "5-6. Evaluate & rerank, iterate if needed",
         "analyse": "7. Generate the final analysis", "verify": "7. Check every claim (code)",
         "next": "Next job"}
TOOLS = {
    "search_cv": "search_cv(query): searches the candidate's CV again. Use other words: synonyms, the tools or "
                 "tasks that imply the skill, the French term (the CV may be in French), e.g. \"REST API\" -> "
                 "\"web services Flask endpoints\".",
    "search_hr_knowledge": "search_hr_knowledge(query): searches the company's HR rules, e.g. the skills the job "
                           "profile accepts as equivalent, how internships count in years of experience.",
}

PLAN_SYSTEM_PROMPT = """You are the screening agent of a recruitment team. You check ONE candidate's CV against ONE job, requirement by requirement, before the final judgement. You get the requirements whose CV evidence is still weak or missing, the best CV passage found so far for each (with its CV section and its relevance from 0 to 1) and the searches already made.

Decide which further searches could still find evidence. Tools:
{tools}

Rules:
1. At most {max_calls} calls, must-haves first. Never repeat a search already made.
2. No call for a requirement already proven by its passage, nor when the CV clearly has nothing related: the final judgement will then say "not evidenced".
3. A skill only found in a skills section is "listed": search the CV for a position or a project that used it.
4. Never search for age, gender, origin, nationality, religion, health or family situation.

Answer with one JSON object: {{"thought": "<one short sentence>", "calls": [{{"tool": "search_cv", "requirement": "M2", "query": "<words to search>"}}]}}. "calls" may be empty."""

AGENT_ANALYSIS_SYSTEM_PROMPT = ANALYSIS_SYSTEM_PROMPT.replace("\n\nRules:", """

Under each requirement you get the CV passages the screening agent retrieved for it, with the CV section they come from and their relevance (0-1). Start from them and check the full CV: a passage from a position, a project or an achievement can show a skill demonstrated; a passage from a skills section or the summary only shows it listed; a passage may also be unrelated to the requirement. Quote the CV word for word.

Rules:""", 1)


class PlannedCall(_Lenient):
    tool: str = "search_cv"
    requirement: str | None = None
    query: str = ""

    @field_validator("tool", "query", mode="before")
    @classmethod
    def _text(cls, value: Any) -> Any:
        return "" if value is None else value


class ScreeningPlan(_Lenient):
    thought: str = Field(default="", description="One short sentence: what is still missing")
    calls: list[PlannedCall] = Field(default_factory=list)

    @field_validator("thought", mode="before")
    @classmethod
    def _thought(cls, value: Any) -> Any:
        return "" if value is None else value

    @field_validator("calls", mode="before")
    @classmethod
    def _calls(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, dict):
            value = [value]
        return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


class ScreeningState(TypedDict, total=False):
    candidate: dict[str, Any]
    open_jobs: list[dict[str, Any]]       # input: the job applied to, or every open job
    chosen_job_id: str | None
    jobs: list[dict[str, Any]]            # the jobs to analyse (step 1)
    considered: list[dict[str, Any]]      # spontaneous application: every open job with its fast grade
    current: int                          # index of the job being analysed
    # the current job
    job_started: float
    round: int                            # planning rounds made (the first one is systematic)
    pending: list[dict[str, Any]]         # planned tool calls
    last_calls: int                       # calls of the last planning round
    evidence: dict[str, list[dict[str, Any]]]  # requirement id -> best CV passages
    knowledge: list[dict[str, Any]]       # HR knowledge excerpts
    queries: dict[str, list[str]]         # requirement id (or "_hr") -> searches made
    semantic: dict[str, Any] | None       # the fast semantic report (fallback, years, strong evidence)
    plans: list[dict[str, Any]]
    job_calls: list[dict[str, Any]]
    job_warnings: list[str]
    judgement: dict[str, Any] | None
    llm_stats: dict[str, Any]
    prompt: str
    route: str
    # the whole run
    reports: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    warnings: list[str]
    llm_calls: int
    timeline: list[dict[str, Any]]


def _job_reset() -> dict[str, Any]:
    return {"round": 0, "pending": [], "last_calls": 0, "evidence": {}, "knowledge": [], "queries": {},
            "semantic": None, "plans": [], "job_calls": [], "job_warnings": [], "judgement": None, "llm_stats": {},
            "prompt": "", "job_started": time.perf_counter()}


def _short(text: str | None, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


class ScreeningAgent:
    """Candidate + job(s) in, one evidence-backed report per job out, with the trace of every step."""

    def __init__(self, llm: Any, embedder: Any, reranker: Any | None, semantic: Any, analyzer: Any,
                 max_rounds: int = MAX_ROUNDS, max_jobs: int = MAX_JOBS) -> None:
        self.llm, self.embedder, self.reranker = llm, embedder, reranker
        self.semantic, self.analyzer = semantic, analyzer   # SemanticMatcher, EvidenceAnalyzer
        self.max_rounds, self.max_jobs = max_rounds, max_jobs
        self.index: CVEvidenceIndex | None = None           # the candidate's CV, built by ``run``
        self._semantic_reports: dict[str, MatchReport] = {}
        self.graph = self._build_graph()

    @classmethod
    def from_container(cls, container: Any) -> "ScreeningAgent":
        return cls(container.llm, container.embedder, container.reranker, container.semantic, container.analyzer)

    # ------------------------------------------------------------------ graph

    def _build_graph(self) -> Any:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:  # same nodes, run by a plain loop
            logger.info("langgraph is not installed: the screening agent runs its nodes with a simple loop")
            return None
        graph = StateGraph(ScreeningState)
        for name in ("understand", "plan", "act", "evaluate", "analyse", "verify", "next_job"):
            graph.add_node(name, getattr(self, name))
        graph.add_edge(START, "understand")
        graph.add_edge("understand", "plan")
        graph.add_edge("plan", "act")
        graph.add_edge("act", "evaluate")
        graph.add_conditional_edges("evaluate", self.route, {"plan": "plan", "analyse": "analyse"})
        graph.add_conditional_edges("analyse", self.route, {"verify": "verify", "next": "next_job"})
        graph.add_edge("verify", "next_job")
        graph.add_conditional_edges("next_job", self.route, {"plan": "plan", "end": END})
        return graph.compile()

    @staticmethod
    def route(state: ScreeningState) -> str:
        return state.get("route") or "end"

    # ------------------------------------------------------------------ helpers

    def _thresholds(self) -> tuple[float, float]:
        """(demonstrated, equivalent) thresholds of the fast matching, on the scores the index produces."""
        t = self.semantic.thresholds
        reranked = self.index.reranked if self.index is not None and self.index.reranked is not None else True
        return (t.demonstrated, t.equivalent) if reranked else (t.cosine_demonstrated, t.cosine_equivalent)

    def _usage(self) -> dict[str, Any]:
        stats = getattr(self.llm, "last_stats", None) or {}
        return {k: stats[k] for k in ("prompt_tokens", "output_tokens") if stats.get(k) is not None}

    @staticmethod
    def _logged(state: ScreeningState, update: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
        event = {"node": event["node"], "step": STEPS[event["node"]], **event}
        return {**update, "timeline": (state.get("timeline") or []) + [event]}

    def _semantic_report(self, job: dict[str, Any], candidate: dict[str, Any]) -> MatchReport:
        if job["id"] not in self._semantic_reports:
            self._semantic_reports[job["id"]] = self.semantic.match(job, candidate)
        return self._semantic_reports[job["id"]]

    def _strong(self, hits: list[dict[str, Any]]) -> bool:
        """Evidence enough for the planner: a relevant passage from a position, project, degree..."""
        high, _ = self._thresholds()
        return any(h["score"] >= high and h["section"] in STRONG_SECTIONS for h in hits)

    def weak_requirements(self, job: dict[str, Any], state: ScreeningState) -> list[RequirementAssessment]:
        """Requirements without strong evidence yet, must-haves first."""
        evidence = state.get("evidence") or {}
        weak = [a for a in requirement_templates(job) if not self._strong(evidence.get(a.id) or [])]
        return sorted(weak, key=lambda a: a.kind != "must")

    # ------------------------------------------------------------------ 1. understand the task

    def understand(self, state: ScreeningState) -> dict[str, Any]:
        start = time.perf_counter()
        candidate, jobs_in, chosen = state["candidate"], state["open_jobs"], state.get("chosen_job_id")
        calls = list(state.get("tool_calls") or [])
        considered: list[dict[str, Any]] = []
        if chosen:
            jobs = [j for j in jobs_in if j["id"] == chosen] or jobs_in[:1]
            task, why = "one_job", f"the candidate applied to {jobs[0]['title']}"
        else:
            graded = []
            for job in jobs_in:
                report = self._semantic_report(job, candidate)
                graded.append((job, report))
                considered.append({"job_id": job["id"], "title": job["title"], "grade": report.grade,
                                   "grade_label": report.grade_label, "score": report.score})
            graded.sort(key=lambda item: (item[1].grade, item[1].score), reverse=True)
            considered.sort(key=lambda c: (c["grade"], c["score"]), reverse=True)
            fit = [job for job, report in graded if report.grade >= FIT_GRADE][: self.max_jobs]
            jobs = fit or [graded[0][0]]
            task = "find_jobs"
            why = (f"spontaneous application: {len(fit)} of {len(jobs_in)} open jobs graded {FIT_GRADE} or more"
                   if fit else f"spontaneous application: no open job graded {FIT_GRADE} or more, the best one "
                               "is analysed so that a recruiter can answer")
            calls.append({"tool": "semantic_matching", "job": None, "round": 0, "requirement": None,
                          "query": f"{len(jobs_in)} open jobs", "found": len(considered),
                          "result": ", ".join(f"{c['title']}: grade {c['grade']}" for c in considered[:5])})
        event = {"node": "understand", "seconds": round(time.perf_counter() - start, 2), "task": task,
                 "jobs": [j["title"] for j in jobs], "why": why}
        return self._logged(state, {"jobs": jobs, "considered": considered, "current": 0, "tool_calls": calls,
                                    **_job_reset()}, event)

    # ------------------------------------------------------------------ 2. decide which tools to use

    def plan_prompt(self, job: dict[str, Any], state: ScreeningState, weak: list[RequirementAssessment]) -> str:
        candidate = state["candidate"]
        profile = candidate.get("profile") or {}
        timeline = positions(profile)
        skills = [s.get("name") if isinstance(s, dict) else str(s) for s in profile.get("skills") or []]
        lines = [f"JOB: {job.get('title', '')}" + (f" ({job['company']})" if job.get("company") else ""),
                 f"CANDIDATE: {candidate.get('name') or 'candidate'} — positions: "
                 + ("; ".join(f"{p.title} at {p.company}" for p in timeline[:5]) or "none found")
                 + f" — total experience: {total_years(timeline)} years",
                 "Skills listed: " + (", ".join(s for s in skills[:25] if s) or "none"), "",
                 "REQUIREMENTS WITH WEAK OR MISSING EVIDENCE:"]
        evidence, queries = state.get("evidence") or {}, state.get("queries") or {}
        for a in weak[:WEAK_SHOWN]:
            lines.append(f"- {a.id} [{'must-have' if a.kind == 'must' else 'nice-to-have'}, {a.importance}]: {a.text}")
            best = (evidence.get(a.id) or [None])[0]
            lines.append(f"    best passage: [{best['label']}] ({best['score']:.2f}) {_short(best['text'])}" if best
                         else "    best passage: none")
            lines.append("    searched: " + ("; ".join(f'"{q}"' for q in queries.get(a.id) or []) or "nothing"))
        if queries.get("_hr"):
            lines.append("HR knowledge already searched: " + "; ".join(f'"{q}"' for q in queries["_hr"]))
        lines += ["", f"Planning round {state.get('round', 1)} of {self.max_rounds}. Which searches next?"]
        return "\n".join(lines)

    def clean_calls(self, calls: list[PlannedCall], job: dict[str, Any],
                    state: ScreeningState) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """The planned calls the code accepts (valid tool, requirement and query; not repeated), and the rest."""
        ids = {a.id.upper(): a.id for a in requirement_templates(job)}
        made = {key: {q.lower() for q in values} for key, values in (state.get("queries") or {}).items()}
        accepted, dropped = [], []
        for call in calls:
            tool, query = call.tool.strip().lower(), re.sub(r"\s+", " ", call.query).strip()[:200]
            requirement = ids.get((call.requirement or "").strip().upper())
            key = requirement if tool == "search_cv" else "_hr"
            reason = (f"unknown tool {call.tool!r}" if tool not in TOOLS
                      else "no query" if len(query) < 3
                      else "no requirement id" if tool == "search_cv" and requirement is None
                      else "protected attribute (KB-HR-003)" if PROTECTED.search(query)
                      else "already searched" if query.lower() in made.get(key or "", set())
                      else "too many calls" if len(accepted) >= MAX_CALLS else None)
            if reason:
                dropped.append({"tool": call.tool, "requirement": call.requirement, "query": query, "why": reason})
                continue
            made.setdefault(key or "", set()).add(query.lower())
            accepted.append({"tool": tool, "requirement": requirement, "query": query, "role": job.get("role")})
        return accepted, dropped

    def plan(self, state: ScreeningState) -> dict[str, Any]:
        start = time.perf_counter()
        job = state["jobs"][state["current"]]
        rnd = state.get("round", 0)
        if rnd == 0:  # systematic: every requirement in the CV, the job's rules in the HR knowledge base
            calls = [{"tool": "search_cv", "requirement": a.id, "query": a.text, "role": None}
                     for a in requirement_templates(job)]
            calls += [{"tool": "search_hr_knowledge", "requirement": None, "query": q, "role": role}
                      for q, role in self.analyzer.knowledge_queries(job)]
            update = {"pending": calls, "round": 1, "last_calls": len(calls), "job_started": time.perf_counter(),
                      "semantic": self._semantic_report(job, state["candidate"]).model_dump(mode="json")}
            event = {"node": "plan", "job": job["title"], "round": 0, "by": "code", "calls": len(calls),
                     "seconds": round(time.perf_counter() - start, 2),
                     "why": "first round: every requirement searched in the CV, the job's rules in the HR knowledge"}
            return self._logged(state, update, event)
        weak = self.weak_requirements(job, state)
        warnings = list(state.get("job_warnings") or [])
        prompt = self.plan_prompt(job, state, weak)
        system = PLAN_SYSTEM_PROMPT.format(tools="\n".join(f"- {t}" for t in TOOLS.values()), max_calls=MAX_CALLS)
        try:
            plan = self.llm.generate_structured(prompt, ScreeningPlan, system_prompt=system, max_tokens=PLAN_TOKENS,
                                                use_adapter=False)
            calls, dropped = self.clean_calls(plan.calls, job, state)
            thought, by = plan.thought, "model"
        except Exception as exc:  # a failed plan: equivalences of the weak must-haves in the HR rules
            error = f"{type(exc).__name__}: {str(exc)[:150]}"
            warnings.append(f"Planning round {rnd} failed ({error}): default searches")
            fallback = [PlannedCall(tool="search_hr_knowledge", requirement=a.id,
                                    query=f"{a.text}: accepted equivalent skills")
                        for a in weak if a.kind == "must"]
            calls, dropped = self.clean_calls(fallback, job, state)
            thought, by = f"planning failed ({error})", "code (planning failed)"
        record = {"round": rnd, "by": by, "thought": thought, "calls": calls, "dropped": dropped,
                  "weak": [a.id for a in weak]}
        event = {"node": "plan", "job": job["title"], "round": rnd, "by": by, "thought": thought,
                 "calls": len(calls), "dropped": len(dropped), "seconds": round(time.perf_counter() - start, 2),
                 **(self._usage() if by == "model" else {})}
        update = {"pending": calls, "round": rnd + 1, "last_calls": len(calls), "job_warnings": warnings,
                  "plans": (state.get("plans") or []) + [record], "llm_calls": state.get("llm_calls", 0) + 1}
        return self._logged(state, update, event)

    # ------------------------------------------------------------------ 3-4. call the tools, retrieve

    def act(self, state: ScreeningState) -> dict[str, Any]:
        start = time.perf_counter()
        job = state["jobs"][state["current"]]
        calls = state.get("pending") or []
        rnd = state.get("round", 1) - 1
        evidence = {k: list(v) for k, v in (state.get("evidence") or {}).items()}
        queries = {k: list(v) for k, v in (state.get("queries") or {}).items()}
        knowledge = list(state.get("knowledge") or [])
        warnings = list(state.get("job_warnings") or [])
        made: list[dict[str, Any]] = []
        cv_calls = [c for c in calls if c["tool"] == "search_cv"]
        assert self.index is not None
        for call, hits in zip(cv_calls, self.index.search_many([c["query"] for c in cv_calls],
                                                               top_k=EVIDENCE_PER_REQUIREMENT), strict=True):
            evidence.setdefault(call["requirement"], []).extend(h.as_dict() for h in hits)
            queries.setdefault(call["requirement"], []).append(call["query"])
            made.append({"tool": "search_cv", "job": job["title"], "round": rnd, "requirement": call["requirement"],
                         "query": call["query"], "found": len(hits),
                         "best": round(hits[0].score, 3) if hits else None})
        seen = {k["id"] for k in knowledge}
        search = self.analyzer.knowledge_search  # (query, job profile) -> HR knowledge chunks, reranked
        for call in (c for c in calls if c["tool"] == "search_hr_knowledge"):
            try:
                hits = search(call["query"], call.get("role")) if search is not None else []
            except Exception as exc:
                warnings.append(f"HR knowledge search failed: {str(exc)[:150]}")
                hits = []
            new = 0
            for hit in hits[:2]:
                if hit.id in seen:
                    continue
                seen.add(hit.id)
                new += 1
                meta = hit.metadata or {}
                knowledge.append({"id": hit.id, "doc_id": meta.get("document_id") or "?", "title": meta.get("title") or "",
                                  "section": meta.get("section"), "excerpt": hit.text,
                                  "score": round(float(hit.relevance_score), 4), "requirement": call.get("requirement")})
            queries.setdefault("_hr", []).append(call["query"])
            made.append({"tool": "search_hr_knowledge", "job": job["title"], "round": rnd,
                         "requirement": call.get("requirement"), "query": call["query"], "found": new,
                         "documents": sorted({(h.metadata or {}).get("document_id") or "?" for h in hits[:2]})})
        event = {"node": "act", "job": job["title"], "round": rnd, "seconds": round(time.perf_counter() - start, 2),
                 "cv_searches": len(cv_calls), "hr_searches": len(calls) - len(cv_calls),
                 "hr_excerpts": len(knowledge)}
        update = {"evidence": evidence, "queries": queries, "knowledge": knowledge, "pending": [],
                  "job_warnings": warnings, "job_calls": (state.get("job_calls") or []) + made,
                  "tool_calls": (state.get("tool_calls") or []) + made}
        return self._logged(state, update, event)

    # ------------------------------------------------------------------ 5-6. evaluate & rerank, iterate

    def evaluate(self, state: ScreeningState) -> dict[str, Any]:
        start = time.perf_counter()
        job = state["jobs"][state["current"]]
        evidence = {}
        for rid, hits in (state.get("evidence") or {}).items():
            best: dict[str, dict[str, Any]] = {}
            for h in hits:  # the same passage found by two searches: keep its best score
                if h["text"] not in best or h["score"] > best[h["text"]]["score"]:
                    best[h["text"]] = h
            evidence[rid] = sorted(best.values(), key=lambda h: h["score"], reverse=True)[:EVIDENCE_PER_REQUIREMENT]
        reranked = {**state, "evidence": evidence}
        weak = self.weak_requirements(job, reranked)  # type: ignore[arg-type]
        weak_must = [a.id for a in weak if a.kind == "must"]
        planned = state.get("round", 1) - 1  # planning rounds by the model so far
        if not weak_must:
            route, why = "analyse", "every must-have has strong evidence"
        elif planned >= self.max_rounds:
            route, why = "analyse", f"limit of {self.max_rounds} planning rounds reached"
        elif planned > 0 and not state.get("last_calls"):
            route, why = "analyse", "the agent chose no further search"
        else:
            route, why = "plan", f"weak must-haves: {', '.join(weak_must)}"
        event = {"node": "evaluate", "job": job["title"], "round": planned, "weak": [a.id for a in weak],
                 "next": route, "why": why, "seconds": round(time.perf_counter() - start, 3)}
        return self._logged(state, {"evidence": evidence, "route": route}, event)

    # ------------------------------------------------------------------ 7. final analysis, checked by code

    def _knowledge_sources(self, state: ScreeningState) -> list[KnowledgeSource]:
        return [KnowledgeSource(doc_id=k["doc_id"], title=k["title"], section=k.get("section"), excerpt=k["excerpt"])
                for k in state.get("knowledge") or []]

    def _job_trace(self, state: ScreeningState) -> dict[str, Any]:
        return {"task": "find_jobs" if not state.get("chosen_job_id") else "one_job",
                "considered": state.get("considered") or [], "planning_rounds": max(state.get("round", 1) - 1, 0),
                "plans": state.get("plans") or [], "tool_calls": state.get("job_calls") or [],
                "evidence": state.get("evidence") or {},
                "hr_knowledge": [{k: v for k, v in item.items() if k != "excerpt"}
                                 for item in state.get("knowledge") or []]}

    def analyse(self, state: ScreeningState) -> dict[str, Any]:
        start = time.perf_counter()
        job, candidate = state["jobs"][state["current"]], state["candidate"]
        source_text = candidate.get("cv_text") or profile_text(candidate["profile"])
        knowledge = self._knowledge_sources(state)
        prompt = self.analyzer.build_prompt(job, candidate["profile"], source_text, knowledge,
                                            evidence=state.get("evidence") or {})
        calls = state.get("llm_calls", 0) + 1
        try:
            judgement, stats = self.analyzer.judge(prompt, AGENT_ANALYSIS_SYSTEM_PROMPT)
        except Exception as exc:  # the pipeline goes on with the fast semantic estimate
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
            fast = state.get("semantic") or self._semantic_report(job, candidate).model_dump(mode="json")
            report = MatchReport.model_validate(fast)
            report.warnings = [*report.warnings, *(state.get("job_warnings") or []),
                               f"AI analysis failed ({error[:160]}); the fast semantic estimate is shown instead"]
            report.knowledge_sources = [KnowledgeSource(doc_id=k.doc_id, title=k.title, section=k.section,
                                                        excerpt=k.excerpt[:300]) for k in knowledge]
            report.trace = {"agent": self._job_trace(state), "prompt": prompt, "prompt_chars": len(prompt),
                            "error": error}
            event = {"node": "analyse", "job": job["title"], "seconds": round(time.perf_counter() - start, 2),
                     "error": error, "fallback": "semantic"}
            return self._logged(state, {"reports": (state.get("reports") or []) + [report.model_dump(mode="json")],
                                        "llm_calls": calls, "route": "next"}, event)
        event = {"node": "analyse", "job": job["title"], "seconds": round(time.perf_counter() - start, 2),
                 **self._usage(), "prompt_chars": len(prompt)}
        return self._logged(state, {"judgement": judgement.model_dump(mode="json"), "llm_stats": stats,
                                    "prompt": prompt, "llm_calls": calls, "route": "verify"}, event)

    @staticmethod
    def _level_from_hits(hits: list[dict[str, Any]], high: float, low: float) -> tuple[str, dict[str, Any] | None]:
        """The evidence level the retrieved passages show (same rules as the fast matching)."""
        level, chosen = "not_evidenced", None
        for h in hits:
            if h["score"] >= high:
                found = "demonstrated" if h["section"] in STRONG_SECTIONS else "listed"
            elif h["score"] >= low:
                found = ("equivalent" if h["section"] in USAGE_SECTIONS
                         else "listed" if h["section"] in LISTING_SECTIONS else "not_evidenced")
            else:
                found = "not_evidenced"
            if chosen is None or LEVEL_ORDER.index(found) > LEVEL_ORDER.index(level):
                level, chosen = found, h
        return level, (chosen if level != "not_evidenced" else None)

    def verify_assessments(self, assessments: list[RequirementAssessment], evidence: dict[str, list[dict[str, Any]]],
                           semantic: dict[str, Any] | None, source_text: str) -> list[dict[str, Any]]:
        """Check the model's claims (rubric KB-RR-002); returns what was changed and why."""
        high, low = self._thresholds()
        fast = {r["id"]: r for r in (semantic or {}).get("requirements") or []}
        changes: list[dict[str, Any]] = []

        def change(a: RequirementAssessment, level: str, why: str, hit: dict[str, Any] | None = None,
                   evidence_text: str | None = None, source: str | None = None) -> None:
            changes.append({"id": a.id, "from": a.level, "to": level, "why": why})
            a.level, a.note = level, why
            if level == "not_evidenced":
                a.evidence, a.evidence_source, a.grounded = None, None, True
            elif hit is not None or evidence_text:
                a.evidence = hit["text"] if hit is not None else evidence_text
                a.evidence_source = hit["label"] if hit is not None else source
                a.grounded = is_grounded(a.evidence, source_text)

        # 1. a positive claim needs a quote that supports the requirement (cross-encoder relevance)
        claims = [a for a in assessments if a.level in rubric.POSITIVE and a.evidence and a.grounded]
        scores = self.index.relevance([(a.text, a.evidence or "") for a in claims]) if self.index else None
        for a, score in zip(claims, scores or [], strict=False):
            a.confidence = round(float(score), 3)
            if score < low:
                level, hit = self._level_from_hits(evidence.get(a.id) or [], high, low)
                change(a, level, f"the quoted excerpt does not support the requirement (relevance {score:.2f}): "
                                 "level from the CV passages retrieved for it", hit)
        # 2. "demonstrated" needs usage context: a quote found only in a skills section or the summary is "listed"
        for a in assessments:
            if a.level == "demonstrated" and a.evidence and self.index is not None:
                where = self.index.sections_of(a.evidence)
                if where and where <= LISTING_SECTIONS:
                    change(a, "listed", "the excerpt is only in a skills section or the summary: listed (KB-RR-002)")
        # 3. strong evidence the model did not use (the fast matching's "demonstrated", precise on the dev pairs)
        for a in assessments:
            f = fast.get(a.id) or {}
            if a.level in ("not_evidenced", "listed") and f.get("level") == "demonstrated" \
                    and (f.get("confidence") or 0) >= high:
                hit = next((h for h in evidence.get(a.id) or []
                            if h["section"] in STRONG_SECTIONS and h["score"] >= low), None)
                if hit is None and not f.get("evidence"):
                    continue  # no excerpt to show the recruiter: the model's level stays
                change(a, "demonstrated", f"strong CV evidence the model did not use (relevance "
                                          f"{f.get('confidence'):.2f}, {f.get('evidence_source') or 'CV'})",
                       hit, f.get("evidence"), f.get("evidence_source"))
                if a.min_years and a.years_found is None:
                    a.years_found = f.get("years_found")
        # 4. a minimum of years the model did not count: the relevant positions found by the fast matching
        for a in assessments:
            f = fast.get(a.id) or {}
            if a.min_years and a.years_found is None and a.level in rubric.POSITIVE \
                    and f.get("years_found") is not None:
                a.years_found = f["years_found"]
                changes.append({"id": a.id, "from": a.level, "to": a.level,
                                "why": f"years from the relevant positions: {a.years_found:g}"})
        for a in assessments:
            rubric.finalize(a)
        return changes

    def verify(self, state: ScreeningState) -> dict[str, Any]:
        start = time.perf_counter()
        job, candidate = state["jobs"][state["current"]], state["candidate"]
        source_text = candidate.get("cv_text") or profile_text(candidate["profile"])
        judgement = MatchJudgement.model_validate(state.get("judgement") or {})
        assessments = self.analyzer.apply_judgement(job, judgement, source_text)
        changes = self.verify_assessments(assessments, state.get("evidence") or {}, state.get("semantic"),
                                          source_text)
        warnings = list(state.get("job_warnings") or [])
        if changes:
            warnings.append(f"The agent's checks changed {len({c['id'] for c in changes})} requirement(s) of the "
                            "model's judgement (see the report's trace)")
        prompt = state.get("prompt") or ""
        trace = {"agent": self._job_trace(state), "prompt": prompt, "prompt_chars": len(prompt),
                 **(state.get("llm_stats") or {}), "checks": {"verification": changes}}
        report = self.analyzer.make_report(job, candidate, judgement, assessments, self._knowledge_sources(state),
                                           warnings, trace, state.get("job_started") or start, method="agent")
        event = {"node": "verify", "job": job["title"], "seconds": round(time.perf_counter() - start, 2),
                 "changes": len(changes), "grade": report.grade, "score": report.score}
        return self._logged(state, {"reports": (state.get("reports") or []) + [report.model_dump(mode="json")]},
                            event)

    def next_job(self, state: ScreeningState) -> dict[str, Any]:
        current = state["current"] + 1
        if current < len(state["jobs"]):
            event = {"node": "next", "job": state["jobs"][current]["title"], "next": "plan"}
            return self._logged(state, {"current": current, "route": "plan", **_job_reset()}, event)
        return self._logged(state, {"route": "end"}, {"node": "next", "next": "end",
                                                      "reports": len(state.get("reports") or [])})

    # ------------------------------------------------------------------ run

    def run(self, candidate: dict[str, Any], jobs: list[dict[str, Any]],
            chosen_job_id: str | None = None) -> dict[str, Any]:
        """``candidate``: {"id", "name", "profile", "cv_text"}; ``jobs``: the job applied to, or the open jobs."""
        if not jobs:
            raise ValueError("No job to screen the candidate against: create (or reopen) a job first")
        start = time.perf_counter()
        self.index = CVEvidenceIndex(self.embedder, self.reranker, candidate.get("cv_text"), candidate.get("profile"))
        self._semantic_reports = {}
        state: ScreeningState = {"candidate": candidate, "open_jobs": jobs, "chosen_job_id": chosen_job_id,
                                 "reports": [], "tool_calls": [], "warnings": [], "llm_calls": 0, "timeline": []}
        if self.graph is not None:
            state = self.graph.invoke(state, {"recursion_limit": 25 * self.max_jobs + 10})
        else:
            state.update(self.understand(state))
            while True:
                state.update(self.plan(state))
                state.update(self.act(state))
                state.update(self.evaluate(state))
                if self.route(state) == "plan":
                    continue
                state.update(self.analyse(state))
                if self.route(state) == "verify":
                    state.update(self.verify(state))
                state.update(self.next_job(state))
                if self.route(state) == "end":
                    break
        reports = [MatchReport.model_validate(r) for r in state.get("reports") or []]
        return {"reports": reports, "considered": state.get("considered") or [],
                "tool_calls": state.get("tool_calls") or [], "llm_calls": state.get("llm_calls", 0),
                "timeline": state.get("timeline") or [], "warnings": state.get("warnings") or [],
                "passages": len(self.index.passages), "passages_from": "cv_text" if self.index.from_cv_text
                else "profile", "seconds": round(time.perf_counter() - start, 1),
                "model": getattr(self.llm, "model", None), "graph": "langgraph" if self.graph is not None else "loop"}
