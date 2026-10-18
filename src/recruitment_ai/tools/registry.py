"""Tools of the AI assistant (diagram part 5, "LangGraph agent with tools").

| Tool                 | Diagram box               | What it does                                                |
|----------------------|---------------------------|-------------------------------------------------------------|
| search_candidates    | CV search tool            | BGE-M3 search in all CVs (Qdrant) + BGE reranker            |
| search_jobs          | Job description search    | the same over the job descriptions                          |
| search_hr_knowledge  | HR knowledge search tool  | policies, scoring rubric, job profiles, skills (RAG)        |
| get_candidate        | Candidate DB tool         | a structured profile from PostgreSQL                        |
| get_job              | Candidate DB tool (jobs)  | a job and its requirements from PostgreSQL                  |
| rank_candidates      | Semantic matching tool    | all candidates of a job, graded with the rubric             |
| match_candidate      | Semantic matching tool    | one candidate vs one job, requirement by requirement        |
| rerank_candidates    | Reranking tool            | orders candidates for a precise criterion (cross-encoder)   |

Arguments are validated (Pydantic) before a tool runs; a failure becomes an observation the agent
can react to, never a crash. Tools only read: decisions stay with recruiters. Contact details
(email, phone) are never given to the model (data minimisation, KB-HR-004).
"""

import json
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    BeforeValidator,
    Field,
    ValidationError,
)
from sqlalchemy.orm import Session

from recruitment_ai.database.repositories import (
    AnalysisRepository,
    CandidateRepository,
    JobRepository,
)
from recruitment_ai.matching.evidence import candidate_snippets, positions, total_years
from recruitment_ai.utils.logging import get_logger

logger = get_logger(__name__)
MAX_RERANK_CANDIDATES = 50


@dataclass
class ToolContext:
    """What the tools work with: a database session and the application's models and stores."""

    db: Session
    container: Any


class ToolResult(BaseModel):
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    data: Any = None
    error: str | None = None
    sources: list[dict[str, Any]] = Field(default_factory=list)
    seconds: float = 0.0

    def observation(self, limit: int = 2500) -> str:
        """What the model reads: compact JSON, cut to ``limit`` characters."""
        text = json.dumps(self.data, ensure_ascii=False, default=str) if self.ok else f"ERROR: {self.error}"
        return text if len(text) <= limit else text[:limit] + " …(truncated)"


class AgentTool:
    def __init__(self, name: str, description: str, args_model: type[BaseModel],
                 func: Callable[..., tuple[Any, list[dict[str, Any]]]]) -> None:
        self.name, self.description, self.args_model, self.func = name, description, args_model, func

    def signature(self) -> str:
        params = []
        for field_name, info in self.args_model.model_fields.items():
            if info.is_required():
                params.append(field_name)
            else:
                params.append(f"{field_name}={info.get_default(call_default_factory=True)!r}")
        return f"{self.name}({', '.join(params)})"

    def run(self, ctx: ToolContext, arguments: dict[str, Any] | None) -> ToolResult:
        start = time.perf_counter()
        arguments = dict(arguments or {})
        try:
            args = self.args_model.model_validate(arguments)
            data, sources = self.func(ctx, **args.model_dump())
            result = ToolResult(tool=self.name, arguments=args.model_dump(), ok=True, data=data, sources=sources)
        except ValidationError as exc:
            details = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            result = ToolResult(tool=self.name, arguments=arguments, ok=False,
                                error=f"invalid arguments ({details}); expected {self.signature()}")
        except Exception as exc:  # error boundary: the agent sees the error and can try another way
            logger.warning(f"Tool {self.name} failed: {exc}")
            result = ToolResult(tool=self.name, arguments=arguments, ok=False, error=f"{type(exc).__name__}: {exc}")
        result.seconds = round(time.perf_counter() - start, 2)
        return result


# --------------------------------------------------------------------------- arguments


# A 7B model does not always use the exact argument names or limits: common variants are accepted
# ("q" for "query", "name" for "candidate"...), result counts are clamped, "A, B" becomes a list.


def _clamp(high: int) -> AfterValidator:
    return AfterValidator(lambda value: max(1, min(value, high)))


def _as_list(value: Any) -> Any:
    if isinstance(value, str):
        return [part.strip() for part in re.split(r",|;| and ", value) if part.strip()]
    return value


QUERY = AliasChoices("query", "q", "question", "text", "search", "criterion")
CANDIDATE = AliasChoices("candidate", "name", "candidate_name", "candidate_id", "id")
JOB = AliasChoices("job", "job_title", "job_id", "title", "position")


class SearchArgs(BaseModel):
    query: str = Field(min_length=2, validation_alias=QUERY)
    top_k: Annotated[int, _clamp(10)] = 5


class KnowledgeArgs(BaseModel):
    query: str = Field(min_length=2, validation_alias=QUERY)
    role: str | None = None
    top_k: Annotated[int, _clamp(6)] = 3


class CandidateArgs(BaseModel):
    candidate: str = Field(min_length=1, validation_alias=CANDIDATE, description="candidate id or name")


class JobArgs(BaseModel):
    job: str = Field(min_length=1, validation_alias=JOB, description="job id or title")


class RankArgs(BaseModel):
    job: str = Field(min_length=1, validation_alias=JOB)
    top_k: Annotated[int, _clamp(20)] = 5


class MatchArgs(BaseModel):
    candidate: str = Field(min_length=1, validation_alias=CANDIDATE)
    job: str = Field(min_length=1, validation_alias=JOB)


class RerankArgs(BaseModel):
    query: str = Field(min_length=2, validation_alias=QUERY)
    candidates: Annotated[list[str], BeforeValidator(_as_list)] = Field(
        default_factory=list, description="ids or names; empty = all candidates")
    top_k: Annotated[int, _clamp(20)] = 5


# --------------------------------------------------------------------------- helpers


def _candidate(ctx: ToolContext, reference: str) -> Any:
    found = CandidateRepository(ctx.db).find(reference)
    if found is None:
        raise LookupError(f"no candidate matches {reference!r}: use search_candidates to find one")
    return found


def _job(ctx: ToolContext, reference: str) -> Any:
    found = JobRepository(ctx.db).find(reference)
    if found is None:
        raise LookupError(f"no job matches {reference!r}: see the list of jobs, or use search_jobs")
    return found


def _candidate_source(candidate_id: str, name: str | None) -> dict[str, Any]:
    return {"type": "candidate", "id": candidate_id, "title": name or "Unnamed candidate"}


def _job_source(job: Any) -> dict[str, Any]:
    return {"type": "job", "id": job.id, "title": job.title}


def _clip(text: str | None, limit: int) -> str | None:
    if not text:
        return text
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


# --------------------------------------------------------------------------- tools


def search_candidates(ctx: ToolContext, query: str, top_k: int) -> tuple[Any, list[dict[str, Any]]]:
    hits = ctx.container.index.search_candidates(query, top_k=top_k)
    data = [{"candidate_id": h["candidate_id"], "name": h["name"], "evidence": _clip(h["evidence"], 300),
             "where": h["label"], "relevance": h["score"]} for h in hits]
    return data, [_candidate_source(h["candidate_id"], h["name"]) for h in hits]


def search_jobs(ctx: ToolContext, query: str, top_k: int) -> tuple[Any, list[dict[str, Any]]]:
    hits = ctx.container.index.search_jobs(query, top_k=min(top_k, 5))
    data = [{"job_id": h["job_id"], "title": h["name"], "evidence": _clip(h["evidence"], 400),
             "relevance": h["score"]} for h in hits]
    return data, [{"type": "job", "id": h["job_id"], "title": h["name"]} for h in hits]


def search_hr_knowledge(ctx: ToolContext, query: str, role: str | None, top_k: int) -> tuple[Any, list[dict[str, Any]]]:
    hits = ctx.container.index.search_knowledge(query, role=role or None, top_k=top_k)
    data, sources = [], []
    for h in hits:
        meta = h.metadata or {}
        doc_id = meta.get("document_id") or "?"
        data.append({"doc_id": doc_id, "title": meta.get("title"), "section": meta.get("section"),
                     "text": _clip(h.text, 700), "relevance": round(h.relevance_score, 4)})
        sources.append({"type": "knowledge", "id": doc_id,
                        "title": f"{meta.get('title') or doc_id} / {meta.get('section') or ''}".strip(" /")})
    return data, sources


def get_candidate(ctx: ToolContext, candidate: str) -> tuple[Any, list[dict[str, Any]]]:
    c = _candidate(ctx, candidate)
    profile = c.profile or {}
    timeline = positions(profile)
    analyses = AnalysisRepository(ctx.db).filter(candidate_id=c.id, limit=10)
    data = {
        "candidate_id": c.id, "name": c.full_name, "location": c.location,
        "summary": _clip(profile.get("summary"), 400),
        "total_years_experience": total_years(timeline),
        "experience": [{"title": p.title, "company": p.company, "from": p.start, "to": p.end, "years": p.years}
                       for p in timeline][:10],
        "education": [" ".join(str(e.get(k)) for k in ("degree", "field_of_study", "institution", "end_date")
                               if e.get(k)) for e in profile.get("education") or []][:5],
        "projects": [p.get("title") for p in profile.get("projects") or [] if p.get("title")][:8],
        "skills": [s.get("name") if isinstance(s, dict) else str(s) for s in profile.get("skills") or []][:40],
        "languages": [x.get("language") for x in profile.get("languages") or [] if isinstance(x, dict)][:6],
        "certifications": [x.get("name") if isinstance(x, dict) else str(x)
                           for x in profile.get("certifications") or []][:8],
        "analyses": [{"job": a.job.title if a.job else a.job_id, "grade": a.final_grade if a.final_grade is not None
                      else a.grade, "score": a.score, "status": a.status} for a in analyses],
    }
    return data, [_candidate_source(c.id, c.full_name)]


def get_job(ctx: ToolContext, job: str) -> tuple[Any, list[dict[str, Any]]]:
    j = _job(ctx, job)
    data = {"job_id": j.id, "title": j.title, "company": j.company, "location": j.location, "role": j.role,
            "requirements": [{"id": r.get("id"), "text": r.get("requirement_text"),
                              "type": "must-have" if r.get("requirement_type") == "required" else "nice-to-have",
                              "importance": r.get("importance"), "min_years": r.get("min_years_experience")}
                             for r in j.requirements or []],
            "description": _clip(j.description, 600)}
    return data, [_job_source(j)]


def rank_candidates(ctx: ToolContext, job: str, top_k: int) -> tuple[Any, list[dict[str, Any]]]:
    from recruitment_ai.services.matching import rank

    j = _job(ctx, job)
    rows = rank(ctx.db, ctx.container, j.id)
    data = {"job": j.title, "candidates_ranked": len(rows), "method": "fast semantic matching (rubric KB-RR-002)",
            "top": [{"candidate_id": r["candidate_id"], "name": r["name"], "grade": r["grade"],
                     "grade_label": r["grade_label"], "score": r["score"],
                     "must_haves_met": f"{r['met']}/{r['must_total']}", "gaps": r["gaps"][:2],
                     "review_status": r["analysis_status"]} for r in rows[:top_k]]}
    return data, [_job_source(j)] + [_candidate_source(r["candidate_id"], r["name"]) for r in rows[:top_k]]


def match_candidate(ctx: ToolContext, candidate: str, job: str) -> tuple[Any, list[dict[str, Any]]]:
    from recruitment_ai.services.jobs import job_payload
    from recruitment_ai.services.matching import candidate_payload

    c, j = _candidate(ctx, candidate), _job(ctx, job)
    stored = AnalysisRepository(ctx.db).filter(candidate_id=c.id, job_id=j.id, limit=1)
    if stored:  # an analysis already exists (maybe reviewed by a recruiter): use it
        a = stored[0]
        report = a.report or {}
        requirements = report.get("requirements") or []
        method = "stored AI analysis" if a.method in ("llm", "agent") else "stored semantic analysis"
        status = a.status
        decision = a.reviews[-1].notes if a.reviews else None
    else:
        r = ctx.container.semantic.match(job_payload(j), candidate_payload(c))
        report = r.model_dump(mode="json")
        requirements = report["requirements"]
        method, status, decision = "fast semantic matching", "not analysed yet", None
    data = {"candidate": c.full_name, "job": j.title, "method": method, "grade": report.get("grade"),
            "grade_label": report.get("grade_label"), "score": report.get("score"), "review_status": status,
            "recruiter_notes": decision,
            "requirements": [{"id": x.get("id"), "requirement": _clip(x.get("text"), 90), "status": x.get("status"),
                              "level": x.get("level"), "evidence": _clip(x.get("evidence"), 160)}
                             for x in requirements],
            "gaps": (report.get("gaps") or [])[:4]}
    return data, [_candidate_source(c.id, c.full_name), _job_source(j)]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def rerank_candidates(ctx: ToolContext, query: str, candidates: list[str], top_k: int) -> tuple[Any, list[dict[str, Any]]]:
    repo = CandidateRepository(ctx.db)
    if candidates:
        chosen, missing = [], []
        for ref in candidates:
            found = repo.find(str(ref))
            if found is None:
                missing.append(ref)
            elif found not in chosen:
                chosen.append(found)
    else:
        chosen, missing = repo.list(limit=MAX_RERANK_CANDIDATES), []
    if not chosen:
        raise LookupError("no candidate to rerank" + (f" (not found: {', '.join(map(str, missing))})" if missing else ""))
    pairs, owners = [], []
    for c in chosen:
        for s in candidate_snippets(c.profile or {}):
            if s.section != "summary":
                pairs.append([query, s.text])
                owners.append((c, s))
    scores = ctx.container.reranker.score_pairs(pairs) if pairs else []
    method = "BGE cross-encoder reranker"
    if scores is None:  # no cross-encoder configured: cosine similarity of BGE-M3 embeddings
        vectors = ctx.container.embedder.embed_documents([query] + [p[1] for p in pairs])
        scores = [_cosine(vectors[0], v) for v in vectors[1:]]
        method = "BGE-M3 cosine similarity (no reranker configured)"
    best: dict[str, tuple[float, Any, Any]] = {}
    for (c, s), score in zip(owners, scores, strict=True):
        if c.id not in best or score > best[c.id][0]:
            best[c.id] = (score, c, s)
    ranked = sorted(best.values(), key=lambda item: item[0], reverse=True)[:top_k]
    data = {"criterion": query, "method": method, "not_found": missing,
            "ranking": [{"candidate_id": c.id, "name": c.full_name, "relevance": round(float(score), 4),
                         "best_evidence": _clip(s.text, 220), "where": s.label} for score, c, s in ranked]}
    return data, [_candidate_source(c.id, c.full_name) for _, c, _ in ranked]


# --------------------------------------------------------------------------- registry

TOOLS = [
    AgentTool("search_candidates", "Semantic search in all candidates' CVs (positions, projects, skills, education); "
              "returns the best evidence per candidate. Use it to find people with a skill or an experience.",
              SearchArgs, search_candidates),
    AgentTool("search_jobs", "Semantic search in the job descriptions and their requirements.", SearchArgs, search_jobs),
    AgentTool("search_hr_knowledge", "Search the company's HR knowledge base: recruitment policies, fit scoring "
              "rubric, experience rules, interview guidelines, salary levels, job profiles, skill definitions. "
              "Optional role: ml_engineer, nlp_llm_engineer, backend_developer, devops_cloud, "
              "data_analyst_scientist.", KnowledgeArgs, search_hr_knowledge),
    AgentTool("get_candidate", "A candidate's structured profile from the database (experience timeline with "
              "years, education, projects, skills, languages, certifications, existing analyses).",
              CandidateArgs, get_candidate),
    AgentTool("get_job", "A job and its requirements (must-haves, nice-to-haves, minimum years).", JobArgs, get_job),
    AgentTool("rank_candidates", "Rank all candidates for a job with the fit scoring rubric (grade 0-3, score, "
              "must-haves met, gaps). Use it for 'who are the best candidates for job X'.", RankArgs, rank_candidates),
    AgentTool("match_candidate", "Compare one candidate with one job requirement by requirement (status, evidence "
              "level, CV evidence, gaps); uses the stored analysis when there is one.", MatchArgs, match_candidate),
    AgentTool("rerank_candidates", "Order candidates (given ids/names, or all) by how well their CV evidence fits a "
              "precise criterion, with the cross-encoder reranker.", RerankArgs, rerank_candidates),
]


class ToolRegistry:
    def __init__(self, tools: list[AgentTool] | None = None) -> None:
        self._tools = {t.name: t for t in (tools if tools is not None else TOOLS)}

    def register(self, tool: AgentTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> AgentTool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def list_tools(self) -> list[AgentTool]:
        return list(self._tools.values())

    def describe(self) -> str:
        return "\n".join(f"- {t.signature()}: {t.description}" for t in self._tools.values())

    def run(self, name: str, ctx: ToolContext, arguments: dict[str, Any] | None) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(tool=name, arguments=arguments or {}, ok=False,
                              error=f"unknown tool {name!r}; available: {', '.join(self.names())}")
        return tool.run(ctx, arguments)
