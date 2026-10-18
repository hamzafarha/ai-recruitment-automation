"""Full evidence-backed analysis (DSO3 + diagram parts 5-6: contextualization and LLM analysis).

1. Retrieval (RAG): the HR knowledge chunks relevant to this job (job profile and accepted
   equivalences, the fit scoring rubric, the experience-counting rules) come from Qdrant.
2. Contextualization: one prompt combines those rules, the job requirements, the candidate's
   experience timeline (years computed with KB-RR-003) and the CV text.
3. LLM analysis (Qwen2.5-7B, base model: the fine-tuned adapter is for CV extraction): for each
   requirement, an evidence level, the CV excerpt that supports it and the relevant years; plus
   strengths, gaps, interview questions and a short summary.
4. Checks done by code, not by the model: every excerpt must be found in the CV (otherwise a
   "demonstrated" claim only counts as "listed"); grade, score and action come from the rubric.
If the LLM fails, the fast semantic estimate is returned, with a warning.

This fixed chain is the "llm" method of the Matching page and the baseline of the evaluation. The
pipeline's screening agent (agents/screening.py) reuses its prompt, judgement and report parts,
with the CV evidence it retrieved for each requirement (``build_prompt(..., evidence=...)``).
"""

import re
import time
from typing import Any

from pydantic import Field, field_validator

from recruitment_ai.extraction.extractor import _Lenient
from recruitment_ai.matching import rubric
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

MAX_CV_CHARS = 7000
MAX_OUTPUT_TOKENS = 1800
UNGROUNDED_NOTE = "excerpt not found in the CV: counted as listed"
NOT_ASSESSED_NOTE = "not assessed by the model"
LEVEL_ALIASES = {
    "demonstrated": "demonstrated", "met": "demonstrated", "strong": "demonstrated",
    "equivalent": "equivalent", "transferable": "equivalent",
    "listed": "listed", "partial": "listed", "mentioned": "listed",
    "not_evidenced": "not_evidenced", "not evidenced": "not_evidenced", "missing": "not_evidenced",
    "none": "not_evidenced", "absent": "not_evidenced", "no": "not_evidenced",
}


def normalize_level(value: Any) -> str:
    key = re.sub(r"[\s\-]+", " ", str(value or "")).strip().lower()
    return LEVEL_ALIASES.get(key) or LEVEL_ALIASES.get(key.replace(" ", "_")) or "not_evidenced"


# --------------------------------------------------------------------------- LLM-facing schema


class RequirementJudgement(_Lenient):
    id: str = ""
    level: str = Field(default="not_evidenced", description="demonstrated | equivalent | listed | not_evidenced")
    evidence: str | None = Field(default=None, description="Short excerpt copied from the CV (max 25 words)")
    years: float | None = Field(default=None, description="Relevant years, only for requirements with a minimum")

    @field_validator("years", mode="before")
    @classmethod
    def _years(cls, value: Any) -> Any:
        if isinstance(value, str):
            match = re.search(r"\d+(\.\d+)?", value.replace(",", "."))
            return float(match.group(0)) if match else None
        return value


class MatchJudgement(_Lenient):
    requirements: list[RequirementJudgement] = Field(default_factory=list, max_length=30)
    strengths: list[str] = Field(default_factory=list, max_length=4)
    gaps: list[str] = Field(default_factory=list, max_length=4)
    interview_questions: list[str] = Field(default_factory=list, max_length=4)
    summary: str | None = None

    @field_validator("requirements", "strengths", "gaps", "interview_questions", mode="before")
    @classmethod
    def _lenient_lists(cls, value: Any, info: Any) -> Any:
        if value is None:
            return []
        if not isinstance(value, list):
            return value
        limit = 30 if info.field_name == "requirements" else 4
        return [v for v in value if v is not None][:limit]


ANALYSIS_SYSTEM_PROMPT = """You are the screening assistant of a recruitment team. You compare ONE candidate's CV with ONE job description, applying the company's fit scoring rubric (KB-RR-002) and experience rules (KB-RR-003) given in the context.

For each requirement, choose its evidence level:
- demonstrated: the skill appears in a job, project or achievement with concrete context;
- equivalent: an equivalent or transferable skill is demonstrated (for example Azure for AWS, PostgreSQL for MySQL; use the equivalences of the job profile);
- listed: the skill only appears in a skills list or a summary, with no usage context;
- not_evidenced: nothing in the CV supports it. This is a gap to check in interview, not a proven weakness.

Rules:
1. Use only the CV. Never invent. "evidence" = a short excerpt (at most 25 words) copied from the CV; null when not_evidenced.
2. "years" only for requirements with a minimum number of years: the candidate's RELEVANT years (positions that used that skill), counted with the experience rules: internships 50 % (at most 1 year), overlapping positions once, projects and studies not counted. Use the experience timeline. Otherwise null.
3. Synonyms count as the same skill ("ML" = "machine learning" = "apprentissage automatique"). A long skills list without matching experience is "listed", not "demonstrated".
4. Never use or mention age, gender, origin, nationality, family situation, religion, health or any other protected attribute.
5. strengths and gaps: at most 4 each, short. interview_questions: at most 4, each targeting one gap.
6. summary: two factual sentences. Never decide the hiring: a recruiter decides.
7. Assess every requirement, using its id. The CV may be in French: answer in English.

Return only the JSON object."""


class EvidenceAnalyzer:
    """LLM analysis of one candidate for one job, grounded in the HR knowledge base."""

    def __init__(self, llm: Any, knowledge_search: Any | None = None, semantic: Any | None = None) -> None:
        self.llm = llm
        self.knowledge_search = knowledge_search  # (query, role) -> list of chunks (RerankResult)
        self.semantic = semantic                  # SemanticMatcher, fallback when the LLM fails

    # ------------------------------------------------------------------ retrieval (RAG)

    @staticmethod
    def knowledge_queries(job: dict[str, Any]) -> list[tuple[str, str | None]]:
        """The searches made in the HR knowledge base for a job: (query, job-profile filter)."""
        return [(f"{job.get('title', '')}: required skills, accepted equivalences and interview topics",
                 job.get("role")),
                ("evidence levels demonstrated listed equivalent and fit grades", None),
                ("counting years of relevant experience, internships and seniority", None)]

    def retrieve_knowledge(self, job: dict[str, Any]) -> tuple[list[KnowledgeSource], list[str]]:
        """HR knowledge relevant to this job: job profile/equivalences, scoring rubric, experience rules."""
        if self.knowledge_search is None:
            return [], ["HR knowledge base not available: analysis made without company rules"]
        sources, seen, warnings = [], set(), []
        for query, role in self.knowledge_queries(job):
            try:
                hits = self.knowledge_search(query, role)
            except Exception as exc:
                warnings.append(f"Knowledge retrieval failed: {exc}")
                break
            for hit in hits[:2]:
                if hit.id in seen:
                    continue
                seen.add(hit.id)
                meta = hit.metadata or {}
                sources.append(KnowledgeSource(doc_id=meta.get("document_id") or "?", title=meta.get("title") or "",
                                               section=meta.get("section"), excerpt=hit.text))
        return sources, warnings

    # ------------------------------------------------------------------ prompt

    @staticmethod
    def build_prompt(job: dict[str, Any], profile: dict[str, Any], cv_text: str, knowledge: list[KnowledgeSource],
                     evidence: dict[str, list[dict[str, Any]]] | None = None) -> str:
        """``evidence`` (screening agent): requirement id -> CV passages retrieved for it, best first
        ({"label", "text", "score"}); without it the prompt is the fixed analysis' one."""
        header = ("REQUIREMENTS:" if evidence is None
                  else "REQUIREMENTS, each with the CV passages retrieved for it (best first, relevance 0-1):")
        lines = [f"JOB: {job.get('title', '')}" + (f" ({job['company']})" if job.get("company") else ""), "",
                 header]
        for a in requirement_templates(job):
            facets = ["must-have" if a.kind == "must" else "nice-to-have", f"{a.importance} importance"]
            if a.min_years:
                facets.append(f"minimum {a.min_years:g} years")
            lines.append(f"- {a.id} [{', '.join(facets)}]: {a.text}")
            if evidence is not None:
                passages = evidence.get(a.id) or []
                lines += [f"    · [{p.get('label') or 'CV'}] ({float(p.get('score') or 0):.2f}) {p.get('text', '')}"
                          for p in passages] or ["    · no passage found"]
        if knowledge:
            lines += ["", "COMPANY RULES (retrieved from the HR knowledge base):"]
            lines += [f"[{k.doc_id} — {k.title} / {k.section}] {k.excerpt[:700]}" for k in knowledge]
        timeline = positions(profile)
        if timeline:
            lines += ["", "CANDIDATE EXPERIENCE TIMELINE (computed from the CV dates):"]
            for p in timeline:
                years = f"{p.years} years" if p.years is not None else "duration unknown"
                lines.append(f"- {p.title} at {p.company}: {p.start or '?'} → {p.end or '?'} ({years}"
                             + (", internship: counts 50 %" if p.internship else "") + ")")
            lines.append(f"Total professional experience (KB-RR-003): {total_years(timeline)} years")
        lines += ["", "CANDIDATE CV:", "=== CV ===", cv_text[:MAX_CV_CHARS], "=== END OF CV ==="]
        return "\n".join(lines)

    # ------------------------------------------------------------------ analysis

    def analyze(self, job: dict[str, Any], candidate: dict[str, Any]) -> MatchReport:
        """``candidate``: {"id", "name", "profile" (CandidateProfile dict), "cv_text" (optional)}."""
        start = time.perf_counter()
        profile = candidate["profile"]
        source_text = candidate.get("cv_text") or profile_text(profile)
        knowledge, warnings = self.retrieve_knowledge(job)
        prompt = self.build_prompt(job, profile, source_text, knowledge)
        # what happened inside the analysis, kept with the report (Reports and "Matching internals" pages)
        trace: dict[str, Any] = {"knowledge_queries": [{"query": q, "job_profile": r}
                                                       for q, r in self.knowledge_queries(job)],
                                 "prompt": prompt, "prompt_chars": len(prompt)}
        try:
            judgement, stats = self.judge(prompt)
        except Exception as exc:
            if self.semantic is None:
                raise
            return self.fallback(job, candidate, exc, knowledge, trace)
        trace.update(stats)
        assessments = self.apply_judgement(job, judgement, source_text)
        return self.make_report(job, candidate, judgement, assessments, knowledge, warnings, trace, start)

    def judge(self, prompt: str,
              system_prompt: str = ANALYSIS_SYSTEM_PROMPT) -> tuple[MatchJudgement, dict[str, Any]]:
        """One call to Qwen (base model): the judgement, and what it cost (for the trace)."""
        llm_start = time.perf_counter()
        judgement = self.llm.generate_structured(prompt, MatchJudgement, system_prompt=system_prompt,
                                                 max_tokens=MAX_OUTPUT_TOKENS, use_adapter=False)
        stats = dict(getattr(self.llm, "last_stats", None) or {})
        return judgement, {"raw_answer": stats.get("raw_answer"),
                           "llm_seconds": round(time.perf_counter() - llm_start, 1),
                           "prompt_tokens": stats.get("prompt_tokens"), "output_tokens": stats.get("output_tokens")}

    def fallback(self, job: dict[str, Any], candidate: dict[str, Any], exc: Exception,
                 knowledge: list[KnowledgeSource], trace: dict[str, Any]) -> MatchReport:
        """The fast semantic estimate, when the model could not judge."""
        report = self.semantic.match(job, candidate)
        report.warnings.append(f"AI analysis failed ({type(exc).__name__}: {str(exc)[:160]}); "
                               "the fast semantic estimate is shown instead")
        report.knowledge_sources = knowledge
        report.trace = {**trace, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        return report

    @staticmethod
    def apply_judgement(job: dict[str, Any], judgement: MatchJudgement,
                        source_text: str) -> list[RequirementAssessment]:
        """The model's level, excerpt and years for each requirement, every excerpt checked against the CV."""
        by_id = {j.id.strip().upper(): j for j in judgement.requirements if j.id.strip()}
        assessments = requirement_templates(job)
        in_order = len(judgement.requirements) == len(assessments)  # ids forgotten: same order as asked
        for i, a in enumerate(assessments):
            j = by_id.get(a.id.upper()) or (judgement.requirements[i] if in_order else None)
            if j is None:
                a.note = NOT_ASSESSED_NOTE
                rubric.finalize(a)
                continue
            a.level = normalize_level(j.level)
            a.evidence = (j.evidence or "").strip() or None
            if a.min_years:
                a.years_found = j.years
            if a.level != "not_evidenced":
                a.grounded = is_grounded(a.evidence, source_text)
                if not a.grounded and a.level in rubric.POSITIVE:
                    a.note = UNGROUNDED_NOTE
                    a.level = "listed"
            else:
                a.evidence = None
            rubric.finalize(a)
        return assessments

    def make_report(self, job: dict[str, Any], candidate: dict[str, Any], judgement: MatchJudgement,
                    assessments: list[RequirementAssessment], knowledge: list[KnowledgeSource], warnings: list[str],
                    trace: dict[str, Any], start: float, method: str = "llm") -> MatchReport:
        """Grade, score and action from the rubric, and the report with the model's texts."""
        g = rubric.grade(assessments)
        grade_value = int(g["grade"])
        downgraded = [a.id for a in assessments if a.note == UNGROUNDED_NOTE]
        if downgraded:
            warnings.append("Some excerpts quoted by the model were not found in the CV: those requirements were "
                            "counted as only listed")
        trace["checks"] = {**(trace.get("checks") or {}), "downgraded_ungrounded": downgraded,
                           "not_assessed": [a.id for a in assessments if a.note == NOT_ASSESSED_NOTE], "rubric": g}
        return MatchReport(
            candidate_id=candidate.get("id"), candidate_name=candidate.get("name"),
            job_id=job.get("id"), job_title=job.get("title") or "", method=method,
            grade=grade_value, grade_label=rubric.GRADE_LABELS[grade_value],
            score=rubric.score(grade_value, float(g["must_coverage"]), float(g["nice_coverage"])),
            must_coverage=float(g["must_coverage"]), nice_coverage=float(g["nice_coverage"]),
            total_years=total_years(positions(candidate["profile"])), experience_ok=bool(g["experience_ok"]),
            requirements=assessments, strengths=judgement.strengths, gaps=judgement.gaps,
            interview_questions=judgement.interview_questions,
            summary=(judgement.summary or "").strip() or rubric.GRADE_LABELS[grade_value],
            action=rubric.ACTIONS[grade_value], action_label=rubric.ACTION_LABELS[rubric.ACTIONS[grade_value]],
            knowledge_sources=[KnowledgeSource(doc_id=k.doc_id, title=k.title, section=k.section,
                                               excerpt=k.excerpt[:300]) for k in knowledge],
            model=getattr(self.llm, "model", None), seconds=round(time.perf_counter() - start, 1),
            warnings=warnings, trace=trace,
        )
