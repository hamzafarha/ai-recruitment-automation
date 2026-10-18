"""Fast semantic matching (DSO3, "Semantic Matching Tool"): no LLM, seconds for a whole job.

For each job requirement and each candidate:
1. retrieval: BGE-M3 cosine similarity between the requirement and the candidate's evidence
   snippets (one per position, project, degree, skills group...); the best ``TOP_K`` are kept;
2. reranking: the BGE cross-encoder scores (requirement, snippet) pairs (more precise than cosine);
3. evidence level from each snippet's score and CV section (rubric KB-RR-002): a strong match
   inside a job or project = "demonstrated", inside a skills list = "listed", a weaker match
   inside a job or project = "equivalent" (related skill used for real); the highest level found
   among the reranked snippets is kept, with its snippet as the evidence;
4. exact skill matches from the EN/FR skill lexicon can raise the level (a skill named in a job
   description is demonstrated even if the sentence is phrased differently);
5. years for requirements with a minimum: relevant positions = those scoring above the
   "equivalent" threshold, counted with KB-RR-003;
6. grade, score and action from the rubric.
Thresholds are calibrated on the dev split by scripts/eval_matching.py.
All candidates of a job are embedded in one request and reranked in one request.
"""

import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from recruitment_ai.matching import rubric
from recruitment_ai.matching.evidence import (
    Snippet,
    candidate_snippets,
    positions,
    requirement_templates,
    total_years,
)
from recruitment_ai.matching.schemas import MatchReport, RequirementAssessment

TOP_K = 3
THRESHOLDS_FILE = Path("data/evaluation/matching/thresholds.json")  # written by scripts/eval_matching.py
CONTEXT_SECTIONS = {"experience", "project", "education", "certification", "languages"}  # evidence with context
USAGE_SECTIONS = {"experience", "project"}  # where a skill is used for real (needed for "equivalent")
LEVEL_ORDER = ["not_evidenced", "listed", "equivalent", "demonstrated"]


@dataclass
class Thresholds:
    """Evidence thresholds on the reranker probability (or on cosine when no reranker is available)."""

    demonstrated: float = 0.35
    equivalent: float = 0.10
    cosine_demonstrated: float = 0.60
    cosine_equivalent: float = 0.50

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def load_thresholds(path: Path = THRESHOLDS_FILE) -> Thresholds:
    """The thresholds calibrated on the dev pairs (defaults until the matching evaluation has run)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        names = {f.name for f in fields(Thresholds)}
        return Thresholds(**{k: float(v) for k, v in data.items() if k in names})
    except (OSError, ValueError, TypeError):
        return Thresholds()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _higher(level: str, other: str) -> str:
    return level if LEVEL_ORDER.index(level) >= LEVEL_ORDER.index(other) else other


def _short(text: str, limit: int = 90) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


class SemanticMatcher:
    def __init__(self, embedder: Any, reranker: Any | None = None, normalizer: Any | None = None,
                 thresholds: Thresholds | None = None) -> None:
        self.embedder, self.reranker, self.normalizer = embedder, reranker, normalizer
        self.thresholds = thresholds or Thresholds()

    # ------------------------------------------------------------------ skill concepts (lexicon)

    def _concepts(self, text: str) -> set[str]:
        if self.normalizer is None or not text:
            return set()
        from recruitment_ai.extraction.extractor import split_skill_phrase

        found = split_skill_phrase(text, self.normalizer)
        return {c.id for c in (self.normalizer.normalize(s) for s in found) if c is not None}

    def _candidate_concepts(self, snippets: list[Snippet]) -> tuple[set[str], set[str]]:
        """(concepts used in jobs/projects, concepts only listed)."""
        used, listed = set(), set()
        for s in snippets:
            if s.section in USAGE_SECTIONS:
                used |= self._concepts(s.text)
            elif s.section == "skills" and self.normalizer is not None:
                for name in s.text.removeprefix("Skills list:").split(","):
                    concept = self.normalizer.normalize(name.strip())
                    if concept is not None:
                        listed.add(concept.id)
        return used, listed

    # ------------------------------------------------------------------ matching

    def score_requirements(self, job: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Raw evidence for every (candidate, requirement): best snippet and its scores.

        Returned per candidate: {"requirements": [...], "snippets": [...], "pair_scores": {...}}; used by
        ``match_many`` and by the evaluation (to calibrate thresholds without re-running the models).
        """
        requirements = requirement_templates(job)
        per_candidate = [candidate_snippets(c["profile"]) for c in candidates]
        texts = [r.text for r in requirements] + [s.text for snippets in per_candidate for s in snippets]
        vectors = self.embedder.embed_documents(texts) if texts else []
        req_vectors = vectors[:len(requirements)]
        offset = len(requirements)
        results, pairs, pair_index = [], [], []
        for ci, snippets in enumerate(per_candidate):
            snippet_vectors = vectors[offset:offset + len(snippets)]
            offset += len(snippets)
            cosines = [[_cosine(rv, sv) for sv in snippet_vectors] for rv in req_vectors]
            chosen = []
            for ri, req in enumerate(requirements):
                order = sorted(range(len(snippets)), key=lambda j, ri=ri: cosines[ri][j], reverse=True)
                keep = order[:TOP_K]
                if req.min_years:  # every position may count towards the years
                    keep += [j for j in order[TOP_K:] if snippets[j].section == "experience"]
                chosen.append(keep)
                for j in keep:
                    pairs.append([req.text, snippets[j].text])
                    pair_index.append((ci, ri, j))
            results.append({"snippets": snippets, "cosines": cosines, "chosen": chosen, "rerank": {}})
        scores = self.reranker.score_pairs(pairs) if self.reranker is not None and pairs else None
        if scores is not None:
            for (ci, ri, j), s in zip(pair_index, scores, strict=True):
                results[ci]["rerank"][(ri, j)] = s
        for result in results:
            result["requirements"] = requirements
            result["reranked"] = scores is not None
        return results

    def assess(self, evidence: dict[str, Any], thresholds: Thresholds | None = None) -> list[RequirementAssessment]:
        """Evidence levels of one candidate from ``score_requirements`` output."""
        t = thresholds or self.thresholds
        snippets: list[Snippet] = evidence["snippets"]
        reranked = evidence["reranked"]
        high, low = (t.demonstrated, t.equivalent) if reranked else (t.cosine_demonstrated, t.cosine_equivalent)
        used, listed = self._candidate_concepts(snippets)
        out = []
        for ri, template in enumerate(evidence["requirements"]):
            a = template.model_copy(deep=True)

            def score_of(j: int, ri: int = ri) -> float:
                return evidence["rerank"].get((ri, j), 0.0) if reranked else evidence["cosines"][ri][j]

            # every reranked snippet counts: a skill used in a job beats the same skill in a skills list
            level, best, best_score = "not_evidenced", None, 0.0
            for j in evidence["chosen"][ri]:
                s, section = score_of(j), snippets[j].section
                if s >= high:
                    found = "demonstrated" if section in CONTEXT_SECTIONS else "listed"
                elif s >= low:
                    found = ("equivalent" if section in USAGE_SECTIONS
                             else "listed" if section == "skills" else "not_evidenced")
                else:
                    found = "not_evidenced"
                if (LEVEL_ORDER.index(found), s) > (LEVEL_ORDER.index(level), best_score):
                    level, best, best_score = found, j, s
            scores = [score_of(j) for j in evidence["chosen"][ri]]
            a.confidence = round(best_score if best is not None else max(scores, default=0.0), 3)
            if best is not None and level != "not_evidenced":
                a.evidence, a.evidence_source = _short(snippets[best].text, 220), snippets[best].label
            concepts = self._concepts(a.text)
            if concepts & used:
                level = _higher(level, "demonstrated")
                a.note = "skill named in a position or project (lexicon match)"
            elif concepts & listed:
                level = _higher(level, "listed")
            a.level = level
            if a.min_years:
                relevant = [snippets[j].position for j in evidence["chosen"][ri]
                            if snippets[j].position is not None and score_of(j) >= low]
                a.years_found = total_years(relevant) if relevant else (0.0 if level != "not_evidenced" else None)
            out.append(rubric.finalize(a))
        return out

    def report(self, assessments: list[RequirementAssessment], job: dict[str, Any], candidate: dict[str, Any],
               seconds: float | None = None) -> MatchReport:
        g = rubric.grade(assessments)
        grade_value = int(g["grade"])
        must = [a for a in assessments if a.kind == "must"] or assessments
        met = [a for a in must if a.status == "met"]
        partial = [a for a in must if a.status == "partial"]
        missing = [a for a in must if a.status == "missing"]
        gaps = [f"{a.id}: {_short(a.text, 70)} — " + (
            f"{a.years_found} years vs {a.min_years:g} required" if a.years_short
            else "only listed" if a.level == "listed" else "not evidenced") for a in partial + missing]
        questions = [
            (f"Which projects gave you the most ownership with {_short(a.text, 60)}?" if a.years_short
             else f"Walk us through how you applied {_short(a.text, 60)} in a real project."
             if a.level == "listed" else f"Can you describe concrete experience with {_short(a.text, 60)}?")
            for a in (partial + missing)[:4]]
        label = rubric.GRADE_LABELS[grade_value]
        score = rubric.score(grade_value, float(g["must_coverage"]), float(g["nice_coverage"]))
        return MatchReport(
            candidate_id=candidate.get("id"), candidate_name=candidate.get("name"),
            job_id=job.get("id"), job_title=job.get("title") or "", method="semantic",
            grade=grade_value, grade_label=label, score=score,
            must_coverage=float(g["must_coverage"]), nice_coverage=float(g["nice_coverage"]),
            total_years=total_years(positions(candidate["profile"])), experience_ok=bool(g["experience_ok"]),
            requirements=assessments,
            strengths=[f"{a.id}: {_short(a.text, 80)}" for a in met[:4]],
            gaps=gaps[:4], interview_questions=questions,
            summary=(f"{label} ({score:g}/100): {len(met)} of {len(must)} must-haves met, {len(partial)} partial, "
                     f"{len(missing)} not evidenced. Fast semantic estimate: run the full AI analysis before a "
                     "decision."),
            action=rubric.ACTIONS[grade_value], action_label=rubric.ACTION_LABELS[rubric.ACTIONS[grade_value]],
            model="BGE-M3 + BGE-reranker-v2-m3", seconds=seconds,
        )

    def match_many(self, job: dict[str, Any], candidates: list[dict[str, Any]]) -> list[MatchReport]:
        """Reports for several candidates of one job (one embedding and one reranking request).

        ``candidates``: [{"id": ..., "name": ..., "profile": CandidateProfile dict}, ...]
        """
        if not candidates:
            return []
        start = time.perf_counter()
        evidence = self.score_requirements(job, candidates)
        seconds = round((time.perf_counter() - start) / len(candidates), 2)
        return [self.report(self.assess(e), job, c, seconds) for e, c in zip(evidence, candidates, strict=True)]

    def match(self, job: dict[str, Any], candidate: dict[str, Any]) -> MatchReport:
        return self.match_many(job, [candidate])[0]
