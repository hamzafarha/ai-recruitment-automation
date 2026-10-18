"""The fit scoring rubric KB-RR-002 as code: evidence weights, coverage, grade, score, action.

Single source of truth for both matching modes (fast semantic and full LLM analysis), so a grade
always means the same thing. Rules, from data/knowledge_base/recruitment_rules/fit_scoring_rubric.md:
- evidence weights: demonstrated 1.0, equivalent 0.75, listed 0.5, not evidenced 0;
- must-have coverage = sum(evidence weight x importance weight) / sum(importance weight),
  importance high 3, medium 2, low 1;
- grade 3 (strong fit): all must-haves demonstrated or equivalent, experience meets the requirement
  (1 year of tolerance), most nice-to-haves present;
- grade 2 (partial fit): at least 70 % must-have coverage, or all must-haves with experience up to
  2 years below the requirement;
- grade 1 (weak fit): some relevant skills but less than 70 % coverage;
- grade 0 (no fit): different field (coverage under 25 %) or a hard requirement missing;
- scores: grade 3 -> 75-100, 2 -> 50-74, 1 -> 25-49, 0 -> 0-24;
- action: 3 shortlist, 2 recruiter review, 0-1 not recommended; never an automatic rejection.
A requirement whose experience is more than 1 year short counts as "partial" (weight capped at 0.5).
"""

from collections.abc import Iterable

from recruitment_ai.matching.schemas import Action, RequirementAssessment, Status

EVIDENCE_WEIGHT = {"demonstrated": 1.0, "equivalent": 0.75, "listed": 0.5, "not_evidenced": 0.0}
IMPORTANCE_WEIGHT = {"high": 3, "medium": 2, "low": 1}
GRADE_LABELS = {3: "Strong fit", 2: "Partial fit", 1: "Weak fit", 0: "No fit"}
SCORE_BANDS = {3: (75.0, 100.0), 2: (50.0, 74.0), 1: (25.0, 49.0), 0: (0.0, 24.0)}
ACTIONS: dict[int, Action] = {3: "shortlist", 2: "review", 1: "not_recommended", 0: "not_recommended"}
ACTION_LABELS = {
    "shortlist": "Recommended for the shortlist (recruiter confirms)",
    "review": "Recruiter review needed: often worth an interview if the gaps are trainable",
    "not_recommended": "Not recommended (never rejected automatically: a recruiter confirms)",
}
EXPERIENCE_TOLERANCE = 1.0  # years below the requirement still treated as meeting it
PARTIAL_EXPERIENCE_GAP = 2.0  # grade 2 accepts all must-haves with experience up to 2 years short
POSITIVE = ("demonstrated", "equivalent")


def years_short(min_years: float | None, years_found: float | None) -> bool:
    """More than EXPERIENCE_TOLERANCE below the required years (unknown years are not penalised)."""
    return bool(min_years) and years_found is not None and years_found < min_years - EXPERIENCE_TOLERANCE


def status_of(level: str, short: bool) -> Status:
    if level == "not_evidenced":
        return "missing"
    if level == "listed" or short:
        return "partial"
    return "met"


def finalize(assessment: RequirementAssessment) -> RequirementAssessment:
    """Fill ``years_short`` and ``status`` from the level and the years."""
    assessment.years_short = years_short(assessment.min_years, assessment.years_found)
    assessment.status = status_of(assessment.level, assessment.years_short)
    return assessment


def effective_weight(assessment: RequirementAssessment) -> float:
    weight = EVIDENCE_WEIGHT[assessment.level]
    return min(weight, 0.5) if assessment.years_short else weight


def coverage(assessments: Iterable[RequirementAssessment]) -> float | None:
    """Importance-weighted evidence coverage (0..1), None for an empty list."""
    items = list(assessments)
    total = sum(IMPORTANCE_WEIGHT[a.importance] for a in items)
    if not total:
        return None
    return sum(effective_weight(a) * IMPORTANCE_WEIGHT[a.importance] for a in items) / total


def grade(assessments: list[RequirementAssessment]) -> dict[str, float | int | bool]:
    """Grade, coverages and experience check of a candidate for a job (rubric KB-RR-002)."""
    must = [a for a in assessments if a.kind == "must"]
    nice = [a for a in assessments if a.kind == "nice"]
    if not must:  # a job without must-haves: every requirement counts as one
        must, nice = list(assessments), []
    must_cov = coverage(must) or 0.0
    nice_cov = coverage(nice) if nice else 1.0
    shortfalls = [a.min_years - a.years_found for a in must
                  if a.min_years and a.years_found is not None and a.years_found < a.min_years]
    max_short = max(shortfalls, default=0.0)
    experience_ok = max_short <= EXPERIENCE_TOLERANCE
    all_must_ok = all(a.level in POSITIVE for a in must)

    if any(a.hard and a.level == "not_evidenced" for a in must):
        value = 0
    elif all_must_ok and experience_ok and nice_cov >= 0.5:
        value = 3
    elif must_cov >= 0.70 or (all_must_ok and max_short <= PARTIAL_EXPERIENCE_GAP):
        value = 2
    elif must_cov >= 0.25:
        value = 1
    else:
        value = 0
    return {"grade": value, "must_coverage": round(must_cov, 4), "nice_coverage": round(nice_cov or 0.0, 4),
            "experience_ok": experience_ok, "max_years_short": round(max_short, 2)}


def score(grade_value: int, must_cov: float, nice_cov: float) -> float:
    """0-100 score: 85 % must-haves, 15 % nice-to-haves, kept inside the grade's band."""
    raw = 100 * (0.85 * must_cov + 0.15 * nice_cov)
    low, high = SCORE_BANDS[grade_value]
    return round(min(max(raw, low), high), 1)
