"""Matching (DSO3): rubric KB-RR-002 as code, experience rules KB-RR-003, semantic matching, the
evidence-checked LLM analysis, job extraction and the report. Fakes only (no model, no GPU)."""

from datetime import date
from types import SimpleNamespace

import pytest

from recruitment_ai.extraction.job_extractor import JobExtraction, extract_job
from recruitment_ai.matching import rubric
from recruitment_ai.matching.analyzer import (
    EvidenceAnalyzer,
    MatchJudgement,
    normalize_level,
)
from recruitment_ai.matching.evidence import (
    candidate_snippets,
    is_grounded,
    positions,
    requirement_templates,
    total_years,
)
from recruitment_ai.matching.schemas import RequirementAssessment
from recruitment_ai.matching.semantic import (
    SemanticMatcher,
    Thresholds,
    load_thresholds,
)
from recruitment_ai.reports.markdown import report_markdown
from tests.fakes import ANN, BOB, JOB, OverlapReranker, ScriptedLLM, WordEmbedder

TODAY = date(2026, 10, 17)


def req(rid: str, level: str, kind: str = "must", importance: str = "high", min_years: float | None = None,
        years: float | None = None, hard: bool = False) -> RequirementAssessment:
    return rubric.finalize(RequirementAssessment(id=rid, text=rid, kind=kind, importance=importance, level=level,
                                                 min_years=min_years, years_found=years, hard=hard))


# --------------------------------------------------------------------------- rubric (KB-RR-002)


def test_strong_fit_needs_every_must_have_and_the_experience():
    items = [req("M1", "demonstrated"), req("M2", "equivalent"), req("N1", "listed", kind="nice")]
    g = rubric.grade(items)
    assert g["grade"] == 3 and g["experience_ok"]
    assert g["must_coverage"] == pytest.approx((1.0 * 3 + 0.75 * 3) / 6)
    assert 75 <= rubric.score(3, g["must_coverage"], g["nice_coverage"]) <= 100


def test_a_missing_hard_requirement_gives_no_fit():
    items = [req("M1", "demonstrated"), req("M2", "not_evidenced", hard=True)]
    assert rubric.grade(items)["grade"] == 0


def test_experience_tolerance_and_partial_status():
    one_year_short = req("M1", "demonstrated", min_years=3, years=2.0)  # within the 1-year tolerance
    assert one_year_short.status == "met" and not one_year_short.years_short
    two_years_short = req("M1", "demonstrated", min_years=5, years=3.0)
    assert two_years_short.status == "partial" and rubric.effective_weight(two_years_short) == 0.5
    g = rubric.grade([two_years_short, req("M2", "demonstrated")])
    assert g["grade"] == 2 and not g["experience_ok"]  # all must-haves, experience up to 2 years short


def test_weak_and_no_fit_bands_and_score_clamping():
    weak = rubric.grade([req("M1", "listed"), req("M2", "not_evidenced")])
    assert weak["grade"] == 1 and weak["must_coverage"] == 0.25
    assert rubric.grade([req("M1", "not_evidenced"), req("M2", "not_evidenced")])["grade"] == 0
    assert rubric.score(1, 0.9, 1.0) == 49.0  # a grade-1 score stays in its band
    assert rubric.score(2, 0.2, 0.0) == 50.0
    assert rubric.status_of("listed", False) == "partial" and rubric.status_of("not_evidenced", False) == "missing"


# --------------------------------------------------------------------------- evidence (KB-RR-003)


def test_experience_counts_overlaps_once_and_internships_half():
    profile = {"experience": [
        {"title": "Engineer", "company": "A", "start_date": "2020-01", "end_date": "2021-12"},
        {"title": "Consultant", "company": "B", "start_date": "2021-01", "end_date": "2021-12"},  # overlaps A
        {"title": "Data Intern", "company": "C", "start_date": "2019-01", "end_date": "2019-12"},
    ]}
    timeline = positions(profile, TODAY)
    assert [p.internship for p in timeline] == [False, False, True]
    assert total_years(timeline) == 2.5  # 24 months + 12 internship months at 50 %


def test_year_only_dates_run_july_to_june_and_present_is_today():
    profile = {"experience": [{"title": "Dev", "company": "A", "start_date": "2018", "end_date": "2020"},
                              {"title": "Dev", "company": "B", "start_date": "2026-01", "end_date": "present"}]}
    timeline = positions(profile, TODAY)
    assert timeline[0].years == 2.0 and timeline[1].years == 0.8
    assert total_years([]) is None


def test_snippets_keep_the_cv_section_and_grounding_check():
    snippets = candidate_snippets(ANN, TODAY)
    sections = [s.section for s in snippets]
    assert sections[:3] == ["summary", "experience", "experience"] and "skills" in sections
    assert "internship" in snippets[2].label
    cv = "Built REST APIs in Python with FastAPI and PostgreSQL."
    assert is_grounded("REST APIs in Python with FastAPI", cv)
    assert not is_grounded("Led a team of 20 Kubernetes engineers", cv)


def test_requirement_templates_give_ids_and_kinds():
    templates = requirement_templates(JOB)
    assert [t.id for t in templates] == ["M1", "M2", "M3", "N4"]
    assert templates[0].min_years == 3 and templates[3].kind == "nice"


# --------------------------------------------------------------------------- semantic matching


def test_semantic_matching_ranks_the_relevant_candidate_first():
    matcher = SemanticMatcher(WordEmbedder(), OverlapReranker())
    job = {**JOB, "id": "j1"}
    reports = matcher.match_many(job, [{"id": "b", "name": "Bob Stone", "profile": BOB},
                                       {"id": "a", "name": "Ann Lee", "profile": ANN}])
    by_name = {r.candidate_name: r for r in reports}
    ann, bob = by_name["Ann Lee"], by_name["Bob Stone"]
    assert ann.grade > bob.grade and ann.score > bob.score
    m1 = next(r for r in ann.requirements if r.id == "M1")
    assert m1.level == "demonstrated" and (m1.evidence_source or "").startswith(("Backend Developer at Acme",
                                                                                 "Software Intern at Beta"))
    assert m1.years_found and m1.years_found >= 3
    assert bob.grade <= 1 and bob.gaps and bob.interview_questions
    assert ann.method == "semantic" and "Fast semantic estimate" in ann.summary


def test_a_skill_only_listed_is_not_demonstrated():
    matcher = SemanticMatcher(WordEmbedder(), OverlapReranker())
    report = matcher.match({**JOB, "id": "j1"}, {"id": "a", "name": "Ann Lee", "profile": ANN})
    kubernetes = next(r for r in report.requirements if r.id == "N4")
    assert kubernetes.level == "listed" and kubernetes.status == "partial"


def test_thresholds_file(tmp_path):
    path = tmp_path / "thresholds.json"
    assert load_thresholds(path) == Thresholds()
    path.write_text('{"demonstrated": 0.5, "equivalent": 0.2, "calibrated_on": "14 dev pairs"}', encoding="utf-8")
    assert load_thresholds(path) == Thresholds(demonstrated=0.5, equivalent=0.2)


# --------------------------------------------------------------------------- LLM analysis


def judgement(prompt: str) -> dict:
    assert "COMPANY RULES" in prompt and "CANDIDATE EXPERIENCE TIMELINE" in prompt and "M1 [must-have" in prompt
    return {"requirements": [
        {"id": "M1", "level": "demonstrated", "evidence": "Built REST APIs in Python", "years": "5 years"},
        {"id": "m2", "level": "Demonstrated", "evidence": "with FastAPI and PostgreSQL"},
        {"id": "M3", "level": "demonstrated", "evidence": "Administered Oracle RAC clusters for banks"},
        {"id": "N4", "level": "missing", "evidence": "nothing"},
    ], "strengths": ["Python APIs"], "gaps": ["Kubernetes"], "interview_questions": ["Kubernetes?"],
        "summary": "Solid backend profile."}


HIT = SimpleNamespace(id="KB-RR-002#1", text="Evidence levels: demonstrated, listed...",
                      metadata={"document_id": "KB-RR-002", "title": "Fit Scoring Rubric",
                                "section": "Evidence levels"})


def test_llm_analysis_checks_every_excerpt_and_grades_with_the_rubric():
    llm = ScriptedLLM([judgement])
    analyzer = EvidenceAnalyzer(llm, knowledge_search=lambda query, role: [HIT])
    cv = ("Ann Lee. Backend Developer at Acme. Built REST APIs in Python with FastAPI and PostgreSQL; "
          "Docker deployments.")
    report = analyzer.analyze({**JOB, "id": "j1"}, {"id": "a", "name": "Ann Lee", "profile": ANN, "cv_text": cv})
    levels = {r.id: (r.level, r.status) for r in report.requirements}
    assert levels["M1"] == ("demonstrated", "met") and levels["M2"] == ("demonstrated", "met")
    assert levels["M3"] == ("listed", "partial")  # the quoted excerpt is not in the CV: downgraded
    assert levels["N4"] == ("not_evidenced", "missing")
    assert next(r for r in report.requirements if r.id == "M1").years_found == 5.0
    assert report.method == "llm" and report.knowledge_sources[0].doc_id == "KB-RR-002"
    assert any("not found in the CV" in w for w in report.warnings)
    assert report.grade == 2 and report.summary == "Solid backend profile."
    assert llm.calls[0]["use_adapter"] is False  # the base model: the adapter is for CV extraction


def test_llm_failure_falls_back_to_the_semantic_estimate():
    semantic = SemanticMatcher(WordEmbedder(), OverlapReranker())
    analyzer = EvidenceAnalyzer(ScriptedLLM([RuntimeError("server stopped")]), semantic=semantic)
    report = analyzer.analyze({**JOB, "id": "j1"}, {"id": "a", "name": "Ann Lee", "profile": ANN})
    assert report.method == "semantic" and "AI analysis failed" in report.warnings[0]


def test_levels_from_the_model_are_normalised():
    assert normalize_level("Not evidenced") == "not_evidenced" and normalize_level("transferable") == "equivalent"
    assert normalize_level("whatever") == "not_evidenced"
    lenient = MatchJudgement.model_validate({"requirements": None, "strengths": ["a", None, "b", "c", "d", "e"]})
    assert lenient.requirements == [] and lenient.strengths == ["a", "b", "c", "d"]


# --------------------------------------------------------------------------- job extraction, report


def test_job_advertisement_becomes_requirements_with_ids():
    llm = ScriptedLLM([{"title": "Data Engineer", "company": "Acme", "requirements": [
        {"text": "Python", "kind": "required", "importance": "high", "min_years": "3+ years"},
        "SQL",
        {"text": "Airflow", "kind": "nice to have", "importance": "LOW"},
        {"text": "Spark", "kind": "preferred", "importance": "medium"}]}])
    job = extract_job("Data Engineer at Acme\nWe need Python (3+ years), SQL; Airflow and Spark are a plus.", llm)
    assert job["title"] == "Data Engineer" and job["company"] == "Acme"
    ids = [(r["id"], r["requirement_type"], r["importance"], r["min_years_experience"]) for r in job["requirements"]]
    assert ids == [("M1", "required", "high", 3.0), ("M2", "required", "medium", None),
                   ("M3", "required", "low", None), ("N1", "preferred", "medium", None)]
    assert llm.calls[0]["schema"] == JobExtraction.__name__ and llm.calls[0]["use_adapter"] is False


def test_report_markdown_has_every_required_part():
    matcher = SemanticMatcher(WordEmbedder(), OverlapReranker())
    report = matcher.match({**JOB, "id": "j1"}, {"id": "a", "name": "Ann Lee", "profile": ANN})
    text = report_markdown(report, {"decision": "interview", "reviewer": "Sam", "interview_at": "2026-10-20 10:00"})
    for part in ("# Screening report — Ann Lee for Python Backend Developer", "**Grade:**", "## Recruiter decision",
                 "| M1 |", "## Gaps to explore in interview", "never makes the hiring decision"):
        assert part in text


def test_report_pdf_is_a_pdf():
    from recruitment_ai.reports.pdf import markdown_to_pdf

    pdf = markdown_to_pdf("# Title\n\n**Grade:** 3\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n- item", title="Report")
    assert pdf.startswith(b"%PDF") and len(pdf) > 1000
