"""The screening agent of the application pipeline (DSO4, diagram part 5) and its CV evidence index.
Fakes only (tests/fakes.py): bag-of-words embeddings, a word-overlap reranker, scripted Qwen answers."""

from types import SimpleNamespace

import pytest

from recruitment_ai.agents.screening import AGENT_ANALYSIS_SYSTEM_PROMPT, ScreeningAgent
from recruitment_ai.matching.analyzer import EvidenceAnalyzer
from recruitment_ai.matching.cv_index import (
    CVEvidenceIndex,
    cv_passages,
    section_of_heading,
)
from recruitment_ai.matching.semantic import SemanticMatcher
from tests.fakes import (
    ANN,
    ANN_CV,
    BOB,
    BOB_CV,
    DESIGN_JOB,
    JOB,
    OverlapReranker,
    ScriptedLLM,
    WordEmbedder,
)

RULE = SimpleNamespace(id="KB-JP-BE#3", text="Backend developer profile: Flask or Django count for FastAPI.",
                       relevance_score=0.8, metadata={"document_id": "KB-JP-BE", "title": "Backend developer profile",
                                                      "section": "Accepted equivalences"})
BACKEND = {**JOB, "id": "backend"}   # requirement ids: M1, M2, M3 (must-haves), N4 (nice-to-have)
DESIGN = {**DESIGN_JOB, "id": "design"}
ANN_JUDGEMENT = {"requirements": [
    {"id": "M1", "level": "demonstrated", "evidence": "Built REST APIs in Python", "years": 5},
    {"id": "M2", "level": "demonstrated", "evidence": "REST APIs in Python with FastAPI"},
    {"id": "M3", "level": "demonstrated", "evidence": "FastAPI and PostgreSQL"},
    {"id": "N4", "level": "listed", "evidence": "Kubernetes"}], "summary": "Strong backend profile."}


def agent_with(llm: ScriptedLLM, knowledge: tuple = (RULE,)) -> ScreeningAgent:
    embedder, reranker = WordEmbedder(), OverlapReranker()
    semantic = SemanticMatcher(embedder, reranker)
    analyzer = EvidenceAnalyzer(llm, knowledge_search=lambda query, role: list(knowledge), semantic=semantic)
    return ScreeningAgent(llm, embedder, reranker, semantic, analyzer)


def ann() -> dict:
    return {"id": "ann", "name": "Ann Lee", "profile": ANN, "cv_text": ANN_CV}


def bob() -> dict:
    return {"id": "bob", "name": "Bob Stone", "profile": BOB, "cv_text": BOB_CV}


# --------------------------------------------------------------------------- the CV as evidence


def test_cv_passages_keep_their_section_and_position():
    cv = ("Ann Lee\nExpérience professionnelle :\nDéveloppeuse backend — Acme (2020 - 2024)\n"
          "• Conception d'API REST en Python avec FastAPI\nTechnologies: Docker, PostgreSQL\nCOMPÉTENCES\n"
          "Python, FastAPI, Kubernetes\nProjets\nMoteur de recherche\n- Recherche sémantique avec BGE-M3\n"
          "Langues: Anglais, Français\n")
    passages = cv_passages(cv)
    assert [(p.section, p.text) for p in passages] == [
        ("header", "Ann Lee"),
        ("experience", "Développeuse backend — Acme (2020 - 2024)"),
        ("experience", "Conception d'API REST en Python avec FastAPI"),
        ("experience", "Docker, PostgreSQL"),          # a position's stack stays part of it
        ("skills", "Python, FastAPI, Kubernetes"),
        ("project", "Moteur de recherche"),
        ("project", "Recherche sémantique avec BGE-M3"),
        ("languages", "Anglais, Français")]
    assert passages[2].label == "Experience — Développeuse backend — Acme (2020 - 2024)"
    assert section_of_heading("TECHNICAL SKILLS") == "skills" and section_of_heading("Experience in APIs") is None


def test_long_blocks_are_cut_and_a_cv_without_text_uses_the_profile():
    long_text = "Experience\n" + " ".join(f"Built service number {i} in Python for the payments team." for i in range(20))
    assert all(len(p.text) <= 320 for p in cv_passages(long_text)) and len(cv_passages(long_text)) > 2
    index = CVEvidenceIndex(WordEmbedder(), None, None, ANN)
    assert not index.from_cv_text and {p.section for p in index.passages} >= {"experience", "skills"}
    hits = index.search("Python FastAPI")
    assert hits and index.reranked is False  # no cross-encoder: cosine similarity
    assert index.sections_of("Python, FastAPI, PostgreSQL") >= {"skills"}


# --------------------------------------------------------------------------- the agent


def test_the_first_round_searches_every_requirement_then_qwen_judges_from_the_passages():
    llm = ScriptedLLM({"MatchJudgement": [ANN_JUDGEMENT]})
    result = agent_with(llm).run(ann(), [BACKEND], "backend")
    report = result["reports"][0]
    assert report.method == "agent" and report.grade == 3 and report.summary == "Strong backend profile."
    assert [c["schema"] for c in llm.calls] == ["MatchJudgement"]  # strong evidence everywhere: no planning round
    assert all(c["use_adapter"] is False for c in llm.calls)       # the base model: the adapter is for extraction
    first = [c for c in result["tool_calls"] if c["round"] == 0]
    assert {c["requirement"] for c in first if c["tool"] == "search_cv"} == {"M1", "M2", "M3", "N4"}
    assert sum(c["tool"] == "search_hr_knowledge" for c in first) == 3
    assert [e["node"] for e in result["timeline"]] == ["understand", "plan", "act", "evaluate", "analyse", "verify",
                                                       "next"]
    prompt = llm.calls[0]["prompt"]
    assert "CV passages retrieved for it" in prompt and "[Experience — Backend Developer — Acme (2020 - 2024)]" in prompt
    assert llm.calls[0]["system"] == AGENT_ANALYSIS_SYSTEM_PROMPT
    assert report.knowledge_sources[0].doc_id == "KB-JP-BE"
    trace = report.trace["agent"]
    assert trace["planning_rounds"] == 0 and trace["evidence"]["M2"][0]["section"] == "experience"
    assert report.trace["checks"]["verification"] == []  # every claim passed the checks


def test_the_checks_correct_the_claims_of_the_model():
    llm = ScriptedLLM({"MatchJudgement": [{"requirements": [
        {"id": "M1", "level": "demonstrated", "evidence": "Built REST APIs in Python"},  # no years given
        {"id": "M2", "level": "demonstrated", "evidence": "Docker deployments"},        # in the CV, but unrelated
        {"id": "M3", "level": "not_evidenced"},                                         # missed
        {"id": "N4", "level": "demonstrated", "evidence": "Kubernetes"}],               # only in the skills section
        "summary": "Backend profile."}]})
    report = agent_with(llm).run(ann(), [BACKEND], "backend")["reports"][0]
    by_id = {r.id: r for r in report.requirements}
    assert {i: r.level for i, r in by_id.items()} == {"M1": "demonstrated", "M2": "demonstrated",
                                                       "M3": "demonstrated", "N4": "listed"}
    passage = "Built REST APIs in Python with FastAPI and PostgreSQL"
    assert by_id["M2"].evidence == passage and by_id["M3"].evidence == passage and by_id["M3"].grounded
    assert by_id["M1"].years_found is not None and by_id["M1"].years_found >= 3  # the relevant positions' years
    changes = {c["id"]: c["why"] for c in report.trace["checks"]["verification"]}
    assert set(changes) == {"M1", "M2", "M3", "N4"}
    assert "does not support" in changes["M2"] and "did not use" in changes["M3"] and "skills section" in changes["N4"]
    assert any("checks changed 4 requirement" in w for w in report.warnings)


def test_weak_evidence_makes_the_agent_plan_further_searches_within_limits():
    plan = {"thought": "Look for programming with other words", "calls": [
        {"tool": "search_cv", "requirement": "M1", "query": "Python scripting automation"},
        {"tool": "search_cv", "requirement": "M1", "query": "Python programming"},          # made in round 0
        {"tool": "search_hr_knowledge", "requirement": "M2", "query": "skills accepted instead of FastAPI"},
        {"tool": "web_search", "requirement": "M2", "query": "fastapi"},
        {"tool": "search_cv", "requirement": "M3", "query": "age of the candidate"}]}
    llm = ScriptedLLM({"ScreeningPlan": [plan, {"thought": "Nothing related in this CV", "calls": []}],
                       "MatchJudgement": [{"requirements": [], "summary": "No backend experience."}]})
    result = agent_with(llm).run(bob(), [BACKEND], "backend")
    assert [c["schema"] for c in llm.calls] == ["ScreeningPlan", "ScreeningPlan", "MatchJudgement"]
    plans = result["reports"][0].trace["agent"]["plans"]
    assert [c["query"] for c in plans[0]["calls"]] == ["Python scripting automation", "skills accepted instead of FastAPI"]
    assert {d["why"] for d in plans[0]["dropped"]} == {"already searched", "unknown tool 'web_search'",
                                                        "protected attribute (KB-HR-003)"}
    assert plans[1]["calls"] == []
    assert [c["round"] for c in result["tool_calls"] if c.get("query") == "Python scripting automation"] == [1]
    assert [e["next"] for e in result["timeline"] if e["node"] == "evaluate"] == ["plan", "plan", "analyse"]
    assert result["reports"][0].grade == 0


def test_a_failed_plan_falls_back_to_the_hr_rules():
    llm = ScriptedLLM({"ScreeningPlan": [RuntimeError("bad JSON")]})
    report = agent_with(llm).run(bob(), [BACKEND], "backend")["reports"][0]
    plan = report.trace["agent"]["plans"][0]
    assert plan["by"] == "code (planning failed)" and {c["tool"] for c in plan["calls"]} == {"search_hr_knowledge"}
    assert any("Planning round 1 failed" in w for w in report.warnings)


def test_a_spontaneous_application_is_screened_for_the_jobs_that_fit():
    llm = ScriptedLLM({"MatchJudgement": [{"requirements": [
        {"id": "M1", "level": "demonstrated", "evidence": "Designed brand identities with Photoshop and Illustrator"},
        {"id": "M2", "level": "equivalent", "evidence": "Designed brand identities"}], "summary": "Designer."}]})
    result = agent_with(llm).run(bob(), [BACKEND, DESIGN])
    assert [c["title"] for c in result["considered"]] == ["Graphic Designer", "Python Backend Developer"]
    assert [r.job_title for r in result["reports"]] == ["Graphic Designer"] and result["reports"][0].grade == 3
    assert result["timeline"][0]["task"] == "find_jobs"
    # no open job fits: the best one is analysed anyway, so that a recruiter can answer the candidate
    alone = agent_with(ScriptedLLM()).run(bob(), [BACKEND])
    assert len(alone["reports"]) == 1 and "no open job graded 2 or more" in alone["timeline"][0]["why"]


def test_a_failed_analysis_gives_the_fast_estimate_and_no_job_is_an_error():
    llm = ScriptedLLM({"MatchJudgement": [RuntimeError("server stopped")]})
    report = agent_with(llm).run(ann(), [BACKEND], "backend")["reports"][0]
    assert report.method == "semantic" and any("AI analysis failed" in w for w in report.warnings)
    assert report.trace["agent"]["tool_calls"]  # the agent's searches are kept with the report
    with pytest.raises(ValueError):
        agent_with(ScriptedLLM()).run(ann(), [])
