"""The "Under the hood" endpoints (/observe/...), the assistant's timeline and the AI analysis trace.

Uses the ``env`` fixture of conftest.py: an empty SQLite database and fake models behind the API.
"""

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from recruitment_ai.config.settings import Settings
from recruitment_ai.services import observe
from tests.fakes import JOB

ROOT = Path(__file__).resolve().parents[2]
V1 = "/api/v1"
CV = "Ann Lee. Built REST APIs in Python with FastAPI and PostgreSQL."


def setup_data(client: TestClient) -> tuple[dict, dict]:
    job = client.post(f"{V1}/jobs", json=JOB).json()
    ann = client.post(f"{V1}/candidates", files={"file": ("cv.pdf", CV.encode(), "application/pdf")}).json()
    return job, ann


def test_database_tables_and_rows(env):
    client, _, _ = env
    setup_data(client)
    tables = {t["table"]: t["rows"] for t in client.get(f"{V1}/observe/tables").json()["tables"]}
    assert tables["candidates"] == 1 and tables["jobs"] == 1 and tables["notifications"] == 2
    rows = client.get(f"{V1}/observe/tables/candidates").json()
    assert rows["total"] == 1 and {"name": "profile", "type": "JSON"} in rows["columns"]
    row = rows["rows"][0]
    assert row["full_name"] == "Ann Lee" and row["profile"]["skills"][0]["name"] == "Python"
    assert row["cv_text"] == CV and isinstance(row["created_at"], str)
    assert client.get(f"{V1}/observe/tables/users").status_code == 404


def test_collections_points_and_vector_map(env):
    client, _, _ = env
    setup_data(client)
    collections = {c["name"]: c for c in client.get(f"{V1}/observe/collections").json()["data"]}
    assert collections["candidates"]["points"] > 3 and collections["candidates"]["dimension"] == 256
    assert collections["hr_knowledge"]["status"] == "missing"  # the knowledge base is not built in this test
    points = client.get(f"{V1}/observe/collections/candidates/points").json()["points"]
    assert all(p["label"].startswith("Ann Lee — ") for p in points)
    one = client.get(f"{V1}/observe/collections/candidates/points/{points[0]['id']}").json()
    assert one["dimension"] == 256 and len(one["vector"]) == 256 and one["norm"] > 0
    drawn = client.get(f"{V1}/observe/collections/candidates/map", params={"color_by": "section"}).json()
    assert len(drawn["points"]) == len(points) and {"x", "y", "group", "label"} <= set(drawn["points"][0])
    assert {p["group"] for p in drawn["points"]} >= {"experience", "skills"} and len(drawn["explained_variance"]) == 2
    assert client.get(f"{V1}/observe/collections/secrets/points").status_code == 404


def test_one_search_step_by_step(env):
    client, _, _ = env
    setup_data(client)
    result = client.get(f"{V1}/observe/retrieval", params={"q": "FastAPI and PostgreSQL", "collection": "candidates",
                                                          "top_k": 3}).json()
    assert result["embedding"]["dimension"] == 256 and len(result["embedding"]["first_values"]) == 12
    assert result["dense"] and [d["dense_rank"] for d in result["dense"]] == list(range(1, len(result["dense"]) + 1))
    reranked = result["reranked"]
    assert [r["rerank_rank"] for r in reranked] == list(range(1, len(reranked) + 1))
    assert reranked[0]["rerank_score"] >= reranked[-1]["rerank_score"]
    assert all(r["moved"] == r["dense_rank"] - r["rerank_rank"] for r in reranked)
    assert "FastAPI" in reranked[0]["text"]


def test_what_feeds_an_analysis(env):
    client, container, _ = env
    container.index.build_knowledge(ROOT / "data" / "knowledge_base")
    job, ann = setup_data(client)
    result = client.post(f"{V1}/observe/matching", json={"candidate_id": ann["id"], "job_id": job["id"]}).json()
    assert [r["id"] for r in result["requirements"]] == ["M1", "M2", "M3", "N1"]
    m2 = next(r for r in result["requirements"] if r["id"] == "M2")
    assert m2["level"] == "demonstrated" and m2["snippets"][0]["rerank"] == 1.0
    assert len(result["knowledge"]) == 3 and all(k["hits"] for k in result["knowledge"])
    assert result["rules_in_prompt"] and "REQUIREMENTS:" in result["prompt"] and CV in result["prompt"]
    assert result["semantic_grade"]["grade"] == 3
    missing = client.post(f"{V1}/observe/matching", json={"candidate_id": "nope", "job_id": job["id"]})
    assert missing.status_code == 404


def test_a_cv_through_document_processing_and_the_adapter_comparison(env):
    client, container, _ = env
    response = client.post(f"{V1}/observe/extraction", data={"compare_base": "true"},
                           files={"file": ("cv.pdf", CV.encode(), "application/pdf")})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["parsing"]["pages"] == [{"page": 1, "method": "text_layer", "characters": len(CV),
                                           "ocr_confidence": None}]
    assert result["text"] == CV and CV in result["prompt"]["user"] and '"skills"' in result["prompt"]["user"]
    assert result["finetuned"]["adapter_used"] and not result["base"]["adapter_used"]
    assert container.extractor.adapter_calls == [True, False]
    comparison = result["comparison"]
    assert comparison["counts"]["skills"] == {"finetuned": 5, "base": 3}
    assert comparison["skills_only_finetuned"] == ["docker", "kubernetes", "postgresql"]
    assert comparison["skills_only_base"] == ["teamwork"]
    assert client.get(f"{V1}/candidates").json()["count"] == 0  # nothing stored
    alone = client.post(f"{V1}/observe/extraction", files={"file": ("cv.pdf", CV.encode(), "application/pdf")}).json()
    assert alone["base"] is None and alone["comparison"] is None


def test_the_assistant_records_its_path_through_the_graph(env):
    client, container, _ = env
    setup_data(client)
    container.llm.structured = [
        {"thought": "Rank them", "action": "Rank_Candidates", "arguments": {"job": "Python Backend Developer"}},
        {"thought": "Enough", "action": "final_answer", "arguments": {}}]
    run = client.post(f"{V1}/agent/ask", json={"question": "Best candidates for the backend job?"}).json()
    timeline = run["timeline"]
    assert [e["node"] for e in timeline] == ["decide", "act", "decide", "respond"]
    assert timeline[0]["decision"]["action"] == "Rank_Candidates" and timeline[0]["next"] == "act"
    assert timeline[1]["tool"] == "rank_candidates" and timeline[1]["ok"] and timeline[1]["result"] == "1 top"
    assert timeline[2]["why"] == "enough to answer" and timeline[3]["output_tokens"] > 0
    assert client.get(f"{V1}/agent/runs/{run['id']}").json()["timeline"] == timeline
    prompts = client.get(f"{V1}/agent/tools").json()["system_prompts"]
    assert "rank_candidates(job, top_k=5)" in prompts["decide"] and "KB-HR-003" in prompts["respond"]


def test_tool_arguments_are_forgiving():
    from recruitment_ai.tools.registry import RerankArgs, SearchArgs

    assert SearchArgs.model_validate({"q": "Python", "top_k": 50}).model_dump() == {"query": "Python", "top_k": 10}
    reranked = RerankArgs.model_validate({"criterion": "LLM", "candidates": "Ann Lee, Bob Stone and Cara Diaz"})
    assert reranked.candidates == ["Ann Lee", "Bob Stone", "Cara Diaz"]


def test_an_ai_analysis_keeps_its_trace(env):
    client, container, _ = env
    job, ann = setup_data(client)
    container.llm.structured = [{"requirements": [
        {"id": "M1", "level": "demonstrated", "evidence": "Built REST APIs in Python", "years": 5},
        {"id": "M2", "level": "demonstrated", "evidence": "Ten years of Rust at NASA"}]}]
    analysis = client.post(f"{V1}/analyses", json={"candidate_id": ann["id"], "job_id": job["id"]}).json()
    trace = analysis["report"]["trace"]
    assert len(trace["knowledge_queries"]) == 3 and "REQUIREMENTS:" in trace["prompt"]
    assert '"Ten years of Rust at NASA"' in trace["raw_answer"] and trace["prompt_tokens"] > 0
    assert trace["checks"]["downgraded_ungrounded"] == ["M2"]
    assert trace["checks"]["not_assessed"] == ["M3", "N1"]


def test_overview_of_the_nine_parts(env, monkeypatch):
    client, _, _ = env
    setup_data(client)
    monkeypatch.setattr(observe, "n8n_status", lambda settings: {"reachable": False, "events_enabled": False})
    o = client.get(f"{V1}/observe/overview").json()
    assert o["tables"]["candidates"] == 1 and o["candidates_by_source"] == {"upload": 1}
    assert o["last_extraction"]["candidate"] == "Ann Lee" and o["vectors"]["candidates"] > 0
    assert o["ai_server"] == {"used": False} and o["n8n"]["reachable"] is False


@pytest.fixture
def fake_n8n(monkeypatch):
    """n8n's answers: CV intake and Events published, New job and Daily digest not."""
    live = "This webhook is not registered for GET requests. Did you mean to make a POST request?"

    def get(url, headers=None, params=None, timeout=None):
        if url.endswith("/healthz"):
            return httpx.Response(200, json={"status": "ok"})
        if "/webhook/" in url:
            path = url.rsplit("/", 1)[1]
            message = (live if path in ("cv-intake", "recruitment-events")
                       else f'The requested webhook "GET {path}" is not registered.')
            return httpx.Response(404, json={"code": 404, "message": message})
        if url.endswith("/api/v1/workflows"):
            assert headers == {"X-N8N-API-KEY": "key"}
            return httpx.Response(200, json={"data": [{"id": "RecruitEvents001", "name": "Recruitment — Events",
                                                       "active": True}]})
        if url.endswith("/api/v1/executions"):
            return httpx.Response(200, json={"data": [{"id": "7", "workflowId": "RecruitEvents001", "status": "success",
                                                       "mode": "webhook", "startedAt": "2026-10-18T09:00:00.000Z",
                                                       "stoppedAt": "2026-10-18T09:00:01.500Z"}]})
        raise AssertionError(url)

    monkeypatch.setattr(observe.httpx, "get", get)


def test_n8n_status(fake_n8n):
    status = observe.n8n_status(Settings(N8N_URL="http://n8n.test", N8N_API_KEY="key"))
    assert status["reachable"] and {w["path"]: w["live"] for w in status["webhooks"]} == {
        "/webhook/cv-intake": True, "/webhook/new-job": False, "/webhook/recruitment-events": True,
        "/webhook/daily-digest": False}
    execution = status["executions"][0]
    assert execution["workflow"] == "Recruitment — Events" and execution["seconds"] == 1.5
    assert execution["link"] == "http://n8n.test/workflow/RecruitEvents001/executions/7"
    assert "key" not in str({k: v for k, v in status.items() if k != "api_key"})  # the key is never returned


def test_n8n_unreachable(monkeypatch):
    def down(*args, **kwargs):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(observe.httpx, "get", down)
    status = observe.n8n_status(Settings(N8N_URL="http://n8n.test"))
    assert status["reachable"] is False and "webhooks" not in status
