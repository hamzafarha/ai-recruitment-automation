"""The application through its API: candidates, jobs, ranking, analyses, reports, human validation,
assistant, search, notifications, n8n endpoints and setup.

Each test gets an empty SQLite database and a container of fakes (the ``env`` fixture of conftest.py):
the real services, search index (embedded Qdrant), matching, analysis and agent code run; only the
models are fakes.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from tests.fakes import JOB

ROOT = Path(__file__).resolve().parents[2]
V1 = "/api/v1"


def upload(client: TestClient, text: str = "Ann Lee. Built REST APIs in Python with FastAPI and PostgreSQL."):
    return client.post(f"{V1}/candidates", files={"file": ("cv.pdf", text.encode(), "application/pdf")})


def create_job(client: TestClient) -> dict:
    response = client.post(f"{V1}/jobs", json=JOB)
    assert response.status_code == 201, response.text
    return response.json()


def test_cv_upload_creates_an_indexed_candidate_and_deletes_the_file(env):
    client, container, _ = env
    response = upload(client)
    assert response.status_code == 201, response.text
    ann = response.json()
    assert ann["full_name"] == "Ann Lee" and ann["indexed"] and ann["experience_count"] == 2
    assert "raw_answer" not in ann["extraction"]  # what the model wrote is not stored with the candidate
    path, existed = container.extractor.seen[0]
    assert existed and not path.exists()  # processed in a temporary folder, then deleted
    listed = client.get(f"{V1}/candidates").json()
    assert listed["count"] == 1 and listed["data"][0]["headline"] == "Backend Developer at Acme"
    detail = client.get(f"{V1}/candidates/{ann['id']}").json()
    assert detail["profile"]["skills"][0]["name"] == "Python" and detail["analyses"] == []
    assert "FastAPI" in client.get(f"{V1}/candidates/{ann['id']}/cv-text").json()["cv_text"]
    hits = client.get(f"{V1}/search/candidates", params={"q": "FastAPI PostgreSQL"}).json()["data"]
    assert hits[0]["candidate_id"] == ann["id"]
    events = [n["event"] for n in client.get(f"{V1}/notifications").json()["data"]]
    assert events == ["candidate.created"]


def test_upload_validation_and_extraction_errors(env):
    client, container, _ = env
    bad = client.post(f"{V1}/candidates", files={"file": ("cv.exe", b"x", "application/octet-stream")})
    assert bad.status_code == 400
    container.extractor.profiles = [RuntimeError("The Kaggle AI server does not answer")]
    stopped = upload(client)
    assert stopped.status_code == 503 and "does not answer" in stopped.json()["detail"]


def test_extract_only_stores_nothing(env):
    client, _, _ = env
    response = client.post(f"{V1}/candidates/extract", files={"file": ("cv.docx", b"text", "application/msword")})
    assert response.status_code == 200 and response.json()["profile"]["full_name"] == "Ann Lee"
    assert "raw_answer" not in response.json()["stats"]
    assert client.get(f"{V1}/candidates").json()["count"] == 0


def test_job_creation_validation_and_status(env):
    client, _, _ = env
    job = create_job(client)
    assert [r["id"] for r in job["requirements"]] == ["M1", "M2", "M3", "N1"]
    assert job["must_haves"] == 3 and job["nice_to_haves"] == 1
    assert client.post(f"{V1}/jobs", json={**JOB, "requirements": []}).status_code == 422
    assert client.patch(f"{V1}/jobs/{job['id']}", json={"status": "closed"}).json()["status"] == "closed"
    assert client.get(f"{V1}/search/jobs", params={"q": "FastAPI REST"}).json()["data"][0]["job_id"] == job["id"]
    assert client.get(f"{V1}/jobs/unknown").status_code == 404


def test_job_from_an_advertisement(env):
    client, container, _ = env
    container.llm.structured = [{"title": "Data Engineer", "requirements": [{"text": "Python", "kind": "must"},
                                                                           {"text": "Airflow", "kind": "nice"}]}]
    ad = "Data Engineer at Acme. " * 5
    draft = client.post(f"{V1}/jobs/extract", json={"text": ad}).json()
    assert draft["title"] == "Data Engineer" and [r["id"] for r in draft["requirements"]] == ["M1", "N1"]
    assert client.get(f"{V1}/jobs").json()["count"] == 0  # a draft, reviewed before it is saved
    assert client.post(f"{V1}/jobs/extract", json={"text": "too short"}).status_code == 422


def test_ranking_analysis_report_and_review(env):
    client, container, _ = env
    job = create_job(client)
    ann, bob = upload(client).json(), upload(client, "Bob Stone, graphic designer").json()
    ranking = client.get(f"{V1}/jobs/{job['id']}/ranking").json()["data"]
    assert [r["name"] for r in ranking] == ["Ann Lee", "Bob Stone"] and ranking[0]["grade"] > ranking[1]["grade"]

    container.llm.structured = [{"requirements": [
        {"id": "M1", "level": "demonstrated", "evidence": "Built REST APIs in Python", "years": 5},
        {"id": "M2", "level": "demonstrated", "evidence": "with FastAPI"},
        {"id": "M3", "level": "demonstrated", "evidence": "PostgreSQL"},
        {"id": "N1", "level": "listed", "evidence": "Kubernetes in a skills list"}],
        "summary": "Strong backend profile.", "gaps": ["Kubernetes only listed"]}]
    response = client.post(f"{V1}/analyses", json={"candidate_id": ann["id"], "job_id": job["id"]})
    assert response.status_code == 201, response.text
    analysis = response.json()
    assert analysis["method"] == "llm" and analysis["status"] == "pending_review"
    assert analysis["report"]["summary"] == "Strong backend profile." and analysis["grade"] >= 2

    markdown = client.get(f"{V1}/analyses/{analysis['id']}/report.md")
    assert markdown.status_code == 200 and "# Screening report — Ann Lee" in markdown.text
    pdf = client.get(f"{V1}/analyses/{analysis['id']}/report.pdf")
    assert pdf.content.startswith(b"%PDF") and "attachment" in pdf.headers["content-disposition"]

    url = f"{V1}/analyses/{analysis['id']}/review"
    assert client.post(url, json={"decision": "reject"}).status_code == 400  # a rejection needs a reason
    assert client.post(url, json={"decision": "maybe"}).status_code == 422
    decided = client.post(url, json={"decision": "interview", "reviewer": "Sam", "adjusted_grade": 3,
                                     "interview_at": "2026-10-20 10:00"}).json()
    assert decided["status"] == "interview" and decided["final_grade"] == 3
    assert decided["reviews"][0]["reviewer"] == "Sam"
    assert "## Recruiter decision" in client.get(f"{V1}/analyses/{analysis['id']}/report.md").text

    semantic = client.post(f"{V1}/analyses", json={"candidate_id": bob["id"], "job_id": job["id"],
                                                   "method": "semantic"}).json()
    assert semantic["method"] == "semantic" and semantic["grade"] <= 1
    pending = client.get(f"{V1}/analyses", params={"status": "pending_review"}).json()["data"]
    assert [a["candidate_name"] for a in pending] == ["Bob Stone"]
    events = {n["event"] for n in client.get(f"{V1}/notifications").json()["data"]}
    assert {"analysis.completed", "review.decided", "job.created"} <= events

    dashboard = client.get(f"{V1}/dashboard").json()
    assert dashboard["candidates"] == 2 and dashboard["analyses"] == 2 and dashboard["pending_reviews"] == 1
    assert client.post(f"{V1}/analyses", json={"candidate_id": "nope", "job_id": job["id"]}).status_code == 404


def test_deleting_a_candidate_removes_their_analyses(env):
    client, _, _ = env
    job = create_job(client)
    ann = upload(client).json()
    client.post(f"{V1}/analyses", json={"candidate_id": ann["id"], "job_id": job["id"], "method": "semantic"})
    assert client.delete(f"{V1}/candidates/{ann['id']}").status_code == 200
    assert client.get(f"{V1}/analyses").json()["count"] == 0
    assert client.get(f"{V1}/search/candidates", params={"q": "FastAPI"}).json()["data"] == []
    assert client.delete(f"{V1}/candidates/{ann['id']}").status_code == 404


def test_assistant_answers_with_tools_and_stores_its_trace(env):
    client, container, _ = env
    create_job(client)
    upload(client)
    upload(client, "Bob Stone")
    container.llm.structured = [
        {"thought": "Rank the candidates of this job", "action": "rank_candidates",
         "arguments": {"job": "Python Backend Developer"}},
        {"thought": "Enough to answer", "action": "final_answer", "arguments": {}}]
    container.llm.text = "**Ann Lee** is the strongest candidate (grade 3)."
    run = client.post(f"{V1}/agent/ask", json={"question": "Who are the best candidates for the backend job?"}).json()
    assert run["answer"].startswith("**Ann Lee**") and [s["tool"] for s in run["steps"]] == ["rank_candidates"]
    step = run["steps"][0]
    assert step["ok"] and step["data"]["top"][0]["name"] == "Ann Lee"
    assert {s["type"] for s in run["sources"]} == {"job", "candidate"}
    assert run["final_thought"] == "Enough to answer" and run["llm_calls"] == 3
    assert all(call["use_adapter"] is False for call in container.llm.calls)
    decide_prompt = container.llm.calls[0]["prompt"]
    assert "Python Backend Developer" in decide_prompt and "Candidates in the database: 2" in decide_prompt
    assert client.get(f"{V1}/agent/runs").json()["data"][0]["id"] == run["id"]
    tools = {t["name"] for t in client.get(f"{V1}/agent/tools").json()["data"]}
    assert {"search_candidates", "search_hr_knowledge", "rank_candidates", "rerank_candidates"} <= tools


def test_assistant_guards(env):
    client, container, _ = env
    upload(client)
    same = {"thought": "", "action": "get_candidate", "arguments": {"candidate": "Ann Lee"}}
    container.llm.structured = [same, {"action": "no_such_tool", "arguments": "{}"}, same]
    container.llm.text = RuntimeError("server stopped")
    run = client.post(f"{V1}/agent/ask", json={"question": "Tell me about Ann Lee"}).json()
    assert [s["tool"] for s in run["steps"]] == ["get_candidate", "no_such_tool"]
    assert run["steps"][0]["data"]["name"] == "Ann Lee" and "email" not in run["steps"][0]["data"]
    assert not run["steps"][1]["ok"] and "unknown tool" in run["steps"][1]["error"]
    assert any("Repeated call" in w for w in run["warnings"])
    assert "here is what the tools found" in run["answer"]  # the model failed: the tool results are shown
    assert client.post(f"{V1}/agent/ask", json={"question": "?"}).status_code == 422


def test_assistant_falls_back_when_planning_fails(env):
    client, container, _ = env
    container.llm.structured = [RuntimeError("bad JSON"), {"action": "final_answer"}]
    run = client.post(f"{V1}/agent/ask", json={"question": "How are internships counted?"}).json()
    assert [s["tool"] for s in run["steps"]] == ["search_hr_knowledge"]
    assert any("Planning step failed" in w for w in run["warnings"])


def test_n8n_endpoints(env):
    # a CV received by n8n goes into the application pipeline: tests/unit/test_pipeline.py
    client, container, _ = env
    upload(client)
    container.llm.structured = [{"title": "Data Engineer", "requirements": [{"text": "Python"}]}]
    job = client.post(f"{V1}/webhooks/n8n/job", json={"text": "We hire a data engineer. " * 5}).json()
    assert job["title"] == "Data Engineer" and job["source"] == "n8n"
    digest = client.get(f"{V1}/webhooks/n8n/digest").json()
    assert digest["title"].startswith("Recruiting digest") and "1 candidates not analysed yet" in digest["message"]
    note = client.post(f"{V1}/notifications", json={"event": "digest", "title": "From n8n", "source": "n8n"}).json()
    assert note["source"] == "n8n"
    assert client.post(f"{V1}/notifications/{note['id']}/read").json()["read"] is True
    assert client.post(f"{V1}/notifications/read-all").json()["marked_read"] >= 1
    assert client.get(f"{V1}/notifications").json()["unread"] == 0


def test_setup_loads_the_demo_once_and_indexes_the_knowledge_base(env):
    from recruitment_ai.services import demo

    _, container, factory = env
    db = factory()
    first = demo.setup(db, container, root=ROOT)
    assert first["knowledge_chunks"] > 30 and first["jobs_created"] == 5 and first["candidates_created"] == 12
    again = demo.setup(db, container, root=ROOT)
    assert again["jobs_created"] == 0 and again["candidates_created"] == 0 and again["candidates_indexed"] == 12
    counts = container.index.counts()
    assert counts["hr_knowledge"] == first["knowledge_chunks"] and counts["jobs"] > 0 and counts["candidates"] > 0
    hits = container.index.search_knowledge("How are internships counted in years of experience?", top_k=3)
    assert hits and all(h.metadata.get("document_id") for h in hits)
    db.close()


def test_status_and_health(env):
    client, _, _ = env
    status = client.get(f"{V1}/admin/status").json()
    assert status["llm_provider"] == "mock" and status["ai_server"] == {"used": False}
    assert set(status["collections"]) == {"hr_knowledge", "candidates", "jobs"}
    assert client.get("/health").json()["status"] == "ok"
