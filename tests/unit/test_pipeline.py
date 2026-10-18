"""The application pipeline through the API: ONE workflow from the CV to the final decision, run by the
API (in the background) or step by step by n8n, and resumed by the recruiter's decision.

Uses the ``env`` fixture of conftest.py: an empty SQLite database and fake models behind the API; the
real pipeline, screening agent, search index (embedded Qdrant), matching and report code run.
"""

import base64

import httpx
from fastapi.testclient import TestClient

from recruitment_ai.services import pipeline
from tests.fakes import ANN_CV, BOB, BOB_CV, DESIGN_JOB, JOB

V1 = "/api/v1"
RESUME = "http://n8n.test/webhook-waiting/42"
STEPS = ["intake", "document", "extract", "store", "acknowledge", "screen", "report", "notify_recruiters",
         "validation", "decision", "notify_candidate", "close"]
JUDGEMENT = {"requirements": [
    {"id": "M1", "level": "demonstrated", "evidence": "Built REST APIs in Python", "years": 5},
    {"id": "M2", "level": "demonstrated", "evidence": "REST APIs in Python with FastAPI"},
    {"id": "M3", "level": "demonstrated", "evidence": "FastAPI and PostgreSQL"},
    {"id": "N1", "level": "listed", "evidence": "Kubernetes"}],
    "summary": "Strong backend profile.", "gaps": ["Kubernetes only listed"]}


def b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def cv(text: str = ANN_CV, **extra: object) -> dict:
    return {"filename": "cv.pdf", "content_base64": b64(text), "source": "upload", **extra}


def statuses(app: dict) -> dict:
    return {s["step"]: s["status"] for s in app["steps"]}


def events(client: TestClient) -> list[str]:
    return [n["event"] for n in client.get(f"{V1}/notifications").json()["data"]]


def step(client: TestClient, app_id: str, name: str, **body: object):
    return client.post(f"{V1}/applications/{app_id}/steps/{name}", json=body)


def until_review(client: TestClient, container, job_id: str) -> tuple[str, str]:
    """What the n8n workflow calls, node by node, until it waits for the recruiter: (application, analysis)."""
    app_id = client.post(f"{V1}/applications", json=cv(source="portal", consent=True, job_id=job_id)).json()["id"]
    assert step(client, app_id, "document", content_base64=b64(ANN_CV)).status_code == 200
    for name in ("extract", "store", "acknowledge"):
        assert step(client, app_id, name).status_code == 200, name
    container.llm.structured = {"MatchJudgement": [JUDGEMENT]}
    for name in ("screen", "report"):
        assert step(client, app_id, name).status_code == 200, name
    waiting = step(client, app_id, "notify_recruiters", resume_url=RESUME).json()["application"]
    return app_id, waiting["analyses"][0]["id"]


def test_one_workflow_from_the_cv_to_the_final_decision(env):
    client, container, _ = env
    job = client.post(f"{V1}/jobs", json=JOB).json()
    container.llm.structured = {"MatchJudgement": [JUDGEMENT]}
    response = client.post(f"{V1}/applications/submit", json=cv(job_id=job["id"]))
    assert response.status_code == 202, response.text
    app_id = response.json()["id"]
    app = client.get(f"{V1}/applications/{app_id}").json()  # the background run is over (the test client waits)
    assert app["status"] == "waiting_review" and app["stage"] == "validation" and app["orchestrator"] == "api"
    assert [s["step"] for s in app["steps"]] == STEPS
    done = statuses(app)
    assert all(done[s] == "done" for s in STEPS[:8]) and done["validation"] == "waiting" and done["close"] == "pending"
    steps = {s["step"]: s for s in app["steps"]}
    assert steps["document"]["result"]["characters"] > 100 and steps["extract"]["result"]["full_name"] == "Ann Lee"
    screen = steps["screen"]["result"]
    assert screen["jobs_analysed"][0]["title"] == "Python Backend Developer" and screen["llm_calls"] == 1
    assert [e["node"] for e in screen["timeline"]][:4] == ["understand", "plan", "act", "evaluate"]
    analysis = app["analyses"][0]
    assert analysis["method"] == "agent" and analysis["status"] == "pending_review" and analysis["grade"] == 3
    assert app["candidate_name"] == "Ann Lee" and app["applied_for"] == "Python Backend Developer"
    assert {"application.received", "application.review_needed"} <= set(events(client))
    assert not {"candidate.created", "analysis.completed"} & set(events(client))  # the pipeline writes its own
    candidate_id = app["candidate_id"]
    assert client.get(f"{V1}/candidates/{candidate_id}").json()["source"] == "upload"
    assert "FastAPI" in client.get(f"{V1}/candidates/{candidate_id}/cv-text").json()["cv_text"]
    stored = client.get(f"{V1}/analyses/{analysis['id']}").json()
    assert stored["application"]["status"] == "waiting_review" and stored["report"]["trace"]["agent"]["tool_calls"]
    assert "Screening agent" in client.get(f"{V1}/analyses/{analysis['id']}/report.md").text

    # the recruiter decides: the same workflow goes on to the end
    decided = client.post(f"{V1}/analyses/{analysis['id']}/review",
                          json={"decision": "interview", "reviewer": "Sam", "interview_at": "2026-10-20 10:00"}).json()
    assert decided["application"]["status"] == "closed"
    app = client.get(f"{V1}/applications/{app_id}").json()
    assert app["decision"] == "interview" and set(statuses(app).values()) == {"done"}
    answer = next(n for n in client.get(f"{V1}/notifications").json()["data"] if n["event"] == "application.answer")
    assert "interview on 2026-10-20 10:00" in answer["message"] and "ann.lee@example.com" in answer["message"]
    assert step(client, app_id, "close").json()["result"]["already_done"]

    # right to erasure: deleting the candidate deletes the application too
    assert client.delete(f"{V1}/candidates/{candidate_id}").status_code == 200
    assert client.get(f"{V1}/applications").json()["count"] == 0


def test_n8n_runs_the_steps_one_by_one_and_the_decision_resumes_it(env, monkeypatch):
    client, container, _ = env
    job = client.post(f"{V1}/jobs", json=JOB).json()
    created = client.post(f"{V1}/applications", json=cv(source="portal", consent=True, job_id=job["id"]))
    assert created.status_code == 201 and created.json()["orchestrator"] == "n8n"
    app_id = created.json()["id"]
    assert step(client, app_id, "extract").status_code == 409     # not yet: document processing comes first
    assert step(client, app_id, "validation").status_code == 409  # a recruiter's decision, not a call
    assert step(client, app_id, "nonsense").status_code == 404
    document = step(client, app_id, "document", content_base64=b64(ANN_CV)).json()
    assert document["result"]["engine"] == "PyMuPDF" and document["application"]["stage"] == "document"
    for name in ("extract", "store"):
        assert step(client, app_id, name).status_code == 200
    ack = step(client, app_id, "acknowledge").json()["result"]
    assert ack["full_name"] == "Ann Lee" and "10 working days" in ack["message"] and ack["to"] == "ann.lee@example.com"
    assert step(client, app_id, "acknowledge").json()["result"]["already_done"]  # a retried node never runs twice
    container.llm.structured = {"MatchJudgement": [JUDGEMENT]}
    assert step(client, app_id, "screen").json()["result"]["jobs_analysed"][0]["grade"] == 3
    report = step(client, app_id, "report").json()["result"]["analyses"][0]
    assert report["markdown"] == f"{V1}/analyses/{report['id']}/report.md"
    waiting = step(client, app_id, "notify_recruiters", resume_url=RESUME).json()["application"]
    assert waiting["status"] == "waiting_review" and waiting["orchestrator"] == "n8n"
    assert step(client, app_id, "decision").json()["result"]["decided"] is False  # no decision yet: n8n waits again
    assert step(client, app_id, "notify_candidate").status_code == 409

    resumed = []
    monkeypatch.setattr(pipeline.httpx, "post",
                        lambda url, json=None, timeout=None: resumed.append((url, json)) or httpx.Response(200))
    client.post(f"{V1}/analyses/{report['id']}/review", json={"decision": "reject", "notes": "Needs Kubernetes"})
    assert resumed == [(RESUME, {"application_id": app_id, "decision": "reject"})]
    assert client.get(f"{V1}/applications/{app_id}").json()["status"] == "waiting_review"  # n8n runs the rest
    decision = step(client, app_id, "decision").json()["result"]
    assert decision["decided"] and decision["decision"] == "reject"
    answer = step(client, app_id, "notify_candidate").json()["result"]
    assert "will not continue" in answer["text"] and "Kubernetes" not in answer["text"]  # the reason stays internal
    closed = step(client, app_id, "close").json()["application"]
    assert closed["status"] == "closed" and closed["decision"] == "reject"


def test_the_api_finishes_the_workflow_when_n8n_does_not_answer(env, monkeypatch):
    client, container, _ = env
    job = client.post(f"{V1}/jobs", json=JOB).json()
    app_id, analysis_id = until_review(client, container, job["id"])

    def refused(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(pipeline.httpx, "post", refused)
    monkeypatch.setattr(pipeline, "RESUME_RETRY_SECONDS", 0)
    decided = client.post(f"{V1}/analyses/{analysis_id}/review", json={"decision": "shortlist"}).json()
    assert decided["application"]["status"] == "closed"
    app = client.get(f"{V1}/applications/{app_id}").json()
    assert app["orchestrator"] == "api" and app["decision"] == "shortlist"
    validation = next(s for s in app["steps"] if s["step"] == "validation")
    assert "n8n did not answer" in validation["note"]
    answer = next(n for n in client.get(f"{V1}/notifications").json()["data"] if n["event"] == "application.answer")
    assert "within 5 working days" in answer["message"]


def test_the_resume_address_follows_n8n_url():
    url = "http://localhost:5678/webhook-waiting/7?signature=abc"  # n8n's public address
    assert pipeline.resume_address(url, "http://n8n:5678") == "http://n8n:5678/webhook-waiting/7?signature=abc"
    assert pipeline.resume_address(url, "http://localhost:5678") == url


def test_a_failed_step_is_recorded_and_the_workflow_continues_from_it(env):
    client, container, _ = env
    job = client.post(f"{V1}/jobs", json=JOB).json()
    container.extractor.text_profiles = [RuntimeError("The Kaggle AI server does not answer")]
    app_id = client.post(f"{V1}/applications/submit", json=cv(job_id=job["id"])).json()["id"]
    app = client.get(f"{V1}/applications/{app_id}").json()
    assert app["status"] == "failed" and app["stage"] == "extract" and "does not answer" in app["error"]
    assert [statuses(app)[s] for s in ("document", "extract", "store")] == ["done", "failed", "pending"]
    assert "application.failed" in events(client)
    # the AI server is back: the CV text was kept between the steps, the file is not needed again
    container.llm.structured = {"MatchJudgement": [JUDGEMENT]}
    continued = client.post(f"{V1}/applications/{app_id}/continue").json()
    assert continued["status"] == "waiting_review" and statuses(continued)["extract"] == "done"
    assert client.post(f"{V1}/applications/{app_id}/continue").json()["status"] == "waiting_review"  # waits


def test_intake_checks(env):
    client, _, _ = env
    job = client.post(f"{V1}/jobs", json=JOB).json()

    def submit(body: dict) -> httpx.Response:
        return client.post(f"{V1}/applications/submit", json=body)

    assert submit(cv(source="portal")).status_code == 400            # no consent (KB-HR-004)
    assert submit(cv(filename="cv.exe")).status_code == 400          # file type
    assert submit(cv(job_id="unknown")).status_code == 400
    assert submit({**cv(), "content_base64": "%" * 12}).status_code == 400
    client.patch(f"{V1}/jobs/{job['id']}", json={"status": "closed"})
    assert "closed" in submit(cv(job_id=job["id"])).json()["detail"]
    assert client.get(f"{V1}/applications").json()["count"] == 0
    assert [s["step"] for s in client.get(f"{V1}/applications/steps").json()["data"]] == STEPS


def test_a_spontaneous_application_goes_to_the_job_that_fits_and_on_hold_keeps_waiting(env):
    client, container, _ = env
    client.post(f"{V1}/jobs", json=JOB)
    design = client.post(f"{V1}/jobs", json=DESIGN_JOB).json()
    container.extractor.text_profiles = [BOB]
    container.llm.structured = {"MatchJudgement": [{"requirements": [
        {"id": "M1", "level": "demonstrated", "evidence": "Designed brand identities with Photoshop and Illustrator"},
        {"id": "M2", "level": "equivalent", "evidence": "Designed brand identities"}], "summary": "Designer."}]}
    app_id = client.post(f"{V1}/applications/submit", json=cv(BOB_CV)).json()["id"]
    app = client.get(f"{V1}/applications/{app_id}").json()
    assert app["status"] == "waiting_review" and app["applied_for"].startswith("spontaneous")
    screen = next(s for s in app["steps"] if s["step"] == "screen")["result"]
    assert [c["title"] for c in screen["jobs_considered"]] == ["Graphic Designer", "Python Backend Developer"]
    assert [a["job_id"] for a in app["analyses"]] == [design["id"]]
    analysis_id = app["analyses"][0]["id"]
    client.post(f"{V1}/analyses/{analysis_id}/review", json={"decision": "hold"})
    assert client.get(f"{V1}/applications/{app_id}").json()["status"] == "waiting_review"  # on hold: still waiting
    client.post(f"{V1}/analyses/{analysis_id}/review", json={"decision": "interview", "interview_at": "2026-11-02 14:00"})
    app = client.get(f"{V1}/applications/{app_id}").json()
    assert app["status"] == "closed" and app["decision"] == "interview"
