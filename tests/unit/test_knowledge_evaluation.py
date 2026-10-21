"""The documents of the HR knowledge base, the evaluation results (data/evaluation/), the base vs fine-tuned
comparison on a stored candidate, and the React interface served by the API."""

import base64

from api import main
from tests.fakes import ANN_CV, JOB

V1 = "/api/v1"


def test_the_exact_knowledge_documents_the_rag_cites(env):
    client, _, _ = env
    docs = client.get(f"{V1}/knowledge/documents").json()
    assert docs["count"] == 15
    rubric = next(d for d in docs["data"] if d["doc_id"] == "KB-RR-002")
    assert rubric["category"] == "recruitment_rules" and "Evidence levels" in rubric["sections"]
    full = client.get(f"{V1}/knowledge/documents/kb-rr-002").json()  # ids are case-insensitive
    assert "## Evidence levels" in full["content"] and not full["content"].startswith("---")
    assert full["chunk_list"][0]["chunk_id"] == "KB-RR-002#0" and full["chunks"] == len(full["chunk_list"])
    assert client.get(f"{V1}/knowledge/documents/KB-XX-999").status_code == 404


def test_every_evaluation_is_served(env):
    client, _, _ = env
    overview = client.get(f"{V1}/evaluation/overview").json()
    assert [c["key"] for c in overview["components"]] == ["extraction", "finetune", "rag", "matching", "screening",
                                                          "assistant"]
    assert overview["headline"]["extraction"]["after"] > 0 and "dense+rerank" in overview["headline"]["rag"]
    ex = client.get(f"{V1}/evaluation/extraction").json()
    assert ex["before"] == "ocr_llm_Qwen2.5-7B-Instruct_dev" and ex["after"].endswith("-finetuned_dev")
    assert ex["metrics"][0]["metric"] == "Overall F1" and {g for g in ex["groups"]} == {"input_type", "source",
                                                                                         "language"}
    assert [s["section"] for s in ex["sections"]] == ["experience", "education", "skills"] and ex["documents"]
    assert client.get(f"{V1}/evaluation/extraction", params={"before": "nope"}).status_code == 404
    ft = client.get(f"{V1}/evaluation/finetune").json()
    assert ft["status"] == "complete" and ft["losses"][0]["step"] == 5 and ft["params"]["LORA_R"] == 16
    rag = client.get(f"{V1}/evaluation/rag").json()
    assert {"dense", "dense+rerank"} <= set(rag["configurations"]) and len(rag["per_query"]) > 40
    matching = client.get(f"{V1}/evaluation/matching").json()
    assert any(name.startswith("agent") for name in matching["systems"]) and matching["agent"]["pairs"] == 36
    assistant = client.get(f"{V1}/evaluation/assistant").json()
    assert len(assistant["scenarios"]) == 20 and assistant["summary"]["overall"]["scenarios"] == 20
    live = client.get(f"{V1}/evaluation/pipeline").json()
    assert live["applications"] == 0 and len(live["steps"]) == 12
    mlflow = client.get(f"{V1}/evaluation/mlflow").json()
    assert mlflow["reachable"] is False and mlflow["tracking_uri"].startswith("sqlite")


def test_base_and_fine_tuned_models_on_a_stored_candidate(env):
    client, container, _ = env
    job = client.post(f"{V1}/jobs", json=JOB).json()
    body = {"filename": "cv.pdf", "content_base64": base64.b64encode(ANN_CV.encode()).decode(), "job_id": job["id"]}
    app_id = client.post(f"{V1}/applications/submit", json=body).json()["id"]
    candidate_id = client.get(f"{V1}/applications/{app_id}").json()["candidate_id"]
    assert client.get(f"{V1}/candidates/{candidate_id}/compare-models").json()["comparison"] is None
    result = client.post(f"{V1}/candidates/{candidate_id}/compare-models").json()
    assert container.extractor.adapter_calls[-1] is False  # one run of the base model, without the adapter
    assert result["comparison"]["counts"]["experience"] == {"finetuned": 2, "base": 1}
    assert result["comparison"]["skills_only_base"] == ["teamwork"]
    assert result["finetuned"]["profile"]["full_name"] == "Ann Lee" and result["base"]["adapter_used"] is False
    assert client.get(f"{V1}/candidates/{candidate_id}/compare-models").json()["comparison"]["at"] == result["at"]
    outputs = client.get(f"{V1}/applications/{app_id}/outputs").json()
    assert outputs["extract"]["model_comparison"]["comparison"]["counts"]["skills"]["base"] == 3
    assert client.post(f"{V1}/candidates/unknown/compare-models").status_code == 404


def test_the_api_serves_the_built_interface(env, tmp_path, monkeypatch):
    client, _, _ = env
    monkeypatch.setattr(main, "WEB_DIST", tmp_path)
    assert "not built" in client.get("/pipeline").json()["detail"]
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<div id='root'></div>", encoding="utf-8")
    (tmp_path / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    page = client.get("/pipeline/abc")
    assert page.status_code == 200 and "root" in page.text  # the single-page app answers every route
    script = client.get("/assets/app.js")
    assert "javascript" in script.headers["content-type"] and "immutable" in script.headers["cache-control"]
    assert client.get(f"{V1}/nothing-here").status_code == 404  # API paths never answer the page
    assert client.get("/health").json()["status"] == "ok"
