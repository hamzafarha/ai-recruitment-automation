"""Unit tests for FastAPI endpoints."""

from fastapi.testclient import TestClient


def test_root_health(api_client: TestClient):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_api_v1_health(api_client: TestClient):
    response = api_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "components" in data
    assert "llm_provider" in data["components"]


def test_candidates_list(api_client: TestClient):
    response = api_client.get("/api/v1/candidates")
    assert response.status_code == 200
    assert "data" in response.json()


def test_document_upload_validation(api_client: TestClient):
    # Test valid text file upload
    response = api_client.post(
        "/api/v1/documents/upload",
        files={"file": ("resume.txt", b"Sample Resume Text", "text/plain")},
    )
    # Since txt is not in default allowed_extensions (pdf, docx, png, jpg, jpeg), should return 400
    assert response.status_code == 400

    # Test allowed pdf upload
    response_pdf = api_client.post(
        "/api/v1/documents/upload",
        files={
            "file": (
                "resume.pdf",
                b"%PDF-1.4 sample pdf binary data",
                "application/pdf",
            )
        },
    )
    assert response_pdf.status_code == 200
    assert response_pdf.json()["status"] == "success"


def test_recruiter_feedback_endpoint(api_client: TestClient):
    payload = {
        "match_id": "test-match-id",
        "recruiter_id": "recruiter_01",
        "decision": "Approved",
        "rating": 5,
        "comments": "Selected for interview stage",
    }
    response = api_client.post("/api/v1/feedback", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["data"]["decision"] == "Approved"
