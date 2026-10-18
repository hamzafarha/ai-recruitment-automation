"""Unit tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

import api.routers.health as health_router


def test_root_health(api_client: TestClient):
    response = api_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.fixture
def services(monkeypatch):
    """Fake PostgreSQL/Qdrant checks: unit tests must not depend on Docker being up
    (a real connection attempt to a stopped database used to hang for minutes)."""
    state = {"db": (True, "PostgreSQL connected successfully."), "qdrant": (True, "Qdrant connected.")}
    monkeypatch.setattr(health_router, "check_db_connection", lambda: state["db"])
    monkeypatch.setattr(health_router.QdrantVectorStore, "health_check", lambda self: state["qdrant"])
    return state


def test_api_v1_health(api_client: TestClient, services):
    response = api_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "llm_provider" in data["components"]


def test_api_v1_health_degraded_when_a_service_is_down(api_client: TestClient, services):
    services["qdrant"] = (False, "Qdrant health check failed: connection refused")
    data = api_client.get("/api/v1/health").json()
    assert data["status"] == "degraded"
    assert data["components"]["qdrant"]["healthy"] is False
    assert data["components"]["postgresql"]["healthy"] is True


def test_candidates_list(api_client: TestClient):
    response = api_client.get("/api/v1/candidates")
    assert response.status_code == 200
    assert "data" in response.json() and "count" in response.json()


def test_removed_placeholder_endpoints_are_gone(api_client: TestClient):
    """The scaffold's fake endpoints (fixed scores, "placeholder" answers) were replaced by real ones."""
    assert api_client.post("/api/v1/feedback", json={}).status_code == 404
    assert api_client.post("/api/v1/matching", json={}).status_code == 404
