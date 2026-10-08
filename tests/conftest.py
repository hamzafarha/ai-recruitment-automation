"""Pytest fixtures and configuration."""

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from recruitment_ai.config.settings import Settings, get_settings
from recruitment_ai.database.connection import Base, get_db

# In-memory SQLite engine with StaticPool so all threads/sessions share the created tables
_test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
Base.metadata.create_all(bind=_test_engine)
_TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_test_engine)


def _override_get_db():
    db = _TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db


@pytest.fixture(autouse=True)
def set_test_env():
    """Ensure safe testing environment variables."""
    os.environ["APP_ENV"] = "testing"
    os.environ["LLM_PROVIDER"] = "mock"
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["COHERE_API_KEY"] = ""
    os.environ["LANGCHAIN_API_KEY"] = ""


@pytest.fixture
def test_settings() -> Settings:
    """Fixture providing isolated Settings instance."""
    return get_settings()


@pytest.fixture
def api_client() -> TestClient:
    """FastAPI TestClient fixture."""
    return TestClient(app)
