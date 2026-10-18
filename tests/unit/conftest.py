"""Shared fixtures of the application tests (API with fakes)."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.main import app
from recruitment_ai.database.connection import Base, get_db
from recruitment_ai.services.container import get_container
from tests.fakes import ANN, BOB, FakeContainer, FakeExtractor


@pytest.fixture
def env():
    """The API on an empty SQLite database and a container of fakes (tests/fakes.py): the real services,
    search index (embedded Qdrant), matching, analysis and agent code run; only the models are fakes.
    Yields (client, container, session factory)."""
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    def session():
        db = factory()
        try:
            yield db
        finally:
            db.close()

    container = FakeContainer(extractor=FakeExtractor([ANN, BOB]))
    previous = dict(app.dependency_overrides)
    app.dependency_overrides[get_db] = session
    app.dependency_overrides[get_container] = lambda: container
    yield TestClient(app), container, factory
    app.dependency_overrides.clear()
    app.dependency_overrides.update(previous)
