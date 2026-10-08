"""Unit tests for database models and repositories using in-memory SQLite."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from recruitment_ai.database.connection import Base
from recruitment_ai.database.models import Candidate, Job
from recruitment_ai.database.repositories import CandidateRepository, JobRepository


def test_database_models_and_repositories():
    """Verify ORM models and repository operations on clean in-memory database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    cand_repo = CandidateRepository(session)
    job_repo = JobRepository(session)

    # Create candidate
    candidate = Candidate(
        full_name="Alice Smith",
        email="alice@example.com",
        location="San Francisco, CA",
    )
    created_cand = cand_repo.create(candidate)
    assert created_cand.id is not None
    assert created_cand.full_name == "Alice Smith"

    # Query candidate
    fetched = cand_repo.get_by_email("alice@example.com")
    assert fetched is not None
    assert fetched.id == created_cand.id

    # Create Job
    job = Job(
        title="Senior Python Architect",
        description="Lead backend architecture",
    )
    created_job = job_repo.create(job)
    assert created_job.id is not None

    session.close()
