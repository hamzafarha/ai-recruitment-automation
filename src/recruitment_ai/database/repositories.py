"""Database repository classes for clean separation of concerns."""

from typing import Generic, TypeVar

from sqlalchemy.orm import Session

from recruitment_ai.database.models import (
    Candidate,
    CandidateMatch,
    Job,
    RecruiterFeedback,
)

ModelType = TypeVar("ModelType")


class BaseRepository(Generic[ModelType]):
    """Base generic repository with common query methods."""

    def __init__(self, model: type[ModelType], db: Session) -> None:
        self.model = model
        self.db = db

    def get(self, entity_id: str) -> ModelType | None:
        """Get single record by ID."""
        return self.db.query(self.model).filter(self.model.id == entity_id).first()  # type: ignore[attr-defined]

    def list(self, skip: int = 0, limit: int = 100) -> list[ModelType]:
        """List records with pagination."""
        return self.db.query(self.model).offset(skip).limit(limit).all()

    def create(self, obj: ModelType) -> ModelType:
        """Add and commit record."""
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def delete(self, entity_id: str) -> bool:
        """Delete record by ID."""
        obj = self.get(entity_id)
        if obj:
            self.db.delete(obj)
            self.db.commit()
            return True
        return False


class CandidateRepository(BaseRepository[Candidate]):
    """Repository for candidate operations."""

    def __init__(self, db: Session) -> None:
        super().__init__(Candidate, db)

    def get_by_email(self, email: str) -> Candidate | None:
        """Find candidate by email address."""
        return self.db.query(Candidate).filter(Candidate.email == email).first()


class JobRepository(BaseRepository[Job]):
    """Repository for job vacancies."""

    def __init__(self, db: Session) -> None:
        super().__init__(Job, db)


class MatchRepository(BaseRepository[CandidateMatch]):
    """Repository for candidate matching results."""

    def __init__(self, db: Session) -> None:
        super().__init__(CandidateMatch, db)

    def get_for_job(self, job_id: str) -> list[CandidateMatch]:
        """Get candidate matches for a specific job."""
        return (
            self.db.query(CandidateMatch)
            .filter(CandidateMatch.job_id == job_id)
            .order_by(CandidateMatch.overall_score.desc())
            .all()
        )


class FeedbackRepository(BaseRepository[RecruiterFeedback]):
    """Repository for recruiter feedback."""

    def __init__(self, db: Session) -> None:
        super().__init__(RecruiterFeedback, db)
