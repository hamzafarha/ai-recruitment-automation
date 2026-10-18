"""Database repositories: the queries the services need, nothing more."""

import re
from typing import Generic, TypeVar

from sqlalchemy import func
from sqlalchemy.orm import Session

from recruitment_ai.database.models import (
    AgentRun,
    Analysis,
    Application,
    Candidate,
    Job,
    Notification,
    Review,
)

ModelType = TypeVar("ModelType")
_ID_PREFIX = re.compile(r"[0-9a-f-]{8,36}")


class BaseRepository(Generic[ModelType]):
    def __init__(self, model: type[ModelType], db: Session) -> None:
        self.model = model
        self.db = db

    def get(self, entity_id: str) -> ModelType | None:
        return self.db.get(self.model, entity_id)

    def list(self, skip: int = 0, limit: int = 500) -> list[ModelType]:
        return (self.db.query(self.model).order_by(self.model.created_at.desc())  # type: ignore[attr-defined]
                .offset(skip).limit(limit).all())

    def create(self, obj: ModelType) -> ModelType:
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def delete(self, entity_id: str) -> bool:
        obj = self.get(entity_id)
        if obj is None:
            return False
        self.db.delete(obj)
        self.db.commit()
        return True

    def count(self) -> int:
        return self.db.query(func.count(self.model.id)).scalar() or 0  # type: ignore[attr-defined]


class CandidateRepository(BaseRepository[Candidate]):
    def __init__(self, db: Session) -> None:
        super().__init__(Candidate, db)

    def get_by_email(self, email: str) -> Candidate | None:
        return self.db.query(Candidate).filter(Candidate.email == email).first()

    def get_by_external_id(self, external_id: str) -> Candidate | None:
        return self.db.query(Candidate).filter(Candidate.external_id == external_id).first()

    def find(self, reference: str) -> Candidate | None:
        """By id, external id, or (case-insensitive) name."""
        reference = reference.strip()
        found = self.get(reference) or self.get_by_external_id(reference)
        if found is None and _ID_PREFIX.fullmatch(reference.lower()):  # a shortened id, e.g. "3f2a9c1e"
            found = self.db.query(Candidate).filter(Candidate.id.startswith(reference.lower())).first()
        if found is None and reference:
            found = (self.db.query(Candidate).filter(func.lower(Candidate.full_name) == reference.lower()).first()
                     or self.db.query(Candidate)
                     .filter(func.lower(Candidate.full_name).contains(reference.lower())).first())
        return found


class JobRepository(BaseRepository[Job]):
    def __init__(self, db: Session) -> None:
        super().__init__(Job, db)

    def get_by_external_id(self, external_id: str) -> Job | None:
        return self.db.query(Job).filter(Job.external_id == external_id).first()

    def find(self, reference: str) -> Job | None:
        """By id, external id, role key or (case-insensitive) title."""
        reference = reference.strip()
        found = self.get(reference) or self.get_by_external_id(reference)
        if found is None and _ID_PREFIX.fullmatch(reference.lower()):
            found = self.db.query(Job).filter(Job.id.startswith(reference.lower())).first()
        if found is None and reference:
            lower = reference.lower()
            found = (self.db.query(Job).filter(func.lower(Job.title) == lower).first()
                     or self.db.query(Job).filter(Job.role == lower.replace(" ", "_")).first()
                     or self.db.query(Job).filter(func.lower(Job.title).contains(lower)).first())
        return found


class AnalysisRepository(BaseRepository[Analysis]):
    def __init__(self, db: Session) -> None:
        super().__init__(Analysis, db)

    def filter(self, job_id: str | None = None, candidate_id: str | None = None, status: str | None = None,
               limit: int = 500) -> list[Analysis]:
        query = self.db.query(Analysis)
        if job_id:
            query = query.filter(Analysis.job_id == job_id)
        if candidate_id:
            query = query.filter(Analysis.candidate_id == candidate_id)
        if status:
            query = query.filter(Analysis.status == status)
        return query.order_by(Analysis.created_at.desc()).limit(limit).all()


class ApplicationRepository(BaseRepository[Application]):
    def __init__(self, db: Session) -> None:
        super().__init__(Application, db)

    def filter(self, status: str | None = None, candidate_id: str | None = None,
               limit: int = 500) -> list[Application]:
        query = self.db.query(Application)
        if status:
            query = query.filter(Application.status == status)
        if candidate_id:
            query = query.filter(Application.candidate_id == candidate_id)
        return query.order_by(Application.created_at.desc()).limit(limit).all()


class ReviewRepository(BaseRepository[Review]):
    def __init__(self, db: Session) -> None:
        super().__init__(Review, db)


class NotificationRepository(BaseRepository[Notification]):
    def __init__(self, db: Session) -> None:
        super().__init__(Notification, db)

    def unread_count(self) -> int:
        return self.db.query(func.count(Notification.id)).filter(Notification.read.is_(False)).scalar() or 0


class AgentRunRepository(BaseRepository[AgentRun]):
    def __init__(self, db: Session) -> None:
        super().__init__(AgentRun, db)
