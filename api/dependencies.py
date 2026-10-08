"""FastAPI dependency injection providers."""

from sqlalchemy.orm import Session

from recruitment_ai.config.settings import Settings, get_settings
from recruitment_ai.services.candidate_service import CandidateService


def get_app_settings() -> Settings:
    """Dependency provider for application settings."""
    return get_settings()


def get_candidate_service(
    db: Session = None,  # type: ignore[assignment]
) -> CandidateService:
    """Dependency provider for CandidateService."""
    # When DB is wired via Depends(get_db), pass session, else instantiate standalone
    return CandidateService(db=db)
