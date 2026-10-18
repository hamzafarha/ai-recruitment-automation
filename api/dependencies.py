"""FastAPI dependency providers (unit tests override them with an SQLite session and fake models)."""

from recruitment_ai.config.settings import Settings, get_settings
from recruitment_ai.database.connection import get_db
from recruitment_ai.services.container import Container, get_container

__all__ = ["get_app_settings", "get_container", "get_db", "Container"]


def get_app_settings() -> Settings:
    """Dependency provider for application settings."""
    return get_settings()
