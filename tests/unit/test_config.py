"""Unit tests for configuration management."""

from recruitment_ai.config.settings import Settings, get_settings


def test_settings_defaults():
    """Verify default configuration attributes."""
    settings = get_settings()
    assert settings.app_name == "AI-Recruitment-Automation"
    assert settings.port == 8000
    assert "pdf" in settings.allowed_extensions
    assert settings.llm_provider in ["mock", "openai", "huggingface"]
    assert "postgresql://" in settings.database_url
    assert "http://" in settings.qdrant_url


def test_settings_overrides():
    """Verify settings allow explicit parameter overrides."""
    custom = Settings(
        APP_NAME="CustomRecruiter",
        POSTGRES_SERVER="db.internal",
        POSTGRES_PORT=5433,
    )
    assert custom.app_name == "CustomRecruiter"
    assert "db.internal:5433" in custom.database_url
