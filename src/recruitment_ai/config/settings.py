"""Central application configuration using Pydantic Settings."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_name: str = Field(default="AI-Recruitment-Automation", alias="APP_NAME")
    app_env: Literal["development", "testing", "production"] = Field(
        default="development", alias="APP_ENV"
    )
    debug: bool = Field(default=True, alias="DEBUG")

    @field_validator("debug", mode="before")
    @classmethod
    def _parse_debug(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            if v.lower() in ("true", "1", "yes", "t", "on"):
                return True
            if v.lower() in ("false", "0", "no", "f", "off"):
                return False
        return False

    api_v1_str: str = Field(default="/api/v1", alias="API_V1_STR")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    secret_key: str = Field(
        default="temporary-dev-secret-key-12345", alias="SECRET_KEY"
    )

    # Ingestion & Filesystem
    upload_dir: Path = Field(default=Path("./data/raw/cvs"), alias="UPLOAD_DIR")
    max_upload_size_mb: int = Field(default=15, alias="MAX_UPLOAD_SIZE_MB")
    allowed_extensions_raw: list[str] | str = Field(
        default=["pdf", "docx", "png", "jpg", "jpeg"],
        alias="ALLOWED_EXTENSIONS",
    )

    @property
    def allowed_extensions(self) -> list[str]:
        """Parsed list of allowed file extensions."""
        if isinstance(self.allowed_extensions_raw, str):
            val = self.allowed_extensions_raw.strip()
            if val.startswith("[") and val.endswith("]"):
                import json

                parsed_list: list[Any] = json.loads(val)
                return [str(x).lstrip(".") for x in parsed_list]
            return [x.strip().lstrip(".") for x in val.split(",") if x.strip()]
        return [x.lstrip(".") for x in self.allowed_extensions_raw]

    # Relational Database (PostgreSQL)
    postgres_server: str = Field(default="localhost", alias="POSTGRES_SERVER")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_user: str = Field(default="postgres", alias="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="recruitment_db", alias="POSTGRES_DB")
    database_url_override: str | None = Field(default=None, alias="DATABASE_URL")

    # Vector Database (Qdrant)
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_collection: str = Field(
        default="candidate_knowledge", alias="QDRANT_COLLECTION"
    )
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")

    # Embedding Settings (Architecture: BAAI/bge-m3)
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    embedding_dimension: int = Field(default=1024, alias="EMBEDDING_DIMENSION")

    # LLM Settings (Architecture: Qwen/Qwen2.5-7B-Instruct & Qwen2.5-VL)
    llm_provider: Literal["mock", "openai", "huggingface"] = Field(
        default="mock", alias="LLM_PROVIDER"
    )
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    # Any OpenAI-compatible server, e.g. local Ollama: http://localhost:11434/v1
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    llm_timeout_seconds: float = Field(default=300.0, alias="LLM_TIMEOUT_SECONDS")
    huggingface_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct", alias="HUGGINGFACE_MODEL"
    )
    multimodal_model: str = Field(
        default="Qwen/Qwen2.5-VL-7B-Instruct", alias="MULTIMODAL_MODEL"
    )

    # Reranking (Architecture: BAAI/bge-reranker-v2-m3)
    cohere_api_key: str | None = Field(default=None, alias="COHERE_API_KEY")
    reranking_model: str = Field(
        default="BAAI/bge-reranker-v2-m3", alias="RERANKING_MODEL"
    )

    # Observability & Tracing
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str | None = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(
        default="ai-recruitment-automation", alias="LANGCHAIN_PROJECT"
    )

    # Frontend
    backend_api_url: str = Field(
        default="http://localhost:8000", alias="BACKEND_API_URL"
    )

    @property
    def database_url(self) -> str:
        """Construct database URL if override is not provided."""
        if (
            self.database_url_override
            and self.postgres_server == "localhost"
            and self.postgres_port == 5432
        ):
            return self.database_url_override
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}@"
            f"{self.postgres_server}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def qdrant_url(self) -> str:
        """Construct Qdrant connection URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
