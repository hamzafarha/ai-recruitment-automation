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
    # Embedded mode (no server, e.g. on Kaggle where Docker is unavailable): ":memory:" or a folder path
    qdrant_location: str | None = Field(default=None, alias="QDRANT_LOCATION")

    # Embedding Settings (Architecture: BAAI/bge-m3)
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    embedding_dimension: int = Field(default=1024, alias="EMBEDDING_DIMENSION")
    # local = BGE-M3 and the reranker on this machine; remote = on the Kaggle AI server
    embedding_provider: Literal["local", "remote"] = Field(default="local", alias="EMBEDDING_PROVIDER")

    # n8n orchestrator: the API sends its events (new candidate, analysis done, decision...) to the
    # "recruitment-events" workflow; n8n calls the API back at http://host.docker.internal:8000
    n8n_url: str = Field(default="http://localhost:5678", alias="N8N_URL")
    n8n_events_enabled: bool = Field(default=False, alias="N8N_EVENTS_ENABLED")
    # optional: an n8n API key (n8n -> Settings -> n8n API) lets the app list the workflows' executions
    n8n_api_key: str | None = Field(default=None, alias="N8N_API_KEY")

    # LLM Settings (Architecture: Qwen2.5 & Qwen2.5-VL, run in Python by Hugging Face Transformers)
    llm_provider: Literal["mock", "openai", "huggingface", "remote"] = Field(
        default="mock", alias="LLM_PROVIDER"
    )
    # "remote": the model runs in a Kaggle GPU session (notebooks/kaggle/model_server.ipynb
    # prints both values; the link changes at every session)
    remote_llm_url: str | None = Field(default=None, alias="REMOTE_LLM_URL")
    remote_llm_token: str | None = Field(default=None, alias="REMOTE_LLM_TOKEN")
    huggingface_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct", alias="HUGGINGFACE_MODEL"
    )
    multimodal_model: str = Field(
        default="Qwen/Qwen2.5-VL-7B-Instruct", alias="MULTIMODAL_MODEL"
    )
    # cuda (GPU: the Kaggle jobs set it), cpu or auto
    llm_device: str = Field(default="cpu", alias="LLM_DEVICE")
    # auto = float32 on CPU, float16 on GPU; bfloat16 halves the memory on CPU
    llm_dtype: Literal["auto", "float32", "bfloat16", "float16"] = Field(
        default="auto", alias="LLM_DTYPE"
    )
    # Optional LoRA adapter produced by our fine-tuning (folder with adapter_config.json)
    llm_adapter: str | None = Field(default=None, alias="LLM_ADAPTER")
    # Largest page image given to the vision model, in pixels (Qwen2.5-VL: 1 token per 28x28 block)
    vlm_max_pixels: int = Field(default=1280 * 28 * 28, alias="VLM_MAX_PIXELS")
    # "openai" provider: any OpenAI-compatible server
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    llm_timeout_seconds: float = Field(default=300.0, alias="LLM_TIMEOUT_SECONDS")

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

    # E-mails to the candidates (services/mailer.py): the acknowledgement and the answer after the
    # recruiter's decision. outbox = stored and shown in the app (Outbox page), not sent; smtp = sent.
    email_mode: Literal["outbox", "smtp"] = Field(default="outbox", alias="EMAIL_MODE")
    smtp_host: str = Field(default="smtp.gmail.com", alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_user: str | None = Field(default=None, alias="SMTP_USER")
    smtp_password: str | None = Field(default=None, alias="SMTP_PASSWORD")
    smtp_starttls: bool = Field(default=True, alias="SMTP_STARTTLS")
    email_from: str | None = Field(default=None, alias="EMAIL_FROM")  # default: SMTP_USER
    email_from_name: str = Field(default="Recruiting team", alias="EMAIL_FROM_NAME")
    # demo safety: every e-mail goes to this address instead of the candidate's (CVs of real people)
    email_redirect_to: str | None = Field(default=None, alias="EMAIL_REDIRECT_TO")
    email_acknowledge: bool = Field(default=True, alias="EMAIL_ACKNOWLEDGE")

    # Interview meetings (services/meetings.py), created when a recruiter schedules an interview.
    # google = Google Calendar event with a Google Meet link (scripts/google_calendar_setup.py once);
    # jitsi = a Jitsi Meet link (no account); auto = google when it is set up, else jitsi; none = no link
    meeting_provider: Literal["auto", "google", "jitsi", "none"] = Field(default="auto", alias="MEETING_PROVIDER")
    interview_minutes: int = Field(default=45, alias="INTERVIEW_MINUTES")
    interview_timezone: str = Field(default="Africa/Tunis", alias="INTERVIEW_TIMEZONE")
    google_client_secrets: str = Field(default="secrets/google_client_secret.json", alias="GOOGLE_CLIENT_SECRETS")
    google_token_file: str = Field(default="secrets/google_token.json", alias="GOOGLE_TOKEN_FILE")
    google_calendar_id: str = Field(default="primary", alias="GOOGLE_CALENDAR_ID")

    # Experiment tracking: every evaluation of data/evaluation/ logged by scripts/mlflow_sync.py
    mlflow_tracking_uri: str = Field(default="sqlite:///mlruns/mlflow.db", alias="MLFLOW_TRACKING_URI")
    mlflow_ui_url: str = Field(default="http://localhost:5000", alias="MLFLOW_UI_URL")

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
