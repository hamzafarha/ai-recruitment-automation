"""HTTP client for the Kaggle AI server (notebooks/kaggle/model_server.ipynb).

The AI server runs every model on Kaggle's GPUs: Qwen2.5 (generation), BGE-M3 (embeddings)
and the BGE reranker. This client holds its link and token and turns network problems into
messages that say what to do. Used by the remote LLM, embedding and reranking adapters.

Settings: REMOTE_LLM_URL and REMOTE_LLM_TOKEN (printed by the notebook; they change at every
Kaggle session).
"""

from typing import Any

import httpx

from recruitment_ai.config.settings import Settings, get_settings


class AIServerClient:
    """Authenticated JSON requests to the AI server."""

    def __init__(self, settings: Settings | None = None, url: str | None = None, token: str | None = None,
                 timeout: float = 60.0) -> None:
        self.settings = settings or get_settings()
        self.url = (url or self.settings.remote_llm_url or "").rstrip("/")
        self.token = token or self.settings.remote_llm_token or ""
        self.timeout = timeout

    def request(self, method: str, path: str, timeout: float | None = None, **kwargs: Any) -> dict[str, Any]:
        if not self.url:
            raise RuntimeError("REMOTE_LLM_URL is empty: start notebooks/kaggle/model_server.ipynb on Kaggle and "
                               "copy the link it prints into .env")
        try:
            response = httpx.request(method, f"{self.url}{path}", headers={"Authorization": f"Bearer {self.token}"},
                                     timeout=timeout or self.timeout, **kwargs)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"The Kaggle AI server does not answer ({type(exc).__name__}): is the Kaggle "
                               "session still running, and is REMOTE_LLM_URL the link of this session?") from exc
        if response.status_code == 401:
            raise RuntimeError("The AI server refused the token: copy REMOTE_LLM_TOKEN from the notebook again")
        if response.status_code >= 400:
            raise RuntimeError(f"The Kaggle AI server answered with error {response.status_code} on {path}: "
                               f"{response.text[:200]}")
        return response.json()

    def info(self) -> dict[str, Any]:
        """Models loaded on the server (quick: used by health checks)."""
        return self.request("GET", "/info", timeout=15)
