"""Remote adapter: the app runs on our PC, the Qwen model in a Kaggle GPU session.

It is a ``HuggingFaceAdapter`` whose generation step is sent to the model server
(``recruitment_ai.llm.server``) instead of a local GPU. Prompt building, JSON repair and
Pydantic validation are inherited unchanged, so a CV extracted through the app gives the same
result as in the evaluation runs.

Settings: LLM_PROVIDER=remote, REMOTE_LLM_URL and REMOTE_LLM_TOKEN (printed by
notebooks/kaggle/model_server.ipynb; the link changes at every Kaggle session).
"""

import base64
from typing import Any

import httpx

from recruitment_ai.config.settings import Settings, get_settings
from recruitment_ai.llm.huggingface_adapter import HuggingFaceAdapter


class RemoteLLMAdapter(HuggingFaceAdapter):
    """Generation by the Kaggle model server; everything else as in HuggingFaceAdapter."""

    def __init__(self, settings: Settings | None = None, url: str | None = None, token: str | None = None) -> None:
        super().__init__(settings or get_settings())
        self.url = (url or self.settings.remote_llm_url or "").rstrip("/")
        self.token = token or self.settings.remote_llm_token or ""
        self._info: dict[str, Any] | None = None

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if not self.url:
            raise RuntimeError("REMOTE_LLM_URL is empty: start notebooks/kaggle/model_server.ipynb on Kaggle and "
                               "copy the link it prints into .env")
        try:
            response = httpx.request(method, f"{self.url}{path}", headers={"Authorization": f"Bearer {self.token}"},
                                     timeout=self.settings.llm_timeout_seconds, **kwargs)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"The Kaggle model server does not answer ({type(exc).__name__}): is the Kaggle "
                               "session still running, and is REMOTE_LLM_URL the link of this session?") from exc
        if response.status_code == 401:
            raise RuntimeError("The model server refused the token: copy REMOTE_LLM_TOKEN from the notebook again")
        response.raise_for_status()
        return response.json()

    def info(self) -> dict[str, Any]:
        """Model, vision support and adapter of the server (asked once)."""
        if self._info is None:
            self._info = self._request("GET", "/info")
            self.model, self.vision, self.adapter = self._info["model"], self._info["vision"], self._info["adapter"]
        return self._info

    @property
    def device(self) -> str:
        return "remote"

    def _messages(self, prompt: str, system_prompt: str | None, n_images: int = 0) -> list[dict[str, Any]]:
        self.info()  # text and vision models format their messages differently
        return super()._messages(prompt, system_prompt, n_images)

    def _generate(self, messages: list[dict[str, Any]], images: list[bytes] | None = None,
                  max_tokens: int = 1024, temperature: float = 0.0) -> str:
        payload = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature,
                   "images": [base64.b64encode(image).decode("ascii") for image in images or []]}
        data = self._request("POST", "/generate", json=payload)
        self.last_stats = {**data["stats"], "server": self.url}
        return data["text"]

    def free_memory(self) -> None:
        """Nothing to free on this side: the server's GPU memory is managed on Kaggle."""
