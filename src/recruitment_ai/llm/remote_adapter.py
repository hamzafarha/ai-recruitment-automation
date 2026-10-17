"""Remote adapter: the app runs on our PC, the Qwen model in a Kaggle GPU session.

It is a ``HuggingFaceAdapter`` whose generation step is sent to the model server
(``recruitment_ai.llm.server``) instead of a local GPU. Prompt building, JSON repair and
Pydantic validation are inherited unchanged, so a CV extracted through the app gives the same
result as in the evaluation runs.

A generation is a job: one request starts it, then short requests ask for the result every
few seconds (a single request would be cut by the Cloudflare link after ~100 s, error 524).

Settings: LLM_PROVIDER=remote, REMOTE_LLM_URL and REMOTE_LLM_TOKEN (printed by
notebooks/kaggle/model_server.ipynb; the link changes at every Kaggle session).
"""

import base64
import time
from typing import Any

import httpx

from recruitment_ai.config.settings import Settings, get_settings
from recruitment_ai.llm.huggingface_adapter import HuggingFaceAdapter


class OutOfMemoryError(RuntimeError):
    """The Kaggle GPUs ran out of memory (very long CV). Same name as torch's error, so the
    extractor retries the CV with a shorter text, as it does with a local GPU."""


class RemoteLLMAdapter(HuggingFaceAdapter):
    """Generation by the Kaggle model server; everything else as in HuggingFaceAdapter."""

    poll_seconds = 3.0

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
                                     timeout=60, **kwargs)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"The Kaggle model server does not answer ({type(exc).__name__}): is the Kaggle "
                               "session still running, and is REMOTE_LLM_URL the link of this session?") from exc
        if response.status_code == 401:
            raise RuntimeError("The model server refused the token: copy REMOTE_LLM_TOKEN from the notebook again")
        if response.status_code >= 400:
            raise RuntimeError(f"The Kaggle model server answered with error {response.status_code} on {path}: "
                               f"{response.text[:200]}")
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
        job = self._request("POST", "/generate", json=payload)["job"]
        deadline = time.monotonic() + max(self.settings.llm_timeout_seconds, 600)
        while time.monotonic() < deadline:
            time.sleep(self.poll_seconds)
            result = self._request("GET", f"/jobs/{job}")
            if result["status"] == "done":
                self.last_stats = {**result["stats"], "server": self.url}
                return result["text"]
            if result["status"] == "error":
                error = OutOfMemoryError if result["error"].startswith("OutOfMemoryError") else RuntimeError
                raise error(f"Generation failed on the Kaggle server: {result['error']}")
        raise RuntimeError("The Kaggle model server took too long to answer this CV")

    def free_memory(self) -> None:
        """Nothing to free on this side: the server frees its GPU memory after a failed job."""
