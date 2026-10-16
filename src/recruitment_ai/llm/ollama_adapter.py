"""Ollama adapter (native /api/chat) implementing LLMProvider.

Used instead of Ollama's OpenAI-compatible endpoint because the native API lets us:
- set the context window (``num_ctx``): Ollama's default is a few thousand tokens, so a long
  CV plus instructions plus JSON schema could be silently truncated;
- constrain the output with a JSON schema (``format``);
- send images, for vision-language models such as Qwen2.5-VL.
"""

import base64
import json
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from recruitment_ai.config.settings import Settings, get_settings
from recruitment_ai.llm.base import LLMMessage, LLMProvider, LLMResponse

T = TypeVar("T", bound=BaseModel)


class OllamaAdapter(LLMProvider):
    """Chat / structured generation with a local Ollama model."""

    def __init__(self, settings: Settings | None = None, model: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = model or self.settings.ollama_model
        self.last_stats: dict[str, Any] = {}

    def _chat(
        self,
        messages: list[dict[str, Any]],
        fmt: dict[str, Any] | str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": temperature, "num_ctx": self.settings.ollama_num_ctx}
        if max_tokens:
            options["num_predict"] = max_tokens
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "stream": False,
            "options": options, "keep_alive": "15m",
        }
        if fmt is not None:
            payload["format"] = fmt
        start = time.perf_counter()
        response = httpx.post(
            f"{self.settings.ollama_base_url.rstrip('/')}/api/chat",
            json=payload, timeout=self.settings.llm_timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        self.last_stats = {
            "model": self.model,
            "seconds": round(time.perf_counter() - start, 2),
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "output_tokens": data.get("eval_count", 0),
            # prompt_eval_count == num_ctx means the prompt was cut: the context window is too small
            "truncated_prompt": data.get("prompt_eval_count", 0) >= self.settings.ollama_num_ctx,
        }
        return data

    @staticmethod
    def _messages(prompt: str, system_prompt: str | None, images: list[bytes] | None = None) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user: dict[str, Any] = {"role": "user", "content": prompt}
        if images:
            user["images"] = [base64.b64encode(img).decode("ascii") for img in images]
        messages.append(user)
        return messages

    def generate(self, prompt: str, system_prompt: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 1024) -> LLMResponse:
        data = self._chat(self._messages(prompt, system_prompt), temperature=temperature, max_tokens=max_tokens)
        return LLMResponse(content=data["message"]["content"], model=self.model,
                           tokens_used=self.last_stats["prompt_tokens"] + self.last_stats["output_tokens"],
                           raw_response=self.last_stats)

    def chat(self, messages: list[LLMMessage], temperature: float = 0.0, max_tokens: int = 1024) -> LLMResponse:
        data = self._chat([{"role": m.role, "content": m.content} for m in messages],
                          temperature=temperature, max_tokens=max_tokens)
        return LLMResponse(content=data["message"]["content"], model=self.model,
                           tokens_used=self.last_stats["prompt_tokens"] + self.last_stats["output_tokens"],
                           raw_response=self.last_stats)

    def generate_structured(self, prompt: str, schema: type[T], system_prompt: str | None = None,
                            images: list[bytes] | None = None, max_tokens: int = 3000) -> T:
        """Generate JSON constrained by ``schema`` (optionally from images) and validate it.

        ``max_tokens`` bounds the answer (a normal CV needs about 1,000): a runaway generation
        fails fast instead of reaching the request timeout.
        """
        data = self._chat(self._messages(prompt, system_prompt, images), fmt=schema.model_json_schema(),
                          max_tokens=max_tokens)
        content = data["message"]["content"]
        if not content.strip():
            raise ValueError("Ollama returned an empty structured response.")
        return schema.model_validate(json.loads(content))
