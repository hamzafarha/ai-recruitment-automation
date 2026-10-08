"""OpenAI adapter implementing LLMProvider."""

from typing import Any, TypeVar

from pydantic import BaseModel

from recruitment_ai.config.settings import Settings, get_settings
from recruitment_ai.llm.base import LLMMessage, LLMProvider, LLMResponse

T = TypeVar("T", bound=BaseModel)


class OpenAIAdapter(LLMProvider):
    """Adapter for OpenAI models."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _get_client(self) -> Any:
        try:
            from openai import OpenAI

            if not self.settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            return OpenAI(api_key=self.settings.openai_api_key)
        except ImportError as exc:
            raise ImportError(
                "openai package not installed. Run pip install openai."
            ) from exc

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        res = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = res.choices[0].message.content or ""
        tokens = res.usage.total_tokens if res.usage else 0
        return LLMResponse(
            content=content,
            model=self.settings.openai_model,
            tokens_used=tokens,
            raw_response=res.model_dump(),
        )

    def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        client = self._get_client()
        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]
        res = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = res.choices[0].message.content or ""
        tokens = res.usage.total_tokens if res.usage else 0
        return LLMResponse(
            content=content,
            model=self.settings.openai_model,
            tokens_used=tokens,
            raw_response=res.model_dump(),
        )

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str | None = None,
    ) -> T:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        completion = client.beta.chat.completions.parse(
            model=self.settings.openai_model,
            messages=messages,
            response_format=schema,
        )
        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Failed to parse structured output from OpenAI.")
        from typing import cast

        return cast(T, parsed)
