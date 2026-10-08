"""Local Hugging Face transformers adapter implementing LLMProvider."""

import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from recruitment_ai.config.settings import Settings, get_settings
from recruitment_ai.llm.base import LLMMessage, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class HuggingFaceAdapter(LLMProvider):
    """Adapter for running local Hugging Face transformer models."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._pipeline: Any = None

    @property
    def pipeline(self) -> Any:
        """Lazy load transformers pipeline only when first invoked."""
        if self._pipeline is None:
            try:
                from transformers import pipeline

                logger.info(
                    f"Loading Hugging Face model pipeline for {self.settings.huggingface_model}"
                )
                self._pipeline = pipeline(
                    "text-generation",
                    model=self.settings.huggingface_model,
                    device_map="auto",
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load Hugging Face model '{self.settings.huggingface_model}': {exc}"
                ) from exc
        return self._pipeline

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        full_prompt = (
            f"System: {system_prompt}\nUser: {prompt}" if system_prompt else prompt
        )
        results = self.pipeline(
            full_prompt,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
        )
        generated_text = results[0]["generated_text"]
        return LLMResponse(
            content=generated_text,
            model=self.settings.huggingface_model,
            tokens_used=len(generated_text.split()),
            raw_response={"status": "success"},
        )

    def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        prompt_lines = [f"{m.role}: {m.content}" for m in messages]
        full_prompt = "\n".join(prompt_lines)
        return self.generate(
            prompt=full_prompt, temperature=temperature, max_tokens=max_tokens
        )

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str | None = None,
    ) -> T:
        import json

        schema_json = json.dumps(schema.model_json_schema())
        guided_prompt = (
            f"{prompt}\n\nReturn strictly valid JSON adhering to schema:\n{schema_json}"
        )
        response = self.generate(guided_prompt, system_prompt=system_prompt)
        # Attempt to locate and parse JSON
        content = response.content.strip()
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            json_str = content[start : end + 1]
            return schema.model_validate_json(json_str)
        raise ValueError("Could not extract valid JSON from Hugging Face model output.")
