"""Mock LLM adapter providing deterministic responses without API keys."""

from typing import Any, TypeVar

from pydantic import BaseModel

from recruitment_ai.config.settings import Settings, get_settings
from recruitment_ai.llm.base import LLMMessage, LLMProvider, LLMResponse

T = TypeVar("T", bound=BaseModel)


class MockLLMAdapter(LLMProvider):
    """Offline mock provider for development and testing."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = "mock-engine"
        self.last_stats: dict[str, Any] = {"model": "mock-engine", "mock": True}

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        return LLMResponse(
            content=f"[Mock Response for prompt: {prompt[:40]}...]",
            model="mock-engine",
            tokens_used=42,
            raw_response={"status": "mocked"},
        )

    def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        last_message = messages[-1].content if messages else ""
        return LLMResponse(
            content=f"[Mock Chat Response for: {last_message[:40]}...]",
            model="mock-engine",
            tokens_used=30,
            raw_response={"status": "mocked"},
        )

    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str | None = None,
        images: list[bytes] | None = None,
        max_tokens: int = 2048,
        use_adapter: bool = True,
    ) -> T:
        """Fake answer of the right shape (accepts the CV extractor's arguments)."""
        import typing

        data: dict[str, Any] = {}
        for field_name, field_info in schema.model_fields.items():
            ann = field_info.annotation
            origin = typing.get_origin(ann)
            args = typing.get_args(ann)
            ann_str = str(ann).lower()

            if origin is typing.Literal or "literal" in ann_str:
                data[field_name] = args[0] if args else "fit"
            elif origin is list or "list" in ann_str:
                data[field_name] = []
            elif origin is dict or "dict" in ann_str:
                data[field_name] = {}
            elif ann is bool or "bool" in ann_str:
                data[field_name] = True
            elif ann is int or "int" in ann_str:
                data[field_name] = 1
            elif ann is float or "float" in ann_str:
                data[field_name] = 85.0
            elif ann is str or "str" in ann_str:
                data[field_name] = f"Sample {field_name}"
            elif field_info.default is not None:
                data[field_name] = field_info.default
            else:
                data[field_name] = None
        return schema.model_validate(data)
