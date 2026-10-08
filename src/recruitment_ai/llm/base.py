"""Abstract LLM provider contracts and message structures."""

from abc import ABC, abstractmethod
from typing import Any, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)


class LLMMessage(BaseModel):
    """Message object for conversational interactions."""

    role: str = Field(..., description="Role: system, user, or assistant")
    content: str


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider."""

    content: str
    model: str
    tokens_used: int = 0
    raw_response: dict[str, Any] = Field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract interface for Language Model providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate text completion from a prompt."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Generate response from a conversation history."""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str | None = None,
    ) -> T:
        """Generate response validated directly into a Pydantic schema."""
        pass
