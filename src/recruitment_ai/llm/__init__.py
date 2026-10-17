"""LLM provider abstraction and adapters."""

from recruitment_ai.llm.base import LLMMessage, LLMProvider, LLMResponse
from recruitment_ai.llm.huggingface_adapter import HuggingFaceAdapter
from recruitment_ai.llm.mock_adapter import MockLLMAdapter
from recruitment_ai.llm.openai_adapter import OpenAIAdapter


def get_llm_provider(provider_type: str | None = None) -> LLMProvider:
    """Factory to instantiate the configured LLM provider."""
    from recruitment_ai.config.settings import get_settings

    settings = get_settings()
    p_type = provider_type or settings.llm_provider

    if p_type == "huggingface":
        return HuggingFaceAdapter(settings)
    if p_type == "remote":
        from recruitment_ai.llm.remote_adapter import RemoteLLMAdapter

        return RemoteLLMAdapter(settings)
    if p_type == "openai":
        return OpenAIAdapter(settings)
    return MockLLMAdapter(settings)


__all__ = [
    "LLMMessage",
    "LLMResponse",
    "LLMProvider",
    "OpenAIAdapter",
    "HuggingFaceAdapter",
    "MockLLMAdapter",
    "get_llm_provider",
]
