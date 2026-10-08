"""Unit tests for LLM provider adapters."""

from recruitment_ai.extraction.schemas import CandidateRecommendation
from recruitment_ai.llm import get_llm_provider
from recruitment_ai.llm.base import LLMMessage
from recruitment_ai.llm.mock_adapter import MockLLMAdapter


def test_mock_llm_generation():
    provider = get_llm_provider("mock")
    assert isinstance(provider, MockLLMAdapter)
    resp = provider.generate("Summarize candidate CV")
    assert resp.model == "mock-engine"
    assert resp.tokens_used > 0
    assert "Mock Response" in resp.content


def test_mock_llm_chat():
    provider = get_llm_provider("mock")
    messages = [
        LLMMessage(role="system", content="You are a recruitment assistant."),
        LLMMessage(role="user", content="Analyze candidate experience."),
    ]
    resp = provider.chat(messages)
    assert resp.model == "mock-engine"
    assert resp.content.startswith("[Mock Chat Response")


def test_mock_llm_structured_output():
    provider = get_llm_provider("mock")
    result = provider.generate_structured(
        "Recommend candidate", CandidateRecommendation
    )
    assert isinstance(result, CandidateRecommendation)
    assert result.decision in ["strong_fit", "fit", "potential_fit", "no_fit"]
