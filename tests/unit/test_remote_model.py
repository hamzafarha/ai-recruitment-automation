"""Tests for using the Kaggle model from the app: model server, remote adapter, extraction API.

No GPU, model or network needed: the "Kaggle" model is a fake behind FastAPI's TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from recruitment_ai.extraction.extractor import SYSTEM_PROMPT, CVExtraction
from recruitment_ai.llm.server import create_app


class FakeQwen:
    """Stands in for the HuggingFaceAdapter loaded in the Kaggle session."""

    model, vision, adapter = "Qwen/Qwen2.5-7B-Instruct", False, None

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.last_stats: dict = {}

    def _generate(self, messages, images=None, max_tokens=1024, temperature=0.0, use_adapter=True):
        self.calls.append({"messages": messages, "images": images, "max_tokens": max_tokens,
                           "use_adapter": use_adapter})
        self.last_stats = {"model": self.model, "seconds": 1.0, "prompt_tokens": 10, "output_tokens": 5}
        return '{"full_name": "Ann Lee", "skills": ["Python"]}'


@pytest.fixture
def kaggle(monkeypatch):
    """A fake model server; the remote adapter's HTTP calls (made by its AIServerClient) are routed to it."""
    from recruitment_ai.llm import remote_client

    qwen = FakeQwen()
    client = TestClient(create_app(qwen, "secret"))

    def through_test_client(method, url, headers=None, timeout=None, **kwargs):
        return client.request(method, url.replace("https://kaggle.example", ""), headers=headers, **kwargs)

    monkeypatch.setattr(remote_client.httpx, "request", through_test_client)
    return qwen, client


def test_model_server_requires_the_token(kaggle):
    _, client = kaggle
    assert client.get("/info").status_code == 401
    assert client.get("/info", headers={"Authorization": "Bearer wrong"}).status_code == 401
    info = client.get("/info", headers={"Authorization": "Bearer secret"}).json()
    assert info["model"] == "Qwen/Qwen2.5-7B-Instruct" and info["requests"] == 0


def test_remote_adapter_extracts_with_the_evaluation_prompt(kaggle):
    from recruitment_ai.llm.remote_adapter import RemoteLLMAdapter

    qwen, _ = kaggle
    llm = RemoteLLMAdapter(url="https://kaggle.example/", token="secret")
    llm.poll_seconds = 0.01
    result = llm.generate_structured("Extract", CVExtraction, system_prompt=SYSTEM_PROMPT, images=[b"png"],
                                     max_tokens=300)
    assert result.full_name == "Ann Lee" and result.skills == ["Python"]
    call = qwen.calls[0]
    assert call["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert '"maxItems":15' in call["messages"][1]["content"]  # same prompt as the Kaggle evaluation runs
    assert call["images"] == [b"png"] and call["max_tokens"] == 300
    assert llm.last_stats["output_tokens"] == 5 and llm.model == "Qwen/Qwen2.5-7B-Instruct"


def test_out_of_memory_on_kaggle_is_reported_as_such(kaggle):
    """The extractor retries such a CV with a shorter text (it recognises the error by its name)."""
    from recruitment_ai.llm.remote_adapter import OutOfMemoryError, RemoteLLMAdapter

    qwen, client = kaggle

    class TorchOutOfMemoryError(RuntimeError):  # what torch raises on the Kaggle GPU
        pass

    TorchOutOfMemoryError.__name__ = "OutOfMemoryError"

    def too_long(messages, images=None, max_tokens=1024, temperature=0.0, use_adapter=True):
        raise TorchOutOfMemoryError("CUDA out of memory")

    qwen._generate = too_long
    llm = RemoteLLMAdapter(url="https://kaggle.example", token="secret")
    llm.poll_seconds = 0.01
    with pytest.raises(OutOfMemoryError):
        llm.generate_structured("x", CVExtraction)
    assert client.get("/info", headers={"Authorization": "Bearer secret"}).json()["running_jobs"] == 0


def test_remote_adapter_explains_what_to_fix(kaggle):
    from recruitment_ai.llm.remote_adapter import RemoteLLMAdapter

    llm = RemoteLLMAdapter(url="https://kaggle.example", token="old-token")
    with pytest.raises(RuntimeError, match="token"):
        llm.generate_structured("x", CVExtraction)
    llm.url = ""  # no link in .env
    with pytest.raises(RuntimeError, match="REMOTE_LLM_URL"):
        llm.generate_structured("x", CVExtraction)


def test_remote_adapter_can_switch_the_adapter_off(kaggle):
    """The fine-tuned adapter is for CV extraction: matching, job reading and the assistant use the base model."""
    from recruitment_ai.llm.remote_adapter import RemoteLLMAdapter

    qwen, _ = kaggle
    llm = RemoteLLMAdapter(url="https://kaggle.example", token="secret")
    llm.poll_seconds = 0.01
    llm.generate_structured("Extract", CVExtraction)
    llm.generate("Hello", use_adapter=False)
    assert [c["use_adapter"] for c in qwen.calls] == [True, False]


class FakeSearchModels:
    """BGE-M3 and the BGE reranker as served by the Kaggle notebook (tiny fakes)."""

    class Settings:
        embedding_model, embedding_dimension, reranking_model = "BAAI/bge-m3", 3, "BAAI/bge-reranker-v2-m3"

    settings = Settings()

    def embed_documents(self, texts):
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    def score_pairs(self, pairs):
        return [1.0 if q.lower() in d.lower() else 0.0 for q, d in pairs]


def test_server_serves_embeddings_and_reranking():
    models = FakeSearchModels()
    client = TestClient(create_app(FakeQwen(), "secret", embedder=models, reranker=models))
    auth = {"Authorization": "Bearer secret"}
    info = client.get("/info", headers=auth).json()
    assert info["embedding_model"] == "BAAI/bge-m3" and info["reranking_model"] == "BAAI/bge-reranker-v2-m3"
    vectors = client.post("/embed", json={"texts": ["ab", "abcd"]}, headers=auth).json()
    assert vectors == {"vectors": [[2.0, 1.0, 0.0], [4.0, 1.0, 0.0]], "dimension": 3}
    scores = client.post("/rerank", json={"pairs": [["python", "Python developer"], ["java", "Go"]]}, headers=auth)
    assert scores.json() == {"scores": [1.0, 0.0]}
    assert client.post("/embed", json={"texts": ["x"]}).status_code == 401


def test_server_without_search_models_answers_503():
    client = TestClient(create_app(FakeQwen(), "secret"))
    auth = {"Authorization": "Bearer secret"}
    assert client.post("/embed", json={"texts": ["x"]}, headers=auth).status_code == 503
    assert client.post("/rerank", json={"pairs": [["a", "b"]]}, headers=auth).status_code == 503


def test_remote_embeddings_and_reranker_use_the_server(monkeypatch):
    from recruitment_ai.embeddings.remote import RemoteEmbeddingService
    from recruitment_ai.llm.remote_client import AIServerClient
    from recruitment_ai.reranking.remote import RemoteReranker
    from recruitment_ai.vectorstore.base import VectorSearchResult

    models = FakeSearchModels()
    client = TestClient(create_app(FakeQwen(), "secret", embedder=models, reranker=models))
    server = AIServerClient(url="https://kaggle.example", token="secret")
    monkeypatch.setattr("recruitment_ai.llm.remote_client.httpx.request",
                        lambda method, url, headers=None, timeout=None, **kw:
                        client.request(method, url.replace("https://kaggle.example", ""), headers=headers, **kw))
    embedder = RemoteEmbeddingService(client=server)
    embedder.require_model()
    assert embedder.embed_query("abc") == [3.0, 1.0, 0.0] and len(embedder.embed_documents(["a"] * 70)) == 70
    reranker = RemoteReranker(client=server)
    docs = [VectorSearchResult(id="1", text="Go developer", score=0.9),
            VectorSearchResult(id="2", text="Python developer", score=0.1)]
    assert [r.id for r in reranker.rerank("python", docs, top_n=2)] == ["2", "1"]
