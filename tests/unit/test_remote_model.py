"""Tests for using the Kaggle model from the app: model server, remote adapter, extraction API.

No GPU, model or network needed: the "Kaggle" model is a fake behind FastAPI's TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from recruitment_ai.extraction.extractor import SYSTEM_PROMPT, CVExtraction
from recruitment_ai.extraction.schemas import CandidateProfile
from recruitment_ai.llm.server import create_app


class FakeQwen:
    """Stands in for the HuggingFaceAdapter loaded in the Kaggle session."""

    model, vision, adapter = "Qwen/Qwen2.5-7B-Instruct", False, None

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.last_stats: dict = {}

    def _generate(self, messages, images=None, max_tokens=1024, temperature=0.0):
        self.calls.append({"messages": messages, "images": images, "max_tokens": max_tokens})
        self.last_stats = {"model": self.model, "seconds": 1.0, "prompt_tokens": 10, "output_tokens": 5}
        return '{"full_name": "Ann Lee", "skills": ["Python"]}'


@pytest.fixture
def kaggle(monkeypatch):
    """A fake model server; the remote adapter's HTTP calls are routed to it."""
    from recruitment_ai.llm import remote_adapter

    qwen = FakeQwen()
    client = TestClient(create_app(qwen, "secret"))

    def through_test_client(method, url, headers=None, timeout=None, **kwargs):
        return client.request(method, url.replace("https://kaggle.example", ""), headers=headers, **kwargs)

    monkeypatch.setattr(remote_adapter.httpx, "request", through_test_client)
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

    def too_long(messages, images=None, max_tokens=1024, temperature=0.0):
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


def test_extract_endpoint_returns_the_profile_and_deletes_the_file(api_client, monkeypatch):
    from api.routers import extraction

    seen = {}

    class FakeExtractor:
        def extract_document(self, path):
            seen.update(path=path, existed=path.exists())
            return CandidateProfile(full_name="Ann Lee"), {"input_type": "text_pdf", "seconds": 1.0}

    monkeypatch.setattr(extraction, "get_extractor", lambda: FakeExtractor())
    res = api_client.post("/api/v1/candidates/extract", files={"file": ("cv.pdf", b"%PDF-1.4", "application/pdf")})
    assert res.status_code == 200 and res.json()["profile"]["full_name"] == "Ann Lee"
    assert seen["existed"] and not seen["path"].exists()  # temporary file, deleted after the extraction
    bad = api_client.post("/api/v1/candidates/extract", files={"file": ("cv.exe", b"x", "application/octet-stream")})
    assert bad.status_code == 400


def test_extract_endpoint_reports_a_stopped_kaggle_server(api_client, monkeypatch):
    from api.routers import extraction

    class StoppedServer:
        def extract_document(self, path):
            raise RuntimeError("The Kaggle model server does not answer")

    monkeypatch.setattr(extraction, "get_extractor", lambda: StoppedServer())
    res = api_client.post("/api/v1/candidates/extract", files={"file": ("cv.pdf", b"%PDF-1.4", "application/pdf")})
    assert res.status_code == 503 and "does not answer" in res.json()["detail"]
