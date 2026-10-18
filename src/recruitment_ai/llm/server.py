"""AI server: lets the app on our PCs use the models running in a Kaggle GPU session.

Started by notebooks/kaggle/model_server.ipynb, behind a temporary Cloudflare link. It serves:
- ``/generate`` + ``/jobs/<id>``: Qwen2.5 generation (``RemoteLLMAdapter``, LLM_PROVIDER=remote);
- ``/embed``: BGE-M3 embeddings (``RemoteEmbeddingService``, EMBEDDING_PROVIDER=remote);
- ``/rerank``: BGE reranker scores (``RemoteReranker``).
The server only runs the models: prompts, JSON parsing, validation, retrieval and scoring stay in
the app, so results are exactly those of the evaluation runs (same code).

A generation takes one to two minutes, but a Cloudflare link closes any request that gets no
answer within about 100 seconds (error 524). So a request only starts a job and returns its id
at once; the app then asks for the result every few seconds. Embedding and reranking take under
a second: they are answered directly.

Every request needs the token printed by the notebook: whoever has the link alone cannot use
the GPU. Generations run one at a time; embeddings/reranking have their own lock, so a search is
not stuck behind a long generation.
"""

import base64
import threading
import time
import uuid
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    messages: list[dict[str, Any]]
    images: list[str] = Field(default_factory=list, description="base64 PNG/JPEG page images (vision model)")
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    use_adapter: bool = Field(default=True, description="False = base model (the adapter is for CV extraction)")


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=512)
    kind: Literal["query", "document"] = "document"


class RerankRequest(BaseModel):
    pairs: list[tuple[str, str]] = Field(min_length=1, max_length=4000, description="(query, document) pairs")


def create_app(llm: Any, token: str, embedder: Any | None = None, reranker: Any | None = None) -> FastAPI:
    """``llm``: a loaded HuggingFaceAdapter; ``embedder``/``reranker``: the project's embedding service
    and reranker (optional: the matching endpoints answer 503 without them)."""
    app = FastAPI(title="Recruitment AI server")
    gpu = threading.Lock()
    tools = threading.Lock()
    jobs: dict[str, dict[str, Any]] = {}
    stats = {"started": time.time(), "requests": 0, "last_request": None}

    def check(authorization: str | None) -> None:
        if authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="missing or wrong token")

    def touch() -> None:
        stats["requests"] += 1
        stats["last_request"] = time.time()

    def run(job_id: str, request: GenerateRequest) -> None:
        images = [base64.b64decode(image) for image in request.images] or None
        try:
            with gpu:  # one generation at a time on the GPUs
                text = llm._generate(request.messages, images, max_tokens=request.max_tokens,
                                     temperature=request.temperature, use_adapter=request.use_adapter)
                jobs[job_id].update(status="done", text=text, stats=dict(llm.last_stats))
        except Exception as exc:  # reported to the app, which decides (e.g. retry a shorter CV)
            jobs[job_id].update(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
            if hasattr(llm, "free_memory"):
                llm.free_memory()
        stats["last_request"] = time.time()

    @app.get("/info")
    def info(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        check(authorization)
        settings = getattr(embedder, "settings", None) or getattr(reranker, "settings", None)
        return {
            "model": llm.model, "vision": llm.vision, "adapter": llm.adapter,
            "embedding_model": settings.embedding_model if embedder is not None and settings else None,
            "embedding_dimension": settings.embedding_dimension if embedder is not None and settings else None,
            "reranking_model": settings.reranking_model if reranker is not None and settings else None,
            "running_jobs": sum(job["status"] == "running" for job in jobs.values()), **stats,
        }

    @app.post("/generate")
    def generate(request: GenerateRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        check(authorization)
        job_id = uuid.uuid4().hex
        jobs[job_id] = {"status": "running"}
        touch()
        threading.Thread(target=run, args=(job_id, request), daemon=True).start()
        return {"job": job_id}

    @app.get("/jobs/{job_id}")
    def job(job_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        check(authorization)
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="unknown job (the server may have restarted)")
        result = jobs[job_id]
        if result["status"] != "running":
            jobs.pop(job_id)  # a finished job is handed out once
        return result

    @app.post("/embed")
    def embed(request: EmbedRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        check(authorization)
        if embedder is None:
            raise HTTPException(status_code=503, detail="no embedding model loaded on the AI server")
        touch()
        with tools:
            vectors = embedder.embed_documents(request.texts)
        return {"vectors": vectors, "dimension": len(vectors[0]) if vectors else 0}

    @app.post("/rerank")
    def rerank(request: RerankRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        check(authorization)
        if reranker is None:
            raise HTTPException(status_code=503, detail="no reranking model loaded on the AI server")
        touch()
        with tools:
            scores = reranker.score_pairs([list(pair) for pair in request.pairs])
        if scores is None:
            raise HTTPException(status_code=503, detail="the reranking model could not score the pairs")
        return {"scores": [float(s) for s in scores]}

    return app
