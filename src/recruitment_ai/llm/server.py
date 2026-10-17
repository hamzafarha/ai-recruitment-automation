"""Model server: lets the app on our PCs use the Qwen model running in a Kaggle GPU session.

Started by notebooks/kaggle/model_server.ipynb, behind a temporary Cloudflare link. The app
reaches it through ``RemoteLLMAdapter`` (LLM_PROVIDER=remote). The server only generates text:
prompts, JSON parsing and validation stay in the app, so the results are exactly those of the
evaluation runs (same prompts, same code).

A generation takes one to two minutes, but a Cloudflare link closes any request that gets no
answer within about 100 seconds (error 524). So a request only starts a job and returns its id
at once; the app then asks for the result every few seconds (``GET /jobs/<id>``).

Every request needs the token printed by the notebook: whoever has the link alone cannot use
the GPU. Jobs run one at a time (a single model on the GPUs).
"""

import base64
import threading
import time
import uuid
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    messages: list[dict[str, Any]]
    images: list[str] = Field(default_factory=list, description="base64 PNG/JPEG page images (vision model)")
    max_tokens: int = Field(default=1024, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


def create_app(llm: Any, token: str) -> FastAPI:
    """``llm``: a loaded HuggingFaceAdapter (its ``_generate`` does the work)."""
    app = FastAPI(title="Recruitment AI model server")
    gpu = threading.Lock()
    jobs: dict[str, dict[str, Any]] = {}
    stats = {"started": time.time(), "requests": 0, "last_request": None}

    def check(authorization: str | None) -> None:
        if authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="missing or wrong token")

    def run(job_id: str, request: GenerateRequest) -> None:
        images = [base64.b64decode(image) for image in request.images] or None
        try:
            with gpu:  # one generation at a time on the GPUs
                text = llm._generate(request.messages, images, max_tokens=request.max_tokens,
                                     temperature=request.temperature)
                jobs[job_id].update(status="done", text=text, stats=dict(llm.last_stats))
        except Exception as exc:  # reported to the app, which decides (e.g. retry a shorter CV)
            jobs[job_id].update(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
            if hasattr(llm, "free_memory"):
                llm.free_memory()
        stats["last_request"] = time.time()

    @app.get("/info")
    def info(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        check(authorization)
        return {"model": llm.model, "vision": llm.vision, "adapter": llm.adapter,
                "running_jobs": sum(job["status"] == "running" for job in jobs.values()), **stats}

    @app.post("/generate")
    def generate(request: GenerateRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        check(authorization)
        job_id = uuid.uuid4().hex
        jobs[job_id] = {"status": "running"}
        stats["requests"] += 1
        stats["last_request"] = time.time()
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

    return app
