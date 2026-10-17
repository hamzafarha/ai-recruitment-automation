"""Model server: lets the app on our PCs use the Qwen model running in a Kaggle GPU session.

Started by notebooks/kaggle/model_server.ipynb, behind a temporary Cloudflare link. The app
reaches it through ``RemoteLLMAdapter`` (LLM_PROVIDER=remote). The server only generates text:
prompts, JSON parsing and validation stay in the app, so the results are exactly those of the
evaluation runs (same prompts, same code).

Every request needs the token printed by the notebook: whoever has the link alone cannot use
the GPU. Requests are served one at a time (a single model on the GPUs).
"""

import base64
import threading
import time
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
    lock = threading.Lock()
    stats = {"started": time.time(), "requests": 0, "last_request": None}

    def check(authorization: str | None) -> None:
        if authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="missing or wrong token")

    @app.get("/info")
    def info(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        check(authorization)
        return {"model": llm.model, "vision": llm.vision, "adapter": llm.adapter, **stats}

    @app.post("/generate")
    def generate(request: GenerateRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        check(authorization)
        images = [base64.b64decode(image) for image in request.images] or None
        with lock:  # one generation at a time on the GPUs
            text = llm._generate(request.messages, images, max_tokens=request.max_tokens,
                                 temperature=request.temperature)
            run_stats = dict(llm.last_stats)
        stats["requests"] += 1
        stats["last_request"] = time.time()
        return {"text": text, "stats": run_stats}

    return app
