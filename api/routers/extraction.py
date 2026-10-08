"""Structured CV extraction endpoints."""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/candidates", tags=["Extraction"])


class ExtractRequest(BaseModel):
    document_path: str
    candidate_id: str | None = None


@router.post("/extract")
def extract_candidate(request: ExtractRequest) -> dict[str, Any]:
    """Trigger structured extraction for a candidate document."""
    return {
        "status": "placeholder",
        "message": "Structured CV extraction endpoint initialized. LLM/Transformer extraction pipeline scheduled for Phase 2.",
        "request": request.model_dump(),
        "schema_target": "CandidateProfile (v1.0.0)",
    }
