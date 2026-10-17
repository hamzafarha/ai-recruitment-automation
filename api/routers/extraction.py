"""CV extraction endpoint (DSO1): upload a CV, get the structured candidate profile."""

import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from recruitment_ai.config.settings import get_settings
from recruitment_ai.utils.security import sanitize_filename, validate_file_upload

router = APIRouter(prefix="/candidates", tags=["Extraction"])


@lru_cache(maxsize=1)
def get_extractor() -> Any:
    """One extractor for the API's lifetime: the OCR model and the skill lexicon load once."""
    from recruitment_ai.extraction.extractor import CVExtractor
    from recruitment_ai.llm import get_llm_provider

    return CVExtractor(llm=get_llm_provider())


@router.post("/extract")
def extract_candidate(file: UploadFile = File(...)) -> dict[str, Any]:
    """Read an uploaded CV (PDF, Word or image; OCR for scans) and extract its profile.

    The document is read on this machine; only its text goes to the LLM (with
    LLM_PROVIDER=remote: Qwen in a Kaggle GPU session, about a minute per CV). The file is
    processed in a temporary folder and deleted at once: CVs are personal data.
    """
    settings = get_settings()
    filename = sanitize_filename(file.filename or "cv")
    content = file.file.read()
    is_valid, reason = validate_file_upload(
        filename=filename, file_size_bytes=len(content),
        allowed_extensions=settings.allowed_extensions, max_size_mb=settings.max_upload_size_mb,
    )
    if not is_valid:
        raise HTTPException(status_code=400, detail=reason)

    extractor = get_extractor()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / filename
        path.write_bytes(content)
        try:
            profile, stats = extractor.extract_document(path)
        except RuntimeError as exc:  # e.g. the Kaggle model server is stopped or its link changed
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ValueError as exc:  # the model's answer could not be read as a profile
            raise HTTPException(status_code=502, detail=f"Unreadable model answer: {exc}"[:500]) from exc
    return {
        "filename": filename,
        "llm_provider": settings.llm_provider,
        "profile": profile.model_dump(mode="json", exclude={"created_at"}),
        "stats": stats,
    }
