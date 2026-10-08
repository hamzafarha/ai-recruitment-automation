"""Document upload and handling endpoints."""

from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from recruitment_ai.config.settings import get_settings
from recruitment_ai.utils.security import (
    sanitize_filename,
    validate_file_path,
    validate_file_upload,
)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload and validate candidate document."""
    settings = get_settings()
    filename = sanitize_filename(file.filename or "uploaded_file")
    content = await file.read()
    file_size = len(content)

    is_valid, reason = validate_file_upload(
        filename=filename,
        file_size_bytes=file_size,
        allowed_extensions=settings.allowed_extensions,
        max_size_mb=settings.max_upload_size_mb,
    )
    if not is_valid:
        raise HTTPException(status_code=400, detail=reason)

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = validate_file_path(upload_dir, upload_dir / filename)

    with open(target_path, "wb") as f:
        f.write(content)

    return {
        "status": "success",
        "message": f"Document '{filename}' successfully uploaded and validated.",
        "filename": filename,
        "size_bytes": file_size,
        "path": str(target_path),
    }
