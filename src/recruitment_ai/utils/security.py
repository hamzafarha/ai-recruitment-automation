"""Security utilities for safe file handling, path traversal protection, and sanitization."""

import os
import re
from pathlib import Path


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and shell injection."""
    clean_name = os.path.basename(filename)
    clean_name = re.sub(r"[^\w\s.-]", "", clean_name).strip()
    if not clean_name:
        clean_name = "unnamed_document"
    return clean_name


def validate_file_path(base_dir: Path, target_path: Path) -> Path:
    """Validate that target_path resides strictly inside base_dir to avoid path traversal."""
    resolved_base = base_dir.resolve()
    resolved_target = target_path.resolve()
    try:
        resolved_target.relative_to(resolved_base)
    except ValueError as exc:
        raise ValueError(
            f"Security violation: path traversal attempt detected for '{target_path}'"
        ) from exc
    return resolved_target


def validate_file_upload(
    filename: str,
    file_size_bytes: int,
    allowed_extensions: list[str],
    max_size_mb: int = 15,
) -> tuple[bool, str]:
    """Validate uploaded document extension and file size."""
    ext = Path(filename).suffix.lower().lstrip(".")
    allowed_cleaned = [e.lower().lstrip(".") for e in allowed_extensions]

    if ext not in allowed_cleaned:
        return (
            False,
            f"Unsupported file format '.{ext}'. Allowed formats: {', '.join(allowed_cleaned)}",
        )

    max_bytes = max_size_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        return (
            False,
            f"File size exceeds maximum allowed limit of {max_size_mb} MB.",
        )

    return True, "File is valid."
