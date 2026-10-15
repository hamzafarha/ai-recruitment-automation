"""Structured logging configuration for AI recruitment pipelines."""

import json
import logging
import sys
from datetime import datetime
from typing import Any

# Enable UTF-8 encoding on console streams
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class JSONFormatter(logging.Formatter):
    """Format logs as structured JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Structured extra fields if passed
        for key in [
            "request_id",
            "component",
            "operation",
            "status",
            "duration_ms",
            "error",
        ]:
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure root logger handler and formatter."""
    handler = logging.StreamHandler(sys.stdout)
    if json_output:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Remove existing handlers to avoid duplicates
    root_logger.handlers = [handler]

    # Third-party HTTP clients log every request at INFO/DEBUG; keep them quiet
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get named logger instance."""
    return logging.getLogger(name)
