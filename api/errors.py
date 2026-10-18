"""Service errors -> HTTP statuses, with messages that say what to do."""

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import HTTPException

from recruitment_ai.services.errors import ConflictError, NotFoundError


@contextmanager
def service_errors(value_status: int = 400) -> Iterator[None]:
    """404 for a missing record, 409 for an action out of turn, ``value_status`` for invalid input, 503 when a
    service is unreachable (Kaggle AI server stopped, Qdrant down...)."""
    try:
        yield
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=value_status, detail=str(exc)[:600]) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)[:600]) from exc
