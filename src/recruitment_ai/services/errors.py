"""Errors of the service layer, mapped to HTTP statuses by the API."""


class NotFoundError(LookupError):
    """A candidate, job, application or analysis that does not exist (HTTP 404)."""


class ConflictError(Exception):
    """An action that does not fit the current state, e.g. a pipeline step out of order (HTTP 409)."""
