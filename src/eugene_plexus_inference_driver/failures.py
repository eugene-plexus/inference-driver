"""Conservative execution outcomes; transport silence never proves no work."""

import math
from contextvars import ContextVar
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx


def retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = (parsedate_to_datetime(value) - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def disposition(error: Exception) -> str:
    status = getattr(error, "upstream_status", None)
    if status == 429 or isinstance(error.__cause__, (httpx.ConnectError, httpx.ConnectTimeout)):
        return "safe"
    if isinstance(status, int) and 400 <= status < 500 and status not in {408, 409, 425}:
        return "terminal"
    return "indeterminate"


request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
