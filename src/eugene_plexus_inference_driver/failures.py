"""Conservative execution outcomes; transport silence never proves no work."""

import math
import re
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


#: A 403 whose words say the CONTENT was refused is the caller's — OpenRouter
#: answers 403 for input its moderation flagged — so only the rest are ours.
_CONTENT_REFUSAL = re.compile(r"flag|moderat|content[ _-]?polic|safety|violat", re.I)


def credential_refused(error: Exception) -> int | None:
    """The backend's status when it refused THIS DRIVER's own credential.

    A 401 (the key), a 402 (the account behind it has no credit), or a 403
    that is not about the content. None otherwise. The caller cannot fix
    any of these and another request with the same key fails the same way;
    before 2026-09-28 they were the caller's 400 (measured live: OpenRouter's
    401 for a bad key read as `invalid_request_error` at the gateway).
    """
    status = getattr(error, "upstream_status", None)
    if not isinstance(status, int):
        return None
    if status in (401, 402):
        return status
    if status == 403 and not _CONTENT_REFUSAL.search(str(error)):
        return status
    return None


def disposition(error: Exception) -> str:
    status = getattr(error, "upstream_status", None)
    if status == 429 or isinstance(error.__cause__, (httpx.ConnectError, httpx.ConnectTimeout)):
        return "safe"
    if isinstance(status, int) and 400 <= status < 500 and status not in {408, 409, 425}:
        return "terminal"
    return "indeterminate"


request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
