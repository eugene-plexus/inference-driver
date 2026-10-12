"""Conservative execution outcomes; transport silence never proves no work."""

import json
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


#: llama-server answers 500 when the model's chat template refuses the
#: request's SHAPE -- the Qwen 3.5+ templates' "System message must be at
#: the beginning" -- and its message says so. Measured 2026-10-02: as a
#: 502 it tripped the gateway's circuit, and every other client of that
#: model got 503 "cooling down" (gateway#7). The same request fails the
#: same way every time and says nothing about the backend's health.
_TEMPLATE_REFUSAL = re.compile(r"Jinja Exception")


#: llama-server's words when the KV pool its slots share is full: the
#: prompts in flight together outgrew `-c`. The engine refuses the next
#: request (a 500) or cuts every stream in the batch it was decoding (an
#: `error` frame, then nothing), and says the same either way. Measured
#: 2026-10-02 on b11211: as a 502 it tripped the gateway's circuit, and on
#: an 8B at 64k with four agents a replica 158 of 204 turns then failed,
#: most of them "cooling down" on healthy replicas (gateway#8). A request
#: that fits alone overflows only beside others, so it is load, not a
#: broken backend: see `capacity_refused`.
_POOL_FULL = re.compile(r"Context size has been exceeded")


#: llama-server's words for a reranker's input that is longer than it takes, in
#: the two forms it answers (MR6, measured 2026-10-12 on b10948, Qwen3-Reranker
#: and bge-reranker-v2-m3, `-c 4096 --parallel 4 -b 2048 -ub 2048`). Over a
#: slot's context, 1024 tokens here, it is the 400 every over-long input gets:
#: "input (1828 tokens) is larger than the max context size (1024 tokens).
#: skipping". Over the physical batch, 2048, it is a **500**: "input (2994
#: tokens) is too large to process. increase the physical batch size (current
#: batch size: 2048)". An embedding model answers the 400 at every size (6001
#: tokens tried), so only a reranker has the 500, and as a 502 it would be
#: retried against every replica of a model that refuses it identically, with
#: "outcome unknown" said of a request nothing computed.
_INPUT_TOO_LONG = re.compile(
    r"input \(\d+ tokens\) is (?:larger than the max context size|too large to process)"
)


def input_too_long(text: str) -> bool:
    """Whether the backend's words are a refusal of input longer than it takes."""
    return _INPUT_TOO_LONG.search(text) is not None


#: The backend's whole message, for each backend that has been MEASURED
#: refusing a request because it carries tools. Tools are assumed to work
#: (`docs/design/tools-assumed.md`, Troy 2026-10-12), so this table is the
#: only thing that turns them off for an engine, and it is deliberately
#: short and exact: a pattern that matched an unrelated refusal would take
#: tools away from a backend that serves them. **A wording goes in here only
#: after it has been seen on the wire, never from reading the source.**
#:
#: * llama-server b10948 started with `--no-jinja` (its own `--help` says jinja
#:   is on by default), measured 2026-10-12 on Qwen3-0.6B, CPU: every
#:   request with a non-empty `tools` -- auto, `required`, `none` and
#:   streamed alike -- is
#:   `HTTP 500 {"error":{"code":500,"message":"tools param requires --jinja
#:   flag","type":"server_error"}}`. **A 500, not a 4xx**, which is why this
#:   keys on the words and not on the status. The same engine's words for a
#:   malformed tool definition ("Failed to parse tools: ...") and for a
#:   request with no `messages` ("'messages' is required") are not these.
#:
#: Ollama's "does not support tools" is NOT here, and stays out until it has been
#: seen on the wire: no Ollama was available to measure on 2026-10-12. The
#: contract (specs 6178255) names llama-server's wording alone.
_TOOLS_REFUSALS: tuple[re.Pattern[str], ...] = (
    re.compile(r"tools param requires --jinja flag\.?"),
)


def _backend_message(body: str) -> str | None:
    """The one message a backend's error body carries, or None.

    OpenAI's shape (`{"error": {"message": ...}}`, which llama-server and Strata
    answer, measured), a bare `{"error": "..."}` or `{"message": ...}`, or a body
    that is plain text. Nothing else: a body this does not understand is not one
    whose words are matched.

    **A body nested too deep to parse is one of those** (`RecursionError`, which
    is not a `ValueError`: 4000 opening brackets raise it on CPython 3.12). It
    arrives from the backend, so it is not ours to crash on, and this runs inside
    the handling of a failed request: an exception here would replace the
    backend's own refusal with a 500 of the driver's.
    """
    text = body.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return text
    except RecursionError:
        return None
    if not isinstance(parsed, dict):
        return None
    error = parsed.get("error", parsed.get("message"))
    if isinstance(error, dict):
        error = error.get("message")
    return error.strip() if isinstance(error, str) and error.strip() else None


def tools_refusal(body: str) -> str | None:
    """The backend's own words, when `body` is a measured refusal of tools.

    The match is on the WHOLE message, not on a phrase somewhere in it, so an
    unrelated error that merely mentions tools, quotes a request, or comes
    from a proxy in front of the backend does not turn tools off.
    """
    message = _backend_message(body)
    if message is not None and any(p.fullmatch(message) for p in _TOOLS_REFUSALS):
        return message
    return None


def every(seconds: float) -> str:
    """An interval in a person's words, for what this driver says it does on a
    timer: `10 minutes`, `1 minute`, `30 seconds`."""
    if seconds >= 60:
        minutes = seconds / 60
        return f"{minutes:g} minute{'' if minutes == 1 else 's'}"
    return f"{seconds:g} second{'' if seconds == 1 else 's'}"


def capacity_refused(error: Exception) -> bool:
    """True when the backend refused for want of room, not for a fault.

    PC3's rule one case further: the request may go to another replica
    (nothing was kept and nothing was answered), and it says nothing about
    this backend's health, so the gateway's circuit must not count it.
    A request bigger than the whole pool is a 400 from llama-server with
    other words, and stays the caller's.
    """
    status = getattr(error, "upstream_status", None)
    if isinstance(status, int) and status < 500:
        return False
    return bool(_POOL_FULL.search(str(error)))


def disposition(error: Exception) -> str:
    status = getattr(error, "upstream_status", None)
    if status == 429 or isinstance(error.__cause__, (httpx.ConnectError, httpx.ConnectTimeout)):
        return "safe"
    if capacity_refused(error):
        return "safe"
    if isinstance(status, int) and 400 <= status < 500 and status not in {408, 409, 425}:
        return "terminal"
    if _TEMPLATE_REFUSAL.search(str(error)) or input_too_long(str(error)):
        return "terminal"
    # A refusal of the request's tools, whatever status the backend gave it
    # (llama-server's is a 500). As that 500 it would be `indeterminate`: a 502
    # the gateway does not replay, but counts against this backend's circuit,
    # for a refusal that says nothing about the backend's health. The same
    # request fails the same way on every replica, so it is the caller's.
    from .engines._subprocess import ToolsRefused  # the engines import this module

    if isinstance(error, ToolsRefused):
        return "terminal"
    return "indeterminate"


request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
