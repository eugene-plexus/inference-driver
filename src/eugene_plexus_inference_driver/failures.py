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


#: llama-server's words while its model is not in memory yet, as the whole of the
#: backend's message: the error object's `message` of a chat or an embeddings body
#: (`{"error":{"message":"Loading model",...}}`, whatever order the keys come in), and
#: the bare message of a rerank (`...for a rerank: Loading model`, which the engine
#: reduces to the backend's message). Measured 2026-10-12 on b10948,
#: Qwen3-0.6B on the CPU, polling from the moment the port was bound: for the
#: ~0.7 s the model took, **every path that needs it answered HTTP 503 with the
#: same 75 bytes** -- chat (streamed or not, with tools or without), embeddings
#: and rerank, and `/health`, `/props` and `/v1/models` too:
#: `{"error":{"message":"Loading model","type":"unavailable_error","code":503}}`
#: (`Server: llama.cpp`, `Connection: close`, no `Retry-After`). A larger model
#: stays in that state for as long as it takes to load, and a replica that is
#: loading is exactly the one a healthy sibling should be tried before
#: (inference-driver#10). Only these words, as the whole message, and only with
#: the 503 (`loading_refused`): a 502 or a 500 that merely mentions loading is
#: not llama-server's refusal, and neither is a body that has the words somewhere
#: else in it (echoed in another field, behind a proxy's own words, or as the message
#: of an object nested in the error's own), which a search of the whole text took for it
#: until round four. The match is on the message and not on the text, as
#: `tools_refusal` matches a refusal of tools. A full stop after it is the one thing
#: allowed, as there.
_LOADING = re.compile(r"Loading model\.?")


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


def _backend_body(error: Exception) -> str | None:
    """What this driver's failure text says the backend answered, or None.

    A failure here reads `<the driver's own words>: <the backend's body>`
    (`openai_compat_http returned 503: {...}`, `... returned 503 for embeddings: {...}`,
    `... returned 503 for a rerank: Loading model`, `... stream error: Loading model`), and the
    driver's own words have no colon in them: the body is what follows the first one. A text
    with no such split says nothing of what the backend answered.
    """
    _words, colon, body = str(error).partition(": ")
    return body if colon else None


def loading_refused(error: Exception) -> bool:
    """True when the backend refused because its model is not in memory yet.

    llama-server's 503 `Loading model` (`_LOADING`). Nothing was computed: the
    engine refuses before it looks at the request, so another replica may take it
    and the same request may be sent again, and "outcome unknown" would say
    something untrue of it. **Keyed on the 503 and on the backend's whole message being
    these words**: the words under another status are not llama-server's, a 503 with other
    words is a backend that may be broken, and a 503 whose body only mentions them is not
    its loading engine either. Each stays `indeterminate`.

    **The words are the backend's whole message, not something its body mentions.** The
    body is read as `tools_refusal` reads one (`_backend_message`): the error object's
    `message`, a bare `error` or `message`, or plain text. A 503 whose body echoes the
    words in another field, has a proxy's own words before them, or carries them in an
    object nested in the error's own is not llama-server's loading engine, and is
    `indeterminate` like any other 503 of unknown cause.
    """
    if getattr(error, "upstream_status", None) != 503:
        return False
    body = _backend_body(error)
    message = _backend_message(body) if body is not None else None
    return message is not None and _LOADING.fullmatch(message) is not None


def disposition(error: Exception) -> str:
    status = getattr(error, "upstream_status", None)
    if status == 429 or isinstance(error.__cause__, (httpx.ConnectError, httpx.ConnectTimeout)):
        return "safe"
    if capacity_refused(error):
        return "safe"
    # A backend still loading its model refused before any work (a 503 that, as a
    # 502 `indeterminate`, kept the gateway from trying the healthy replica
    # beside it, and said "work may have occurred" of a request nothing touched).
    if loading_refused(error):
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
