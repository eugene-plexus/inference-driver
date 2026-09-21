"""BackendEngine — uniform contract every backend implementation honors.

An *engine* is the protocol-level adapter: how this driver talks to its
backend (HTTP, subprocess, whatever). The user-facing concept is the
*provider* (the subscription/service they're wrapping); a `Provider`
in `providers.py` declares which engine to use and any
provider-specific knobs (deny patterns, default URLs, friendly labels).

Engines are stateless: the gateway owns conversation state and
passes the full prompt every call. Each engine knows how to:

  - declare its own config fields (`field_specs`)
  - construct itself from a config getter (`from_config`)
  - generate / stream / list_models against its backend

`generate`, `stream` and `list_models` are uniform across engines so
the route handlers don't have to care which backend is wired in. The
classmethods (`field_specs`, `from_config`) let `config.py` and
`app.py` walk the registry without hardcoding per-engine logic.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Protocol

from .._generated.models import (
    BackendKind,
    ConfigField,
    EmbedResponse,
    GenerateRequest,
    GenerateResponse,
    ToolCallDelta,
)

DEFAULT_REQUEST_TIMEOUT_SECONDS = 660.0
"""The one place this number is written (R2.5).

It was written in four -- the schema default and each of the three
engines' ``or 120`` -- which is how a number drifts. It lives here
rather than in `config.py` because `config.py` imports the engines and
not the other way round.

**660 is deliberately ABOVE the gateway's 600.** Two deadlines sit on
the path, and until R2.5 they were ordered the wrong way round: this
one was 120 s against the gateway's 180 s, so the driver always fired
first and the knob an operator was told to turn governed nothing. The
front door owns the answer; this is the backstop behind it.
"""


class StreamChunk(Protocol):
    """One event in the SSE stream emitted by `BackendEngine.stream`."""

    text: str
    """The newly-generated text fragment for this event. Empty for non-token events."""

    done: bool
    """If True, this is the final event; `result` is set."""

    result: GenerateResponse | None
    """Set on the final event, capturing the assembled response and usage."""


@dataclass(frozen=True)
class Chunk:
    """The concrete `StreamChunk` every engine yields.

    `StreamChunk` above is a Protocol so a test can substitute anything
    with the right shape; this is what the shipped engines actually
    emit, so the three of them do not each invent one.

    Two kinds of event, matching the contract in
    `inference-driver.yaml`: a token event carries `text` with
    `done=False`, and the final event carries `done=True` plus
    `result`. A token event whose text is empty is legal and is simply
    not forwarded -- upstream SSE feeds routinely open with a delta
    that carries only a role.
    """

    text: str = ""
    toolCalls: list[ToolCallDelta] | None = None
    """Tool-call fragments on this frame, when the backend is streaming
    a call rather than text. A frame carries text or fragments, never
    both -- upstream sends them in separate deltas and combining them
    here would invent a shape no backend produces."""
    done: bool = False
    result: GenerateResponse | None = None


class BackendEngine(Protocol):
    """The interface every backend implementation honors.

    Engines are constructed once at process startup based on the
    operator's `provider` choice (mapped through `providers.py` to an
    engine class + provider-specific kwargs). They are stateless across
    requests.
    """

    backend_kind: BackendKind
    """Reported in `/v1/info` and on every `GenerateResponse` so ops can
    see *which protocol* the driver is speaking. Distinct from the
    user-facing `provider` — many providers share one backend kind."""

    supports_tool_calling: bool = False
    """Whether this engine can carry `tools` to its backend and report
    `toolCalls` back.

    Reported as `capabilities.toolCalling` on `/v1/info`, and the
    gateway refuses a tools request against a driver that says False
    rather than dropping the field. **Defaults to False so a new engine
    is honest by omission** -- the failure mode of the opposite default
    is a harness receiving a plain answer it cannot distinguish from the
    model declining to call anything."""

    supports_streaming: bool
    """Whether `stream()` emits genuinely incremental tokens.

    Reported as `capabilities.streaming` on `/v1/info`, whose contract
    wording is "whether `/v1/generate/stream` emits **true incremental**
    tokens". It is allowed to be False and must be honest when it is:
    the stream contract explicitly permits a backend to emit the whole
    response as one token event followed by `done`, so a batching
    backend still works — it just cannot promise early delivery, and a
    flag that claimed otherwise would be useless for the one thing it
    is for."""

    @classmethod
    def field_specs(cls, *, applicable_providers: list[str]) -> list[ConfigField]:
        """Config fields this engine reads from `ConfigStore`. The
        schema builder wires `showWhen` against `provider` so each
        field is hidden unless one of `applicable_providers` is the
        currently-selected provider."""
        ...

    @classmethod
    def from_config(cls, get: Any, **provider_kwargs: Any) -> BackendEngine:
        """Construct from a `key -> value` getter (which transparently
        merges runtime config + transient overrides). `provider_kwargs`
        carries any provider-specific knobs the registry pinned for
        this engine instance — e.g. the OpenAI-compatible engine takes
        `default_base_url`, `fixed_temperature_pattern`, `backend_kind`. CLI engines
        ignore them."""
        ...

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Single-shot generation. Returns the full response."""
        ...

    def stream(self, request: GenerateRequest) -> AsyncGenerator[StreamChunk, None]:
        """Streamed generation. Yields chunks ending with one where `done` is True.

        Declared with a plain `def` returning an `AsyncIterator`, not
        `async def`. An async-generator function is not a coroutine: the
        call returns the iterator directly. Writing `async def` here
        typed the call as `Coroutine[..., AsyncIterator[...]]`, so every
        caller looked like it needed an extra `await` -- which nothing
        noticed for nine milestones because nothing called it.

        `AsyncGenerator` rather than `AsyncIterator`, because cleanup on
        abandonment is load-bearing: a client that disconnects mid-answer
        must leave no subprocess generating and no upstream response
        open, and `aclose()` running the implementation's `finally` is
        what guarantees that. An iterator without it would satisfy the
        types and leak the process.
        """
        ...

    supports_embeddings: bool = False
    """Whether `embed()` works against this backend.

    **Not a property of the model, and not readable from anything.**
    Measured 2026-09-12: `llama-server`'s `/props` carries no pooling or
    embedding field at all, and an Ollama runner started for chat
    answers "This server does not support embeddings. Start it with
    --embeddings". So the HTTP engine determines it by trying once and
    caching -- see `OpenAiCompatibleHttpEngine.probe_embeddings`.

    Defaults to False so a new engine is honest by omission, the same
    reasoning as `supports_tool_calling`: the failure mode of the
    opposite default is a caller receiving something that is not an
    embedding.
    """

    async def embed(self, inputs: list[str]) -> EmbedResponse:
        """Vectors for each input, in the order the inputs arrived.

        Order is the contract. An embedding carries no identity of its
        own, so position is the only thing that lets a caller match a
        vector back to the text it came from.

        Floats, never base64. The base64 encoding the OpenAI SDKs ask
        for by default is a transport detail the gateway applies, so
        that a caller gets the same behaviour whether or not the backend
        underneath implements it.
        """
        ...

    async def context_window(self) -> int | None:
        """The context window the backend resolved, or None if unknown.

        Reported as `capabilities.maxContextTokens` on `/v1/info`.
        **None is a real answer and the correct one by default** -- a
        backend this driver cannot interrogate has no window it can
        promise, and the gateway skips it rather than substituting a
        guess. An engine that returned a plausible-looking number it had
        not read from the backend would be worse than one that returned
        nothing, because the whole point of the field is that a harness
        can size a prompt against it.

        Same defaulting logic as `supports_tool_calling`: honest by
        omission, so a new engine has to opt in to making a claim.
        """
        ...

    async def list_models(self) -> list[str]:
        """Return the model IDs this backend offers, post-policy-filter.

        Used by `GET /v1/config/schema` to populate `modelId.enumValues`
        so the UI can render a dropdown. Returns `[]` on failure
        (transport error, bad auth) — the schema falls back to free-text.
        """
        ...


log = logging.getLogger(__name__)


def refuse_unsupported_settings(request: GenerateRequest, *, unsupported: set[str]) -> None:
    """An explicit caller constraint cannot be treated as an adapter default."""
    from ._subprocess import CliError

    names = {
        "maxTokens": "max_tokens/max_completion_tokens",
        "topP": "top_p",
        "toolChoice": "tool_choice",
        "responseFormat": "response_format",
    }
    for field in request.callerSettings or []:
        if field in unsupported:
            raise CliError(
                f"{names.get(field, field)}: this engine cannot honor this explicit setting; "
                "remove it or select a backend that supports it.",
                upstream_status=400,
            )


def warn_dropped_sampling(
    request: GenerateRequest, *, engine: str, model_id: str, warned: set[str]
) -> None:
    """Say which output-affecting parameters this backend cannot carry.

    The gateway owns `topP` and `seed` and sends them explicitly; an
    agentic CLI has no knob for either, because the harness on the other
    side of the pipe owns the sampler. The contract's answer to that is
    "dropped with a logged warning" rather than a refusal -- refusing
    would fail a request the backend can perfectly well answer -- so the
    one thing that must not happen is what happened before 2026-09-19,
    which is nothing at all.

    **Once per parameter per engine, not once per request.** A line on
    every call is a line per generation on a busy backend, and a warning
    at that rate is one an operator filters out, which costs the warning
    its only job.

    A2: callerSettings distinguishes explicit constraints from inherited defaults.
    CLI engines now refuse explicit controls they cannot honor before spawning.
    Implicit profile/default settings retain the legacy behavior described above.
    """
    refuse_unsupported_settings(
        request,
        unsupported={
            "maxTokens",
            "temperature",
            "topP",
            "seed",
            "stop",
            "tools",
            "toolChoice",
            "responseFormat",
        },
    )
    for field, value in (("topP", request.topP), ("seed", request.seed)):
        if value is None or field in warned:
            continue
        warned.add(field)
        log.warning(
            "%s cannot carry `%s` to %r -- the harness on the other side of the "
            "pipe owns its own sampler -- so the parameter was dropped and the "
            "request was answered. Said once per parameter for the life of this "
            "engine.",
            engine,
            field,
            model_id,
        )
