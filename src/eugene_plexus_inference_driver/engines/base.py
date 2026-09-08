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

from collections.abc import AsyncIterator
from typing import Any, Protocol

from .._generated.models import (
    BackendKind,
    ConfigField,
    GenerateRequest,
    GenerateResponse,
)


class StreamChunk(Protocol):
    """One event in the SSE stream emitted by `BackendEngine.stream`."""

    text: str
    """The newly-generated text fragment for this event. Empty for non-token events."""

    done: bool
    """If True, this is the final event; `result` is set."""

    result: GenerateResponse | None
    """Set on the final event, capturing the assembled response and usage."""


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
        `default_base_url`, `deny_pattern`, `backend_kind`. CLI engines
        ignore them."""
        ...

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Single-shot generation. Returns the full response."""
        ...

    async def stream(self, request: GenerateRequest) -> AsyncIterator[StreamChunk]:
        """Streamed generation. Yields chunks ending with one where `done` is True."""
        ...

    async def list_models(self) -> list[str]:
        """Return the model IDs this backend offers, post-policy-filter.

        Used by `GET /v1/config/schema` to populate `modelId.enumValues`
        so the UI can render a dropdown. Returns `[]` on failure
        (transport error, bad auth) — the schema falls back to free-text.
        """
        ...
