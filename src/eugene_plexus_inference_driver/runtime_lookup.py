"""Resolve a supervised runtime's name to the URL it listens on.

This closes the gap M2 left open: launching a model from the library
declared a runtime and stopped there, so the model loaded, served, and
was unreachable because nothing pointed a driver at it. Now a driver
configured with `runtimeName` asks the agent where that runtime lives —
at startup, and again whenever the engine is rebuilt (an admin restart,
a config test) — instead of carrying a literal `baseUrl`.

The point is not saving a copy-paste. The agent assigns the port when
the operator does not pick one, so a hand-typed URL encodes a number the
operator was never told and does not own; it is wrong the moment the
runtime moves. A name survives that.

The lookup is synchronous on purpose. It runs where the engine is
constructed — the lifespan, `/v1/admin/restart`, `/v1/config/test` —
none of which is a hot path, and a short cap keeps a dead agent from
wedging startup. It also resolves regardless of the runtime's status: a
runtime's URL is derived from its declaration and is stable while the
engine loads, and the gateway already routes only to `ready` runtimes.
"""

from __future__ import annotations

import logging
from urllib.parse import quote

import httpx

from ._http import sync_client_for

log = logging.getLogger(__name__)

# The agent answers `GET /v1/runtimes/{name}` from memory; anything slower
# than this is the agent being down, and the driver should come up
# degraded rather than wait on it.
_LOOKUP_TIMEOUT_SECONDS = 5.0


class RuntimeResolutionError(RuntimeError):
    """A `runtimeName` could not be turned into a URL.

    Raised during engine construction, so the driver lands in degraded
    mode with this message on `/healthz` and the config endpoints stay
    reachable. Every message names the way forward.
    """


def resolve_runtime_url(
    name: str,
    *,
    agent_url: str,
    service_token: str | None,
    timeout: float = _LOOKUP_TIMEOUT_SECONDS,
) -> str:
    """The URL the named runtime listens on, from the agent's `Runtime.url`.

    Reads are open to any service token on the agent (the same rule that
    lets the gateway resolve topology), so the driver presents the
    service token it was spawned with. Without one — a standalone dev
    run — the request goes out bare and the agent's 401 is reported as
    what it is.
    """
    base = agent_url.rstrip("/")
    url = f"{base}/v1/runtimes/{quote(name, safe='')}"
    headers = {"Accept": "application/json"}
    if service_token:
        headers["Authorization"] = f"Bearer {service_token}"

    try:
        # `sync_client_for` rather than a bare `httpx.Client`: the bare
        # one parses certifi's PEM bundle on construction (~104 ms), and
        # this call runs on the event loop from `/v1/config/test`. It
        # also declines the user's proxy, because the agent is this
        # machine or this install's LAN and a corporate `HTTP_PROXY`
        # would swallow the lookup and report the agent as down.
        with sync_client_for(base, timeout=timeout) as client:
            response = client.get(url, headers=headers)
    except httpx.HTTPError as e:
        raise RuntimeResolutionError(
            f"could not resolve runtime {name!r}: the agent at {base} did not answer ({e}). "
            f"The driver is up in degraded mode; check EUGENE_PLEXUS_DRIVER_AGENT_URL and "
            f"that the agent is running, then restart the driver."
        ) from e

    if response.status_code == 404:
        raise RuntimeResolutionError(
            f"runtime {name!r} is not declared on the agent at {base}. `runtimeName` must "
            f"match a runtime's `name` exactly; GET /v1/runtimes on the agent lists them."
        )
    if response.status_code in (401, 403):
        raise RuntimeResolutionError(
            f"the agent at {base} rejected this driver's credentials ({response.status_code}) "
            f"while resolving runtime {name!r}. A driver spawned by the agent carries a "
            f"service token; a standalone driver has none and cannot follow a runtime by name "
            f"— set `baseUrl` instead."
        )
    if response.status_code >= 400:
        raise RuntimeResolutionError(
            f"the agent at {base} returned {response.status_code} for runtime {name!r}: "
            f"{response.text[:200]}"
        )

    try:
        body = response.json()
    except ValueError as e:
        raise RuntimeResolutionError(
            f"the agent at {base} returned non-JSON for runtime {name!r}: {response.text[:200]!r}"
        ) from e

    runtime_url = body.get("url") if isinstance(body, dict) else None
    if not isinstance(runtime_url, str) or not runtime_url.strip():
        raise RuntimeResolutionError(
            f"runtime {name!r} has no URL yet: the agent has not assigned it a port. Re-save "
            f"the runtime on the agent so one is allocated, then restart the driver."
        )
    resolved = runtime_url.rstrip("/")
    log.info("runtime %r resolves to %s (status %s)", name, resolved, body.get("status"))
    return resolved


__all__ = ["RuntimeResolutionError", "resolve_runtime_url"]
