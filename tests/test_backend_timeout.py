"""R2.5 — a backend that is still computing has not failed.

The reproduction, written before the fix (roadmap §1). Four properties,
each of which the code fails today:

1. A read timeout from the upstream engine is **identified** as a
   timeout rather than folded into the generic transport branch. The
   measured symptom is that `str(httpx.ReadTimeout(""))` is the empty
   string, so the driver's own error read literally
   ``"openai_compat_http request failed: "`` -- an anonymous failure
   for the one case where the cause is exactly known.
2. The message names the deadline that fired and the knob that moves
   it, because an operator whose CPU box needs four minutes cannot act
   on a blank.
3. The route answers **504**, not 502. 502 means "the backend is
   broken and the next one might not be"; 504 means "this one had not
   finished", which the next one will not either.
4. The CLI backends' own timeout carries the same identity.
"""

from __future__ import annotations

import httpx
import pytest
import respx
from fastapi import HTTPException

from eugene_plexus_inference_driver._generated.models import (
    BackendKind,
    GenerateRequest,
    Message,
    Role,
)
from eugene_plexus_inference_driver.engines._subprocess import BackendTimeout, CliError
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine
from eugene_plexus_inference_driver.routes.generate import _backend_error


def _request() -> GenerateRequest:
    return GenerateRequest(messages=[Message(role=Role.user, content="say PING")])


def _engine(**kw: object) -> OpenAiCompatibleHttpEngine:
    return OpenAiCompatibleHttpEngine(
        base_url="http://127.0.0.1:9/",
        api_key=None,
        model_id="m",
        backend_kind=BackendKind.openai_compat_http,
        timeout_seconds=42.0,
        auth_required=False,
        **kw,  # type: ignore[arg-type]
    )


def test_backend_timeout_is_a_subclass_so_existing_handlers_still_catch_it() -> None:
    """Every `except CliError` in the routes keeps working unchanged."""
    assert issubclass(BackendTimeout, CliError)


@pytest.mark.asyncio
@respx.mock
async def test_a_read_timeout_is_identified_and_names_its_deadline() -> None:
    respx.post("http://127.0.0.1:9/v1/chat/completions").mock(side_effect=httpx.ReadTimeout(""))
    engine = _engine()
    with pytest.raises(BackendTimeout) as caught:
        await engine.generate(_request())
    message = str(caught.value)
    # The defect verbatim: "openai_compat_http request failed: " with
    # nothing after the colon.
    assert not message.endswith(": ")
    assert "42" in message, message
    assert "requestTimeoutSeconds" in message, message


@pytest.mark.asyncio
@respx.mock
async def test_a_streaming_read_timeout_is_identified_too() -> None:
    respx.post("http://127.0.0.1:9/v1/chat/completions").mock(side_effect=httpx.ReadTimeout(""))
    engine = _engine()
    with pytest.raises(BackendTimeout) as caught:
        async for _ in engine.stream(_request()):
            pass
    assert "requestTimeoutSeconds" in str(caught.value)


@pytest.mark.asyncio
@respx.mock
async def test_an_embeddings_read_timeout_is_identified_too() -> None:
    respx.post("http://127.0.0.1:9/v1/embeddings").mock(side_effect=httpx.ReadTimeout(""))
    engine = _engine()
    with pytest.raises(BackendTimeout) as caught:
        await engine.embed(["hello"])
    assert "requestTimeoutSeconds" in str(caught.value)


def test_a_timeout_answers_504_and_a_transport_failure_still_answers_502() -> None:
    timed_out = _backend_error(BackendTimeout("took too long", limit_seconds=42.0), "x")
    assert isinstance(timed_out, HTTPException)
    assert timed_out.status_code == 504

    refused = _backend_error(CliError("connection refused"), "x")
    assert refused.status_code == 502


@pytest.mark.asyncio
@respx.mock
async def test_the_504_problem_carries_the_engines_own_words() -> None:
    """End to end, because the value of the 504 is what it SAYS, and
    the message is built in the engine while the status is chosen in
    the route -- a test of either alone proves nothing about the pair."""
    respx.post("http://127.0.0.1:9/v1/chat/completions").mock(side_effect=httpx.ReadTimeout(""))
    with pytest.raises(BackendTimeout) as caught:
        await _engine().generate(_request())

    problem = _backend_error(caught.value, "openai_compat_http").detail
    assert problem["status"] == 504
    body = f"{problem['title']} {problem['detail']}".lower()
    assert "42" in body
    assert "requesttimeoutseconds" in body.replace(" ", "")


@pytest.mark.asyncio
@respx.mock
async def test_a_connect_timeout_is_a_dead_host_and_keeps_its_502() -> None:
    """The one timeout that is not "still computing". Nothing was handed
    to an engine, so the next backend is a real rescue and the cascade
    must still fire — which means this must NOT be a `BackendTimeout`."""
    respx.post("http://127.0.0.1:9/v1/chat/completions").mock(
        side_effect=httpx.ConnectTimeout("no route to host")
    )
    with pytest.raises(CliError) as caught:
        await _engine().generate(_request())

    assert not isinstance(caught.value, BackendTimeout)
    assert _backend_error(caught.value, "x").status_code == 502


@pytest.mark.asyncio
@respx.mock
async def test_an_over_long_prompt_still_hard_fails_as_400() -> None:
    """The 400 that step 7 earned: nothing in R2.5 may soften it."""
    respx.post("http://127.0.0.1:9/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": {"message": "n_ctx 512"}})
    )
    with pytest.raises(CliError) as caught:
        await _engine().generate(_request())
    assert _backend_error(caught.value, "x").status_code == 400


def test_the_driver_holds_the_LONGER_of_the_two_deadlines() -> None:
    """The backstop, not the decision. See the gateway's mirror of this
    test; `scripts/still-computing-acceptance.sh` asks both live."""
    from eugene_plexus_inference_driver.config import FIELDS
    from eugene_plexus_inference_driver.engines.base import DEFAULT_REQUEST_TIMEOUT_SECONDS

    field = next(f for f in FIELDS if f.key == "requestTimeoutSeconds")
    assert field.default == DEFAULT_REQUEST_TIMEOUT_SECONDS
    assert DEFAULT_REQUEST_TIMEOUT_SECONDS > 600.0, (
        "the gateway's deadline is 600 s; a driver deadline at or below it fires first "
        "and makes the gateway's knob inert -- which is the defect R2.5 fixed"
    )


def test_every_engine_reads_the_one_constant() -> None:
    """It was written in four places. A provider added later that types
    `120` again re-creates the ordering defect silently."""
    import inspect

    from eugene_plexus_inference_driver.engines.base import DEFAULT_REQUEST_TIMEOUT_SECONDS
    from eugene_plexus_inference_driver.engines.claude_code_cli import ClaudeCodeCliEngine
    from eugene_plexus_inference_driver.engines.codex_cli import CodexCliEngine

    for engine in (ClaudeCodeCliEngine, CodexCliEngine, OpenAiCompatibleHttpEngine):
        default = inspect.signature(engine.__init__).parameters["timeout_seconds"].default
        assert default == DEFAULT_REQUEST_TIMEOUT_SECONDS, engine.__name__
