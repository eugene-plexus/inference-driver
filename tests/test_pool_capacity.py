"""A full KV pool is load, not a broken backend (CB3, gateway#8).

llama-server's slots share one KV pool (`-c`) by default. When the prompts
in flight together outgrow it, the engine refuses the next request with a
500, or cuts every stream it was decoding with an `error` frame and closes.
Both say "Context size has been exceeded". As a 502 that tripped the
gateway's circuit, and on an 8B at 64k with four agents a replica 158 of 204
turns failed, most refused as "cooling down" by healthy replicas.

Every byte below is what llama-server b11211 sent (a 1B at `-c 4096`, four
~2,000-token requests at once; 2026-10-02).
"""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from eugene_plexus_inference_driver._generated.models import (
    BackendKind,
    GenerateRequest,
    Message,
    Role,
)
from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine
from eugene_plexus_inference_driver.failures import capacity_refused, disposition
from eugene_plexus_inference_driver.routes.generate import _backend_error

URL = "http://127.0.0.1:9/v1/chat/completions"

#: b11211's batch answer, verbatim.
POOL_FULL_BODY = (
    '{"error":{"code":500,"message":"Context size has been exceeded.","type":"server_error"}}'
)
#: Its streamed answer: a token, then the error frame, then the close.
TOKEN = b'data: {"choices":[{"index":0,"delta":{"content":"word0"}}],"object":"chat.completion.chunk"}\n\n'
POOL_FULL_FRAME = b"data: " + POOL_FULL_BODY.encode() + b"\n\n"


def _engine() -> OpenAiCompatibleHttpEngine:
    return OpenAiCompatibleHttpEngine(
        base_url="http://127.0.0.1:9/",
        api_key=None,
        model_id="m",
        backend_kind=BackendKind.openai_compat_http,
        timeout_seconds=42.0,
        auth_required=False,
    )


def _request() -> GenerateRequest:
    return GenerateRequest(messages=[Message(role=Role.user, content="Repeat this back")])


def test_a_full_pool_is_capacity_not_a_fault():
    error = CliError(f"openai_compat_http returned 500: {POOL_FULL_BODY}", upstream_status=500)
    assert capacity_refused(error)
    # Cascade-eligible: nothing was kept and nothing was answered.
    assert disposition(error) == "safe"
    failure = _backend_error(error, "openai_compat_http")
    assert failure.status_code == 503
    assert failure.detail["type"].endswith("#backend-capacity")
    assert failure.detail["retryDisposition"] == "safe"
    assert "Context size has been exceeded" in failure.detail["detail"]


def test_any_other_500_is_still_the_backends():
    error = CliError(
        'openai_compat_http returned 500: {"error":{"message":"boom"}}', upstream_status=500
    )
    assert not capacity_refused(error)
    assert disposition(error) == "indeterminate"
    assert _backend_error(error, "openai_compat_http").status_code == 502


def test_a_4xx_with_the_same_words_stays_the_callers():
    """A request bigger than the whole pool is llama-server's 400, and a
    400 is the request's whatever it says."""
    error = CliError("returned 400: Context size has been exceeded.", upstream_status=400)
    assert not capacity_refused(error)
    assert disposition(error) == "terminal"


@pytest.mark.asyncio
@respx.mock
async def test_a_stream_cut_for_room_says_so():
    """Before this the error frame was a frame with no choices, the cut
    read "ended without [DONE]", and its reason was lost."""
    respx.post(URL).mock(
        return_value=httpx.Response(
            200,
            content=TOKEN + POOL_FULL_FRAME,
            headers={"content-type": "text/event-stream"},
        )
    )
    with pytest.raises(CliError) as caught:
        async for _ in _engine().stream(_request()):
            pass
    assert capacity_refused(caught.value), str(caught.value)
    assert _backend_error(caught.value, "x").detail["type"].endswith("#backend-capacity")


@pytest.mark.asyncio
@respx.mock
async def test_a_choice_carrying_an_error_is_not_this():
    """OpenRouter's mid-stream error rides on a choice with
    `finish_reason: "error"`; that path is unchanged."""
    frame = {
        "error": {"code": 502, "message": "provider went away"},
        "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
    }
    respx.post(URL).mock(
        return_value=httpx.Response(
            200,
            content=TOKEN + b"data: " + json.dumps(frame).encode() + b"\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )
    )
    chunks = [c async for c in _engine().stream(_request())]
    assert chunks, "the stream ended normally"
