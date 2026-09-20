"""The two sampling knobs, and the finish reason that lost its name.

R3 item 4, driver half. `GenerateRequest` grew `topP` and `seed` on
2026-09-19 because `gateway.yaml` had promised since M0 that both were
carried to backends that support them and dropped with a logged warning
where they are not, and neither half was true: there was no field to
carry them and no warning anywhere.

And `FinishReason` grew `content_filter`, which had been folded into
`error`. Two different states with two different remedies -- something
broke, versus a classifier stopped the answer on purpose -- reported as
one value, which the gateway then flattened to `stop`.

These are unit tests of the shaping, on both the batch and the streaming
path, because the two build their payload from the same helper and
report their finish reason from two different places.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest
import respx

from eugene_plexus_inference_driver._generated.models import (
    FinishReason,
    GenerateRequest,
    Message,
    Role,
)
from eugene_plexus_inference_driver.engines.openai_compat_http import (
    OPENAI_FIXED_TEMPERATURE_PATTERN,
    OpenAiCompatibleHttpEngine,
)

BASE = "http://127.0.0.1:11434"


def _body(finish_reason: str = "stop") -> dict[str, object]:
    return {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "llama3.1:70b",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "PING"},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _engine(**kwargs: object) -> OpenAiCompatibleHttpEngine:
    return OpenAiCompatibleHttpEngine(
        api_key="sk-stub",
        base_url=BASE,
        model_id="llama3.1:70b",
        **kwargs,  # type: ignore[arg-type]
    )


def _request(**kwargs: object) -> GenerateRequest:
    return GenerateRequest(messages=[Message(role=Role.user, content="hi")], **kwargs)  # type: ignore[arg-type]


def _sent(route: respx.Route, index: int = 0) -> dict[str, object]:
    return json.loads(route.calls[index].request.read())


# --------------------------------------------------------------------------- #
# the two knobs reach the backend
# --------------------------------------------------------------------------- #


@respx.mock
async def test_top_p_and_seed_reach_the_backend() -> None:
    route = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_body())
    )
    await _engine().generate(_request(topP=0.42, seed=1234))

    payload = _sent(route)
    assert payload["top_p"] == 0.42
    assert payload["seed"] == 1234


@respx.mock
async def test_top_p_and_seed_reach_the_backend_when_streaming() -> None:
    """`_payload_for` is shared by both paths, so this asserts the
    sharing rather than a second implementation -- which is exactly why
    the helper exists: a streamed answer differing from a non-streamed
    one for the same request is the quietest bug this adapter can have.
    """
    stream = "data: " + json.dumps(
        {"choices": [{"delta": {"content": "PING"}, "finish_reason": None}]}
    )
    done = "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    route = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=f"{stream}\n\n{done}\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )
    )
    async for _ in _engine().stream(_request(topP=0.9, seed=7)):
        pass

    payload = _sent(route)
    assert payload["top_p"] == 0.9
    assert payload["seed"] == 7
    assert payload["stream"] is True


@respx.mock
async def test_an_unset_knob_is_absent_rather_than_null() -> None:
    """A JSON null is not the same as an omitted key to every backend,
    and the gateway owns these values: if one reaches a backend, the
    gateway put it there. Sending `"seed": null` would be the driver
    making a statement the caller never made."""
    route = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_body())
    )
    await _engine().generate(_request())

    payload = _sent(route)
    assert "top_p" not in payload
    assert "seed" not in payload


@respx.mock
async def test_a_zero_seed_is_sent(caplog: pytest.LogCaptureFixture) -> None:
    """`seed=0` is a real, commonly used seed and is falsy. The check
    that decides whether to send it has to be `is not None`, and this
    is the case that tells the two apart."""
    route = respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_body())
    )
    await _engine().generate(_request(seed=0))

    assert _sent(route)["seed"] == 0


@respx.mock
async def test_a_fixed_temperature_model_drops_top_p_too(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """OpenAI's reasoning models reject `top_p` for the same reason they
    reject `temperature`: the sampler is not the caller's to tune. The
    adapter already dropped one and would have 400ed on the other, so
    the drop moves with the flag rather than being a second rule.
    """
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_body())
    )
    adapter = OpenAiCompatibleHttpEngine(
        api_key="sk-test",
        model_id="o3-mini",
        fixed_temperature_pattern=OPENAI_FIXED_TEMPERATURE_PATTERN,
    )
    with caplog.at_level(logging.WARNING):
        await adapter.generate(_request(temperature=0.7, topP=0.5, seed=3))

    payload = _sent(route)
    assert "temperature" not in payload
    assert "top_p" not in payload
    # The seed is untouched: reasoning models accept it, and widening
    # the drop to every sampling parameter would be the over-correction.
    assert payload["seed"] == 3
    assert "top_p" in caplog.text


@respx.mock
async def test_a_tunable_model_keeps_top_p() -> None:
    """The pair. Without it, an adapter that dropped `top_p` for every
    model would pass the test above."""
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_body())
    )
    adapter = OpenAiCompatibleHttpEngine(
        api_key="sk-test",
        model_id="gpt-4o",
        fixed_temperature_pattern=OPENAI_FIXED_TEMPERATURE_PATTERN,
    )
    await adapter.generate(_request(temperature=0.7, topP=0.5))

    payload = _sent(route)
    assert payload["top_p"] == 0.5
    assert payload["temperature"] == 0.7


# --------------------------------------------------------------------------- #
# content_filter is its own state
# --------------------------------------------------------------------------- #


@respx.mock
async def test_a_filtered_answer_is_not_a_backend_error() -> None:
    """The reproduction. `content_filter` mapped to `error`, which the
    gateway then renders as `stop`, so a refusal reached the caller as a
    natural end."""
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_body(finish_reason="content_filter"))
    )
    response = await _engine().generate(_request())

    assert response.finishReason is FinishReason.content_filter


@respx.mock
async def test_a_filtered_answer_is_not_a_backend_error_when_streaming() -> None:
    chunk = "data: " + json.dumps(
        {"choices": [{"delta": {"content": "I can"}, "finish_reason": None}]}
    )
    done = "data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "content_filter"}]})
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=f"{chunk}\n\n{done}\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )
    )
    final = None
    async for event in _engine().stream(_request()):
        if event.done:
            final = event.result
    assert final is not None
    assert final.finishReason is FinishReason.content_filter


@respx.mock
async def test_an_unknown_finish_reason_is_still_a_stop() -> None:
    """The pair for the map itself: adding a row must not turn the
    fallback into something else. A backend inventing a value we have
    never seen still ends the turn cleanly with the text it produced."""
    respx.post(f"{BASE}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_body(finish_reason="something_new"))
    )
    response = await _engine().generate(_request())

    assert response.finishReason is FinishReason.stop


# --------------------------------------------------------------------------- #
# an adapter with no such knob says so
# --------------------------------------------------------------------------- #


def test_the_cli_engines_warn_once_per_parameter(caplog: pytest.LogCaptureFixture) -> None:
    """`gateway.yaml` promises *dropped with a warning where they do
    not* support it. The agentic CLIs are the backends that do not: the
    harness on the other side of the pipe owns its own sampler. Before
    2026-09-19 the field did not exist, so nothing was dropped and
    nothing was said -- the promise had neither half.

    Once per parameter, not once per request: a line per generation is
    a warning an operator filters out, which costs it its only job.
    """
    from eugene_plexus_inference_driver.engines.base import warn_dropped_sampling

    warned: set[str] = set()
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            warn_dropped_sampling(
                _request(topP=0.5, seed=9),
                engine="claude_code_cli",
                model_id="claude-opus-4-7",
                warned=warned,
            )

    lines = [r for r in caplog.records if "cannot carry" in r.getMessage()]
    assert len(lines) == 2, [r.getMessage() for r in lines]
    assert warned == {"topP", "seed"}


def test_nothing_is_said_when_nothing_was_asked_for(caplog: pytest.LogCaptureFixture) -> None:
    """The pair. A warning on every request regardless would pass the
    test above on its count and be noise on every call that never set
    either field -- which is nearly all of them."""
    from eugene_plexus_inference_driver.engines.base import warn_dropped_sampling

    warned: set[str] = set()
    with caplog.at_level(logging.WARNING):
        warn_dropped_sampling(_request(), engine="codex_cli", model_id="gpt-5-codex", warned=warned)

    assert [r for r in caplog.records if "cannot carry" in r.getMessage()] == []
    assert warned == set()


def test_a_zero_seed_still_warns(caplog: pytest.LogCaptureFixture) -> None:
    """`seed=0` is falsy and is a real seed. A truthiness check here
    would silently drop it AND stay silent about dropping it, which is
    the same defect twice."""
    from eugene_plexus_inference_driver.engines.base import warn_dropped_sampling

    warned: set[str] = set()
    with caplog.at_level(logging.WARNING):
        warn_dropped_sampling(
            _request(seed=0), engine="codex_cli", model_id="gpt-5-codex", warned=warned
        )

    assert warned == {"seed"}
