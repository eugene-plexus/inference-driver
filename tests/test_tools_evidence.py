"""Tools are assumed, and turned off only on the backend's own word.

Troy, 2026-10-12: "we should always assume in 2026 that a model & engine can use
tools". So no engine is refused tools for being untested, and the one thing that
turns them off is a backend that refused a request BECAUSE IT CARRIED TOOLS, in
words this driver has measured (`failures.tools_refusal`). The engine then
reports `toolCalling` false, says so once at INFO, and the request is the caller's
400 with the backend's words. **Not for good**: the driver does not restart when
its runtime does (the agent restarts an engine under a driver that goes on), so
the backend is asked again every ten minutes and an answer turns tools back on
(`test_tools_reask.py`).

**The bytes below were measured on 2026-10-12**, llama-server b10948 on
Qwen3-0.6B, CPU. Started with `--no-jinja` it answers every request with a
non-empty `tools` -- auto, `required`, `none`, streamed -- with HTTP **500**
`{"error":{"code":500,"message":"tools param requires --jinja flag",
"type":"server_error"}}`. A 500, not a 4xx, so the wording is what counts and
not the status. The same build with jinja on (its default) answers tools with a
200, a malformed tool definition with a different 500, a tool whose schema it
cannot parse with a different 400, and a request with no `messages` with a 400;
none of those is a refusal of tools, and each is a negative case here, because a
pattern that took tools away from a backend that serves them would be worse than
the problem it solves.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver._generated.models import (
    BackendKind,
    FunctionDefinition,
    GenerateRequest,
    Message,
    Role,
    Tool,
)
from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.engines._subprocess import CliError, ToolsRefused
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine
from eugene_plexus_inference_driver.failures import disposition, tools_refusal
from eugene_plexus_inference_driver.routes.generate import _backend_error
from eugene_plexus_inference_driver.settings import Settings
from tests.test_accounts import _wait_for_catalogue

BASE = "http://llama.local"
CHAT = f"{BASE}/v1/chat/completions"
WORDS = "tools param requires --jinja flag"

#: `--no-jinja`, verbatim.
NO_JINJA = {"error": {"code": 500, "message": WORDS, "type": "server_error"}}
#: Jinja on, a tool definition with a number for a name, verbatim: a 500 that mentions tools
#: and is not a refusal of them.
BAD_TOOLS = {
    "error": {
        "code": 500,
        "message": (
            "Failed to parse tools: [json.exception.type_error.302] type must be string, but is "
            'number; tools = [\n  {\n    "type": "function",\n    "function": {\n      "name": 5\n'
            "    }\n  }\n]"
        ),
        "type": "server_error",
    }
}
#: Jinja on, a tool whose schema has a type that is not one (sent through this driver, to the
#: real engine, 2026-10-12), verbatim: a 400 that CARRIES tools and says nothing of refusing them.
UNPARSEABLE_SCHEMA = {
    "error": {
        "code": 400,
        "message": (
            "Unable to generate parser for this template. Automatic parser generation failed: "
            "JSON schema error at #/properties/a: unrecognized type not-a-json-schema-type"
        ),
        "type": "invalid_request_error",
    }
}
#: No `messages`, verbatim: a 400 that is the request's and has nothing to do with tools.
NO_MESSAGES = {
    "error": {"code": 400, "message": "'messages' is required", "type": "invalid_request_error"}
}
#: Strata's own words for `response_format` together with tools (its source, `_openai`).
STRATA_FORMAT_WITH_TOOLS = {
    "error": {
        "type": "invalid_request_error",
        "message": "structured response_format with tools/MCP is not supported",
    }
}
#: llama-server while its model is still loading, verbatim (measured 2026-10-12 on b10948, key order
#: and all): a 503 that mentions no tools and refuses nothing about them. Nothing was computed, so
#: it is safe to send again (`test_loading_is_safe.py`), and it is not evidence about tools.
LOADING = {"error": {"message": "Loading model", "type": "unavailable_error", "code": 503}}
OK = {
    "model": "qwen",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}
WEATHER = Tool(
    type="function",
    function=FunctionDefinition(
        name="get_weather",
        description="Current weather for a city.",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    ),
)


def _engine(**kwargs: Any) -> OpenAiCompatibleHttpEngine:
    return OpenAiCompatibleHttpEngine(
        base_url=BASE,
        model_id="qwen",
        backend_kind=BackendKind.openai_compat_http,
        auth_required=False,
        **kwargs,
    )


def _ask(*, tools: bool = True, **fields: Any) -> GenerateRequest:
    return GenerateRequest(
        messages=[Message(role=Role.user, content="What is the weather in Paris?")],
        tools=[WEATHER] if tools else None,
        **fields,
    )


def _response(status: int, body: Any) -> httpx.Response:
    return (
        httpx.Response(status, text=body)
        if isinstance(body, str)
        else (httpx.Response(status, json=body))
    )


def _carries_tools(request: httpx.Request) -> bool:
    return bool(json.loads(request.content).get("tools"))


# --------------------------------------------------------------------------- #
# the wording table
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("body", "words"),
    [
        (json.dumps(NO_JINJA), WORDS),
        (WORDS, WORDS),
        (WORDS + ".", WORDS + "."),
        (json.dumps({"error": WORDS}), WORDS),
        (json.dumps({"message": WORDS}), WORDS),
        (f"  {WORDS}\n", WORDS),
    ],
    ids=["measured", "plain", "period", "bare-error", "bare-message", "whitespace"],
)
def test_the_measured_wording_is_found_in_every_shape_it_could_arrive_in(
    body: str, words: str
) -> None:
    assert tools_refusal(body) == words


@pytest.mark.parametrize(
    "body",
    [
        "",
        "null",
        "[]",
        "{}",
        json.dumps(NO_MESSAGES),
        json.dumps(BAD_TOOLS),
        json.dumps(UNPARSEABLE_SCHEMA),
        json.dumps(STRATA_FORMAT_WITH_TOOLS),
        json.dumps(LOADING),
        json.dumps({"error": {"message": f"upstream said: {WORDS}"}}),
        f"bad gateway: {WORDS}",
        json.dumps({"error": {"message": f"{WORDS}; and more"}}),
        # A sibling nobody has measured: the table is the measured wording only.
        json.dumps({"error": {"message": "tool_choice param requires --jinja flag"}}),
        # Ollama's, named in the contract and not measured (no Ollama on 2026-10-12).
        json.dumps(
            {"error": {"message": "registry.ollama.ai/library/gemma3:4b does not support tools"}}
        ),
        json.dumps({"detail": WORDS}),
        # Nested past what `json.loads` can parse. It raises `RecursionError`, which is
        # not a `ValueError`: 4000 brackets do it on CPython 3.12. A backend's body is
        # not ours to crash on, here or anywhere a failed request is being handled.
        "[" * 4000,
        '{"error":' * 4000,
        "[" * 100_000,
    ],
    ids=[
        "empty",
        "null",
        "list",
        "object",
        "no-messages",
        "bad-tools",
        "unparseable-schema",
        "strata-format-with-tools",
        "loading",
        "wrapped-by-a-proxy",
        "plain-wrapped",
        "with-a-suffix",
        "tool-choice-sibling",
        "ollama-unmeasured",
        "other-key",
        "deep-array",
        "deep-object",
        "very-deep-array",
    ],
)
def test_nothing_else_is_a_refusal_of_tools(body: str) -> None:
    assert tools_refusal(body) is None


# --------------------------------------------------------------------------- #
# the engine learns it, and says so once
# --------------------------------------------------------------------------- #


@respx.mock
@pytest.mark.anyio
async def test_the_measured_refusal_turns_tools_off_until_the_backend_answers_again() -> None:
    respx.post(CHAT).mock(return_value=httpx.Response(500, json=NO_JINJA))
    engine = _engine()
    assert engine.supports_tool_calling is True
    assert {"tools", "toolChoice", "parallelToolCalls"} <= set(engine.supported_settings)
    try:
        with pytest.raises(ToolsRefused) as refused:
            await engine.generate(_ask())
    finally:
        await engine.aclose()

    error = refused.value
    assert error.upstream_status == 500
    assert error.words == WORDS
    # The backend's own status and words, then what Eugene did about them. Said plainly,
    # and not as a claim that the driver restarts: it does not, with its runtime.
    assert "returned 500" in str(error) and WORDS in str(error)
    assert "Eugene treats tools as off for 'qwen'" in str(error)
    assert "asks the backend again every 10 minutes" in str(error)
    assert "until this driver restarts" not in str(error)
    assert engine.supports_tool_calling is False
    # `tools` is what the backend refused. What rides with it says nothing on a request
    # that carries none, so it stays offered (`test_tools_reask.py`).
    offered = set(engine.supported_settings)
    assert "tools" not in offered and {"toolChoice", "parallelToolCalls"} <= offered
    assert engine.tools_refused_words() == WORDS
    assert engine.tools_refused_words("qwen") == WORDS

    # A new engine starts by assuming tools again (a config change builds one).
    fresh = _engine()
    assert fresh.supports_tool_calling is True and fresh.tools_refused_words() is None
    await fresh.aclose()


@respx.mock
@pytest.mark.anyio
async def test_the_stream_learns_it_before_its_first_frame() -> None:
    respx.post(CHAT).mock(return_value=httpx.Response(500, json=NO_JINJA))
    engine = _engine()
    frames: list[object] = []
    try:
        with pytest.raises(ToolsRefused) as refused:
            async for chunk in engine.stream(_ask()):
                frames.append(chunk)
    finally:
        await engine.aclose()
    assert frames == [], "nothing was streamed: it can still be a status code"
    assert refused.value.words == WORDS
    assert engine.supports_tool_calling is False


@respx.mock
@pytest.mark.anyio
async def test_it_is_said_once_at_info(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(
        logging.INFO, logger="eugene_plexus_inference_driver.engines.openai_compat_http"
    )
    respx.post(CHAT).mock(return_value=httpx.Response(500, json=NO_JINJA))
    engine = _engine()
    try:
        for _ in range(3):  # the engine direct: the backend refuses each time and is heard once
            with pytest.raises(ToolsRefused):
                await engine.generate(_ask())
    finally:
        await engine.aclose()
    said = [r for r in caplog.records if r.levelno == logging.INFO and WORDS in r.getMessage()]
    assert len(said) == 1, [r.getMessage() for r in caplog.records]
    assert "toolCalling false" in said[0].getMessage()


# --------------------------------------------------------------------------- #
# what the caller sees
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("status", [500, 400])
def test_a_refusal_of_tools_is_the_callers_400_whatever_status_it_came_with(status: int) -> None:
    """llama-server's is a 500, which as a 502 would be counted against the circuit of a
    healthy engine (`indeterminate`: not replayed elsewhere, but counted), as `Jinja
    Exception` was before it was made the caller's."""
    error = ToolsRefused(
        f"openai_compat_http returned {status}: {json.dumps(NO_JINJA)}. The backend refused it.",
        words=WORDS,
        upstream_status=status,
    )
    assert disposition(error) == "terminal"
    failure = _backend_error(error, "openai_compat_http")
    assert failure.status_code == 400
    assert failure.detail["type"].endswith("#backend-rejected-request")
    assert failure.detail["retryDisposition"] == "terminal"
    assert WORDS in failure.detail["detail"]


def test_the_same_500_without_the_engine_having_decided_is_still_the_backends() -> None:
    """Only the engine, which knows the request carried tools, gives it the identity."""
    plain = CliError(
        f"openai_compat_http returned 500: {json.dumps(NO_JINJA)}", upstream_status=500
    )
    assert disposition(plain) == "indeterminate"
    assert _backend_error(plain, "openai_compat_http").status_code == 502


def _custom_app(tmp_path: Path, **config: Any) -> TestClient:
    path = tmp_path / "custom.yaml"
    path.write_text(
        json.dumps(
            {
                "provider": "openai_compat_custom",
                "baseUrl": BASE,
                "backendLocality": "local",
                **config,
            }
        ),
        "utf-8",
    )
    return TestClient(create_app(settings=Settings(config_file=path)))


def _llama(chat_side_effect: Any) -> respx.Route:
    """A llama-server by its `/props` and its list; every other path it lacks is a 404."""
    respx.get(f"{BASE}/props").mock(
        return_value=httpx.Response(
            200,
            json={
                "default_generation_settings": {"n_ctx": 8192},
                "modalities": {"vision": False, "audio": False},
            },
        )
    )
    respx.get(f"{BASE}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "qwen"}, {"id": "other"}]})
    )
    chat = respx.post(CHAT).mock(side_effect=chat_side_effect)
    respx.route(host="llama.local").mock(return_value=httpx.Response(404))
    return chat


def _body(*, tools: bool = True, **fields: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        **fields,
    }
    if tools:
        body["tools"] = [WEATHER.model_dump(mode="json", exclude_none=True)]
    return body


@respx.mock
@pytest.mark.parametrize("path", ["/v1/generate", "/v1/generate/stream"])
def test_through_the_app_the_first_request_is_a_400_and_info_then_says_false(
    tmp_path: Path, path: str
) -> None:
    chat = _llama(
        lambda request: (
            httpx.Response(500, json=NO_JINJA)
            if _carries_tools(request)
            else httpx.Response(200, json=OK)
        )
    )
    with _custom_app(tmp_path, modelId="qwen") as client:
        (before,) = client.get("/v1/info").json()["models"]
        assert before["capabilities"]["toolCalling"] is True
        assert "tools" in before["capabilities"]["supportedSettings"]
        # llama-server honours what it is sent: nothing to name (absent, not an empty list).
        assert "settingsNotHonoured" not in before["capabilities"]

        first = client.post(path, json=_body())
        assert first.status_code == 400, first.text
        detail = first.json()["detail"]
        assert detail["type"].endswith("#backend-rejected-request")
        assert detail["retryDisposition"] == "terminal"
        assert WORDS in detail["detail"]

        (after,) = client.get("/v1/info").json()["models"]
        assert after["capabilities"]["toolCalling"] is False
        settings = set(after["capabilities"]["supportedSettings"])
        assert "tools" not in settings
        assert {"toolChoice", "parallelToolCalls"} <= settings, "they say nothing without tools"

        # Later requests are refused at the driver's door, in the backend's words,
        # without another trip to a backend that has already answered.
        second = client.post(path, json=_body())
        assert second.status_code == 400, second.text
        refusal = second.json()["detail"]
        assert refusal["type"].endswith("#tools-unsupported")
        assert (
            WORDS in refusal["detail"]
            and "refused a request for its tools earlier" in refusal["detail"]
        )
        # What the person is told: the driver's own report, the cadence of asking again, and
        # no claim that a restart of the driver is what brings tools back.
        assert "asks the backend again every 10 minutes" in refusal["detail"]
        assert "restart" not in refusal["detail"]
        assert sum(1 for call in chat.calls if _carries_tools(call.request)) == 1

        # And chat without tools keeps working on the same engine.
        plain = client.post("/v1/generate", json=_body(tools=False))
        assert plain.status_code == 200, plain.text


# --------------------------------------------------------------------------- #
# narrowness: an unrelated refusal must not take tools away
# --------------------------------------------------------------------------- #

NOT_EVIDENCE = [
    pytest.param(400, NO_MESSAGES, True, "terminal", id="an-unrelated-400"),
    pytest.param(
        500,
        BAD_TOOLS,
        True,
        "indeterminate",
        id="a-malformed-tool-definition-measured-with-jinja-on",
    ),
    pytest.param(
        400, UNPARSEABLE_SCHEMA, True, "terminal", id="a-schema-the-engine-cannot-parse-measured"
    ),
    pytest.param(400, STRATA_FORMAT_WITH_TOOLS, True, "terminal", id="stratas-format-with-tools"),
    # Loading is nothing to do with tools -- and it is the one here that is safe to send again.
    pytest.param(503, LOADING, True, "safe", id="a-backend-still-loading"),
    pytest.param(
        502,
        {"error": {"message": f"upstream said: {WORDS}"}},
        True,
        "indeterminate",
        id="the-words-wrapped",
    ),
    pytest.param(
        500, f"backend failure: {WORDS}", True, "indeterminate", id="the-words-inside-plain-text"
    ),
    pytest.param(
        500,
        {"error": {"message": "tool_choice param requires --jinja flag"}},
        True,
        "indeterminate",
        id="unmeasured",
    ),
    pytest.param(
        500, NO_JINJA, False, "indeterminate", id="the-exact-words-to-a-request-with-no-tools"
    ),
    # A body `json.loads` cannot get to the bottom of (RecursionError, not ValueError). The
    # backend's own refusal must still be what the caller gets, not a crash of the driver's.
    pytest.param(500, "[" * 4000, True, "indeterminate", id="a-body-nested-too-deep-to-parse"),
]


@respx.mock
@pytest.mark.anyio
@pytest.mark.parametrize(("status", "body", "tools", "retry"), NOT_EVIDENCE)
@pytest.mark.parametrize("stream", [False, True])
async def test_an_unrelated_refusal_leaves_tools_on(
    status: int, body: Any, tools: bool, retry: str, stream: bool
) -> None:
    respx.post(CHAT).mock(return_value=_response(status, body))
    engine = _engine()
    try:
        with pytest.raises(CliError) as refused:
            if stream:
                _ = [c async for c in engine.stream(_ask(tools=tools))]
            else:
                await engine.generate(_ask(tools=tools))
    finally:
        await engine.aclose()

    assert type(refused.value) is CliError, "not the identity of a refusal of tools"
    assert refused.value.upstream_status == status
    assert engine.supports_tool_calling is True
    assert engine.tools_refused_words() is None
    assert {"tools", "toolChoice", "parallelToolCalls"} <= set(engine.supported_settings)
    # The status keeps its meaning: a 4xx is the caller's, anything else is the backend's, and
    # what nothing was computed for (llama-server still loading) is safe to send again.
    assert disposition(refused.value) == retry


@respx.mock
@pytest.mark.anyio
async def test_a_refusal_after_a_good_answer_still_counts() -> None:
    """What the backend said is believed whenever it says it, not only on a first request."""
    respx.post(CHAT).mock(
        side_effect=[httpx.Response(200, json=OK), httpx.Response(500, json=NO_JINJA)]
    )
    engine = _engine()
    try:
        await engine.generate(_ask())
        assert engine.supports_tool_calling is True
        with pytest.raises(ToolsRefused):
            await engine.generate(_ask())
    finally:
        await engine.aclose()
    assert engine.supports_tool_calling is False


@respx.mock
@pytest.mark.anyio
async def test_an_attachment_request_still_shows_the_backends_words_for_tools() -> None:
    """Upstream bodies are withheld from attachment requests, in case one echoes the file. This
    one is the whole of a short sentence this driver recognises, so it cannot hold one."""
    respx.post(CHAT).mock(return_value=httpx.Response(500, json=NO_JINJA))
    png = (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8Dw"
        "HwAFAAH/q842iQAAAABJRU5ErkJggg=="
    )
    request = GenerateRequest.model_validate(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {"type": "image_url", "image_url": {"url": png}},
                    ],
                }
            ],
            "tools": [WEATHER.model_dump(mode="json", exclude_none=True)],
        }
    )
    engine = _engine()
    try:
        with pytest.raises(ToolsRefused) as refused:
            await engine.generate(request)
    finally:
        await engine.aclose()
    assert WORDS in str(refused.value)
    assert "Upstream body omitted" not in str(refused.value)


# --------------------------------------------------------------------------- #
# an account: per model, and never written into its saved list
# --------------------------------------------------------------------------- #


def _account(path: Path) -> OpenAiCompatibleHttpEngine:
    return OpenAiCompatibleHttpEngine(
        base_url=BASE,
        model_id=None,
        backend_kind=BackendKind.openai_compat_http,
        auth_required=False,
        filter_models=False,
        catalogue_path=path,
    )


def _refuses_tools_for_a(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if body["model"] == "a" and body.get("tools"):
        return httpx.Response(500, json=NO_JINJA)
    return httpx.Response(200, json=OK)


@respx.mock
@pytest.mark.anyio
async def test_an_account_turns_tools_off_for_the_one_model_and_never_saves_it_to_the_list(
    tmp_path: Path,
) -> None:
    respx.get(f"{BASE}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}]})
    )
    respx.post(CHAT).mock(side_effect=_refuses_tools_for_a)
    saved = tmp_path / "models.json"
    engine = _account(saved)
    assert engine.catalogue is not None
    try:
        await engine.catalogue.refresh()
        with pytest.raises(ToolsRefused):
            await engine.generate(_ask(model="a"))

        def served() -> dict[str, Any]:
            assert engine.catalogue is not None
            return {m.id: m.capabilities for m in engine.catalogue.exposed()}

        caps = served()
        assert caps["a"] is not None and caps["a"].toolCalling is False
        assert "tools" not in (caps["a"].supportedSettings or [])
        # The companions stay offered: they say nothing on a request without tools.
        assert {"toolChoice", "parallelToolCalls"} <= set(caps["a"].supportedSettings or [])
        assert caps["b"] is not None and caps["b"].toolCalling is True
        assert "tools" in (caps["b"].supportedSettings or [])
        assert engine.tools_refused_words("a") == WORDS and engine.tools_refused_words("b") is None
        # The model the route resolves is the served one, so its door says no too.
        found = engine.catalogue.find("a")
        assert found is not None and found.capabilities is not None
        assert found.capabilities.toolCalling is False

        # Whatever saves the list next, what this engine learned is not in it: the
        # served model is a copy, and the stored one is the provider's own word. (A
        # refresh replaces the stored list before anything writes it, so this is
        # saved by hand to ask the question.)
        engine.catalogue._save()
        on_disk = {m["id"]: m for m in json.loads(saved.read_text("utf-8"))["models"]}
        assert on_disk["a"]["capabilities"]["toolCalling"] is True
        # A re-read of the provider's list keeps what this engine learned.
        await engine.catalogue.refresh()
        assert served()["a"].toolCalling is False  # type: ignore[union-attr]
        on_disk = {m["id"]: m for m in json.loads(saved.read_text("utf-8"))["models"]}
        assert on_disk["a"]["capabilities"]["toolCalling"] is True
        # The other model still takes tools.
        await engine.generate(_ask(model="b"))
    finally:
        await engine.aclose()

    # A new engine (the driver started again), reading the same saved list, assumes tools again.
    fresh = _account(saved)
    assert fresh.catalogue is not None
    found = fresh.catalogue.find("a")
    assert found is not None and found.capabilities is not None
    assert found.capabilities.toolCalling is True
    await fresh.aclose()


@respx.mock
@pytest.mark.anyio
async def test_what_an_ollama_account_learned_is_never_written_into_the_entry_it_stores(
    tmp_path: Path,
) -> None:
    """An Ollama account builds each entry from `engine_defaults(<the model's name>)`, which
    used to carry that name as the model's id, so a learned refusal for the model took `tools`
    out of the entry it STORED (and saved) the next time the list was read. The stored entry is
    the provider's word; what the engine learned is laid over it when it is served."""
    ollama = "http://ollama.local"
    respx.get(f"{ollama}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "qwen3:8b"}]})
    )
    respx.post(f"{ollama}/api/show").mock(
        return_value=httpx.Response(200, json={"capabilities": ["completion", "tools"]})
    )
    saved = tmp_path / "models.json"
    engine = OpenAiCompatibleHttpEngine(
        base_url=ollama,
        model_id=None,
        backend_kind=BackendKind.openai_compat_http,
        auth_required=False,
        catalogue_source="ollama",
        catalogue_path=saved,
    )
    assert engine.catalogue is not None
    try:
        await engine.catalogue.refresh()
        engine._learn_tools_refused(engine.resolve_model("qwen3:8b"), WORDS, 500)
        await engine.catalogue.refresh()  # the list is read again, as it is on its interval

        stored = engine.catalogue._by_id["qwen3:8b"].capabilities
        assert stored is not None and stored.toolCalling is True
        assert "tools" in (stored.supportedSettings or [])
        on_disk = json.loads(saved.read_text("utf-8"))["models"][0]["capabilities"]
        assert "tools" in on_disk["supportedSettings"] and on_disk["toolCalling"] is True

        served = engine.catalogue.find("qwen3:8b")
        assert served is not None and served.capabilities is not None
        assert served.capabilities.toolCalling is False
        assert "tools" not in (served.capabilities.supportedSettings or [])
    finally:
        await engine.aclose()


@respx.mock
def test_through_the_app_an_account_refuses_one_models_tools_at_the_door(tmp_path: Path) -> None:
    respx.get(f"{BASE}/props").mock(return_value=httpx.Response(404))
    respx.get(f"{BASE}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "a"}, {"id": "b"}]})
    )
    chat = respx.post(CHAT).mock(side_effect=_refuses_tools_for_a)
    respx.route(host="llama.local").mock(return_value=httpx.Response(404))
    with _custom_app(tmp_path) as client:
        _wait_for_catalogue(client)

        def tools_of(info: dict[str, Any]) -> dict[str, bool]:
            return {m["id"]: m["capabilities"]["toolCalling"] for m in info["models"]}

        assert tools_of(client.get("/v1/info").json()) == {"a": True, "b": True}
        first = client.post("/v1/generate", json=_body(model="a"))
        assert first.status_code == 400 and WORDS in first.json()["detail"]["detail"]
        assert tools_of(client.get("/v1/info").json()) == {"a": False, "b": True}

        second = client.post("/v1/generate", json=_body(model="a"))
        assert second.status_code == 400
        assert second.json()["detail"]["type"].endswith("#tools-unsupported")
        assert WORDS in second.json()["detail"]["detail"]
        other = client.post("/v1/generate", json=_body(model="b"))
        assert other.status_code == 200, other.text
        assert sum(1 for c in chat.calls if json.loads(c.request.content)["model"] == "a") == 1
