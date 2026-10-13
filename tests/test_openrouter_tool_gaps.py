"""An OpenRouter model's listing leaves out a tool setting: named, not refused (Troy, 2026-10-12).

`docs/design/tools-assumed.md`: a setting a backend does not honour is **sent where it can
be, and named**, never refused, because a refusal is for what cannot be sent at all and a
harness that sends `parallel_tool_calls: false` on every request (Codex does) can then use no
model that is refused it. OpenRouter's listing says per model what each takes, and on its
public list of 2026-10-12 (659 models, 390 chat models naming `tools`) **377 leave out
`parallel_tool_calls`** and 8 leave out `tool_choice`. Until round three an explicit one was
refused for every one of them (`this engine cannot honor this explicit setting`), and the
gateway, reading `supportedSettings`, routed Codex around them all.

**What the rule rests on, and what it does not** (round four). The listing is OpenRouter saying
which parameters it lists for a model, not what any provider does with a field it was sent:
OpenAI's and Mistral's providers take `parallel_tool_calls`, and none of the 99 `openai/*`
models that name `tools` lists it. What holds is OpenRouter's documented rule for
`require_parameters`, which Eugene sets on every request that carries a setting the caller sent:
it routes only to providers that support every parameter in the request, so a parameter the
listing leaves out cannot be delivered (not tried: no key was used). Hence, for a model whose
listing names `tools` and leaves a companion out, **while Eugene asks for `require_parameters`**:

* the setting is **offered** (`supportedSettings`) and **listed**
  (`capabilities.settingsNotHonoured`: `parallelToolCalls` with `false`, `toolChoice` with
  `none`, `required` and `function`), on the served entry only -- never in the list saved to disk;
* it is **not delivered**: OpenRouter lists no provider of this model that takes it, and Eugene
  asks OpenRouter only for providers that take every parameter it sends (`require_parameters`),
  so it is left out of the request, once per field in the log;
* it is **named** on `x-eugene-plexus-ignored-settings`, but only when the caller sent it
  (`callerSettings`) with a value the listing names, as the gateway names it.

With `require_parameters` off, nothing is withheld and nothing is named: the settings are sent,
an unknown counting as yes. **`tool_choice: none` is not honoured by sending no tools** there, as
it is for Strata and Ollama (which ignore `tool_choice` outright): a strict upstream (Bedrock's
Converse, behind `amazon/nova-*`) refuses a history that carries tool calls or results without
tool definitions. It is named like `required` and a named function.

A model whose listing names both companions is sent them as asked and has no gap; a model
that names no `tools` is the provider's word on tools and is unchanged (a request that carries
none is not refused for `tool_choice` or `parallel_tool_calls` there: they are dropped, once per
field in the log); a listing that says nothing is offered every tool setting and has none.
"""

from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver._generated.models import (
    BackendKind,
    DriverModel,
    GenerateRequest,
    Message,
    Role,
)
from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.engines.openai_compat_http import (
    OpenAiCompatibleHttpEngine,
    _Target,
    unlisted_tool_settings,
)
from eugene_plexus_inference_driver.settings import Settings
from tests.test_accounts import _completion, _or_model, _wait_for_catalogue, _write

OPENROUTER = "https://openrouter.ai/api"
HEADER = "x-eugene-plexus-ignored-settings"
CHAT = f"{OPENROUTER}/v1/chat/completions"
ENGINE_LOG = "eugene_plexus_inference_driver.engines.openai_compat_http"
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}
NAMED = {"type": "function", "function": {"name": "get_weather"}}
PARALLEL_GAP = {"setting": "parallelToolCalls", "values": [False]}
CHOICE_GAP = {"setting": "toolChoice", "values": ["none", "required", "function"]}
#: What the log, the docstrings and these tests say of a setting that is left out, and nothing
#: else: a statement about the listing and about what Eugene asks of OpenRouter, never about what
#: a provider does with the field (round four, 2026-10-12).
NOT_DELIVERED = (
    "not delivered: OpenRouter lists no provider of this model that takes it, and Eugene asks "
    "OpenRouter only for providers that take every parameter it sends (require_parameters)"
)
SSE_OK = (
    'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
    'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)

#: The models of this account, by what each one's listing names (OpenRouter's own spellings).
LISTED = {
    "vendor/no-parallel": ["max_tokens", "tools", "tool_choice"],
    "vendor/no-choice": ["max_tokens", "tools", "parallel_tool_calls"],
    "vendor/neither": ["max_tokens", "tools"],
    "vendor/full": ["max_tokens", "tools", "tool_choice", "parallel_tool_calls"],
    "vendor/no-tools": ["max_tokens"],
}


def _silent() -> dict[str, Any]:
    entry = _or_model("vendor/silent")
    del entry["supported_parameters"]  # the key is simply not there
    return entry


def _listing() -> dict[str, Any]:
    models = [_or_model(name, supported_parameters=params) for name, params in LISTED.items()]
    models.append(_silent())
    models.append(
        _or_model(
            "vendor/embed",
            architecture={"input_modalities": ["text"], "output_modalities": ["embeddings"]},
            supported_parameters=["tools"],
        )
    )
    return {"data": models}


def _answer(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    if body.get("stream"):
        return httpx.Response(200, text=SSE_OK, headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json=_completion(body["model"]))


def _mock_listing() -> None:
    respx.get(f"{OPENROUTER}/v1/models/user").mock(
        return_value=httpx.Response(200, json=_listing())
    )
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.get(f"{OPENROUTER}/v1/videos/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )


@contextlib.contextmanager
def _driver(tmp_path: Path, answer: Any = _answer) -> Iterator[tuple[TestClient, respx.Route]]:
    """The real app for an OpenRouter account, once its list has been read, and the route its
    chat requests reach."""
    _mock_listing()
    chat = respx.post(CHAT).mock(side_effect=answer)
    config = tmp_path / "openrouter.yaml"
    _write(config, {"provider": "openrouter", "apiKey": "sk-or-test"})
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        _wait_for_catalogue(client)
        yield client, chat


def _body(model: str, *, tools: bool = True, **fields: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        **fields,
    }
    if tools:
        body["tools"] = [WEATHER]
    return body


def _sent(route: respx.Route) -> dict[str, Any]:
    return json.loads(route.calls[-1].request.content)


def _post(client: TestClient, stream: bool, body: dict[str, Any]) -> httpx.Response:
    return client.post("/v1/generate/stream" if stream else "/v1/generate", json=body)


# --------------------------------------------------------------------------- #
# the rule, on its own
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("listed", "gaps"),
    [
        (["tools"], [CHOICE_GAP, PARALLEL_GAP]),
        (["tools", "toolChoice"], [PARALLEL_GAP]),
        (["tools", "parallelToolCalls"], [CHOICE_GAP]),
        (["tools", "toolChoice", "parallelToolCalls"], []),
        (["maxTokens", "tools", "seed"], [CHOICE_GAP, PARALLEL_GAP]),
        # No `tools`: the provider's word on tools, which is `toolCalling` false, and what
        # rides with them says nothing there.
        (["toolChoice", "parallelToolCalls"], []),
        (["maxTokens"], []),
        ([], []),
    ],
    ids=[
        "tools-only",
        "tools-and-choice",
        "tools-and-parallel",
        "all-three",
        "tools-among-others",
        "companions-without-tools",
        "no-tools",
        "empty",
    ],
)
def test_the_gap_is_what_a_listing_that_names_tools_leaves_out(
    listed: list[str], gaps: list[dict[str, Any]]
) -> None:
    assert [g.model_dump(mode="json") for g in unlisted_tool_settings(listed)] == gaps


def test_a_fresh_list_each_time() -> None:
    unlisted_tool_settings(["tools"])[0].values.append("auto")
    assert unlisted_tool_settings(["tools"])[0].values == ["none", "required", "function"]


# --------------------------------------------------------------------------- #
# what /v1/info says
# --------------------------------------------------------------------------- #


@respx.mock
def test_info_offers_and_lists_the_gap_of_each_model_and_the_saved_list_has_none(
    tmp_path: Path,
) -> None:
    with _driver(tmp_path) as (client, _):
        info = client.get("/v1/info").json()
    caps = {m["id"]: m["capabilities"] for m in info["models"]}
    expected = {
        "vendor/no-parallel": [PARALLEL_GAP],
        "vendor/no-choice": [CHOICE_GAP],
        "vendor/neither": [CHOICE_GAP, PARALLEL_GAP],
    }
    for model, gaps in expected.items():
        assert caps[model]["settingsNotHonoured"] == gaps, model
        # A gap is named, not refused, so it is offered too (the contract says so).
        assert {"tools", "toolChoice", "parallelToolCalls"} <= set(caps[model]["supportedSettings"])
        assert caps[model]["toolCalling"] is True
    # Nothing to name: it lists both, it lists no tools, it says nothing, it does not chat.
    for model in ("vendor/full", "vendor/no-tools", "vendor/silent", "vendor/embed"):
        assert "settingsNotHonoured" not in caps[model], model
    # A model whose listing names no tools is the provider's word on tools: `toolCalling` false
    # and no `tools`. What rides with tools is offered all the same (round five): a request that
    # carries none has it dropped, and the gateway admits by this list (`test_tools_off_rule.py`).
    assert caps["vendor/no-tools"]["toolCalling"] is False
    assert "tools" not in caps["vendor/no-tools"]["supportedSettings"]
    assert {"toolChoice", "parallelToolCalls"} <= set(caps["vendor/no-tools"]["supportedSettings"])
    # And every other model's settings are what its listing said.
    assert set(caps["vendor/no-parallel"]["supportedSettings"]) == {
        "maxTokens",
        "tools",
        "toolChoice",
        "parallelToolCalls",
    }

    # The provider's own word is what is saved. What this engine adds is laid over it
    # when it is served, and a list read again from OpenRouter must not carry it.
    saved = json.loads((tmp_path / "openrouter.catalogue.json").read_text("utf-8"))
    for model in saved["models"]:
        assert "settingsNotHonoured" not in model["capabilities"], model["id"]
    stored = {m["id"]: m["capabilities"]["supportedSettings"] for m in saved["models"]}
    assert "parallelToolCalls" not in stored["vendor/no-parallel"]
    assert "toolChoice" not in stored["vendor/no-choice"]
    assert stored["vendor/no-tools"] == ["maxTokens"], "what is offered beside it is not saved"


@respx.mock
def test_serving_a_model_never_changes_the_entry_the_driver_stores(tmp_path: Path) -> None:
    """What the engine lays over a listing is laid over a copy. The stored entries are the
    provider's own word, saved to disk and read again; one that came to carry the gap, or an
    offered setting the listing never named, would be saved with it by the next write and would
    outlive the engine that said it. Looked at in memory after `/v1/info` has served every one."""
    with _driver(tmp_path) as (client, _):
        info = client.get("/v1/info").json()
        stored = {m.id: m for m in client.app.state.adapter.catalogue._models}  # type: ignore[attr-defined]
    served = {m["id"]: m["capabilities"] for m in info["models"]}
    assert served["vendor/neither"]["settingsNotHonoured"], "the served entry carries it"
    assert stored.keys() == served.keys()
    for model_id, entry in stored.items():
        assert entry.capabilities is not None
        assert entry.capabilities.settingsNotHonoured is None, model_id
    neither = stored["vendor/neither"].capabilities
    assert neither is not None and set(neither.supportedSettings or []) == {"maxTokens", "tools"}
    no_tools = stored["vendor/no-tools"].capabilities
    assert no_tools is not None and set(no_tools.supportedSettings or []) == {"maxTokens"}
    assert set(served["vendor/no-tools"]["supportedSettings"]) == {
        "maxTokens",
        "toolChoice",
        "parallelToolCalls",
    }, "served with what rides with tools offered, on a copy"


# --------------------------------------------------------------------------- #
# parallel_tool_calls: not refused, not delivered, named
# --------------------------------------------------------------------------- #


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
def test_an_explicit_parallel_false_is_answered_not_sent_and_named(
    tmp_path: Path, stream: bool
) -> None:
    """Codex's shape: `parallel_tool_calls: false` on every request, with tools. Until round
    three it was a 400 for 377 of 390 OpenRouter tool models."""
    with _driver(tmp_path) as (client, chat):
        response = _post(
            client,
            stream,
            _body(
                "vendor/no-parallel",
                parallelToolCalls=False,
                toolChoice="auto",
                callerSettings=["tools", "toolChoice", "parallelToolCalls"],
            ),
        )
    assert response.status_code == 200, response.text
    assert response.headers[HEADER] == "parallelToolCalls"
    sent = _sent(chat)
    assert "parallel_tool_calls" not in sent, "OpenRouter lists no provider that takes it"
    assert [t["function"]["name"] for t in sent["tools"]] == ["get_weather"]
    assert sent["tool_choice"] == "auto", "what the listing names is sent as asked"
    # Why it is not delivered: a caller's setting makes OpenRouter strict about the rest.
    assert sent["provider"] == {"require_parameters": True}


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("fields", "why"),
    [
        ({"parallelToolCalls": True, "callerSettings": ["tools", "parallelToolCalls"]}, "true"),
        ({"parallelToolCalls": False, "callerSettings": ["tools"]}, "inherited"),
        ({"callerSettings": ["tools"]}, "absent"),
    ],
    ids=["true-is-what-it-does-anyway", "a-profiles-false-is-not-named", "nothing-sent"],
)
def test_only_a_false_the_caller_sent_is_named(
    tmp_path: Path, stream: bool, fields: dict[str, Any], why: str
) -> None:
    with _driver(tmp_path) as (client, chat):
        response = _post(client, stream, _body("vendor/no-parallel", **fields))
    assert response.status_code == 200, response.text
    assert HEADER not in response.headers
    assert "parallel_tool_calls" not in _sent(chat), "not delivered where the listing lacks it"


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
def test_a_model_whose_listing_names_parallel_tool_calls_is_sent_it_as_asked(
    tmp_path: Path, stream: bool
) -> None:
    with _driver(tmp_path) as (client, chat):
        response = _post(
            client,
            stream,
            _body(
                "vendor/full",
                parallelToolCalls=False,
                toolChoice="required",
                callerSettings=["tools", "toolChoice", "parallelToolCalls"],
            ),
        )
    assert response.status_code == 200, response.text
    assert HEADER not in response.headers, "this listing leaves nothing out"
    sent = _sent(chat)
    assert sent["parallel_tool_calls"] is False and sent["tool_choice"] == "required"


# --------------------------------------------------------------------------- #
# tool_choice: the other companion, the same way, and `none` too
# --------------------------------------------------------------------------- #


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("choice", "header"),
    [("none", "toolChoice"), ("required", "toolChoice"), (NAMED, "toolChoice"), ("auto", None)],
    ids=["none", "required", "named", "auto"],
)
def test_a_tool_choice_the_listing_leaves_out_is_not_sent_and_every_value_but_auto_is_named(
    tmp_path: Path, stream: bool, choice: Any, header: str | None
) -> None:
    with _driver(tmp_path) as (client, chat):
        response = _post(
            client,
            stream,
            _body("vendor/no-choice", toolChoice=choice, callerSettings=["tools", "toolChoice"]),
        )
    assert response.status_code == 200, response.text
    assert response.headers.get(HEADER) == header
    sent = _sent(chat)
    assert "tool_choice" not in sent, "OpenRouter lists no provider that takes it"
    assert [t["function"]["name"] for t in sent["tools"]] == ["get_weather"], (
        "tools stay, `none` too"
    )


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
def test_tool_choice_none_keeps_the_tools_and_is_named_not_honoured(
    tmp_path: Path, stream: bool
) -> None:
    """Round three honoured `none` here by sending no tools, as it does for Strata and Ollama,
    which ignore `tool_choice` outright. Here that is a trap: the history may carry tool calls
    and results, which a strict upstream refuses without the tool definitions. So the tools go,
    `tool_choice` is not delivered, the model may call one, and the caller is told."""
    with _driver(tmp_path) as (client, chat):
        response = _post(
            client,
            stream,
            _body(
                "vendor/neither",
                toolChoice="none",
                parallelToolCalls=False,
                callerSettings=["tools", "toolChoice", "parallelToolCalls"],
            ),
        )
    assert response.status_code == 200, response.text
    assert response.headers[HEADER] == "toolChoice, parallelToolCalls"
    sent = _sent(chat)
    assert [t["function"]["name"] for t in sent["tools"]] == ["get_weather"]
    assert not {"tool_choice", "parallel_tool_calls"} & sent.keys()


#: A history with a call and its result, and what a strict upstream (Bedrock's Converse, behind
#: `amazon/nova-*`) answers a request that carries one and no tool definitions: a 400 on the
#: history. A stand-in for it: the words are not a recording.
HISTORY = [
    {"role": "user", "content": "What is the weather in Paris?"},
    {
        "role": "assistant",
        "content": None,
        "toolCalls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
            }
        ],
    },
    {"role": "tool", "toolCallId": "call_1", "content": '{"temp_c":17,"sky":"light rain"}'},
    {"role": "user", "content": "And is that warm for May?"},
]


def _strict_upstream(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    history_has_calls = any(
        m.get("tool_calls") or m.get("role") == "tool" for m in body["messages"]
    )
    if history_has_calls and not body.get("tools"):
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": 400,
                    "message": "tool calls and results in the messages need the tool definitions",
                }
            },
        )
    return _answer(request)


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("model", ["vendor/no-choice", "vendor/neither"])
def test_through_the_app_a_tool_history_with_tool_choice_none_still_has_its_tools(
    tmp_path: Path, stream: bool, model: str
) -> None:
    """The reason `none` is named here and not honoured by stripping the tools: the same request
    with the tools taken off is the upstream's 400, and the turn after a tool result is exactly
    where an agent says `none` ("answer now, no more calls")."""
    request = _body(
        model,
        messages=HISTORY,
        toolChoice="none",
        callerSettings=["tools", "toolChoice"],
    )
    with _driver(tmp_path, answer=_strict_upstream) as (client, chat):
        response = _post(client, stream, request)
    assert response.status_code == 200, response.text
    assert response.headers[HEADER] == "toolChoice"
    sent = _sent(chat)
    assert [t["function"]["name"] for t in sent["tools"]] == ["get_weather"]
    assert "tool_choice" not in sent
    roles = [m["role"] for m in sent["messages"]]
    assert roles == ["user", "assistant", "tool", "user"]
    assert sent["messages"][1]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert sent["messages"][2]["tool_call_id"] == "call_1"


@respx.mock
def test_a_listing_with_neither_names_both_in_the_order_of_its_gaps(tmp_path: Path) -> None:
    with _driver(tmp_path) as (client, chat):
        response = client.post(
            "/v1/generate",
            json=_body(
                "vendor/neither",
                toolChoice="required",
                parallelToolCalls=False,
                callerSettings=["tools", "toolChoice", "parallelToolCalls"],
            ),
        )
    assert response.status_code == 200, response.text
    assert response.headers[HEADER] == "toolChoice, parallelToolCalls"
    assert not {"tool_choice", "parallel_tool_calls"} & _sent(chat).keys()


# --------------------------------------------------------------------------- #
# only while Eugene asks OpenRouter for providers that take every parameter it sends
# --------------------------------------------------------------------------- #


async def _account(*, require_parameters: bool) -> OpenAiCompatibleHttpEngine:
    """An OpenRouter account as the registry builds one (`providers.py`), but for
    `require_parameters`, with its list read."""
    _mock_listing()
    engine = OpenAiCompatibleHttpEngine(
        api_key="sk-or-test",
        base_url=OPENROUTER,
        model_id=None,
        backend_kind=BackendKind.openai_compat_http,
        dialect="openrouter",
        catalogue_source="openrouter",
        filter_models=False,
        require_parameters=require_parameters,
    )
    assert engine.catalogue is not None
    assert await engine.catalogue.refresh()
    return engine


def _ask(model: str, **fields: Any) -> GenerateRequest:
    return GenerateRequest.model_validate(
        {
            "model": model,
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": [WEATHER],
            **fields,
        }
    )


EXPLICIT = {
    "toolChoice": "required",
    "parallelToolCalls": False,
    "callerSettings": ["tools", "toolChoice", "parallelToolCalls"],
}


@respx.mock
@pytest.mark.anyio
async def test_with_require_parameters_the_settings_are_withheld_and_named() -> None:
    engine = await _account(require_parameters=True)
    try:
        request = _ask("vendor/neither", **EXPLICIT)
        payload = engine._payload_for(request, engine.resolve_model("vendor/neither"))
        assert not {"tool_choice", "parallel_tool_calls"} & payload.keys()
        assert payload["provider"] == {"require_parameters": True}
        assert engine.ignored_settings(request) == ["toolChoice", "parallelToolCalls"]
        entry = engine.catalogue.find("vendor/neither")  # type: ignore[union-attr]
        assert entry is not None and entry.capabilities is not None
        assert [g.setting for g in entry.capabilities.settingsNotHonoured or []] == [
            "toolChoice",
            "parallelToolCalls",
        ]
    finally:
        await engine.aclose()


@respx.mock
@pytest.mark.anyio
async def test_without_require_parameters_the_settings_are_sent_and_nothing_is_named() -> None:
    """The premise is OpenRouter's rule for `require_parameters`, so with it off there is no
    rule: a provider that takes the field, though the listing does not name it (OpenAI's and
    Mistral's take `parallel_tool_calls`), is sent it, and none is named as not honoured. An
    unknown counts as yes (`docs/design/tools-assumed.md`): the settings stay offered."""
    engine = await _account(require_parameters=False)
    try:
        request = _ask("vendor/neither", **EXPLICIT)
        payload = engine._payload_for(request, engine.resolve_model("vendor/neither"))
        assert payload["tool_choice"] == "required"
        assert payload["parallel_tool_calls"] is False
        assert "provider" not in payload, "nothing asks OpenRouter to hold to the listing"
        assert engine.ignored_settings(request) == []
        entry = engine.catalogue.find("vendor/neither")  # type: ignore[union-attr]
        assert entry is not None and entry.capabilities is not None
        assert entry.capabilities.settingsNotHonoured is None
        assert {"tools", "toolChoice", "parallelToolCalls"} <= set(
            entry.capabilities.supportedSettings or []
        )
        # `none` is a value like the others: sent, with its tools.
        none = _ask("vendor/neither", toolChoice="none", callerSettings=["tools", "toolChoice"])
        sent = engine._payload_for(none, engine.resolve_model("vendor/neither"))
        assert sent["tool_choice"] == "none" and sent["tools"]
    finally:
        await engine.aclose()


@respx.mock
@pytest.mark.anyio
@pytest.mark.parametrize("require_parameters", [True, False])
async def test_what_is_withheld_follows_require_parameters_whatever_the_entry_says(
    require_parameters: bool,
) -> None:
    """`_withheld_settings` itself, given an entry that carries the gaps: whether they are
    withheld is the engine's `require_parameters` and not what the entry says."""
    engine = await _account(require_parameters=require_parameters)
    try:
        entry = engine.catalogue.find("vendor/neither")  # type: ignore[union-attr]
        assert entry is not None and entry.capabilities is not None
        carrying = DriverModel.model_validate(
            {
                "id": "vendor/neither",
                "surfaces": ["chat"],
                "capabilities": {
                    "supportedSettings": ["tools", "toolChoice", "parallelToolCalls"],
                    "settingsNotHonoured": [CHOICE_GAP, PARALLEL_GAP],
                },
            }
        )
        target = _Target(
            id="vendor/neither", upstream="vendor/neither", temperature_fixed=False, entry=carrying
        )
        withheld = engine._withheld_settings(target, _ask("vendor/neither", **EXPLICIT))
        assert withheld == (
            frozenset({"toolChoice", "parallelToolCalls"}) if require_parameters else frozenset()
        )
    finally:
        await engine.aclose()


@respx.mock
@pytest.mark.anyio
async def test_the_log_says_why_in_one_set_of_words_and_claims_nothing_about_providers(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=ENGINE_LOG)
    engine = await _account(require_parameters=True)
    try:
        request = _ask("vendor/neither", **EXPLICIT)
        for _ in range(3):
            engine._payload_for(request, engine.resolve_model("vendor/neither"))
    finally:
        await engine.aclose()
    said = [r.getMessage() for r in caplog.records if "vendor/neither" in r.getMessage()]
    assert sum("`tool_choice`" in m for m in said) == 1, said
    assert sum("`parallel_tool_calls`" in m for m in said) == 1, said
    for message in said:
        assert NOT_DELIVERED in message
        assert "measured" not in message.lower() and "takes none" not in message
        assert "x-eugene-plexus-ignored-settings" in message


# --------------------------------------------------------------------------- #
# only when the request itself asks OpenRouter to hold to the listing (round five)
# --------------------------------------------------------------------------- #
#
# The premise of withholding is OpenRouter's rule for `require_parameters`: a parameter the
# listing leaves out would leave the request no provider. Eugene puts `require_parameters` in the
# payload only for a request that carries a setting the caller sent (`callerSettings`), so a
# request that lists none asks OpenRouter for nothing strict, and has no reason to lose a setting:
# it is sent, an unknown counting as yes, and nothing is named or logged as not delivered.


@respx.mock
@pytest.mark.anyio
@pytest.mark.parametrize(
    ("caller_settings", "asks"),
    [
        (None, False),
        ([], False),
        (["maxTokens"], True),
        (["tools", "toolChoice", "parallelToolCalls"], True),
    ],
    ids=["absent", "empty", "another-setting", "the-tool-settings"],
)
async def test_a_setting_is_withheld_exactly_when_the_payload_asks_for_require_parameters(
    caller_settings: list[str] | None, asks: bool
) -> None:
    engine = await _account(require_parameters=True)
    try:
        fields: dict[str, Any] = {"toolChoice": "required", "parallelToolCalls": False}
        if caller_settings is not None:
            fields["callerSettings"] = caller_settings
        request = _ask("vendor/neither", **fields)
        payload = engine._payload_for(request, engine.resolve_model("vendor/neither"))
        assert ("provider" in payload) is asks
        if asks:
            assert payload["provider"] == {"require_parameters": True}
            assert not {"tool_choice", "parallel_tool_calls"} & payload.keys()
        else:
            assert payload["tool_choice"] == "required"
            assert payload["parallel_tool_calls"] is False
    finally:
        await engine.aclose()


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("fields", [{}, {"callerSettings": []}], ids=["absent", "empty"])
def test_with_no_caller_settings_the_listed_out_settings_are_sent_and_nothing_is_named(
    tmp_path: Path, stream: bool, fields: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    """The through-the-app shape of a caller that is not the gateway (which always lists what it
    sends): `vendor/neither`'s listing leaves out both, and a request that names neither in
    `callerSettings` is sent both as asked."""
    caplog.set_level(logging.WARNING, logger=ENGINE_LOG)
    with _driver(tmp_path) as (client, chat):
        response = _post(
            client,
            stream,
            _body("vendor/neither", toolChoice="required", parallelToolCalls=False, **fields),
        )
    assert response.status_code == 200, response.text
    assert HEADER not in response.headers, "nothing the caller listed, so nothing is named"
    sent = _sent(chat)
    assert sent["tool_choice"] == "required" and sent["parallel_tool_calls"] is False
    assert "provider" not in sent, "nothing asks OpenRouter to hold to the listing"
    assert [t["function"]["name"] for t in sent["tools"]] == ["get_weather"]
    assert not [r for r in caplog.records if "not delivered" in r.getMessage()]


@respx.mock
@pytest.mark.anyio
async def test_the_not_delivered_line_is_said_only_when_something_was_withheld(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=ENGINE_LOG)
    engine = await _account(require_parameters=True)
    try:
        target = engine.resolve_model("vendor/neither")
        sends = {"toolChoice": "required", "parallelToolCalls": False}
        for _ in range(3):
            engine._payload_for(_ask("vendor/neither", **sends), target)
        assert not [r for r in caplog.records if "not delivered" in r.getMessage()], (
            "no caller setting: nothing was withheld, so nothing is said"
        )
        for _ in range(3):
            engine._payload_for(_ask("vendor/neither", callerSettings=["tools"], **sends), target)
    finally:
        await engine.aclose()
    said = [r.getMessage() for r in caplog.records if "not delivered" in r.getMessage()]
    assert sum("`tool_choice`" in m for m in said) == 1, said
    assert sum("`parallel_tool_calls`" in m for m in said) == 1, said


# --------------------------------------------------------------------------- #
# what has not moved
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_silent_listing_is_sent_every_tool_setting_and_names_nothing(tmp_path: Path) -> None:
    with _driver(tmp_path) as (client, chat):
        response = client.post(
            "/v1/generate",
            json=_body(
                "vendor/silent",
                toolChoice="required",
                parallelToolCalls=False,
                callerSettings=["tools", "toolChoice", "parallelToolCalls"],
            ),
        )
    assert response.status_code == 200, response.text
    assert HEADER not in response.headers
    sent = _sent(chat)
    assert sent["tool_choice"] == "required" and sent["parallel_tool_calls"] is False


@respx.mock
def test_a_model_that_names_no_tools_is_still_the_providers_no(tmp_path: Path) -> None:
    with _driver(tmp_path) as (client, chat):
        response = client.post(
            "/v1/generate", json=_body("vendor/no-tools", callerSettings=["tools"])
        )
    assert response.status_code == 400
    assert response.json()["detail"]["type"].endswith("#tools-unsupported")
    assert not chat.called


@respx.mock
def test_a_setting_the_listing_leaves_out_that_is_not_a_tool_setting_is_still_refused(
    tmp_path: Path,
) -> None:
    """Only the tool companions are named instead of refused: a sampler changes the answer."""
    with _driver(tmp_path) as (client, chat):
        response = client.post(
            "/v1/generate",
            json=_body("vendor/no-parallel", tools=False, topP=0.5, callerSettings=["topP"]),
        )
    assert response.status_code == 400
    assert "top_p" in response.text and not chat.called


# --------------------------------------------------------------------------- #
# a model whose listing names no tools: what rides with tools is dropped from a request that
# carries none, as it is for a backend that refused them in its own words
# --------------------------------------------------------------------------- #


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
def test_a_request_with_no_tools_to_a_model_with_none_is_not_refused_for_tool_settings(
    tmp_path: Path, stream: bool, caplog: pytest.LogCaptureFixture
) -> None:
    """Codex sends `tool_choice` and `parallel_tool_calls` on every request, tools or not, as
    explicit settings. For a model whose listing names no tools both were a 400
    (`this engine cannot honor this explicit setting`), although they say nothing without
    tools: the rule for a model whose tools the backend refused (`test_tools_reask.py`), for
    any model whose tools are off, whether learned or listed. Dropped, once per field."""
    caplog.set_level(logging.WARNING, logger=ENGINE_LOG)
    plain = _body(
        "vendor/no-tools",
        tools=False,
        toolChoice="auto",
        parallelToolCalls=False,
        callerSettings=["toolChoice", "parallelToolCalls"],
    )
    with _driver(tmp_path) as (client, chat):
        for _ in range(3):
            served = _post(client, stream, plain)
            assert served.status_code == 200, served.text
            assert HEADER not in served.headers, "nothing was sent that was not honoured"
        sent = _sent(chat)
        assert not {"tools", "tool_choice", "parallel_tool_calls"} & sent.keys()

        # With tools it is still the door's refusal, and nothing is sent.
        calls = len(chat.calls)
        carried = _post(
            client, stream, _body("vendor/no-tools", toolChoice="auto", callerSettings=["tools"])
        )
        assert carried.status_code == 400
        assert carried.json()["detail"]["type"].endswith("#tools-unsupported")
        assert len(chat.calls) == calls

    left_out = [r.getMessage() for r in caplog.records if "omitting `" in r.getMessage()]
    assert sum("`tool_choice`" in m for m in left_out) == 1, left_out
    assert sum("`parallel_tool_calls`" in m for m in left_out) == 1, left_out
    # Its cause, in words: what OpenRouter's listing for this model says, not a guess.
    assert all("OpenRouter's own listing" in m and "names no tools" in m for m in left_out)
    assert all("carries no tools" in m for m in left_out)


@respx.mock
def test_a_model_that_takes_tools_is_sent_what_rides_with_them_when_it_carries_none(
    tmp_path: Path,
) -> None:
    """The other half: only a model whose tools are off has them dropped. One that takes tools
    is sent what the caller sent (its provider answers for itself)."""
    with _driver(tmp_path) as (client, chat):
        response = client.post(
            "/v1/generate",
            json=_body(
                "vendor/full",
                tools=False,
                toolChoice="auto",
                parallelToolCalls=False,
                callerSettings=["toolChoice", "parallelToolCalls"],
            ),
        )
    assert response.status_code == 200, response.text
    sent = _sent(chat)
    assert sent["tool_choice"] == "auto" and sent["parallel_tool_calls"] is False


# --------------------------------------------------------------------------- #
# the engine's own hook, called directly
# --------------------------------------------------------------------------- #


@respx.mock
def test_the_hook_names_nothing_for_a_model_that_is_not_served_or_not_named(
    tmp_path: Path,
) -> None:
    """The route refuses those before anything is sent (404, and a 400 for no `model`), so
    there is nothing ignored, and the hook must not raise on the way."""
    with _driver(tmp_path) as (client, _):
        engine = client.app.state.adapter  # type: ignore[attr-defined]

        def ask(model: str | None) -> GenerateRequest:
            return GenerateRequest(
                model=model,
                messages=[Message(role=Role.user, content="hi")],
                parallelToolCalls=False,
                callerSettings=["parallelToolCalls"],
            )

        assert engine.ignored_settings(ask("vendor/no-parallel")) == ["parallelToolCalls"]
        assert engine.ignored_settings(ask("vendor/full")) == []
        assert engine.ignored_settings(ask("vendor/not-listed")) == []
        assert engine.ignored_settings(ask(None)) == []


@respx.mock
def test_a_direct_caller_is_named_only_what_it_lists_in_caller_settings(tmp_path: Path) -> None:
    """The header names what the caller SENT, and what the caller sent is what it lists in
    `callerSettings`: a request that lists nothing is named nothing, so a caller that is not the
    gateway (which always lists them) must list what it wants named."""
    with _driver(tmp_path) as (client, _):
        unlisted = client.post(
            "/v1/generate",
            json=_body("vendor/neither", toolChoice="required", parallelToolCalls=False),
        )
        assert unlisted.status_code == 200, unlisted.text
        assert HEADER not in unlisted.headers

        listed = client.post(
            "/v1/generate",
            json=_body(
                "vendor/neither",
                toolChoice="required",
                parallelToolCalls=False,
                callerSettings=["toolChoice", "parallelToolCalls"],
            ),
        )
        assert listed.headers[HEADER] == "toolChoice, parallelToolCalls"
