"""An OpenRouter model's listing leaves out a tool setting: named, not refused (Troy, 2026-10-12).

`docs/design/tools-assumed.md`: a setting a backend does not honour is **sent where it can
be, and named**, never refused, because a refusal is for what cannot be sent at all and a
harness that sends `parallel_tool_calls: false` on every request (Codex does) can then use no
model that is refused it. OpenRouter's listing says per model what each takes, and on its
public list of 2026-10-12 (659 models, 390 chat models naming `tools`) **377 leave out
`parallel_tool_calls`** and 8 leave out `tool_choice`. Until round three an explicit one was
refused for every one of them (`this engine cannot honor this explicit setting`), and the
gateway, reading `supportedSettings`, routed Codex around them all.

What the driver does now, for a model whose listing names `tools` and leaves a companion out:

* the setting is **offered** (`supportedSettings`) and **listed**
  (`capabilities.settingsNotHonoured`: `parallelToolCalls` with `false`, `toolChoice` with
  `required` and `function`), on the served entry only -- never in the list saved to disk;
* it is **not sent**: the provider takes no such parameter, so it would be dropped without a
  word, or, with `require_parameters` (set whenever the caller sent a setting), would by
  OpenRouter's documentation leave it no provider to route to (not measured: no key was used);
* it is **named** on `x-eugene-plexus-ignored-settings`, but only when the caller sent it
  (`callerSettings`) with a value the listing names, as the gateway names it.

A model whose listing names both companions is sent them as asked and has no gap; a model
that names no `tools` is the provider's word on tools and is unchanged; a listing that says
nothing is offered every tool setting and has none.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.engines.openai_compat_http import unlisted_tool_settings
from eugene_plexus_inference_driver.settings import Settings
from tests.test_accounts import _completion, _or_model, _wait_for_catalogue, _write

OPENROUTER = "https://openrouter.ai/api"
HEADER = "x-eugene-plexus-ignored-settings"
CHAT = f"{OPENROUTER}/v1/chat/completions"
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
CHOICE_GAP = {"setting": "toolChoice", "values": ["required", "function"]}
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


@contextlib.contextmanager
def _driver(tmp_path: Path) -> Iterator[tuple[TestClient, respx.Route]]:
    """The real app for an OpenRouter account, once its list has been read, and the route its
    chat requests reach."""
    respx.get(f"{OPENROUTER}/v1/models/user").mock(
        return_value=httpx.Response(200, json=_listing())
    )
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.get(f"{OPENROUTER}/v1/videos/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    chat = respx.post(CHAT).mock(side_effect=_answer)
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
    assert unlisted_tool_settings(["tools"])[0].values == ["required", "function"]


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
        # A measured gap is named, not refused, so it is offered too (the contract says so).
        assert {"tools", "toolChoice", "parallelToolCalls"} <= set(caps[model]["supportedSettings"])
        assert caps[model]["toolCalling"] is True
    # Nothing to name: it lists both, it lists no tools, it says nothing, it does not chat.
    for model in ("vendor/full", "vendor/no-tools", "vendor/silent", "vendor/embed"):
        assert "settingsNotHonoured" not in caps[model], model
    # And the other models' settings are what their listings said.
    assert "toolChoice" not in caps["vendor/no-tools"]["supportedSettings"]
    assert "parallelToolCalls" not in caps["vendor/no-tools"]["supportedSettings"]
    assert caps["vendor/no-tools"]["toolCalling"] is False

    # The provider's own word is what is saved. What this engine adds is laid over it
    # when it is served, and a list read again from OpenRouter must not carry it.
    saved = json.loads((tmp_path / "openrouter.catalogue.json").read_text("utf-8"))
    for model in saved["models"]:
        assert "settingsNotHonoured" not in model["capabilities"], model["id"]
    stored = {m["id"]: m["capabilities"]["supportedSettings"] for m in saved["models"]}
    assert "parallelToolCalls" not in stored["vendor/no-parallel"]
    assert "toolChoice" not in stored["vendor/no-choice"]


# --------------------------------------------------------------------------- #
# parallel_tool_calls: not refused, not sent, named
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
    assert "parallel_tool_calls" not in sent, "the provider takes no such parameter"
    assert [t["function"]["name"] for t in sent["tools"]] == ["get_weather"]
    assert sent["tool_choice"] == "auto", "what the listing names is sent as asked"
    # Why it must not be sent: a caller's setting makes OpenRouter strict about the rest.
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
    assert "parallel_tool_calls" not in _sent(chat), "never sent where the listing lacks it"


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
# tool_choice: the other companion, the same way
# --------------------------------------------------------------------------- #


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("choice", "header"),
    [("required", "toolChoice"), (NAMED, "toolChoice"), ("auto", None)],
    ids=["required", "named", "auto"],
)
def test_a_tool_choice_the_listing_leaves_out_is_not_sent_and_named_unless_it_is_auto(
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
    assert "tool_choice" not in sent, "the provider takes no such parameter"
    assert [t["function"]["name"] for t in sent["tools"]] == ["get_weather"]


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
def test_tool_choice_none_is_honoured_by_sending_no_tools_and_names_nothing(
    tmp_path: Path, stream: bool
) -> None:
    """A provider that never reads `tool_choice` would answer `none` as `auto`, so the tools
    are not offered at all (as for Strata and Ollama), and what rode with them goes too."""
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
    assert HEADER not in response.headers, "it was honoured, so nothing was ignored"
    sent = _sent(chat)
    assert not {"tools", "tool_choice", "parallel_tool_calls"} & sent.keys()


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
# the engine's own hook, called directly
# --------------------------------------------------------------------------- #


@respx.mock
def test_the_hook_names_nothing_for_a_model_that_is_not_served_or_not_named(
    tmp_path: Path,
) -> None:
    """The route refuses those before anything is sent (404, and a 400 for no `model`), so
    there is nothing ignored, and the hook must not raise on the way."""
    from eugene_plexus_inference_driver._generated.models import GenerateRequest

    with _driver(tmp_path) as (client, _):
        engine = client.app.state.adapter  # type: ignore[attr-defined]

        def ask(model: str | None) -> GenerateRequest:
            return GenerateRequest.model_validate(
                {
                    "model": model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "parallelToolCalls": False,
                    "callerSettings": ["parallelToolCalls"],
                }
            )

        assert engine.ignored_settings(ask("vendor/no-parallel")) == ["parallelToolCalls"]
        assert engine.ignored_settings(ask("vendor/full")) == []
        assert engine.ignored_settings(ask("vendor/not-listed")) == []
        assert engine.ignored_settings(ask(None)) == []
