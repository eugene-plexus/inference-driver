"""A listing that says nothing is not a no (Troy, 2026-10-12, round two of "tools are assumed").

`docs/design/tools-assumed.md`: a chat model is taken to use tools unless there is evidence
otherwise, and evidence is the backend's own word, a provider's listing that SAYS so, or this
driver's own inability to carry tools. A listing that is silent is none of those. Round one left
three places where an absent field read as "no tools":

* an OpenRouter entry with no `supported_parameters` list at all;
* Gemini's Gemma models, `toolCalling` being `not gemma`, a name prefix nobody measured;
* the route's own door, which read a catalogue entry with no `toolCalling` (a list saved by an
  older driver) as false.

A listing that SAYS something and leaves tools out stays a no where the provider means it as one:
OpenRouter's `supported_parameters` without `tools` (it lists what each model accepts) and Ollama's
capabilities without `tools`. **LM Studio's is not one** (round three): its `capabilities` names
what a model is trained for, which is not the server refusing tools, and a list that named other
things and not `tool_use` read as no tools, a refusal nobody had seen.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx
from fastapi import HTTPException
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver._generated.models import (
    BackendKind,
    Capabilities,
    DriverModel,
    FunctionDefinition,
    GenerateRequest,
    Message,
    Role,
    Tool,
)
from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.engines import _gemini_wire as gemini
from eugene_plexus_inference_driver.engines._catalogue import (
    EngineDefaults,
    from_lmstudio,
    from_openrouter,
    ollama_entry,
)
from eugene_plexus_inference_driver.routes.generate import _refuse_unsupported_tools
from eugene_plexus_inference_driver.settings import Settings
from tests.test_accounts import _completion, _or_model, _wait_for_catalogue, _write

OPENROUTER = "https://openrouter.ai/api"
TOOL_SETTINGS = {"tools", "toolChoice", "parallelToolCalls"}
WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def _entry(**overrides: Any) -> dict[str, Any]:
    """An OpenRouter entry, and `overrides` that may remove the list altogether."""
    entry = _or_model("vendor/model", **overrides)
    for key, value in overrides.items():
        if value is ...:
            del entry[key]
    return entry


def _only(entry: dict[str, Any]) -> DriverModel:
    (model,) = from_openrouter({"data": [entry]})
    return model


# --------------------------------------------------------------------------- #
# OpenRouter: the key absent is not the key without `tools`
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "listing",
    [..., None, [], "tools", {"tools": True}],
    ids=["key-absent", "null", "empty", "a-string", "an-object"],
)
def test_an_openrouter_chat_model_whose_listing_names_no_parameters_takes_tools(
    listing: Any,
) -> None:
    model = _only(_entry(supported_parameters=listing))
    caps = model.capabilities
    assert caps is not None
    assert model.surfaces == ["chat"]
    assert caps.toolCalling is True
    # Offered, so that a `true` here is one an explicit `tools` is not then refused against.
    assert set(caps.supportedSettings or []) >= TOOL_SETTINGS
    assert "maxTokens" in (caps.supportedSettings or [])


@pytest.mark.parametrize(
    "listing",
    [["max_tokens", "temperature"], ["tools_unrelated", "tool_choice"]],
    ids=["others", "tool-choice-only"],
)
def test_an_openrouter_list_that_leaves_tools_out_is_openrouters_no(listing: list[str]) -> None:
    caps = _only(_entry(supported_parameters=listing)).capabilities
    assert caps is not None
    assert caps.toolCalling is False
    assert "tools" not in (caps.supportedSettings or [])


def test_an_openrouter_list_that_names_tools_is_a_yes() -> None:
    caps = _only(_entry(supported_parameters=["max_tokens", "tools"])).capabilities
    assert caps is not None and caps.toolCalling is True
    assert "tools" in (caps.supportedSettings or [])


def test_a_model_that_does_not_chat_is_not_a_tool_model_listing_or_not() -> None:
    speech = {"input_modalities": ["text"], "output_modalities": ["speech"]}
    for listing in (..., []):
        caps = _only(_entry(architecture=speech, supported_parameters=listing)).capabilities
        assert caps is not None
        assert not caps.toolCalling
        assert not set(caps.supportedSettings or []) & TOOL_SETTINGS


@respx.mock
def test_through_the_app_a_silent_listing_is_carried_and_a_listed_no_is_the_providers_word(
    tmp_path: Path,
) -> None:
    """The whole path for an account: the listing, `/v1/info`, the door, the wire."""
    silent = _or_model("vendor/silent")
    del silent["supported_parameters"]  # the key is simply not there
    listing = {
        "data": [
            silent,
            _or_model("vendor/says-no", supported_parameters=["max_tokens"]),
            _or_model("vendor/says-yes", supported_parameters=["max_tokens", "tools"]),
        ]
    }
    respx.get(f"{OPENROUTER}/v1/models/user").mock(return_value=httpx.Response(200, json=listing))
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.get(f"{OPENROUTER}/v1/videos/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    chat = respx.post(f"{OPENROUTER}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion("vendor/silent"))
    )
    config = tmp_path / "openrouter.yaml"
    _write(config, {"provider": "openrouter", "apiKey": "sk-or-test"})

    def ask(model: str) -> dict[str, Any]:
        return {
            "model": model,
            "messages": [{"role": "user", "content": "Weather in Paris?"}],
            "tools": [WEATHER],
            "toolChoice": "auto",
            "callerSettings": ["tools", "toolChoice"],
        }

    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        info = _wait_for_catalogue(client)
        by_id = {m["id"]: m["capabilities"] for m in info["models"]}
        assert by_id["vendor/silent"]["toolCalling"] is True
        assert by_id["vendor/says-yes"]["toolCalling"] is True
        assert by_id["vendor/says-no"]["toolCalling"] is False

        silent = client.post("/v1/generate", json=ask("vendor/silent"))
        assert silent.status_code == 200, silent.text
        sent = json.loads(chat.calls.last.request.content)
        assert [t["function"]["name"] for t in sent["tools"]] == ["get_weather"]

        no = client.post("/v1/generate", json=ask("vendor/says-no"))
        assert no.status_code == 400
        assert no.json()["detail"]["type"].endswith("#tools-unsupported")
        assert chat.call_count == 1, "the listed no was refused at the door, nothing was sent"


# --------------------------------------------------------------------------- #
# Gemini: a name prefix is not a measurement
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "model_id", ["gemma-3-27b-it", "gemma-4-e4b-it", "gemini-3.5-flash", "gemini-3.1-flash-lite"]
)
def test_every_gemini_chat_model_takes_tools_gemma_included(model_id: str) -> None:
    model = gemini.model_from_listing(
        {
            "name": f"models/{model_id}",
            "inputTokenLimit": 8192,
            "supportedGenerationMethods": ["generateContent", "streamGenerateContent"],
        }
    )
    assert model is not None and model.capabilities is not None
    assert "chat" in model.surfaces
    assert model.capabilities.toolCalling is True
    assert {"tools", "toolChoice"} <= set(model.capabilities.supportedSettings or [])


def test_gemma_still_cannot_read_what_it_was_never_listed_to_read() -> None:
    """Only tools were assumed. Attachments are routed only where something confirms them."""
    model = gemini.model_from_listing(
        {
            "name": "models/gemma-3-27b-it",
            "supportedGenerationMethods": ["generateContent"],
        }
    )
    assert model is not None and model.capabilities is not None
    caps = model.capabilities
    assert (caps.imageInput, caps.audioInput, caps.fileInput) == (False, False, False)


# --------------------------------------------------------------------------- #
# Ollama reads its capabilities; LM Studio's say nothing of tools being refused
# --------------------------------------------------------------------------- #

DEFAULTS = EngineDefaults(
    supported_settings=["maxTokens", "tools"], tool_calling=True, streaming=True
)


@pytest.mark.parametrize(
    ("show", "tools"),
    [
        ({"capabilities": ["completion", "tools"]}, True),
        # A list that names what the model can do and leaves tools out is Ollama's own word.
        ({"capabilities": ["completion"]}, False),
        (None, True),  # `/api/show` failed: nothing was said
        ({}, True),  # an older Ollama: no capabilities at all
        ({"capabilities": "completion"}, True),  # not a list: not a statement
    ],
    ids=["lists-tools", "lists-others", "show-failed", "no-capabilities", "not-a-list"],
)
def test_ollama_reads_its_capabilities_only_when_it_gave_a_list(
    show: dict[str, Any] | None, tools: bool
) -> None:
    caps = ollama_entry("m:1b", show, DEFAULTS).capabilities
    assert caps is not None and caps.toolCalling is tools


@pytest.mark.parametrize(
    "capabilities",
    [
        ["tool_use"],
        ["vision", "tool_use"],
        [],
        None,
        "tool_use",
        # Names something and leaves `tool_use` out: it was read as no tools, and is not a
        # refusal of anything. Only the backend refusing tools, in its own words, is evidence.
        ["vision"],
        ["vision", "reasoning"],
        ["completion"],
    ],
    ids=[
        "tool-use",
        "vision-and-tool-use",
        "empty",
        "absent",
        "not-a-list",
        "vision-only",
        "vision-and-reasoning",
        "completion-only",
    ],
)
@pytest.mark.parametrize("kind", ["llm", "vlm"])
def test_lm_studio_takes_tools_whatever_its_capability_list_names(
    capabilities: Any, kind: str
) -> None:
    entry: dict[str, Any] = {"id": "qwen", "type": kind}
    if capabilities is not None:
        entry["capabilities"] = capabilities
    (model,) = from_lmstudio({"data": [entry]}, DEFAULTS)
    caps = model.capabilities
    assert caps is not None and caps.toolCalling is True
    assert "tools" in (caps.supportedSettings or [])


def test_lm_studio_still_reads_what_its_listing_does_say() -> None:
    """Only tools moved: the type, the window and the vision pairing are still read from it."""
    entries = [
        {"id": "chat", "type": "llm", "capabilities": ["tool_use"], "loaded_context_length": 8192},
        {"id": "eyes", "type": "vlm", "capabilities": ["vision"]},
        {"id": "vectors", "type": "embeddings"},
    ]
    chat, eyes, vectors = from_lmstudio({"data": entries}, DEFAULTS)
    assert chat.capabilities is not None and chat.capabilities.maxContextTokens == 8192
    assert eyes.capabilities is not None and eyes.capabilities.imageInput is True
    assert eyes.inputModalities == ["text", "image"]
    assert vectors.surfaces == ["embeddings"]
    assert vectors.capabilities is not None and not vectors.capabilities.toolCalling


@respx.mock
def test_through_the_app_an_lm_studio_model_that_names_no_tool_use_gets_its_tools(
    tmp_path: Path,
) -> None:
    """The whole path for a driver the agent runs for LM Studio: the listing, `/v1/info`, the
    door and the wire. The model's `capabilities` name vision and not `tool_use`; the request
    carrying tools reaches LM Studio, which answers for itself (it was a 400 at the door)."""
    lm = "http://127.0.0.1:1234"
    respx.get(f"{lm}/api/v0/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"id": "qwen2.5-vl-7b", "type": "vlm", "capabilities": ["vision"]},
                    {"id": "qwen3-coder-30b", "type": "llm", "capabilities": ["tool_use"]},
                ]
            },
        )
    )
    chat = respx.post(f"{lm}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion("qwen2.5-vl-7b"))
    )
    config = tmp_path / "lmstudio.yaml"
    _write(config, {"provider": "lmstudio_local"})
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        info = _wait_for_catalogue(client)
        by_id = {m["id"]: m["capabilities"] for m in info["models"]}
        assert by_id["qwen2.5-vl-7b"]["toolCalling"] is True
        assert "tools" in by_id["qwen2.5-vl-7b"]["supportedSettings"]
        response = client.post(
            "/v1/generate",
            json={
                "model": "qwen2.5-vl-7b",
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": [WEATHER],
                "callerSettings": ["tools"],
            },
        )
    assert response.status_code == 200, response.text
    sent = json.loads(chat.calls.last.request.content)
    assert [t["function"]["name"] for t in sent["tools"]] == ["get_weather"]


# --------------------------------------------------------------------------- #
# the door: only an explicit false refuses
# --------------------------------------------------------------------------- #


def _request() -> GenerateRequest:
    return GenerateRequest(
        messages=[Message(role=Role.user, content="hi")],
        tools=[Tool(type="function", function=FunctionDefinition(name="t", parameters={}))],
    )


def _engine(*, supports: bool = True, kind: str = "openai_compat_http") -> Any:
    return SimpleNamespace(backend_kind=BackendKind(kind), supports_tool_calling=supports)


def _entry_with(tool_calling: bool | None, *, capabilities: bool = True) -> DriverModel:
    return DriverModel(
        id="m",
        surfaces=["chat"],
        capabilities=Capabilities(toolCalling=tool_calling) if capabilities else None,
    )


@pytest.mark.parametrize(
    "entry",
    [_entry_with(True), _entry_with(None), _entry_with(None, capabilities=False)],
    ids=["true", "no-answer", "no-capabilities"],
)
def test_a_catalogue_entry_that_does_not_say_no_is_not_refused(entry: DriverModel) -> None:
    """A list saved by an older driver has no `toolCalling`: nobody said no, so it is let through
    (the backend answers for itself)."""
    _refuse_unsupported_tools(_engine(), _request(), entry)


def test_an_entry_that_says_false_is_refused_at_the_door_in_the_drivers_words() -> None:
    with pytest.raises(HTTPException) as refused:
        _refuse_unsupported_tools(_engine(), _request(), _entry_with(False))
    assert refused.value.status_code == 400
    detail = refused.value.detail
    assert detail["type"].endswith("#tools-unsupported")
    assert "cannot carry tool definitions" in detail["detail"]


def test_an_engine_with_no_way_to_carry_tools_says_so_as_its_own_limit() -> None:
    """The subscription CLIs: evidence of the third kind, said as Eugene's limit and never as
    the model's."""
    with pytest.raises(HTTPException) as refused:
        _refuse_unsupported_tools(_engine(supports=False, kind="claude_code_cli"), _request(), None)
    assert "claude_code_cli" in refused.value.detail["detail"]
    assert "cannot carry tool definitions" in refused.value.detail["detail"]


def test_a_request_with_no_tools_is_never_refused_for_them() -> None:
    plain = GenerateRequest(messages=[Message(role=Role.user, content="hi")])
    _refuse_unsupported_tools(_engine(supports=False), plain, _entry_with(False))
    _refuse_unsupported_tools(_engine(supports=False), plain, None)
