"""One rule for every chat model whose tools are off (Troy, 2026-10-12, round five).

`docs/design/tools-assumed.md`: tools are assumed, and off only on evidence. Round four dropped
`tool_choice` and `parallel_tool_calls` from a request that carries no tools, for a model whose
tools the backend refused, whose provider's listing names none, or whose endpoint serves it
without them -- and left the gateway no way to know it: those two settings were missing from the
model's `supportedSettings` for most of those causes (an OpenRouter listing that names no tools,
OpenAI's Responses-only models), and the subscription CLIs listed no settings at all and refused
both outright. The gateway (round five) admits by `supportedSettings`, so the listing is what makes
Codex, which sends both on every request tools or not, work with such a model. One rule, for every
engine family, whatever turned the tools off:

* the model's `supportedSettings` lists `toolChoice` and `parallelToolCalls` (and not `tools`);
* a request that carries no tools (`None` or `[]`) has them dropped, once per field in the log,
  with the cause in words;
* a request that carries tools is still refused, and says why.

The families here are the ones that can have a chat model with its tools off: a backend that
refused them in its own words (learned), an OpenRouter listing that names none, an Ollama model
whose capabilities list none, OpenAI's Responses-only models (a single model and an account),
and the two subscription CLIs, whose CLI is never run: `run_cli` and `stream_cli_lines` are
replaced, and the binary is a name that does not exist.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi import HTTPException
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver._generated.models import (
    BackendKind,
    DriverModel,
    FunctionDefinition,
    GenerateRequest,
    Message,
    Role,
    Tool,
)
from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.engines import claude_code_cli, codex_cli
from eugene_plexus_inference_driver.engines._subprocess import CliError, CliResult
from eugene_plexus_inference_driver.engines.claude_code_cli import ClaudeCodeCliEngine
from eugene_plexus_inference_driver.engines.codex_cli import CodexCliEngine
from eugene_plexus_inference_driver.engines.openai_compat_http import (
    OPENAI_FIXED_TEMPERATURE_PATTERN,
    OpenAiCompatibleHttpEngine,
    _Target,
)
from eugene_plexus_inference_driver.routes.generate import _refuse_unsupported_tools
from eugene_plexus_inference_driver.settings import Settings
from tests.test_accounts import _completion, _or_model, _wait_for_catalogue, _write
from tests.test_adapters import CLAUDE_OK_ENVELOPE, CODEX_OK_STREAM, _patch_run_cli
from tests.test_tools_evidence import WORDS

OPENROUTER = "https://openrouter.ai/api"
OLLAMA = "http://ollama.local"
OPENAI = "https://api.openai.com"
LLAMA = "http://llama.local"
ENGINE_LOG = "eugene_plexus_inference_driver.engines.openai_compat_http"
BASE_LOG = "eugene_plexus_inference_driver.engines.base"
NO_SUCH_CLI = "r5-no-such-executable"
COMPANIONS = {"toolChoice", "parallelToolCalls"}
WEATHER = Tool(
    type="function",
    function=FunctionDefinition(
        name="get_weather",
        description="Current weather for a city.",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
    ),
)
OK = {
    "model": "m",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}
#: What each setting is sent as on the wire, which is what a refusal names.
WIRE = {"toolChoice": "tool_choice", "parallelToolCalls": "parallel_tool_calls"}
VALUE: dict[str, Any] = {"toolChoice": "auto", "parallelToolCalls": False}


def _ask(model: str | None, *, tools: list[Tool] | None = None, **fields: Any) -> GenerateRequest:
    return GenerateRequest(
        model=model,
        messages=[Message(role=Role.user, content="What is the weather in Paris?")],
        tools=tools,
        **fields,
    )


def _sent(route: respx.Route) -> dict[str, Any]:
    return json.loads(route.calls[-1].request.content)


def _said(caplog: pytest.LogCaptureFixture, needle: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if needle in r.getMessage()]


def _mock_openrouter(listing: dict[str, Any], model: str) -> respx.Route:
    """An OpenRouter that lists these models and answers chat for `model`; the route is chat's."""
    respx.get(f"{OPENROUTER}/v1/models/user").mock(return_value=httpx.Response(200, json=listing))
    respx.get(f"{OPENROUTER}/v1/images/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    respx.get(f"{OPENROUTER}/v1/videos/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    return respx.post(f"{OPENROUTER}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion(model))
    )


# --------------------------------------------------------------------------- #
# the HTTP engine's families
# --------------------------------------------------------------------------- #


@dataclass
class Family:
    """One way a chat model's tools come to be off, as an engine serving it."""

    engine: OpenAiCompatibleHttpEngine
    #: What a request names: None for a single-model driver.
    model: str | None
    #: Where the backend's chat route is mocked, to see what is sent.
    chat: respx.Route
    #: Part of what the engine says of why, which is the observed cause.
    cause: str
    #: Whether the entry also offers `tools`. Ollama's does: its entry is built from the engine's
    #: own defaults, which are older than this rule and not what it is about.
    offers_tools: bool = False

    def entry(self) -> DriverModel | None:
        if self.model is None:
            return None
        assert self.engine.catalogue is not None
        return self.engine.catalogue.find(self.model)

    def listed(self) -> set[str]:
        entry = self.entry()
        if entry is None:
            return set(self.engine.supported_settings)
        assert entry.capabilities is not None
        return set(entry.capabilities.supportedSettings or [])

    def tools_are_off(self) -> bool:
        entry = self.entry()
        if entry is None:
            return self.engine.supports_tool_calling is False
        assert entry.capabilities is not None
        return entry.capabilities.toolCalling is False


async def _learned() -> Family:
    """llama-server started with `--no-jinja`: it refused a request's tools, in its own words."""
    chat = respx.post(f"{LLAMA}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OK)
    )
    engine = OpenAiCompatibleHttpEngine(
        base_url=LLAMA,
        model_id="qwen",
        backend_kind=BackendKind.openai_compat_http,
        auth_required=False,
    )
    engine._learn_tools_refused(engine.resolve_model(None), WORDS, 500)
    return Family(engine, None, chat, "refused tools in its own words")


def _openrouter_engine(**more: Any) -> OpenAiCompatibleHttpEngine:
    return OpenAiCompatibleHttpEngine(
        api_key="sk-or-test",
        base_url=OPENROUTER,
        model_id=None,
        backend_kind=BackendKind.openai_compat_http,
        dialect="openrouter",
        catalogue_source="openrouter",
        filter_models=False,
        require_parameters=True,
        **more,
    )


async def _openrouter() -> Family:
    """An account whose listing for the model names no tools (`supported_parameters`)."""
    listing = {"data": [_or_model("vendor/no-tools", supported_parameters=["max_tokens"])]}
    chat = _mock_openrouter(listing, "vendor/no-tools")
    engine = _openrouter_engine()
    assert engine.catalogue is not None and await engine.catalogue.refresh()
    return Family(engine, "vendor/no-tools", chat, "OpenRouter's own listing for this model")


async def _ollama() -> Family:
    """An Ollama model whose `/api/show` lists no `tools` among its capabilities."""
    respx.get(f"{OLLAMA}/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "gemma3:4b"}]})
    )
    respx.post(f"{OLLAMA}/api/show").mock(
        return_value=httpx.Response(200, json={"capabilities": ["completion"]})
    )
    chat = respx.post(f"{OLLAMA}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OK)
    )
    engine = OpenAiCompatibleHttpEngine(
        base_url=OLLAMA,
        model_id=None,
        backend_kind=BackendKind.openai_compat_http,
        auth_required=False,
        catalogue_source="ollama",
    )
    assert engine.catalogue is not None and await engine.catalogue.refresh()
    return Family(engine, "gemma3:4b", chat, "Ollama's own capabilities", offers_tools=True)


def _openai(model_id: str | None) -> OpenAiCompatibleHttpEngine:
    return OpenAiCompatibleHttpEngine(
        api_key="sk-test",
        base_url=OPENAI,
        model_id=model_id,
        fixed_temperature_pattern=OPENAI_FIXED_TEMPERATURE_PATTERN,
        backend_kind=BackendKind.openai_api,
    )


async def _responses_only() -> Family:
    """OpenAI's own Chat Completions serves `gpt-6-astra` without tools (Responses only)."""
    chat = respx.post(f"{OPENAI}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OK)
    )
    return Family(_openai("gpt-6-astra"), None, chat, "Responses API only")


async def _responses_only_account() -> Family:
    respx.get(f"{OPENAI}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "gpt-6-astra"}]})
    )
    chat = respx.post(f"{OPENAI}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OK)
    )
    engine = _openai(None)
    assert engine.catalogue is not None and await engine.catalogue.refresh()
    return Family(engine, "gpt-6-astra", chat, "Responses API only")


FAMILIES: dict[str, Callable[[], Awaitable[Family]]] = {
    "learned": _learned,
    "openrouter-listing": _openrouter,
    "ollama-capabilities": _ollama,
    "responses-only": _responses_only,
    "responses-only-account": _responses_only_account,
}


# --------------------------------------------------------------------------- #
# the listing: what the gateway reads to admit a request
# --------------------------------------------------------------------------- #


@respx.mock
@pytest.mark.anyio
@pytest.mark.parametrize("name", list(FAMILIES))
async def test_a_model_whose_tools_are_off_lists_what_rides_with_them(name: str) -> None:
    """The gateway admits by `supportedSettings` (round five). Before this, an OpenRouter listing
    that names no tools, and OpenAI's Responses-only models, left both settings out, so a request
    carrying only `tool_choice` or `parallel_tool_calls` (Codex's, tools or not) was turned away
    from a model that drops them and answers."""
    family = await FAMILIES[name]()
    try:
        assert family.tools_are_off()
        assert family.listed() >= COMPANIONS
        if not family.offers_tools:
            assert "tools" not in family.listed()
    finally:
        await family.engine.aclose()


@respx.mock
def test_an_openrouter_account_lays_the_companions_over_a_copy_and_saves_the_providers_word(
    tmp_path: Path,
) -> None:
    """What this engine adds when it serves a model is not what OpenRouter listed, so it is never
    written into the stored entry or the list saved to disk, as for a learned refusal."""
    listing = {"data": [_or_model("vendor/no-tools", supported_parameters=["max_tokens"])]}
    _mock_openrouter(listing, "vendor/no-tools")
    config = tmp_path / "openrouter.yaml"
    _write(config, {"provider": "openrouter", "apiKey": "sk-or-test"})
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        info = _wait_for_catalogue(client)
        (model,) = [m for m in info["models"] if m["id"] == "vendor/no-tools"]
        assert model["capabilities"]["toolCalling"] is False
        assert set(model["capabilities"]["supportedSettings"]) == {"maxTokens", *COMPANIONS}
        stored = client.app.state.adapter.catalogue._by_id["vendor/no-tools"]  # type: ignore[attr-defined]
        assert stored.capabilities.supportedSettings == ["maxTokens"]
    saved = json.loads((tmp_path / "openrouter.catalogue.json").read_text("utf-8"))
    assert saved["models"][0]["capabilities"]["supportedSettings"] == ["maxTokens"]


@respx.mock
@pytest.mark.anyio
async def test_only_a_chat_model_is_offered_what_rides_with_tools() -> None:
    """The rule is for a CHAT model whose tools are off. A model that answers embeddings has no
    tools to be off, and its listing names none (`toolCalling` false all the same): what rides
    with tools means nothing there, and is not offered."""
    listing = {
        "data": [
            _or_model("vendor/no-tools", supported_parameters=["max_tokens"]),
            _or_model(
                "vendor/embeds",
                architecture={"input_modalities": ["text"], "output_modalities": ["embeddings"]},
                supported_parameters=[],
            ),
        ]
    }
    _mock_openrouter(listing, "vendor/no-tools")
    engine = _openrouter_engine()
    try:
        assert engine.catalogue is not None and await engine.catalogue.refresh()
        embeds = engine.catalogue.find("vendor/embeds")
        assert embeds is not None and embeds.capabilities is not None
        assert embeds.surfaces == ["embeddings"]
        assert embeds.capabilities.toolCalling is False, "the premise: its listing names no tools"
        assert not COMPANIONS & set(embeds.capabilities.supportedSettings or [])
        chat = engine.catalogue.find("vendor/no-tools")
        assert chat is not None and chat.capabilities is not None
        assert set(chat.capabilities.supportedSettings or []) >= COMPANIONS
    finally:
        await engine.aclose()


# --------------------------------------------------------------------------- #
# no tools: dropped, once per field, with the cause in words
# --------------------------------------------------------------------------- #


@respx.mock
@pytest.mark.anyio
@pytest.mark.parametrize("tools", [None, []], ids=["no-tools", "empty-tools"])
@pytest.mark.parametrize("name", list(FAMILIES))
async def test_a_request_with_no_tools_is_served_without_what_rides_with_them(
    name: str, tools: list[Tool] | None, caplog: pytest.LogCaptureFixture
) -> None:
    """`tools: []` is no tools: Codex sends it, and an empty list is the one thing a check on
    `is None` takes for tools."""
    caplog.set_level(logging.WARNING, logger=ENGINE_LOG)
    family = await FAMILIES[name]()
    try:
        for _ in range(3):
            await family.engine.generate(
                _ask(
                    family.model,
                    tools=tools,
                    toolChoice="auto",
                    parallelToolCalls=False,
                    callerSettings=["toolChoice", "parallelToolCalls"],
                )
            )
        assert len(family.chat.calls) == 3
        assert not {"tools", "tool_choice", "parallel_tool_calls"} & _sent(family.chat).keys()
    finally:
        await family.engine.aclose()
    left_out = _said(caplog, "omitting `")
    assert sum("`tool_choice`" in m for m in left_out) == 1, left_out
    assert sum("`parallel_tool_calls`" in m for m in left_out) == 1, left_out
    assert all(family.cause in m and "carries no tools" in m for m in left_out), left_out


@respx.mock
@pytest.mark.anyio
@pytest.mark.parametrize("name", list(FAMILIES))
async def test_a_request_that_carries_tools_is_never_told_it_carried_none(
    name: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The drop is for a request that carries no tools, and says so. One that carries tools is
    refused where the caller listed what rides with them (above); where it listed nothing there
    is nothing to refuse, and what it sent is what is sent -- not dropped, on a line that says it
    carried no tools when it did. (The route's door refuses the tools themselves to such a model,
    before any of this: this is the engine's rule on its own.)"""
    caplog.set_level(logging.WARNING, logger=ENGINE_LOG)
    family = await FAMILIES[name]()
    try:
        await family.engine.generate(
            _ask(family.model, tools=[WEATHER], toolChoice="auto", parallelToolCalls=True)
        )
        sent = _sent(family.chat)
    finally:
        await family.engine.aclose()
    assert sent["tool_choice"] == "auto" and sent["parallel_tool_calls"] is True
    assert [t["function"]["name"] for t in sent["tools"]] == ["get_weather"]
    assert not _said(caplog, "carries no tools")


@respx.mock
def test_through_the_app_an_empty_tools_list_is_no_tools_at_the_door_too(tmp_path: Path) -> None:
    """Codex's shape for a turn with nothing to call: `tools: []` with both settings. The door
    refuses a request that carries tools to a model whose tools are off; an empty list carries
    none, so it is served, with what rides with them left out."""
    listing = {"data": [_or_model("vendor/no-tools", supported_parameters=["max_tokens"])]}
    chat = _mock_openrouter(listing, "vendor/no-tools")
    config = tmp_path / "openrouter.yaml"
    _write(config, {"provider": "openrouter", "apiKey": "sk-or-test"})
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        _wait_for_catalogue(client)
        served = client.post(
            "/v1/generate",
            json={
                "model": "vendor/no-tools",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [],
                "toolChoice": "auto",
                "parallelToolCalls": False,
                "callerSettings": ["toolChoice", "parallelToolCalls"],
            },
        )
    assert served.status_code == 200, served.text
    assert not {"tools", "tool_choice", "parallel_tool_calls"} & _sent(chat).keys()


# --------------------------------------------------------------------------- #
# tools: still refused, and the refusal says why
# --------------------------------------------------------------------------- #


@respx.mock
@pytest.mark.anyio
@pytest.mark.parametrize("field", ["toolChoice", "parallelToolCalls"])
@pytest.mark.parametrize("name", list(FAMILIES))
async def test_with_tools_an_explicit_companion_is_refused_not_dropped(
    name: str, field: str
) -> None:
    """The other half of the rule. Only a request that carries no tools has them dropped: with
    tools, an explicit `tool_choice` is refused before anything is sent, naming its cause. (The
    route refuses the tools themselves before it ever gets here; this is the engine's own check,
    which a caller that is not the route meets.)"""
    family = await FAMILIES[name]()
    try:
        calls = len(family.chat.calls)
        request = _ask(
            family.model, tools=[WEATHER], callerSettings=[field], **{field: VALUE[field]}
        )
        with pytest.raises(CliError) as refused:
            await family.engine.generate(request)
        assert refused.value.upstream_status == 400
        assert WIRE[field] in str(refused.value)
        assert family.cause in str(refused.value), "the cause, in words"
        assert len(family.chat.calls) == calls, "nothing was sent"
    finally:
        await family.engine.aclose()


@respx.mock
@pytest.mark.anyio
@pytest.mark.parametrize("name", list(FAMILIES))
async def test_the_door_names_the_cause_it_observed_and_not_a_generic_limit(name: str) -> None:
    """A request with tools to a model whose tools are off is the driver's 400. For an OpenRouter
    listing that names no tools it said "cannot carry tool definitions", which is what a backend
    that has no way to carry them says; it is the listing's own words that are the cause."""
    family = await FAMILIES[name]()
    try:
        body = _ask(family.model, tools=[WEATHER])
        with pytest.raises(HTTPException) as refused:
            _refuse_unsupported_tools(family.engine, body, family.entry())
    finally:
        await family.engine.aclose()
    assert refused.value.status_code == 400
    problem: Any = refused.value.detail
    assert problem["type"].endswith("#tools-unsupported")
    assert "cannot carry tool definitions" not in problem["detail"]
    assert family.cause in problem["detail"] or WORDS in problem["detail"], problem["detail"]


@respx.mock
def test_through_the_app_an_openrouter_model_with_no_tools_is_refused_in_the_listings_words(
    tmp_path: Path,
) -> None:
    listing = {"data": [_or_model("vendor/no-tools", supported_parameters=["max_tokens"])]}
    chat = _mock_openrouter(listing, "vendor/no-tools")
    config = tmp_path / "openrouter.yaml"
    _write(config, {"provider": "openrouter", "apiKey": "sk-or-test"})
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        _wait_for_catalogue(client)
        refused = client.post(
            "/v1/generate",
            json={
                "model": "vendor/no-tools",
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": [WEATHER.model_dump(mode="json", exclude_none=True)],
                "toolChoice": "auto",
                "callerSettings": ["tools", "toolChoice"],
            },
        )
    assert refused.status_code == 400, refused.text
    detail = refused.json()["detail"]
    assert detail["type"].endswith("#tools-unsupported")
    assert "OpenRouter's own listing for this model names no tools" in detail["detail"]
    assert "cannot carry tool definitions" not in detail["detail"]
    assert not chat.called


# --------------------------------------------------------------------------- #
# an entry that says nothing of tools is not off
# --------------------------------------------------------------------------- #


def _saved_by_an_older_driver(path: Path) -> Path:
    """A model list on disk from before `toolCalling` was written: its entry has settings and no
    `toolCalling` at all, which is nobody's word on tools."""
    path.write_text(
        json.dumps(
            {
                "origin": f"openrouter|{OPENROUTER}",
                "source": "openrouter",
                "refreshedAt": "2026-10-12T00:00:00+00:00",
                "models": [
                    {
                        "id": "vendor/older",
                        "surfaces": ["chat"],
                        "capabilities": {
                            "supportedSettings": [
                                "maxTokens",
                                "tools",
                                "toolChoice",
                                "parallelToolCalls",
                            ]
                        },
                    },
                    {
                        # Nothing of tools in its settings either, and still no `toolCalling`: an
                        # absent answer is not the provider's "no", whatever the settings leave out.
                        "id": "vendor/older-bare",
                        "surfaces": ["chat"],
                        "capabilities": {"supportedSettings": ["maxTokens"]},
                    },
                ],
            }
        ),
        "utf-8",
    )
    return path


@respx.mock
@pytest.mark.anyio
async def test_toolcalling_none_from_an_older_saved_list_is_not_off(tmp_path: Path) -> None:
    """Only an explicit `false` is evidence. A list saved by an older driver has no `toolCalling`:
    the model is served as its settings say, and what rides with tools is SENT to it, with or
    without them, as to any model that takes tools -- not dropped, and not refused."""
    chat = respx.post(f"{OPENROUTER}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion("vendor/older"))
    )
    engine = _openrouter_engine(catalogue_path=_saved_by_an_older_driver(tmp_path / "models.json"))
    try:
        assert engine.catalogue is not None
        entry = engine.catalogue.find("vendor/older")
        assert entry is not None and entry.capabilities is not None
        assert entry.capabilities.toolCalling is None
        assert engine.tools_off_because("vendor/older") is None
        assert entry.capabilities.supportedSettings == [
            "maxTokens",
            "tools",
            "toolChoice",
            "parallelToolCalls",
        ], "nothing laid over what the list said"

        # A list that says nothing of tools in its settings either is served as it says it: not
        # a model whose tools are off, so what rides with tools is not offered beside it.
        bare = engine.catalogue.find("vendor/older-bare")
        assert bare is not None and bare.capabilities is not None
        assert bare.capabilities.toolCalling is None
        assert engine.tools_off_because("vendor/older-bare") is None
        assert bare.capabilities.supportedSettings == ["maxTokens"]

        await engine.generate(
            _ask(
                "vendor/older",
                toolChoice="auto",
                parallelToolCalls=False,
                callerSettings=["toolChoice", "parallelToolCalls"],
            )
        )
        sent = _sent(chat)
        assert sent["tool_choice"] == "auto" and sent["parallel_tool_calls"] is False

        await engine.generate(
            _ask(
                "vendor/older",
                tools=[WEATHER],
                toolChoice="auto",
                callerSettings=["tools", "toolChoice"],
            )
        )
        sent = _sent(chat)
        assert sent["tool_choice"] == "auto"
        assert [t["function"]["name"] for t in sent["tools"]] == ["get_weather"]
        # And the route's own door lets the tools through to it.
        _refuse_unsupported_tools(engine, _ask("vendor/older", tools=[WEATHER]), entry)
    finally:
        await engine.aclose()


@respx.mock
@pytest.mark.anyio
async def test_what_is_dropped_does_not_depend_on_the_entry_having_been_served() -> None:
    """The engine's rule is in `_payload_for`, not in the list it was handed: an entry that
    leaves the settings out (a raw listing, as OpenRouter wrote it) is not refused them on a
    request that carries no tools, and the rule holds however the entry reached the engine."""
    family = await _openrouter()
    try:
        raw = DriverModel.model_validate(
            {
                "id": "vendor/no-tools",
                "surfaces": ["chat"],
                "capabilities": {"supportedSettings": ["maxTokens"], "toolCalling": False},
            }
        )
        target = _Target(
            id="vendor/no-tools", upstream="vendor/no-tools", temperature_fixed=False, entry=raw
        )
        request = _ask(
            "vendor/no-tools",
            toolChoice="auto",
            parallelToolCalls=False,
            callerSettings=["toolChoice", "parallelToolCalls"],
        )
        payload = family.engine._payload_for(request, target)
        assert not {"tools", "tool_choice", "parallel_tool_calls"} & payload.keys()
    finally:
        await family.engine.aclose()


# --------------------------------------------------------------------------- #
# the subscription CLIs: the same rule, and their CLI is never run
# --------------------------------------------------------------------------- #

#: provider key, the config key that says where its CLI is, the engine, and its module.
CLI_ENGINES = [
    pytest.param(
        "claude_subscription",
        "claudeCodeCliPath",
        ClaudeCodeCliEngine,
        claude_code_cli,
        id="claude_code_cli",
    ),
    pytest.param("chatgpt_subscription", "codexCliPath", CodexCliEngine, codex_cli, id="codex_cli"),
]


def _cli(engine_type: Any) -> Any:
    return engine_type(binary_path=NO_SUCH_CLI, model_id="a-model")


def _fake_cli(monkeypatch: pytest.MonkeyPatch, engine_type: Any, module: Any) -> dict[str, Any]:
    """Both ways the engine could start its CLI, replaced by something that records the call and
    answers like a good CLI, so that a request that is NOT refused is served and a request that is
    refused is seen not to have started anything."""
    claude = engine_type is ClaudeCodeCliEngine
    answer = json.dumps(CLAUDE_OK_ENVELOPE).encode() if claude else CODEX_OK_STREAM
    captured = _patch_run_cli(
        monkeypatch,
        lambda argv: CliResult(stdout=answer, stderr=b"", returncode=0, elapsed_ms=1),
    )
    lines = [json.dumps(CLAUDE_OK_ENVELOPE)] if claude else CODEX_OK_STREAM.decode().splitlines()

    async def stream(argv: list[str], **kwargs: Any) -> Any:
        captured["argv"] = argv
        for line in lines:
            yield line

    monkeypatch.setattr(module, "stream_cli_lines", stream)
    return captured


@pytest.mark.parametrize(("provider", "path_key", "engine_type", "module"), CLI_ENGINES)
def test_a_cli_engine_lists_what_rides_with_tools_and_not_tools(
    provider: str, path_key: str, engine_type: Any, module: Any
) -> None:
    engine = _cli(engine_type)
    assert engine.supports_tool_calling is False
    assert set(engine.supported_settings) == COMPANIONS


@pytest.mark.parametrize(("provider", "path_key", "engine_type", "module"), CLI_ENGINES)
def test_info_says_a_subscription_models_tools_are_off_and_lists_what_rides_with_them(
    tmp_path: Path, provider: str, path_key: str, engine_type: Any, module: Any
) -> None:
    """`GET /v1/info` for a subscription driver (nothing is generated, and its CLI is a name
    that does not exist): `toolCalling` false, which is Eugene's limit, and the two settings that
    say nothing without tools are offered, so the gateway admits a request that carries only
    them."""
    config = tmp_path / "cli.yaml"
    _write(config, {"provider": provider, "modelId": "a-model", path_key: NO_SUCH_CLI})
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        info = client.get("/v1/info").json()
    assert info["backend"] == engine_type.backend_kind.value
    (model,) = info["models"]
    assert model["capabilities"]["toolCalling"] is False
    assert model["capabilities"]["supportedSettings"] == ["toolChoice", "parallelToolCalls"]


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True], ids=["generate", "stream"])
@pytest.mark.parametrize("tools", [None, []], ids=["no-tools", "empty-tools"])
@pytest.mark.parametrize(("provider", "path_key", "engine_type", "module"), CLI_ENGINES)
async def test_a_cli_engine_drops_what_rides_with_tools_from_a_request_with_none(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    provider: str,
    path_key: str,
    engine_type: Any,
    module: Any,
    tools: list[Tool] | None,
    stream: bool,
) -> None:
    caplog.set_level(logging.WARNING, logger=BASE_LOG)
    captured = _fake_cli(monkeypatch, engine_type, module)
    engine = _cli(engine_type)
    for _ in range(3):
        request = _ask(
            None,
            tools=tools,
            toolChoice="auto",
            parallelToolCalls=False,
            callerSettings=["toolChoice", "parallelToolCalls"],
        )
        if stream:
            chunks = [c async for c in engine.stream(request)]
            assert chunks[-1].done and chunks[-1].result.content == "PING"
        else:
            assert (await engine.generate(request)).content == "PING"
    assert captured["argv"] is not None, "answered by the stand-in for its CLI"
    left_out = _said(caplog, "was dropped from a request that carries no tools")
    assert sum("`toolChoice`" in m for m in left_out) == 1, left_out
    assert sum("`parallelToolCalls`" in m for m in left_out) == 1, left_out
    # The cause, in words, and as Eugene's limit and never the model's.
    kind = engine_type.backend_kind.value
    assert all(kind in m and "no way to carry tools" in m for m in left_out), left_out


@pytest.mark.anyio
@pytest.mark.parametrize("stream", [False, True], ids=["generate", "stream"])
@pytest.mark.parametrize("field", ["toolChoice", "parallelToolCalls"])
@pytest.mark.parametrize(("provider", "path_key", "engine_type", "module"), CLI_ENGINES)
async def test_a_cli_engine_still_refuses_what_rides_with_tools_when_it_carries_tools(
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    path_key: str,
    engine_type: Any,
    module: Any,
    field: str,
    stream: bool,
) -> None:
    captured = _fake_cli(monkeypatch, engine_type, module)
    engine = _cli(engine_type)
    request = _ask(None, tools=[WEATHER], callerSettings=[field], **{field: VALUE[field]})
    with pytest.raises(CliError) as refused:
        if stream:
            _ = [c async for c in engine.stream(request)]
        else:
            await engine.generate(request)
    assert refused.value.upstream_status == 400
    assert WIRE[field] in str(refused.value) and "no way to carry tools" in str(refused.value)
    assert captured["argv"] is None, "refused before its CLI was started"


@pytest.mark.parametrize(("provider", "path_key", "engine_type", "module"), CLI_ENGINES)
def test_through_the_app_a_cli_driver_serves_a_request_with_no_tools_and_refuses_one_with(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    provider: str,
    path_key: str,
    engine_type: Any,
    module: Any,
) -> None:
    captured = _fake_cli(monkeypatch, engine_type, module)
    config = tmp_path / "cli.yaml"
    _write(config, {"provider": provider, "modelId": "a-model", path_key: NO_SUCH_CLI})
    plain = {
        "messages": [{"role": "user", "content": "hi"}],
        "toolChoice": "auto",
        "parallelToolCalls": False,
        "callerSettings": ["toolChoice", "parallelToolCalls"],
    }
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        served = client.post("/v1/generate", json=plain)
        assert served.status_code == 200, served.text
        assert captured["argv"] is not None
        captured["argv"] = None
        refused = client.post(
            "/v1/generate",
            json={
                **plain,
                "tools": [WEATHER.model_dump(mode="json", exclude_none=True)],
                "callerSettings": ["tools", "toolChoice"],
            },
        )
        assert refused.status_code == 400, refused.text
        assert refused.json()["detail"]["type"].endswith("#tools-unsupported")
        assert engine_type.backend_kind.value in refused.json()["detail"]["detail"]
        assert captured["argv"] is None, "nothing was started"
