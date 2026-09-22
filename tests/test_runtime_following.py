"""Following a supervised runtime by name — the M2 routing gap, closed.

Before this, launching a model from the library declared a runtime and
stopped there: the gateway builds its routing table from the drivers,
and nothing pointed a driver at a port the agent only chose at launch.
Now a driver configured with `runtimeName` asks the agent where that
runtime lives, at startup and whenever the engine is rebuilt.

The agent is mocked with respx. No process in these tests is real, and
that is the point: the seam under test is the driver's, not the agent's.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver._generated.models import ConfigValueType
from eugene_plexus_inference_driver.app import build_engine_with, create_app
from eugene_plexus_inference_driver.config import as_schema
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine
from eugene_plexus_inference_driver.providers import PROVIDERS
from eugene_plexus_inference_driver.runtime_lookup import (
    RuntimeResolutionError,
    resolve_runtime_url,
)
from eugene_plexus_inference_driver.settings import Settings

AGENT = "http://agent.test:8079"
RUNTIME_URL = "http://127.0.0.1:8090"

# What the agent's `GET /v1/runtimes/{name}` returns for a declared
# runtime. `url` carries pydantic's trailing slash, as the real agent's
# AnyUrl does, and `status` is deliberately not `ready`: the URL is
# derived from the declaration and is stable while the engine loads.
RUNTIME_BODY = {
    "name": "qwen3-8b",
    "engine": "vllm",
    "modelPath": "/home/troy/models/Qwen3-8B",
    "modelAlias": "Qwen3-8B",
    "host": "127.0.0.1",
    "port": 8090,
    "status": "loading",
    "url": RUNTIME_URL + "/",
}

MODELS_BODY = {"object": "list", "data": [{"id": "Qwen3-8B", "object": "model"}]}


def _write_config(path: Path, **fields: object) -> None:
    lines = [f"{key}: {value}" for key, value in fields.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# the lookup itself
# --------------------------------------------------------------------------- #


@respx.mock
def test_resolve_returns_the_runtime_url_without_its_trailing_slash() -> None:
    route = respx.get(f"{AGENT}/v1/runtimes/qwen3-8b").mock(
        return_value=httpx.Response(200, json=RUNTIME_BODY)
    )
    assert resolve_runtime_url("qwen3-8b", agent_url=AGENT, service_token="tok") == RUNTIME_URL
    # Reads on the agent accept any service token; the driver presents
    # the one it was spawned with.
    assert route.calls.last.request.headers["Authorization"] == "Bearer tok"


@respx.mock
def test_resolve_sends_no_bearer_when_there_is_no_token() -> None:
    route = respx.get(f"{AGENT}/v1/runtimes/qwen3-8b").mock(
        return_value=httpx.Response(200, json=RUNTIME_BODY)
    )
    resolve_runtime_url("qwen3-8b", agent_url=AGENT, service_token=None)
    assert "Authorization" not in route.calls.last.request.headers


@respx.mock
def test_resolve_names_the_runtime_when_the_agent_has_none() -> None:
    respx.get(f"{AGENT}/v1/runtimes/nope").mock(return_value=httpx.Response(404))
    with pytest.raises(RuntimeResolutionError, match="'nope' is not declared"):
        resolve_runtime_url("nope", agent_url=AGENT, service_token=None)


@respx.mock
def test_resolve_names_the_agent_when_it_does_not_answer() -> None:
    respx.get(f"{AGENT}/v1/runtimes/qwen3-8b").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(RuntimeResolutionError, match="did not answer") as caught:
        resolve_runtime_url("qwen3-8b", agent_url=AGENT, service_token=None)
    assert "EUGENE_PLEXUS_DRIVER_AGENT_URL" in str(caught.value)


@respx.mock
def test_resolve_explains_a_rejected_token() -> None:
    respx.get(f"{AGENT}/v1/runtimes/qwen3-8b").mock(return_value=httpx.Response(401))
    with pytest.raises(RuntimeResolutionError, match="rejected this driver's credentials"):
        resolve_runtime_url("qwen3-8b", agent_url=AGENT, service_token=None)


@respx.mock
def test_resolve_refuses_a_runtime_with_no_port_yet() -> None:
    body = {k: v for k, v in RUNTIME_BODY.items() if k not in ("url", "port")}
    respx.get(f"{AGENT}/v1/runtimes/qwen3-8b").mock(return_value=httpx.Response(200, json=body))
    with pytest.raises(RuntimeResolutionError, match="has no URL yet"):
        resolve_runtime_url("qwen3-8b", agent_url=AGENT, service_token=None)


@respx.mock
def test_resolve_url_encodes_the_name() -> None:
    route = respx.get(f"{AGENT}/v1/runtimes/two%20words").mock(
        return_value=httpx.Response(200, json=RUNTIME_BODY)
    )
    resolve_runtime_url("two words", agent_url=AGENT, service_token=None)
    assert route.called


# --------------------------------------------------------------------------- #
# the engine: runtimeName wins over baseUrl
# --------------------------------------------------------------------------- #


def test_engine_prefers_the_resolved_runtime_url_over_base_url() -> None:
    """The literal URL is the one that goes stale, so the name wins."""
    config = {
        "provider": "openai_compat_custom",
        "runtimeName": "qwen3-8b",
        "baseUrl": "http://127.0.0.1:9",
        "modelId": "Qwen3-8B",
    }
    engine = build_engine_with(config.get, resolve_runtime=lambda name: RUNTIME_URL)
    assert isinstance(engine, OpenAiCompatibleHttpEngine)
    assert engine._base_url == RUNTIME_URL
    assert engine.runtime == "qwen3-8b"


def test_engine_falls_back_to_base_url_when_no_runtime_is_named() -> None:
    config = {"provider": "openai_compat_custom", "baseUrl": "http://127.0.0.1:9"}
    engine = build_engine_with(config.get, resolve_runtime=lambda name: RUNTIME_URL)
    assert isinstance(engine, OpenAiCompatibleHttpEngine)
    assert engine._base_url == "http://127.0.0.1:9"
    assert engine.runtime is None


def test_engine_with_neither_names_both_fields() -> None:
    config = {"provider": "openai_compat_custom"}
    with pytest.raises(Exception, match="runtimeName") as caught:
        build_engine_with(config.get, resolve_runtime=lambda name: RUNTIME_URL)
    assert "baseUrl" in str(caught.value)


def test_a_runtime_name_on_a_cli_provider_is_ignored_not_fatal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stale value left by a provider switch. The field is only shown
    for the custom provider; a subscription-backed driver must not fail
    over it, and must not claim a runtime it is not fronting."""
    config = {"provider": "claude_subscription", "runtimeName": "qwen3-8b"}
    calls: list[str] = []

    def resolve(name: str) -> str:
        calls.append(name)
        return RUNTIME_URL

    engine = build_engine_with(config.get, resolve_runtime=resolve)
    assert calls == []
    assert getattr(engine, "runtime", None) is None
    assert "does not front a supervised runtime" in caplog.text


def test_a_runtime_name_with_no_resolver_is_a_clear_error() -> None:
    config = {"provider": "openai_compat_custom", "runtimeName": "qwen3-8b"}
    with pytest.raises(ValueError, match="no way to reach the agent"):
        build_engine_with(config.get)


# --------------------------------------------------------------------------- #
# the schema: the field exists, on the custom provider only
# --------------------------------------------------------------------------- #


def test_schema_offers_runtime_name_for_the_custom_provider() -> None:
    fields = {f.key: f for f in as_schema().fields}
    runtime_name = fields["runtimeName"]
    # `runtime_name` is what tells the UI to source a dropdown from the
    # agent's /v1/runtimes rather than a component list — a runtime is
    # deliberately not a component.
    assert runtime_name.valueType == ConfigValueType.runtime_name
    assert runtime_name.showWhen is not None
    # Widened for the System One BYO provider: both custom providers
    # share the one field (two providers declaring `runtimeName` would
    # be two fields with one key).
    assert runtime_name.showWhen.equals == ["openai_compat_custom", "systemone_custom"]
    assert runtime_name.requiresRestart is True

    # `baseUrl` is no longer the only way in, so it is no longer required.
    base_url = fields["baseUrl"]
    assert not base_url.required
    assert base_url.showWhen is not None
    assert base_url.showWhen.equals == ["openai_compat_custom", "systemone_custom"]

    # And it is the custom provider's, not every HTTP provider's: a
    # cloud API is not a runtime we supervise.
    custom = PROVIDERS["openai_compat_custom"]
    assert {f.key for f in custom.extra_field_specs} == {"runtimeName", "baseUrl"}
    assert not PROVIDERS["openai"].extra_field_specs


# --------------------------------------------------------------------------- #
# the app: resolve at startup, report on /v1/info, degrade on failure
# --------------------------------------------------------------------------- #


@respx.mock
def test_driver_follows_a_runtime_by_name_at_startup(tmp_path: Path) -> None:
    respx.get(f"{AGENT}/v1/runtimes/qwen3-8b").mock(
        return_value=httpx.Response(200, json=RUNTIME_BODY)
    )
    respx.get(f"{RUNTIME_URL}/v1/models").mock(return_value=httpx.Response(200, json=MODELS_BODY))

    config = tmp_path / "config.yaml"
    _write_config(
        config,
        provider="openai_compat_custom",
        runtimeName="qwen3-8b",
        modelId="Qwen3-8B",
    )
    app = create_app(settings=Settings(config_file=config, agent_url=AGENT))
    with TestClient(app) as client:
        assert client.get("/healthz").json()["status"] == "ok"

        engine = app.state.adapter
        assert isinstance(engine, OpenAiCompatibleHttpEngine)
        # Resolved from the agent, not typed by anyone.
        assert engine._base_url == RUNTIME_URL

        info = client.get("/v1/info").json()
        assert info["backend"] == "openai_compat_http"
        assert info["provider"] == "openai_compat_custom"
        # What it serves, for the gateway and the UI to show which engine
        # process is behind this driver.
        assert info["runtime"] == "qwen3-8b"

        # The model list came from the resolved runtime.
        assert app.state.available_models == ["Qwen3-8B"]


@respx.mock
def test_driver_degrades_when_the_runtime_cannot_be_resolved(tmp_path: Path) -> None:
    """The OpenClaw rule, applied to a new failure: the driver comes up,
    says why on /healthz, and keeps its config endpoints reachable so the
    operator can fix `runtimeName` from the UI."""
    respx.get(f"{AGENT}/v1/runtimes/ghost").mock(return_value=httpx.Response(404))

    config = tmp_path / "config.yaml"
    _write_config(config, provider="openai_compat_custom", runtimeName="ghost")
    app = create_app(settings=Settings(config_file=config, agent_url=AGENT))
    with TestClient(app) as client:
        health = client.get("/healthz").json()
        assert health["status"] == "degraded"
        assert "'ghost' is not declared" in health["details"]["adapter_error"]

        # Degraded /v1/info still says which runtime it was meant to front.
        info = client.get("/v1/info").json()
        assert info["runtime"] == "ghost"
        assert info["backend"] == "openai_compat_http"

        # And the fix is one PATCH away.
        patched = client.patch("/v1/config", json={"runtimeName": "qwen3-8b"})
        assert "runtimeName" in patched.json()["applied"]
        assert client.get("/v1/config").json()["runtimeName"] == "qwen3-8b"


@respx.mock
def test_config_test_resolves_a_pending_runtime_name(tmp_path: Path) -> None:
    """`/v1/config/test` builds a temporary engine from saved config plus
    overrides, through the same resolver, so a `runtimeName` can be
    checked against the agent before it is committed."""
    respx.get(f"{AGENT}/v1/runtimes/ghost").mock(return_value=httpx.Response(404))
    respx.get(f"{AGENT}/v1/runtimes/qwen3-8b").mock(
        return_value=httpx.Response(200, json=RUNTIME_BODY)
    )
    respx.get(f"{RUNTIME_URL}/v1/models").mock(return_value=httpx.Response(200, json=MODELS_BODY))
    respx.post(f"{RUNTIME_URL}/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "Qwen3-8B",
                "choices": [{"message": {"role": "assistant", "content": "PING"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )

    config = tmp_path / "config.yaml"
    _write_config(config, provider="openai_compat_custom", runtimeName="qwen3-8b")
    app = create_app(settings=Settings(config_file=config, agent_url=AGENT))
    with TestClient(app) as client:
        good = client.post("/v1/config/test").json()
        assert good["ok"] is True, good
        assert good["sampleOutput"] == "PING"

        bad = client.post("/v1/config/test", json={"overrides": {"runtimeName": "ghost"}}).json()
        assert bad["ok"] is False
        assert "'ghost' is not declared" in bad["error"]
