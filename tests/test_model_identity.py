"""The public/upstream model identity split (`upstreamModelId`).

The field exists for backends whose served name is not ours to choose —
`mlx_lm.server` answers only to upstream's `default_model` sentinel or
the model's absolute path — so the invariant under test is one sentence:
**the backend sees the upstream id, everyone else sees the public id.**

The two failure directions are asserted separately, because each has a
plausible wrong fix:

  * The wire carrying the public alias means the backend refuses a name
    it never heard of, or worse resolves it to something else.
  * A response echoing the upstream id means two MLX runtimes serving
    different models both answer `default_model`, which is the exact
    collision the split exists to prevent — and the gateway then reports
    it to the caller (`response.modelId or body.model`).

When NO upstream id is configured, the old echo behavior must survive
byte-identically: a backend that reports serving something other than
what was configured is telling the truth about what answered, and every
install predating the split relies on the passthrough.
"""

from __future__ import annotations

import json

import httpx
import respx

from eugene_plexus_inference_driver._generated.models import (
    BackendKind,
    GenerateRequest,
    Message,
    Role,
)
from eugene_plexus_inference_driver.config import as_schema
from eugene_plexus_inference_driver.engines.claude_code_cli import ClaudeCodeCliEngine
from eugene_plexus_inference_driver.engines.codex_cli import CodexCliEngine
from eugene_plexus_inference_driver.engines.openai_compat_http import (
    OPENAI_FIXED_TEMPERATURE_PATTERN,
    OpenAiCompatibleHttpEngine,
)


def _request(prompt: str = "say PING") -> GenerateRequest:
    return GenerateRequest(messages=[Message(role=Role.user, content=prompt)])


def _mlx_style_body(content: str = "PING") -> dict:
    """What mlx_lm.server sends back: `model` echoes the sentinel."""
    return {
        "id": "chatcmpl-mlx",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "default_model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


def _engine(**overrides) -> OpenAiCompatibleHttpEngine:
    kwargs = {
        "api_key": None,
        "auth_required": False,
        # Never dialed; respx intercepts every request to it.
        "base_url": "http://127.0.0.1:9999",
        "model_id": "qwen3-0.6b-4bit",
        "upstream_model_id": "default_model",
        "timeout_seconds": 30.0,
    }
    kwargs.update(overrides)
    return OpenAiCompatibleHttpEngine(**kwargs)


# --- config schema ---------------------------------------------------------


def test_schema_lists_upstream_model_id_as_plain_string() -> None:
    fields = {f.key: f for f in as_schema().fields}
    assert "upstreamModelId" in fields
    assert fields["upstreamModelId"].valueType.value == "string"
    assert fields["upstreamModelId"].requiresRestart is True


def test_discovered_models_suggest_on_both_identity_fields() -> None:
    """The discovered names are upstream names — but in the no-split
    case `modelId` IS the upstream name, so both fields carry them."""
    schema = as_schema(available_models=["a", "b"])
    by_key = {f.key: f for f in schema.fields}
    assert by_key["modelId"].suggestions == ["a", "b"]
    assert by_key["upstreamModelId"].suggestions == ["a", "b"]


# --- the wire sees the upstream id ----------------------------------------


@respx.mock
async def test_generate_sends_upstream_and_reports_public() -> None:
    route = respx.post("http://127.0.0.1:9999/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_mlx_style_body())
    )
    engine = _engine()

    response = await engine.generate(_request())

    sent = json.loads(route.calls[0].request.read())
    assert sent["model"] == "default_model"
    # The backend echoed its sentinel and the caller must never see it.
    assert response.modelId == "qwen3-0.6b-4bit"


@respx.mock
async def test_stream_terminal_frame_reports_public_id() -> None:
    frames = [
        'data: {"model": "default_model", "choices": [{"index": 0, "delta": {"role": "assistant"}}]}',
        'data: {"model": "default_model", "choices": [{"index": 0, "delta": {"content": "PI"}}]}',
        'data: {"model": "default_model", "choices": [{"index": 0, "delta": {"content": "NG"}, "finish_reason": "stop"}]}',
        "data: [DONE]",
    ]
    respx.post("http://127.0.0.1:9999/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="\n\n".join(frames) + "\n\n",
        )
    )
    engine = _engine()

    chunks = [chunk async for chunk in engine.stream(_request())]

    done = [c for c in chunks if c.done]
    assert len(done) == 1
    assert done[0].result is not None
    assert done[0].result.modelId == "qwen3-0.6b-4bit"
    assert "".join(c.text for c in chunks if c.text) == "PING"


@respx.mock
async def test_embed_sends_upstream_and_reports_public() -> None:
    route = respx.post("http://127.0.0.1:9999/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "model": "default_model",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
    )
    engine = _engine()

    result = await engine.embed(["hello"])

    sent = json.loads(route.calls[0].request.read())
    assert sent["model"] == "default_model"
    assert result.modelId == "qwen3-0.6b-4bit"


# --- nothing configured: the old echo survives -----------------------------


@respx.mock
async def test_without_upstream_id_the_backend_echo_is_kept() -> None:
    """An honest backend reporting a different served model stays
    visible — the passthrough every pre-split install relies on."""
    body = _mlx_style_body()
    body["model"] = "what-actually-answered"
    respx.post("http://127.0.0.1:9999/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=body)
    )
    engine = _engine(upstream_model_id=None, model_id="what-i-configured")

    response = await engine.generate(_request())

    assert response.modelId == "what-actually-answered"


@respx.mock
async def test_without_upstream_id_the_wire_carries_model_id() -> None:
    route = respx.post("http://127.0.0.1:9999/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_mlx_style_body())
    )
    engine = _engine(upstream_model_id=None, model_id="what-i-configured")

    await engine.generate(_request())

    sent = json.loads(route.calls[0].request.read())
    assert sent["model"] == "what-i-configured"


# --- from_config resolves the fallback once --------------------------------


def test_from_config_empty_string_means_unset() -> None:
    """The UI clears a field to `""`, and `""` must mean "send modelId
    verbatim", not "ask the backend for the empty string"."""
    config = {
        "baseUrl": "http://127.0.0.1:9999",
        "modelId": "public-alias",
        "upstreamModelId": "",
    }
    engine = OpenAiCompatibleHttpEngine.from_config(
        config.get,
        default_base_url=None,
        fixed_temperature_pattern=None,
        backend_kind=BackendKind.openai_compat_http,
        auth_required=False,
    )
    assert engine._upstream_model_id == "public-alias"


# --- the temperature-fixed pattern reads the UPSTREAM name -----------------


def test_fixed_temperature_matches_the_name_the_backend_sees() -> None:
    """The pattern describes what the BACKEND rejects, and the backend
    only ever sees the upstream name. A public alias that happens to
    look like an o-series model must not freeze the sampler."""
    frozen = OpenAiCompatibleHttpEngine(
        auth_required=False,
        model_id="my-alias",
        upstream_model_id="o3-mini",
        fixed_temperature_pattern=OPENAI_FIXED_TEMPERATURE_PATTERN,
    )
    assert frozen._temperature_is_fixed is True

    free = OpenAiCompatibleHttpEngine(
        auth_required=False,
        model_id="o3-mini-lookalike-alias",
        upstream_model_id="qwen3-8b",
        fixed_temperature_pattern=OPENAI_FIXED_TEMPERATURE_PATTERN,
    )
    # `o3-mini-lookalike-alias` matches the o-series pattern; the
    # upstream name does not, and the upstream name is what decides.
    assert free._temperature_is_fixed is False


# --- CLI engines: argv is the wire boundary --------------------------------


def test_claude_cli_argv_carries_the_upstream_name() -> None:
    engine = ClaudeCodeCliEngine(
        binary_path="claude",
        model_id="public-alias",
        upstream_model_id="claude-opus-4-7",
    )
    argv = engine._build_argv(system_prompt="")
    model_flag = argv[argv.index("--model") + 1]
    assert model_flag == "claude-opus-4-7"


def test_codex_cli_argv_carries_the_upstream_name() -> None:
    engine = CodexCliEngine(
        binary_path="codex",
        model_id="public-alias",
        upstream_model_id="gpt-5",
    )
    argv = engine._build_argv("say PING")
    model_flag = argv[argv.index("--model") + 1]
    assert model_flag == "gpt-5"


# --- /v1/info advertises both ----------------------------------------------


def test_info_advertises_public_and_upstream(tmp_path, monkeypatch) -> None:
    from fastapi.testclient import TestClient

    config_file = tmp_path / "driver.yaml"
    config_file.write_text(
        "provider: openai_compat_custom\n"
        "baseUrl: http://127.0.0.1:9999\n"
        "modelId: public-alias\n"
        "upstreamModelId: default_model\n"
        "backendLocality: local\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EUGENE_PLEXUS_DRIVER_CONFIG_FILE", str(config_file))

    from eugene_plexus_inference_driver.app import create_app

    with TestClient(create_app()) as client:
        info = client.get("/v1/info").json()

    assert info["modelId"] == "public-alias"
    assert info["upstreamModelId"] == "default_model"
