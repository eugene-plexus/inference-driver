"""Unit tests for the OpenAI-compatible HTTP adapter.

Mocks the upstream HTTP via `respx` so these tests don't hit OpenAI.
End-to-end verification against a real key is gated behind
EUGENE_PLEXUS_DRIVER_LIVE_API=1 and skipped by default.
"""

from __future__ import annotations

import os

import httpx
import pytest
import respx

from eugene_plexus_inference_driver._generated.models import (
    BackendKind,
    FinishReason,
    GenerateRequest,
    Message,
    Role,
)
from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.engines.openai_compat_http import (
    OPENAI_DENY_PATTERN,
    OpenAiCompatibleHttpEngine,
)


def _request(prompt: str = "say PING", system: str | None = None) -> GenerateRequest:
    messages = []
    if system is not None:
        messages.append(Message(role=Role.system, content=system))
    messages.append(Message(role=Role.user, content=prompt))
    return GenerateRequest(messages=messages)


OK_BODY = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "PING"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 1,
        "total_tokens": 13,
    },
}


@respx.mock
async def test_openai_adapter_parses_chat_completion() -> None:
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OK_BODY)
    )
    adapter = OpenAiCompatibleHttpEngine(api_key="sk-test", model_id="gpt-4o", timeout_seconds=30.0)

    response = await adapter.generate(_request(system="You are helpful."))

    assert response.content == "PING"
    assert response.finishReason == FinishReason.stop
    assert response.backend == BackendKind.openai_api
    assert response.modelId == "gpt-4o"
    assert response.usage is not None
    assert response.usage.promptTokens == 12
    assert response.usage.completionTokens == 1

    assert route.called
    call = route.calls[0]
    sent = call.request
    assert sent.headers.get("authorization") == "Bearer sk-test"
    body = httpx._utils.URLPattern  # noqa: F841 — silencing httpx unused-import patterns
    payload = call.request.read()
    import json as _json

    parsed = _json.loads(payload)
    assert parsed["model"] == "gpt-4o"
    assert parsed["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "say PING"},
    ]


@respx.mock
async def test_openai_adapter_uses_configured_base_url() -> None:
    route = respx.post("https://api.minimax.io/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OK_BODY)
    )
    adapter = OpenAiCompatibleHttpEngine(
        api_key="sk-test",
        base_url="https://api.minimax.io",
        model_id="abab6.5-chat",
    )
    await adapter.generate(_request())
    assert route.called


@respx.mock
async def test_openai_adapter_raises_on_http_error() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "invalid key"}})
    )
    adapter = OpenAiCompatibleHttpEngine(api_key="sk-bad")
    with pytest.raises(CliError, match="401"):
        await adapter.generate(_request())


@respx.mock
async def test_openai_adapter_raises_when_no_choices() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    adapter = OpenAiCompatibleHttpEngine(api_key="sk-test")
    with pytest.raises(CliError, match="no choices"):
        await adapter.generate(_request())


@respx.mock
async def test_openai_adapter_passes_roles_through_unchanged() -> None:
    """`Role` is now exactly OpenAI's three roles, so the mapping is
    identity and the multi-turn shape survives the hop. Was a test that
    the bicameral `hemisphere` role collapsed to `assistant`; that role
    left the contract with the consciousness program, but the property
    worth pinning — no re-shaping on the way to the backend — did not."""
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OK_BODY)
    )
    adapter = OpenAiCompatibleHttpEngine(api_key="sk-test")
    await adapter.generate(
        GenerateRequest(
            messages=[
                Message(role=Role.system, content="You are a helpful assistant."),
                Message(role=Role.user, content="hi"),
                Message(role=Role.assistant, content="Hello!"),
                Message(role=Role.user, content="reconsider"),
            ]
        )
    )
    import json as _json

    sent_payload = _json.loads(route.calls[0].request.read())
    assert [m["role"] for m in sent_payload["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert [m["content"] for m in sent_payload["messages"]] == [
        "You are a helpful assistant.",
        "hi",
        "Hello!",
        "reconsider",
    ]


def test_openai_adapter_rejects_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(CliError, match="no API key"):
        OpenAiCompatibleHttpEngine()


def test_openai_adapter_falls_back_to_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    adapter = OpenAiCompatibleHttpEngine()
    assert adapter is not None  # constructor accepts the env-var key without raising


@respx.mock
async def test_openai_adapter_uses_max_completion_tokens_against_openai() -> None:
    """OpenAI's chat-completions API rejects `max_tokens` for newer
    models (gpt-5, o1, o3, ...) — it requires `max_completion_tokens`.
    Pin that the adapter sends the new field name when the base URL
    points at OpenAI proper."""
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OK_BODY)
    )
    adapter = OpenAiCompatibleHttpEngine(api_key="sk-test", model_id="gpt-4o")
    await adapter.generate(
        GenerateRequest(messages=[Message(role=Role.user, content="hi")], maxTokens=512)
    )

    import json as _json

    payload = _json.loads(route.calls[0].request.read())
    assert payload.get("max_completion_tokens") == 512
    assert "max_tokens" not in payload  # legacy field must NOT be sent


@pytest.mark.parametrize(
    "model_id",
    [
        # o-series: reject temperature outright
        "o1",
        "o1-mini",
        "o1-preview",
        "o3",
        "o3-mini",
        # gpt-5 family — schema-accepts temperature but only allows the
        # default value (1). Both the simple names and the dated /
        # variant suffixes OpenAI's /v1/models actually returns must
        # all be rejected at construction.
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-pro",
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.1-codex",
        "gpt-5.5-pro-2026-04-23",
        "gpt-5-2025-08-07",
        "gpt-5-chat-latest",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano-2025-08-07",
    ],
)
def test_openai_adapter_rejects_models_without_controllable_temperature(
    model_id: str,
) -> None:
    """Eugene Plexus relies on temperature as the per-pass divergence
    knob. Models that either reject the parameter (o-series) or only
    accept its default value (gpt-5 family) won't move with our
    NT-modulated temperature in v0.2+, so the adapter refuses to
    construct against one — driver lands in degraded mode with the
    explanation."""
    with pytest.raises(CliError, match="temperature"):
        OpenAiCompatibleHttpEngine(
            api_key="sk-test",
            model_id=model_id,
            deny_pattern=OPENAI_DENY_PATTERN,
        )


@respx.mock
async def test_openai_adapter_uses_legacy_max_tokens_against_self_hosted() -> None:
    """Self-hosted OpenAI-compatible servers (Ollama, vLLM, LM Studio)
    implement the older spec and only know `max_tokens`. Pin that the
    adapter sends the legacy field name when the base URL doesn't
    point at OpenAI proper."""
    route = respx.post("http://127.0.0.1:11434/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OK_BODY)
    )
    adapter = OpenAiCompatibleHttpEngine(
        api_key="sk-ollama-stub",
        base_url="http://127.0.0.1:11434",
        model_id="llama3.1:70b",
    )
    await adapter.generate(
        GenerateRequest(messages=[Message(role=Role.user, content="hi")], maxTokens=512)
    )

    import json as _json

    payload = _json.loads(route.calls[0].request.read())
    assert payload.get("max_tokens") == 512
    assert "max_completion_tokens" not in payload


# ---------------------------------------------------------------------------
# list_models — live-fetched from /v1/models with our compatibility filter
# ---------------------------------------------------------------------------


@respx.mock
async def test_openai_adapter_list_models_filters_temperature_uncontrollable() -> None:
    """Live fetch returns the post-filter set: chat-plausible model IDs
    that ALSO pass our temperature-controllability bar. Embeddings,
    audio, image and the o-series / gpt-5 families are stripped."""
    fake_catalog = {
        "data": [
            {"id": "gpt-4o", "object": "model"},
            {"id": "gpt-4o-mini", "object": "model"},
            {"id": "gpt-4.1", "object": "model"},
            {"id": "gpt-4-turbo", "object": "model"},
            {"id": "gpt-3.5-turbo", "object": "model"},
            # Filtered: temperature-uncontrollable. OpenAI's /v1/models
            # actually returns these dated and dotted variants in real
            # life — pin that the pattern catches all of them.
            {"id": "gpt-5", "object": "model"},
            {"id": "gpt-5-mini", "object": "model"},
            {"id": "gpt-5-2025-08-07", "object": "model"},
            {"id": "gpt-5-chat-latest", "object": "model"},
            {"id": "gpt-5.1", "object": "model"},
            {"id": "gpt-5.2-pro", "object": "model"},
            {"id": "gpt-5.5-pro-2026-04-23", "object": "model"},
            {"id": "o1", "object": "model"},
            {"id": "o1-mini", "object": "model"},
            {"id": "o3", "object": "model"},
            # Filtered: not chat
            {"id": "text-embedding-3-large", "object": "model"},
            {"id": "dall-e-3", "object": "model"},
            {"id": "whisper-1", "object": "model"},
            {"id": "tts-1", "object": "model"},
            {"id": "omni-moderation-latest", "object": "model"},
            {"id": "babbage-002", "object": "model"},
        ]
    }
    respx.get("https://api.openai.com/v1/models").mock(
        return_value=httpx.Response(200, json=fake_catalog)
    )
    adapter = OpenAiCompatibleHttpEngine(
        api_key="sk-test",
        model_id="gpt-4o",
        deny_pattern=OPENAI_DENY_PATTERN,
    )

    models = await adapter.list_models()

    # Allowed: chat-plausible AND temperature-controllable.
    assert "gpt-4o" in models
    assert "gpt-4o-mini" in models
    assert "gpt-4.1" in models
    assert "gpt-4-turbo" in models
    assert "gpt-3.5-turbo" in models
    # Filtered: o-series and gpt-5 family (including the dotted /
    # dated variants OpenAI's /v1/models actually returns).
    for forbidden in (
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-2025-08-07",
        "gpt-5-chat-latest",
        "gpt-5.1",
        "gpt-5.2-pro",
        "gpt-5.5-pro-2026-04-23",
        "o1",
        "o1-mini",
        "o3",
    ):
        assert forbidden not in models
    # Filtered: non-chat.
    assert "text-embedding-3-large" not in models
    assert "dall-e-3" not in models
    assert "whisper-1" not in models
    assert "tts-1" not in models
    assert "omni-moderation-latest" not in models
    assert "babbage-002" not in models
    # Sorted for stable rendering.
    assert models == sorted(models)


@respx.mock
async def test_openai_adapter_list_models_returns_empty_on_transport_error() -> None:
    """If the backend can't be reached, list_models returns an empty
    list — the schema endpoint falls back to free-text input. Driver
    stays up either way."""
    respx.get("https://api.openai.com/v1/models").mock(
        return_value=httpx.Response(500, text="server exploded")
    )
    adapter = OpenAiCompatibleHttpEngine(api_key="sk-test", model_id="gpt-4o")
    assert await adapter.list_models() == []


# ---------------------------------------------------------------------------
# Live test — opt-in via env var (uses real OpenAI quota)
# ---------------------------------------------------------------------------

LIVE = os.environ.get("EUGENE_PLEXUS_DRIVER_LIVE_API") == "1"


@pytest.mark.skipif(not LIVE, reason="set EUGENE_PLEXUS_DRIVER_LIVE_API=1 to run live")
async def test_openai_live_call() -> None:
    adapter = OpenAiCompatibleHttpEngine(timeout_seconds=60.0)
    response = await adapter.generate(
        _request(prompt="Reply with exactly the four characters: PING", system="You are concise.")
    )
    assert "PING" in response.content
