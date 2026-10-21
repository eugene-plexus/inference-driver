"""The chat fields P2c carries, where they are honoured, and what comes back.

Measured 2026-09-28 against OpenRouter (`provider-accounts-measurement.md`
section 7): which models list each setting, OpenAI's `logprobs` shape beside
`delta`, and `url_citation` annotations from a provider's web search. Every
test here fails against the driver as it was before P2c, whose request had
none of these fields and whose parser read neither `logprobs` nor
`annotations`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver._generated.models import GenerateRequest, ReasoningEffort
from eugene_plexus_inference_driver.app import create_app
from eugene_plexus_inference_driver.engines._catalogue import from_openrouter
from eugene_plexus_inference_driver.engines._subprocess import CliError
from eugene_plexus_inference_driver.engines.base import warn_dropped_sampling
from eugene_plexus_inference_driver.settings import Settings

OPENROUTER = "https://openrouter.ai/api"
OPENAI = "https://api.openai.com"
ALL = [
    "logprobs",
    "top_logprobs",
    "logit_bias",
    "reasoning_effort",
    "verbosity",
    "prediction",
    "web_search_options",
    "max_tokens",
    "temperature",
]
FIELDS = {
    "logprobs": True,
    "topLogprobs": 2,
    "logitBias": {"50256": -100},
    "reasoningEffort": "low",
    "verbosity": "low",
    "prediction": {"type": "content", "content": "def f(): pass"},
    "webSearchOptions": {"search_context_size": "low"},
}
SETTINGS = [
    "logprobs",
    "logitBias",
    "reasoningEffort",
    "verbosity",
    "prediction",
    "webSearchOptions",
]
LOGPROBS = {
    "content": [
        {
            "token": "Red",
            "bytes": [82, 101, 100],
            "logprob": -0.01,
            "top_logprobs": [{"token": "Red", "bytes": [82, 101, 100], "logprob": -0.01}],
        }
    ],
    "refusal": None,
}
CITATION = {
    "type": "url_citation",
    "url_citation": {
        "url": "https://example.org/canberra",
        "title": "Canberra",
        "start_index": 0,
        "end_index": 0,
    },
}
#: OpenRouter's own cache of a parsed PDF: not a citation, not carried.
FILE_NOTE = {"type": "file", "file": {"hash": "abc", "name": "a.pdf", "content": []}}


def _or_model(model_id: str, params: list[str]) -> dict[str, Any]:
    return {
        "id": model_id,
        "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "supported_parameters": params,
    }


LISTING = {"data": [_or_model("acme/everything", ALL), _or_model("acme/plain", ["max_tokens"])]}


def ask(model: str, **extra: Any) -> dict[str, Any]:
    fields = {k: v for k, v in FIELDS.items()}
    fields.update(extra)
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Name a colour."}],
        "callerSettings": [s for s in SETTINGS if fields.get(s) is not None],
        **fields,
    }


def _account(tmp_path: Path, provider: str = "openrouter") -> Path:
    config = tmp_path / f"{provider}.yaml"
    config.write_text(json.dumps({"provider": provider, "apiKey": "sk-test"}), "utf-8")
    return config


def _ready(client: TestClient) -> None:
    for _ in range(250):
        if (client.get("/v1/info").json().get("catalogue") or {}).get("refreshedAt"):
            return
        time.sleep(0.02)
    raise AssertionError("catalogue never read")


def _completion(**choice: Any) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Red", **choice.pop("message", {})},
                "finish_reason": "stop",
                **choice,
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


def _sse(*frames: dict[str, Any]) -> httpx.Response:
    body = "".join(f"data: {json.dumps(f)}\n\n" for f in frames) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def _serve(
    config: Path,
    upstream: httpx.Response,
    base: str = OPENROUTER,
    listing: dict[str, Any] = LISTING,
):
    if base == OPENROUTER:
        respx.get(f"{OPENROUTER}/v1/models/user").mock(
            return_value=httpx.Response(200, json=listing)
        )
        respx.get(f"{OPENROUTER}/v1/images/models").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        respx.get(f"{OPENROUTER}/v1/videos/models").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
    else:
        respx.get(f"{base}/v1/models").mock(
            return_value=httpx.Response(
                200, json={"data": [{"id": "gpt-4o-mini", "object": "model"}]}
            )
        )
    route = respx.post(f"{base}/v1/chat/completions").mock(return_value=upstream)
    return TestClient(create_app(settings=Settings(config_file=config))), route


# --------------------------------------------------------------------------- #
# Who is claimed for which setting
# --------------------------------------------------------------------------- #


def test_the_listing_names_which_settings_a_model_takes() -> None:
    everything, plain = from_openrouter(LISTING)
    assert everything.capabilities is not None and plain.capabilities is not None
    assert set(SETTINGS) <= set(everything.capabilities.supportedSettings or [])
    assert not set(SETTINGS) & set(plain.capabilities.supportedSettings or [])


@respx.mock
def test_a_local_engine_is_claimed_for_none_of_them(tmp_path: Path) -> None:
    config = tmp_path / "local.yaml"
    config.write_text(
        json.dumps({"provider": "openai_compat_custom", "baseUrl": "http://local", "modelId": "m"}),
        "utf-8",
    )
    respx.get("http://local/v1/models").mock(return_value=httpx.Response(404))
    respx.get("http://local/props").mock(return_value=httpx.Response(404))
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        models = client.get("/v1/info").json()["models"]
    settings = set(models[0]["capabilities"]["supportedSettings"])
    assert not settings & set(SETTINGS), settings & set(SETTINGS)
    assert {"topK", "minP"} <= settings


# --------------------------------------------------------------------------- #
# Carried to a model that lists them, refused for one that does not
# --------------------------------------------------------------------------- #


@respx.mock
def test_each_setting_reaches_the_backend_in_openais_names(tmp_path: Path) -> None:
    client, upstream = _serve(_account(tmp_path), httpx.Response(200, json=_completion()))
    with client:
        _ready(client)
        response = client.post("/v1/generate", json=ask("acme/everything"))
    assert response.status_code == 200, response.text
    sent = json.loads(upstream.calls[0].request.content)
    assert sent["logprobs"] is True and sent["top_logprobs"] == 2
    assert sent["logit_bias"] == {"50256": -100}
    assert sent["reasoning_effort"] == "low" and sent["verbosity"] == "low"
    assert sent["prediction"] == {"type": "content", "content": "def f(): pass"}
    assert sent["web_search_options"] == {"search_context_size": "low"}
    # Only providers that honour every one of them, as A2 asks.
    assert sent["provider"] == {"require_parameters": True}


@respx.mock
def test_reasoning_effort_max_reaches_the_backend(tmp_path: Path) -> None:
    """`max` joined `ReasoningEffort` on 2026-10-03 (specs 8c41b85): GPT-6
    and OpenRouter take it, and before the pin a caller sending it was a 422
    here -- the value never reached a backend that would have honoured it."""
    client, upstream = _serve(_account(tmp_path), httpx.Response(200, json=_completion()))
    with client:
        _ready(client)
        response = client.post("/v1/generate", json=ask("acme/everything", reasoningEffort="max"))
    assert response.status_code == 200, response.text
    sent = json.loads(upstream.calls[0].request.content)
    assert sent["reasoning_effort"] == "max"


@pytest.mark.parametrize(
    ("setting", "wire"),
    [
        ("logprobs", "logprobs"),
        ("logitBias", "logit_bias"),
        ("verbosity", "verbosity"),
        ("prediction", "prediction"),
        ("webSearchOptions", "web_search_options"),
    ],
)
@respx.mock
def test_a_setting_the_model_does_not_list_is_refused_not_dropped(
    tmp_path: Path, setting: str, wire: str
) -> None:
    client, upstream = _serve(_account(tmp_path), httpx.Response(200, json=_completion()))
    body = {
        "model": "acme/plain",
        "messages": [{"role": "user", "content": "hi"}],
        setting: FIELDS[setting],
        "callerSettings": [setting],
    }
    with client:
        _ready(client)
        response = client.post("/v1/generate", json=body)
    assert response.status_code == 400, response.text
    assert wire in response.text
    assert not upstream.called


def test_an_agentic_cli_refuses_each_of_them() -> None:
    for setting in set(SETTINGS) - {"reasoningEffort"}:
        request = GenerateRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "hi"}],
                setting: FIELDS[setting],
                "callerSettings": [setting],
            }
        )
        with pytest.raises(CliError):
            warn_dropped_sampling(request, engine="claude_code_cli", model_id="m", warned=set())


@pytest.mark.parametrize("effort", list(ReasoningEffort))
@pytest.mark.parametrize("stream", [False, True])
@respx.mock
def test_unlisted_reasoning_effort_is_omitted_instead_of_refused(
    tmp_path: Path, effort: ReasoningEffort, stream: bool
) -> None:
    upstream = (
        _sse(_frame({"content": "ok"})) if stream else httpx.Response(200, json=_completion())
    )
    client, route = _serve(_account(tmp_path), upstream)
    with client:
        _ready(client)
        response = client.post(
            "/v1/generate/stream" if stream else "/v1/generate",
            json={
                "model": "acme/plain",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoningEffort": effort,
                "callerSettings": ["reasoningEffort"],
            },
        )
    assert response.status_code == 200, response.text
    assert "reasoning_effort" not in json.loads(route.calls[-1].request.content)
    assert "provider" not in json.loads(route.calls[-1].request.content)
    assert response.headers["x-eugene-plexus-ignored-settings"] == "reasoningEffort"
    if stream:
        assert "event: done" in response.text


def test_an_agentic_cli_ignores_effort_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    request = GenerateRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "reasoningEffort": "none",
            "callerSettings": ["reasoningEffort"],
        }
    )
    warn_dropped_sampling(request, engine="claude_code_cli", model_id="m", warned=set())
    assert "reasoningEffort" in caplog.text


# --------------------------------------------------------------------------- #
# Hints: carried to OpenAI's own API, dropped elsewhere
# --------------------------------------------------------------------------- #

HINTS = {
    "promptCacheKey": "k-1",
    "promptCacheRetention": "24h",
    "serviceTier": "flex",
    "safetyIdentifier": "user-hash",
}


@respx.mock
def test_hints_are_carried_to_openai(tmp_path: Path) -> None:
    config = tmp_path / "openai.yaml"
    config.write_text(
        json.dumps({"provider": "openai", "apiKey": "sk-test", "modelId": "gpt-4o-mini"}), "utf-8"
    )
    respx.get(f"{OPENAI}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "gpt-4o-mini", "object": "model"}]})
    )
    upstream = respx.post(f"{OPENAI}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion())
    )
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        response = client.post(
            "/v1/generate",
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], **HINTS},
        )
    assert response.status_code == 200, response.text
    sent = json.loads(upstream.calls[0].request.content)
    assert (sent["prompt_cache_key"], sent["prompt_cache_retention"]) == ("k-1", "24h")
    assert (sent["service_tier"], sent["safety_identifier"]) == ("flex", "user-hash")


@respx.mock
def test_hints_are_dropped_where_no_backend_takes_them(tmp_path: Path) -> None:
    client, upstream = _serve(_account(tmp_path), httpx.Response(200, json=_completion()))
    with client:
        _ready(client)
        response = client.post(
            "/v1/generate",
            json={"model": "acme/plain", "messages": [{"role": "user", "content": "hi"}], **HINTS},
        )
    assert response.status_code == 200, response.text
    sent = json.loads(upstream.calls[0].request.content)
    assert not {
        "prompt_cache_key",
        "prompt_cache_retention",
        "service_tier",
        "safety_identifier",
    } & set(sent)


#: Two models whose listings say nothing about caching, as OpenRouter's say none:
#: whether one takes `cache_control` is the provider's name, not a listed parameter.
CACHE_LISTING = {
    "data": [
        _or_model("anthropic/claude-sonnet-4.5", ["max_tokens", "tools"]),
        _or_model("openai/gpt-5", ["max_tokens", "tools"]),
    ]
}


def _cached_ask(model: str, **hint: Any) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        "cacheControl": hint,
    }


@respx.mock
def test_cache_control_reaches_openrouter_anthropic_models_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """PC6: the client's mark rides as one request-level `cache_control`, which
    OpenRouter places after the last message, to an `anthropic/*` model only. A
    model that cannot carry it is sent none and the drop is said once."""
    client, upstream = _serve(
        _account(tmp_path), httpx.Response(200, json=_completion()), listing=CACHE_LISTING
    )
    with client, caplog.at_level("WARNING"):
        _ready(client)
        models = {m["id"]: m for m in client.get("/v1/info").json()["models"]}
        for _ in range(2):
            first = client.post("/v1/generate", json=_cached_ask("anthropic/claude-sonnet-4.5"))
            second = client.post("/v1/generate", json=_cached_ask("openai/gpt-5", ttl="1h"))
            assert first.status_code == 200 and second.status_code == 200, (first.text, second.text)
    sent = {}
    for call in upstream.calls:
        body = json.loads(call.request.content)
        sent.setdefault(body["model"], []).append(body)
    claude, gpt = sent["anthropic/claude-sonnet-4.5"], sent["openai/gpt-5"]
    assert all(b["cache_control"] == {"type": "ephemeral"} for b in claude)
    assert all("cache_control" not in json.dumps(b["messages"]) for b in claude)
    assert all("cache_control" not in b for b in gpt)
    # Listed for the model that carries it, so the gateway can say what arrived.
    assert (
        "cacheControl" in models["anthropic/claude-sonnet-4.5"]["capabilities"]["supportedSettings"]
    )
    assert "cacheControl" not in models["openai/gpt-5"]["capabilities"]["supportedSettings"]
    assert sum("`cache_control`" in r.getMessage() for r in caplog.records) == 1


@respx.mock
def test_the_cache_ttl_is_sent_with_the_mark(tmp_path: Path) -> None:
    client, upstream = _serve(
        _account(tmp_path), httpx.Response(200, json=_completion()), listing=CACHE_LISTING
    )
    with client:
        _ready(client)
        response = client.post(
            "/v1/generate", json=_cached_ask("anthropic/claude-sonnet-4.5", ttl="1h")
        )
    assert response.status_code == 200, response.text
    assert json.loads(upstream.calls[0].request.content)["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }


@respx.mock
def test_a_request_that_marked_nothing_sends_no_cache_control(tmp_path: Path) -> None:
    """Carry the client's, add none."""
    client, upstream = _serve(
        _account(tmp_path), httpx.Response(200, json=_completion()), listing=CACHE_LISTING
    )
    with client:
        _ready(client)
        response = client.post(
            "/v1/generate",
            json={
                "model": "anthropic/claude-sonnet-4.5",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert response.status_code == 200, response.text
    assert "cache_control" not in json.loads(upstream.calls[0].request.content)


@respx.mock
def test_a_stream_carries_the_mark_too(tmp_path: Path) -> None:
    client, upstream = _serve(
        _account(tmp_path),
        _sse({"choices": [{"delta": {"content": "Red"}, "finish_reason": "stop"}]}),
        listing=CACHE_LISTING,
    )
    with client:
        _ready(client)
        response = client.post(
            "/v1/generate/stream", json=_cached_ask("anthropic/claude-sonnet-4.5", ttl="5m")
        )
    assert response.status_code == 200, response.text
    assert json.loads(upstream.calls[0].request.content)["cache_control"] == {
        "type": "ephemeral",
        "ttl": "5m",
    }


@respx.mock
def test_openai_itself_drops_the_mark_with_a_warning(tmp_path: Path) -> None:
    config = tmp_path / "openai.yaml"
    config.write_text(
        json.dumps({"provider": "openai", "apiKey": "sk-test", "modelId": "gpt-4o-mini"}), "utf-8"
    )
    respx.get(f"{OPENAI}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "gpt-4o-mini", "object": "model"}]})
    )
    upstream = respx.post(f"{OPENAI}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion())
    )
    with TestClient(create_app(settings=Settings(config_file=config))) as client:
        response = client.post("/v1/generate", json=_cached_ask("gpt-4o-mini"))
        models = client.get("/v1/info").json()["models"]
    assert response.status_code == 200, response.text
    assert "cache_control" not in json.loads(upstream.calls[0].request.content)
    assert all("cacheControl" not in m["capabilities"]["supportedSettings"] for m in models)


# --------------------------------------------------------------------------- #
# What comes back
# --------------------------------------------------------------------------- #


@respx.mock
def test_a_batch_answer_carries_its_logprobs_and_citations(tmp_path: Path) -> None:
    upstream = httpx.Response(
        200, json=_completion(logprobs=LOGPROBS, message={"annotations": [CITATION, FILE_NOTE]})
    )
    client, _ = _serve(_account(tmp_path), upstream)
    with client:
        _ready(client)
        body = client.post("/v1/generate", json=ask("acme/everything")).json()
    assert body["logprobs"]["content"] == LOGPROBS["content"]
    assert body["annotations"] == [CITATION]


def _frame(delta: dict[str, Any], **choice: Any) -> dict[str, Any]:
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": None, **choice}]}


@respx.mock
def test_a_stream_carries_logprobs_with_their_tokens_and_citations_as_they_come(
    tmp_path: Path,
) -> None:
    second = {"content": [{"token": " sky", "logprob": -0.5, "top_logprobs": []}], "refusal": None}
    upstream = _sse(
        _frame({"role": "assistant", "content": ""}),
        _frame({"content": "Red"}, logprobs=LOGPROBS),
        _frame({"annotations": [CITATION, FILE_NOTE]}),
        _frame({"content": " sky"}, logprobs=second),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )
    client, _ = _serve(_account(tmp_path), upstream)
    with client:
        _ready(client)
        response = client.post("/v1/generate/stream", json=ask("acme/everything"))
    assert response.status_code == 200, response.text
    events = [b for b in response.text.split("\n\n") if b.strip()]
    tokens = [json.loads(e.split("data: ", 1)[1]) for e in events if e.startswith("event: token")]
    texts = [t for t in tokens if t.get("text")]
    assert texts[0]["text"] == "Red"
    assert texts[0]["logprobs"]["content"] == LOGPROBS["content"]
    assert texts[1]["logprobs"]["content"] == second["content"]
    assert [t["annotations"] for t in tokens if "annotations" in t] == [[CITATION]]
    done = json.loads(next(e for e in events if e.startswith("event: done")).split("data: ", 1)[1])
    assert done["logprobs"]["content"] == LOGPROBS["content"] + second["content"]
    assert done["annotations"] == [CITATION]


@respx.mock
def test_a_request_without_them_sends_none(tmp_path: Path) -> None:
    client, upstream = _serve(_account(tmp_path), httpx.Response(200, json=_completion()))
    with client:
        _ready(client)
        client.post(
            "/v1/generate",
            json={"model": "acme/plain", "messages": [{"role": "user", "content": "hi"}]},
        )
    sent = json.loads(upstream.calls[0].request.content)
    assert not {
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "reasoning_effort",
        "verbosity",
        "prediction",
        "web_search_options",
    } & set(sent)
