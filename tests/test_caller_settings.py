"""Explicit settings fail before HTTP/subprocess invocation when unsupported."""

import re
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_inference_driver.engines.claude_code_cli import ClaudeCodeCliEngine
from eugene_plexus_inference_driver.engines.codex_cli import CodexCliEngine
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine


@pytest.mark.parametrize("engine_type", [ClaudeCodeCliEngine, CodexCliEngine])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("maxTokens", 25),
        ("temperature", 0.2),
        ("topP", 0.9),
        ("seed", 0),
        ("stop", ["END"]),
        ("responseFormat", {"type": "json_object"}),
    ],
)
def test_cli_refuses_explicit_control_before_spawn(
    app: FastAPI, engine_type: Any, stream: bool, field: str, value: Any
) -> None:
    # Missing executable is deliberate: spawning would produce a different error.
    with TestClient(app) as client:
        app.state.adapter = engine_type(binary_path="nonexistent-a2-test-executable")
        response = client.post(
            "/v1/generate" + ("/stream" if stream else ""),
            json={
                "messages": [{"role": "user", "content": "PRIVATE PROMPT"}],
                field: value,
                "callerSettings": [field],
            },
        )
        assert response.status_code == 400, response.text
        assert "cannot honor this explicit setting" in response.text
        assert "PRIVATE PROMPT" not in response.text


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("field", ["temperature", "topP"])
def test_fixed_sampler_refuses_explicit_setting(app: FastAPI, stream: bool, field: str) -> None:
    with TestClient(app) as client:
        app.state.adapter = OpenAiCompatibleHttpEngine(
            base_url="http://127.0.0.1:1",
            model_id="fixture",
            auth_required=False,
            fixed_temperature_pattern=re.compile("fixture"),
        )
        response = client.post(
            "/v1/generate" + ("/stream" if stream else ""),
            json={
                "messages": [{"role": "user", "content": "PRIVATE PROMPT"}],
                field: 0.2,
                "callerSettings": [field],
            },
        )
        assert response.status_code == 400, response.text
        assert "cannot honor this explicit setting" in response.text
        assert "PRIVATE PROMPT" not in response.text
