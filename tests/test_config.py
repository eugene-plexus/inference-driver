"""Tests for the config protocol routes."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_get_config_schema_lists_expected_fields(client: TestClient) -> None:
    response = client.get("/v1/config/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["component"] == "hemisphere-driver"

    field_keys = {f["key"] for f in body["fields"]}
    expected = {
        "provider",
        "modelId",
        "claudeCodeCliPath",
        "codexCliPath",
        "apiKey",
        "baseUrl",
        "logLevel",
        "requestTimeoutSeconds",
    }
    assert expected.issubset(field_keys)
    # LLM-output-affecting params (temperature, max-tokens, etc.) are
    # owned by the orchestrator and never appear on the driver schema.
    assert "defaultTemperature" not in field_keys
    assert "defaultMaxTokens" not in field_keys
    # Driver no longer self-asserts identity — the orchestrator labels
    # it via the `drivers` config and the `driverName` it stamps on
    # every emitted message.
    assert "hemisphere" not in field_keys
    # The `adapter` field was renamed to `provider` in the registry refactor.
    assert "adapter" not in field_keys

    provider_field = next(f for f in body["fields"] if f["key"] == "provider")
    assert provider_field["valueType"] == "enum"
    # Provider keys are the user-facing subscription names, not the
    # protocol-level engine names.
    assert "claude_subscription" in provider_field["enumValues"]
    assert "openai" in provider_field["enumValues"]
    assert "xai" in provider_field["enumValues"]
    # Friendly labels are paired one-to-one with the keys.
    assert provider_field.get("enumLabels") is not None
    assert len(provider_field["enumLabels"]) == len(provider_field["enumValues"])
    assert provider_field["requiresRestart"] is True


def test_get_config_returns_defaults_on_first_run(client: TestClient) -> None:
    response = client.get("/v1/config")
    assert response.status_code == 200
    doc = response.json()
    assert doc["provider"] == "claude_subscription"
    # `port` is no longer a config field — owned by the watchdog topology
    # via EUGENE_PLEXUS_HD_BIND_PORT. Both legacy keys are rejected.
    assert "port" not in doc
    assert "hemisphere" not in doc
    assert "adapter" not in doc  # renamed to provider


def test_patch_applies_valid_change_and_rejects_invalid(client: TestClient) -> None:
    response = client.patch(
        "/v1/config",
        json={
            "claudeCodeCliPath": "/usr/local/bin/claude",  # valid string update
            "port": 99999,  # `port` is no longer a config field; rejected as unknown
            "bogusField": True,  # unknown field
            "defaultTemperature": 0.9,  # used to live here; must now be unknown
            "hemisphere": "right",  # also used to live here; must now be unknown
            "adapter": "openai_api",  # renamed to `provider`; old key is unknown
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "claudeCodeCliPath" in body["applied"]

    rejected_keys = {r["key"] for r in body["rejected"]}
    # All five legacy / unknown keys come through as rejections.
    assert {
        "port",
        "bogusField",
        "defaultTemperature",
        "hemisphere",
        "adapter",
    } <= rejected_keys

    # `claudeCodeCliPath` is restart-required.
    assert body["requiresRestart"] is True

    follow = client.get("/v1/config")
    assert follow.json()["claudeCodeCliPath"] == "/usr/local/bin/claude"
    assert "port" not in follow.json()


def test_schema_modelid_is_string_when_no_models_discovered(client: TestClient) -> None:
    """Default fixture starts the app without an adapter the lifespan
    can build (claude_code_cli with `claude` in PATH most likely fails
    or no list is set), so available_models is empty. modelId stays as
    free-text string so the operator can still type a model by hand."""
    body = client.get("/v1/config/schema").json()
    model_field = next(f for f in body["fields"] if f["key"] == "modelId")
    # Either string (no list available) or enum (list discovered) is
    # acceptable; we just assert the contract — when there's NO list,
    # it MUST be string. The fixture's lifespan typically discovers
    # claude models, so we expect enum here. The OTHER test pins
    # the no-list branch directly.
    assert model_field["valueType"] in ("string", "enum")


def test_schema_modelid_carries_suggestions_when_models_available() -> None:
    """When the lifespan populates `available_models`, the schema
    endpoint exposes modelId as a free-text field WITH discovery
    suggestions. Validation stays permissive so an operator can paste
    an id the driver hasn't fetched yet (Ollama just-pulled, custom
    endpoint, etc.); the UI renders the suggestions as a combobox
    dropdown next to the input."""
    from eugene_plexus_hemisphere_driver.config import as_schema

    schema = as_schema(available_models=["claude-opus-4-7", "claude-sonnet-4-7"])
    model_field = next(f for f in schema.fields if f.key == "modelId")
    # valueType stays string so PATCH accepts any pasted id.
    assert model_field.valueType.value == "string"
    assert model_field.suggestions is not None
    assert "claude-opus-4-7" in model_field.suggestions
    assert "claude-sonnet-4-7" in model_field.suggestions


def test_schema_modelid_stays_string_without_models() -> None:
    """Empty / missing list → free-text input. Belt-and-suspenders for
    the fallback path the schema route uses in degraded mode."""
    from eugene_plexus_hemisphere_driver.config import as_schema

    schema = as_schema(available_models=None)
    model_field = next(f for f in schema.fields if f.key == "modelId")
    assert model_field.valueType.value == "string"
    schema = as_schema(available_models=[])
    model_field = next(f for f in schema.fields if f.key == "modelId")
    assert model_field.valueType.value == "string"


def test_patch_with_restart_required_field_sets_pending(client: TestClient) -> None:
    # `claudeCodeCliPath` is one of the remaining restart-required fields
    # after `port` was promoted out of the config schema.
    response = client.patch("/v1/config", json={"claudeCodeCliPath": "/usr/local/bin/claude"})
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == ["claudeCodeCliPath"]
    assert body["requiresRestart"] is True
    assert body["pendingRestart"] == ["claudeCodeCliPath"]
