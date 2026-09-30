"""Settings never lie (Troy, 2026-09-29: fundamental).

For the inference-driver's config trio:

- **A field is shown wherever it is read.** `baseUrl` and `runtimeName`
  were shown only for the two custom providers while every
  OpenAI-compatible engine reads both, so a stale address sent an
  account's key somewhere the page did not show; `apiKey` was hidden for
  TypeSafe, which will not run without one.
- **An unset value says what it does for this provider**: its own address,
  the key in this machine's environment (presence only, never the key),
  every model the account lists.
- **An empty key is no key**: it read `"<redacted>"`.
- **The keys the agent manages are read-only**, and a PATCH of one is
  refused rather than kept until the agent's next start rewrites it.
- **A restart is pending only for this PATCH's keys**, while they differ
  from what the process runs on, and the schema says which value is in
  effect -- as does `/v1/info`, which named a provider not yet running.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from eugene_plexus_inference_driver._generated.models import ConfigUpdateRequest
from eugene_plexus_inference_driver.config import (
    MANAGED_KEYS_ENV,
    ConfigStore,
    as_schema,
    managed_keys,
)


def _store(tmp_path: Path, values: dict[str, Any] | None = None) -> ConfigStore:
    store = ConfigStore(tmp_path / "driver.yaml")
    store.load()
    if values:
        store.apply_patch(ConfigUpdateRequest.model_validate(values))
    return store


def _fields(store: ConfigStore, **kwargs: Any) -> dict[str, Any]:
    schema = as_schema(values=store.values(), pending=store.pending_restart(), **kwargs)
    return {f.key: f for f in schema.fields}


def test_a_field_is_shown_for_every_provider_that_reads_it() -> None:
    fields = {f.key: f for f in as_schema().fields}
    assert "openrouter" in fields["baseUrl"].showWhen.equals
    assert "openai" in fields["runtimeName"].showWhen.equals
    assert "typesafe" in fields["apiKey"].showWhen.equals
    assert "elevenlabs" in fields["catalogueRefreshMinutes"].showWhen.equals
    # And hidden where nothing reads it.
    assert "elevenlabs" not in fields["modelId"].showWhen.equals
    assert "typesafe" not in fields["thinkingMode"].showWhen.equals
    assert "openai" not in fields["backendLocality"].showWhen.equals


def test_an_unset_address_is_the_providers_own(tmp_path: Path) -> None:
    store = _store(tmp_path, {"provider": "openrouter"})
    base = _fields(store)["baseUrl"]
    assert base.unsetResolvesTo == "https://openrouter.ai/api"
    assert "openrouter.ai" in base.unsetMeans
    custom = _fields(_store(tmp_path / "c", {"provider": "openai_compat_custom"}))["baseUrl"]
    assert custom.unsetResolvesTo is None and "nowhere" in custom.unsetMeans


def test_a_key_from_the_environment_is_named_never_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-the-environment")
    store = _store(tmp_path, {"provider": "openai"})
    key = _fields(store)["apiKey"]
    assert "OPENAI_API_KEY" in key.unsetMeans
    assert "sk-from" not in key.model_dump_json()
    monkeypatch.delenv("OPENAI_API_KEY")
    assert "refuses" in _fields(store)["apiKey"].unsetMeans
    # A saved key: nothing to explain.
    store.apply_patch(ConfigUpdateRequest.model_validate({"apiKey": "sk-saved"}))
    assert _fields(store)["apiKey"].unsetMeans is None


def test_an_empty_key_is_no_key(tmp_path: Path) -> None:
    store = _store(tmp_path, {"provider": "openai", "apiKey": ""})
    assert store.as_document().model_dump().get("apiKey") is None


def test_the_agents_keys_are_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MANAGED_KEYS_ENV, "provider,runtimeName,modelId")
    store = _store(tmp_path)
    result = store.apply_patch(
        ConfigUpdateRequest.model_validate({"modelId": "other", "logLevel": "DEBUG"})
    )
    assert result.applied == ["logLevel"]
    assert [r.key for r in result.rejected] == ["modelId"]
    assert "agent" in result.rejected[0].message
    fields = _fields(store, managed=managed_keys())
    assert fields["modelId"].managedBy and fields["logLevel"].managedBy is None


def test_a_restart_is_pending_only_while_the_value_differs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.apply_patch(ConfigUpdateRequest.model_validate({"provider": "openai"}))
    assert first.pendingRestart == ["provider"]
    field = _fields(store)["provider"]
    assert field.pendingRestart is True and field.inEffect == "claude_subscription"
    assert store.started("provider") == "claude_subscription"
    live = store.apply_patch(ConfigUpdateRequest.model_validate({"catalogueInclude": ["a*"]}))
    assert live.requiresRestart is False and live.pendingRestart == []


def test_nan_is_refused(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = store.apply_patch(
        ConfigUpdateRequest.model_validate({"requestTimeoutSeconds": float("nan")})
    )
    assert result.applied == [] and "finite" in result.rejected[0].message


def test_info_names_the_provider_running_not_the_one_saved(client: Any) -> None:
    """`provider` is read at start: saved but not restarted, it is not what
    answers, and `/v1/info` named it beside the running engine's backend."""
    saved = client.patch("/v1/config", json={"provider": "openrouter"})
    assert saved.json()["pendingRestart"] == ["provider"]
    assert client.get("/v1/info").json()["provider"] == "claude_subscription"
