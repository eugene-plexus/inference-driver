"""Runtime configuration: schema declaration + file-backed state + PATCH apply.

Implements the shared Eugene Plexus config protocol:

* `GET /v1/config/schema` -> field metadata for UI rendering (`as_schema()`)
* `GET /v1/config` -> current effective values, secrets redacted (`as_document()`)
* `PATCH /v1/config` -> partial update, per-key validation (`apply_patch()`)

v0.2 at-rest encryption (Phase 6): when constructed with a master key,
the store encrypts `sensitive: true` fields as libsodium-secretbox
envelopes before writing to disk, and decrypts them transparently
on load. In-memory `_values` always holds plaintext so engine
construction (`apiKey` -> `Authorization: Bearer ...`) doesn't need
to know about the at-rest format. Plaintext-on-disk configs from v0.1
load fine and auto-upgrade to envelopes on the next save.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import yaml

from . import security
from ._generated.models import (
    ConfigDocument,
    ConfigField,
    ConfigFieldError,
    ConfigSchema,
    ConfigUpdateRequest,
    ConfigUpdateResult,
    ConfigValueType,
)
from .engines.claude_code_cli import ClaudeCodeCliEngine
from .engines.codex_cli import CodexCliEngine
from .engines.openai_compat_http import OpenAiCompatibleHttpEngine
from .providers import PROVIDERS, collect_extra_field_specs, providers_using

log = logging.getLogger(__name__)

REDACTED = "<redacted>"

CATEGORY_LABELS: dict[str, str] = {
    "adapter": "Provider",
    "network": "Network",
    "logging": "Logging",
}


def _provider_field() -> ConfigField:
    """The user-facing dropdown — which subscription / service this
    driver wraps. Values are stable registry keys; labels are
    operator-friendly. Each downstream engine field carries its own
    `showWhen` against this field so irrelevant inputs disappear."""
    keys = list(PROVIDERS.keys())
    labels = [PROVIDERS[k].label for k in keys]
    return ConfigField(
        key="provider",
        label="Provider",
        description=(
            "Which LLM subscription or service this driver wraps. "
            "The fields shown below adapt to your choice — Claude / "
            "ChatGPT subscriptions ask for the local CLI binary; "
            "OpenAI / xAI / OpenRouter / MiniMax / etc. ask for an "
            "API key; the Custom option lets you point at any "
            "OpenAI-compatible URL. Switching this changes which "
            "other fields are relevant; restart required so the "
            "engine reconnects."
        ),
        category="adapter",
        valueType=ConfigValueType.enum,
        default="claude_subscription",
        enumValues=keys,
        enumLabels=labels,
        required=True,
        requiresRestart=True,
    )


def _modelid_field() -> ConfigField:
    """The model picker. Always shown; per-engine model lists are
    discovered live and supplied via `as_schema(available_models=...)`."""
    return ConfigField(
        key="modelId",
        label="Model",
        description=(
            "Which specific model to ask the backend for (e.g. "
            "\"gpt-4o\", \"claude-opus-4-7\", \"grok-2\", "
            "\"llama3.1:70b\"). The list below is discovered from the "
            "selected provider; pick the empty entry to fall back to "
            "the engine's built-in default."
        ),
        category="adapter",
        valueType=ConfigValueType.string,
        requiresRestart=True,
    )


def _common_fields() -> list[ConfigField]:
    """Provider-agnostic fields — logging and timeouts.

    The bind port deliberately is NOT in this list. Ports are owned by
    the watchdog topology (`watchdog.yaml`), passed to spawned children
    via `EUGENE_PLEXUS_HD_BIND_PORT`. Two sources of truth on `port`
    was an OpenClaw-style trap waiting to bite — the watchdog spawns at
    one port, the driver's own config says another, and the orchestrator
    can't reach the driver. Now there's one source.
    """
    return [
        ConfigField(
            key="thinkingMode",
            label="Thinking mode",
            description=(
                "Controls how much internal reasoning this driver's model "
                "produces before answering. `auto` defers to the model's "
                "natural behavior. `off` instructs the model NOT to emit "
                "`<think>...</think>` blocks or scratchpad reasoning — "
                "useful for reasoning-tag models (DeepSeek R1, Qwen QwQ, "
                "Kimi, etc.) where the internal thinking otherwise leaks "
                "into the response. `low` / `medium` / `high` ask for "
                "progressively more deliberation. v0.2.x applies this as "
                "a system-prompt directive; native API budget control "
                "(Anthropic extended thinking, OpenAI `reasoning_effort`) "
                "lands in v0.3+."
            ),
            category="adapter",
            valueType=ConfigValueType.enum,
            default="auto",
            enumValues=["auto", "off", "low", "medium", "high"],
            enumLabels=["Auto", "Off", "Low", "Medium", "High"],
            requiresRestart=True,
        ),
        ConfigField(
            key="logLevel",
            label="Log level",
            description=(
                "How chatty the driver's terminal output is. `DEBUG` "
                "dumps the full upstream payload going out and the "
                "full response coming back for every backend call — "
                "shows you exactly what the LLM saw (post role-"
                "coercion, post-thinking-directive injection, post-"
                "CLI flattening), which the orchestrator's copy-trace "
                "does not. `INFO` is the normal operating level; "
                "`WARNING` and `ERROR` go progressively quieter."
            ),
            category="logging",
            valueType=ConfigValueType.enum,
            default="INFO",
            enumValues=["DEBUG", "INFO", "WARNING", "ERROR"],
            requiresRestart=True,
        ),
        ConfigField(
            key="requestTimeoutSeconds",
            label="Backend timeout",
            description=(
                "How long the driver will wait on a single LLM call "
                "before giving up and returning an error. Long-running "
                "CLI invocations and large reasoning models can take "
                "a while; 120s is the v0.1 default. Bump it if you "
                "see timeouts on complex prompts."
            ),
            category="network",
            valueType=ConfigValueType.duration,
            default=120,
            minimum=5,
            maximum=900,
            requiresRestart=True,
        ),
    ]


def _build_fields() -> list[ConfigField]:
    """Compose the full FIELDS list from the registry. Order:
    provider -> per-engine fields (API key, CLI paths) -> per-provider
    extras (custom baseUrl) -> modelId -> common.

    Why modelId comes last among adapter fields: most providers need a
    valid API key (or local URL) before their model list is reachable,
    so the UI shows credentials first, then the model picker."""
    out: list[ConfigField] = [_provider_field()]
    seen: set[type] = set()
    for engine_cls in (ClaudeCodeCliEngine, CodexCliEngine, OpenAiCompatibleHttpEngine):
        if engine_cls in seen:
            continue
        seen.add(engine_cls)
        applicable = providers_using(engine_cls)
        if not applicable:
            continue
        out.extend(engine_cls.field_specs(applicable_providers=applicable))
    out.extend(collect_extra_field_specs())
    out.append(_modelid_field())
    out.extend(_common_fields())
    return out

# Schema for hemisphere-driver's config surface. Built dynamically from
# the provider registry — the order here is the order the UI renders.
FIELDS: list[ConfigField] = _build_fields()
# NOTE on what's NOT in this schema:
# Temperature, max-tokens, stop sequences and other parameters that alter
# LLM output are owned by the *caller* (the orchestrator) and arrive on
# every `GenerateRequest`. The driver applies what it's given and never
# substitutes a local default. In v0.2+ the orchestrator's NT system will
# modulate these per-request — placing defaults here would make that
# layering invisible and a future NT signal trivially overridable from a
# config file.

_FIELDS_BY_KEY: dict[str, ConfigField] = {f.key: f for f in FIELDS}


def as_schema(*, available_models: list[str] | None = None) -> ConfigSchema:
    """Return the driver's schema, surfacing discovered models as
    `suggestions` on the `modelId` field when the caller supplies them.

    The list comes from the adapter's `list_models()` (live for
    openai_api, hardcoded for the CLIs) and arrives at the schema
    endpoint via `app.state.available_models`. modelId stays a
    free-text `string` field either way — the operator can paste a
    model the driver hasn't fetched yet (just-pulled in Ollama,
    just-deployed in a custom endpoint) and the validator accepts it.
    The UI renders the suggestions as a combobox dropdown beside the
    free-text input.
    """
    fields = list(FIELDS)
    if available_models:
        fields = [
            _with_model_suggestions(f, available_models) if f.key == "modelId" else f
            for f in fields
        ]
    return ConfigSchema(
        component="hemisphere-driver",
        fields=fields,
        categories=CATEGORY_LABELS,
    )


def _with_model_suggestions(model_field: ConfigField, models: list[str]) -> ConfigField:
    """Return a copy of `modelId` carrying the given models as
    discovery-time `suggestions`. valueType stays `string` so the
    operator can paste an id the driver hasn't fetched yet."""
    return model_field.model_copy(
        update={
            "suggestions": list(models),
        }
    )


def _defaults() -> dict[str, Any]:
    return {f.key: f.default for f in FIELDS if f.default is not None}


def _validate_value(field: ConfigField, value: Any) -> str | None:
    """Return None if valid, otherwise an error message."""
    if value is None:
        return None  # null clears to default

    vt = field.valueType

    if vt == ConfigValueType.string or vt == ConfigValueType.url or vt == ConfigValueType.file_path:
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        if field.pattern is not None:
            import re

            if re.search(field.pattern, value) is None:
                return f"value does not match pattern {field.pattern!r}"
        return None

    if vt == ConfigValueType.secret:
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        if value == REDACTED:
            return "refusing to write the literal redacted value back"
        return None

    if vt == ConfigValueType.integer:
        if isinstance(value, bool) or not isinstance(value, int):
            return f"expected integer, got {type(value).__name__}"
        if field.minimum is not None and value < field.minimum:
            return f"must be >= {field.minimum}"
        if field.maximum is not None and value > field.maximum:
            return f"must be <= {field.maximum}"
        return None

    if vt == ConfigValueType.number or vt == ConfigValueType.duration:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return f"expected number, got {type(value).__name__}"
        if field.minimum is not None and value < field.minimum:
            return f"must be >= {field.minimum}"
        if field.maximum is not None and value > field.maximum:
            return f"must be <= {field.maximum}"
        return None

    if vt == ConfigValueType.boolean:
        if not isinstance(value, bool):
            return f"expected boolean, got {type(value).__name__}"
        return None

    if vt == ConfigValueType.enum:
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        allowed = field.enumValues or []
        if value not in allowed:
            return f"must be one of {allowed}"
        return None

    return f"unsupported valueType: {vt}"


class ConfigStore:
    """File-backed config state. Thread-safe for the simple read/write pattern.

    When constructed with `master_key`, sensitive fields are sealed in
    a libsodium-secretbox envelope before being written to disk and
    transparently decrypted on load. Without a master key (dev runs,
    pre-login window) the store still works — plaintext on disk,
    matching the v0.1 behavior. Envelope-shaped values on disk that
    can't be decrypted (no key, wrong key, tampered ciphertext) get
    skipped with a warning; the engine then sees the field as unset
    and `from_config` produces a clear "no API key" error.
    """

    def __init__(self, path: Path, *, master_key: bytes | None = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._values: dict[str, Any] = _defaults()
        self._pending_restart: set[str] = set()
        self._master_key = master_key

    def load(self) -> None:
        """Load from the configured file, creating it with defaults if absent."""
        with self._lock:
            if self._path.exists():
                raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
                if not isinstance(raw, dict):
                    raise ValueError(f"config file {self._path} must be a YAML mapping at the root")
                merged = _defaults()
                for k, v in raw.items():
                    if k not in _FIELDS_BY_KEY:
                        continue
                    merged[k] = self._decrypt_loaded(k, v)
                self._values = merged
            else:
                self._values = _defaults()
                self._write_locked()

    def _decrypt_loaded(self, key: str, value: Any) -> Any:
        """Resolve an on-disk value to its in-memory (plaintext) form.

        Plaintext-on-disk (v0.1 configs, non-sensitive fields): pass
        through unchanged. Envelope-shape values: decrypt if we have a
        key; otherwise log a warning and drop to None so the engine
        sees an unconfigured field rather than a raw envelope dict.
        """
        if not security.is_envelope(value):
            return value
        if self._master_key is None:
            log.warning(
                "config field %r is encrypted on disk but no master key is "
                "available; treating as unset (engine will fail clearly)",
                key,
            )
            return None
        try:
            envelope = security.Envelope.from_dict(value)
            return security.open_envelope(envelope, self._master_key)
        except ValueError as e:
            log.warning(
                "config field %r failed to decrypt (%s); treating as unset", key, e
            )
            return None

    def as_document(self) -> ConfigDocument:
        with self._lock:
            out: dict[str, Any] = {}
            for key, value in self._values.items():
                field = _FIELDS_BY_KEY.get(key)
                if field is not None and field.sensitive and value is not None:
                    out[key] = REDACTED
                else:
                    out[key] = value
            return ConfigDocument.model_validate(out)

    def apply_patch(self, request: ConfigUpdateRequest) -> ConfigUpdateResult:
        applied: list[str] = []
        rejected: list[ConfigFieldError] = []
        pending_restart: list[str] = []

        # ConfigUpdateRequest is a free-form mapping; iterate its raw dict form.
        patch: dict[str, Any] = request.model_dump()

        with self._lock:
            for key, new_value in patch.items():
                field = _FIELDS_BY_KEY.get(key)
                if field is None:
                    rejected.append(ConfigFieldError(key=key, message="unknown field"))
                    continue

                err = _validate_value(field, new_value)
                if err is not None:
                    rejected.append(ConfigFieldError(key=key, message=err))
                    continue

                if new_value is None and field.default is not None:
                    self._values[key] = field.default
                else:
                    self._values[key] = new_value

                applied.append(key)
                if field.requiresRestart:
                    self._pending_restart.add(key)
                    pending_restart.append(key)

            if applied:
                self._write_locked()

            requires_restart = bool(self._pending_restart)
            return ConfigUpdateResult(
                applied=applied,
                rejected=rejected,
                requiresRestart=requires_restart,
                pendingRestart=sorted(self._pending_restart),
            )

    def get(self, key: str) -> Any:
        with self._lock:
            return self._values.get(key)

    def _write_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Build the on-disk dict by encrypting sensitive fields when
        # we have a key. Non-sensitive fields and (rarely) sensitive
        # ones during the no-master-key window pass through as
        # plaintext — matching v0.1 behavior so dev / pre-login flows
        # keep working.
        on_disk: dict[str, Any] = {}
        for key, value in self._values.items():
            field = _FIELDS_BY_KEY.get(key)
            if (
                field is not None
                and field.sensitive
                and value is not None
                and self._master_key is not None
                and isinstance(value, str)
                and value != ""
            ):
                envelope = security.seal(value, self._master_key)
                on_disk[key] = envelope.to_dict()
            else:
                on_disk[key] = value
        with self._path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(on_disk, f, sort_keys=True, default_flow_style=False)
