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
import math
import os
import threading
from pathlib import Path
from typing import Any

import yaml

from . import _private_files, security
from ._generated.models import (
    ConfigDocument,
    ConfigField,
    ConfigFieldError,
    ConfigFieldShowWhen,
    ConfigSchema,
    ConfigUpdateRequest,
    ConfigUpdateResult,
    ConfigValueType,
)
from .engines.base import DEFAULT_REQUEST_TIMEOUT_SECONDS
from .engines.claude_code_cli import ClaudeCodeCliEngine
from .engines.codex_cli import CodexCliEngine
from .engines.elevenlabs_http import ElevenLabsHttpEngine
from .engines.gemini_api import GeminiApiEngine
from .engines.openai_compat_http import OpenAiCompatibleHttpEngine
from .engines.systemone_http import SystemOneHttpEngine
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
            '"gpt-4o", "claude-opus-4-7", "grok-2", '
            '"llama3.1:70b"). The list below is discovered from the '
            "selected provider. **Leave it empty to use every model** an "
            "API provider or local server lists (OpenRouter, OpenAI, "
            "Ollama, LM Studio, a custom URL): this driver then serves "
            "them all, each published as <this driver's name>/<model>. "
            "For the Claude and ChatGPT subscriptions, empty falls back "
            "to the CLI's own default model."
        ),
        category="adapter",
        valueType=ConfigValueType.string,
        requiresRestart=True,
    )


def _upstream_modelid_field() -> ConfigField:
    """The backend half of the model-identity split.

    Almost every install leaves this unset and the backend is asked for
    `modelId` verbatim, exactly as before the field existed. It exists
    for backends whose served name is not ours to choose — an MLX
    runtime answers only to upstream's `default_model` sentinel — where
    the public alias in `modelId` must stay the routing key while the
    wire asks for something else.
    """
    return ConfigField(
        key="upstreamModelId",
        label="Upstream model name",
        description=(
            "What this driver actually asks its backend for, when that "
            "differs from the public Model above. Leave empty to send "
            "the Model value verbatim (the normal case). Supervised MLX "
            'runtimes set this to "default_model" automatically, because '
            "mlx_lm.server has no flag to serve a chosen name. Never "
            "used for routing, and never reported on responses — callers "
            "always see the public Model."
        ),
        category="adapter",
        valueType=ConfigValueType.string,
        requiresRestart=True,
    )


def _common_fields() -> list[ConfigField]:
    """Provider-agnostic fields — logging and timeouts.

    The bind port deliberately is NOT in this list. Ports are owned by
    the agent topology (`agent.yaml`), passed to spawned children
    via `EUGENE_PLEXUS_DRIVER_BIND_PORT`. Two sources of truth on `port`
    was an OpenClaw-style trap waiting to bite — the agent spawns at
    one port, the driver's own config says another, and the gateway
    can't reach the driver. Now there's one source.
    """
    return [
        ConfigField(
            key="backendLocality",
            label="Endpoint trust",
            description=(
                "For custom, Ollama and LM Studio endpoints, explicitly confirm whether "
                "inference stays inside your trusted local installation. A local URL is "
                "not proof: a proxy may forward to cloud. Unknown endpoints cannot serve "
                "local-only keys. Supervised runtimes are classified local automatically; "
                "cloud APIs and subscription CLIs always remain external. Ollama sends its "
                "`:cloud` models to ollama.com, and this setting covers every model the "
                "backend lists: before confirming an Ollama local, start it with "
                "OLLAMA_NO_CLOUD=1 or pull no cloud models. Restart applies this setting to "
                "the active engine."
            ),
            category="adapter",
            valueType=ConfigValueType.enum,
            enumValues=["unknown", "local", "external"],
            enumLabels=["Unknown", "Confirmed local", "External / cloud"],
            default="unknown",
            requiresRestart=True,
        ),
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
                "CLI flattening), which the gateway's copy-trace "
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
                "How long this driver waits on a single call to its "
                "backend. A backstop: the gateway holds the deadline "
                "that normally decides, and this sits one minute above "
                "it so the gateway's is the one that fires. Raise both "
                "if a model on the processor needs longer. When it does "
                "fire, the request is not retried on another backend — "
                "the next one would take the same time on the same "
                "prompt — so it comes back as one 504 saying so."
            ),
            category="network",
            valueType=ConfigValueType.duration,
            # **Above the gateway's 600 s on purpose (R2.5).** As found,
            # this was 120 s against the gateway's 180 s, so the driver
            # always fired first and the gateway's knob governed
            # nothing: an operator who raised the documented setting saw
            # no change, and the failure arrived as an anonymous
            # transport error that the cascade then recomputed on every
            # replica and every tier.
            default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
            minimum=5,
            maximum=3600,
            requiresRestart=True,
        ),
    ]


def _add_field(out: list[ConfigField], field: ConfigField) -> None:
    """Add `field`, or widen the one already there with its key.

    Two engine classes can read the same key -- ElevenLabs' `apiKey` and the
    OpenAI-compatible engine's (P3a) -- and a key is one field in the
    schema, so the second one's providers are added to the first one's
    `showWhen` rather than appended as a duplicate the UI would render twice.
    The second's description replaces the first's, because it is written to
    cover both.
    """
    for index, existing in enumerate(out):
        if existing.key != field.key:
            continue
        if existing.showWhen is not None and field.showWhen is not None:
            merged = [*existing.showWhen.equals, *field.showWhen.equals]
            widened = existing.showWhen.model_copy(update={"equals": list(dict.fromkeys(merged))})
            out[index] = existing.model_copy(
                update={"showWhen": widened, "description": field.description}
            )
        return
    out.append(field)


def _build_fields() -> list[ConfigField]:
    """Compose the full FIELDS list from the registry. Order:
    provider -> per-engine fields (API key, CLI paths) -> per-provider
    extras (custom baseUrl) -> modelId -> common.

    Why modelId comes last among adapter fields: most providers need a
    valid API key (or local URL) before their model list is reachable,
    so the UI shows credentials first, then the model picker."""
    out: list[ConfigField] = [_provider_field()]
    seen: set[type] = set()
    for engine_cls in (
        ClaudeCodeCliEngine,
        CodexCliEngine,
        OpenAiCompatibleHttpEngine,
        SystemOneHttpEngine,
        ElevenLabsHttpEngine,
        GeminiApiEngine,
    ):
        if engine_cls in seen:
            continue
        seen.add(engine_cls)
        applicable = providers_using(engine_cls)
        if not applicable:
            continue
        for field in engine_cls.field_specs(applicable_providers=applicable):
            _add_field(out, field)
    out.extend(collect_extra_field_specs())
    out.append(_modelid_field())
    out.append(_upstream_modelid_field())
    out.extend(_common_fields())
    return [_shown_where_read(f) for f in out]


def _providers(*engines: Any) -> list[str]:
    return [p for engine in engines for p in providers_using(engine)]


def _read_by() -> dict[str, list[str] | None]:
    """Which providers read each key, from the engines that read it.

    **A field is shown wherever it is read** (settings never lie,
    2026-09-30). `baseUrl` and `runtimeName` were shown only for the two
    custom providers while every OpenAI-compatible engine reads both -- so
    a stale address, invisible on the page, sent an account's key to it --
    and `apiKey` was hidden for TypeSafe, which refuses to run without one.
    `None` means every provider. A key not listed keeps its own condition.
    """
    http = _providers(
        OpenAiCompatibleHttpEngine, ElevenLabsHttpEngine, SystemOneHttpEngine, GeminiApiEngine
    )
    follows_runtimes = _providers(OpenAiCompatibleHttpEngine, SystemOneHttpEngine)
    everything_but_speech = [p for p in PROVIDERS if p not in providers_using(ElevenLabsHttpEngine)]
    no_upstream_name = [
        p for p in everything_but_speech if p not in providers_using(GeminiApiEngine)
    ]
    thinks = _providers(ClaudeCodeCliEngine, CodexCliEngine, OpenAiCompatibleHttpEngine)
    return {
        "baseUrl": http,
        "runtimeName": follows_runtimes,
        "apiKey": http,
        "catalogueRefreshMinutes": _providers(
            OpenAiCompatibleHttpEngine, ElevenLabsHttpEngine, GeminiApiEngine
        ),
        # Gemini reads the same stall clock for its streams.
        "streamStallSeconds": _providers(OpenAiCompatibleHttpEngine, GeminiApiEngine),
        # ElevenLabs reads neither: it serves the voices its account lists.
        "modelId": everything_but_speech,
        # Gemini serves a model by the id Google lists: there is no split.
        "upstreamModelId": no_upstream_name,
        "thinkingMode": thinks,
        # Honoured only for a backend that could be either; every other
        # provider is external (or, fronting a runtime here, local) whatever
        # this says.
        "backendLocality": [
            "ollama_local",
            "lmstudio_local",
            "openai_compat_custom",
            "systemone_custom",
        ],
    }


def _shown_where_read(field: ConfigField) -> ConfigField:
    readers = _read_by()
    if field.key not in readers:
        return field
    providers = readers[field.key]
    show = (
        None
        if providers is None
        else ConfigFieldShowWhen(key="provider", equals=[p for p in PROVIDERS if p in providers])
    )
    return field.model_copy(update={"showWhen": show})


# Schema for inference-driver's config surface. Built dynamically from
# the provider registry — the order here is the order the UI renders.
FIELDS: list[ConfigField] = _build_fields()
# NOTE on what's NOT in this schema:
# Temperature, max-tokens, stop sequences and other parameters that alter
# LLM output are owned by the *caller* (the gateway) and arrive on
# every `GenerateRequest`. The driver applies what it's given and never
# substitutes a local default. In v0.2+ the gateway's NT system will
# modulate these per-request — placing defaults here would make that
# layering invisible and a future NT signal trivially overridable from a
# config file.

_FIELDS_BY_KEY: dict[str, ConfigField] = {f.key: f for f in FIELDS}


#: Which environment variable each engine falls back to for its key.
_KEY_ENV: dict[Any, str] = {
    OpenAiCompatibleHttpEngine: "OPENAI_API_KEY",
    ElevenLabsHttpEngine: "ELEVENLABS_API_KEY",
    GeminiApiEngine: "GEMINI_API_KEY",
}
#: Handed to a companion driver by the agent that declared it, naming the
#: keys that agent rewrites (settings never lie, 2026-09-30).
MANAGED_KEYS_ENV = "EUGENE_PLEXUS_DRIVER_MANAGED_KEYS"
MANAGED_BY = (
    "Set by this machine's agent from the runtime this driver fronts, and rewritten at "
    "its next start: change the runtime instead (the model's profile, or Backends)."
)


def managed_keys() -> frozenset[str]:
    raw = os.environ.get(MANAGED_KEYS_ENV, "")
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


def _unset_facts(key: str, values: dict[str, Any]) -> dict[str, Any]:
    """What an unset `key` does for the provider this driver runs now."""
    provider = PROVIDERS.get(str(values.get("provider") or ""))
    if provider is None:
        return {}
    engine = provider.engine_class
    if key == "baseUrl":
        url = provider.engine_kwargs.get("default_base_url")
        if url:
            return {
                "unsetMeans": f"Not set: uses {provider.label}'s own address, {url}.",
                "unsetResolvesTo": url,
            }
        return {
            "unsetMeans": "Not set: requests go to the runtime named above, and without one "
            "this driver has nowhere to send them."
        }
    if key == "runtimeName":
        return {"unsetMeans": "Not set: requests go to the address below, or the provider's own."}
    if key == "apiKey":
        env = _KEY_ENV.get(engine)
        if env and os.environ.get(env):
            # Presence only -- never the key (Troy, 2026-09-30).
            return {"unsetMeans": f"Not set here: uses the {env} in this machine's environment."}
        if provider.engine_kwargs.get("auth_required", True):
            return {"unsetMeans": f"Not set: {provider.label} refuses requests without a key."}
        return {"unsetMeans": "Not set: no key is sent."}
    if key == "modelId":
        if engine in (ClaudeCodeCliEngine, CodexCliEngine):
            return {"unsetMeans": "Not set: the CLI's own default model."}
        if engine is SystemOneHttpEngine:
            return {"unsetMeans": "Not set: asks for kev-latest.", "unsetResolvesTo": "kev-latest"}
        return {
            "unsetMeans": "Not set: this driver serves every model the account lists, less "
            "any the model filters leave out."
        }
    if key == "upstreamModelId":
        return {"unsetMeans": "Not set: the same as the model id."}
    if key == "decisionMaxConcurrent":
        return {"unsetMeans": "Not set: no limit is advertised."}
    return {}


def as_schema(
    *,
    available_models: list[str] | None = None,
    values: dict[str, Any] | None = None,
    pending: dict[str, Any] | None = None,
    managed: frozenset[str] = frozenset(),
) -> ConfigSchema:
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
        # Both halves of the identity split get the discovered list: the
        # names come from the backend, so they are upstream names — but
        # in the common no-split case `modelId` IS the upstream name,
        # and dropping its suggestions would regress every existing
        # install to teach a field almost nobody sets.
        fields = [
            _with_model_suggestions(f, available_models)
            if f.key in ("modelId", "upstreamModelId")
            else f
            for f in fields
        ]
    live: list[ConfigField] = []
    for field in fields:
        update: dict[str, Any] = {}
        if values is not None and values.get(field.key) in (None, ""):
            update.update(_unset_facts(field.key, values))
        if field.key in managed:
            update["managedBy"] = MANAGED_BY
        if pending and field.key in pending:
            update["pendingRestart"] = True
            if not field.sensitive:
                update["inEffect"] = pending[field.key]
        live.append(field.model_copy(update=update) if update else field)
    return ConfigSchema(
        component="inference-driver",
        fields=live,
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


def _is_unset(field: ConfigField, value: Any) -> bool:
    """None, or an empty secret: an empty key is no key, and it used to
    read `"<redacted>"` -- a key saved that never was."""
    if value is None:
        return True
    return (
        field.valueType == ConfigValueType.secret and isinstance(value, str) and not value.strip()
    )


def _defaults() -> dict[str, Any]:
    return {f.key: f.default for f in FIELDS if f.default is not None}


def _validate_value(field: ConfigField, value: Any) -> str | None:
    """Return None if valid, otherwise an error message."""
    if value is None:
        return None  # null clears to default

    vt = field.valueType

    if vt in (
        ConfigValueType.string,
        ConfigValueType.url,
        ConfigValueType.file_path,
        # A runtime's name. The UI sources a dropdown from the agent's
        # `/v1/runtimes`; the wire value is the plain name, and it is not
        # checked against the agent here — the engine resolves it at
        # construction and reports a missing runtime by name.
        ConfigValueType.runtime_name,
    ):
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
        if not math.isfinite(value):
            # JSON's `NaN` parses and compares false with everything, so it
            # passed the range check.
            return "must be a finite number"
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

    if vt == ConfigValueType.string_list:
        if not isinstance(value, list):
            return f"expected a list of strings, got {type(value).__name__}"
        for i, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                return f"entry {i} must be a non-empty string"
            if len(item) > 256:
                return f"entry {i} is longer than 256 characters"
        if len(value) > 100:
            return "at most 100 entries"
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
        # What this process runs on for every `requiresRestart` field, as
        # loaded; a field is pending while its saved value differs.
        self._started: dict[str, Any] = dict(self._values)
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
                    field = _FIELDS_BY_KEY.get(k)
                    if field is None:
                        continue
                    value = self._decrypt_loaded(k, v)
                    # A null in the file is the default, as a PATCH of null
                    # is; an empty secret is no secret.
                    if _is_unset(field, value):
                        value = field.default
                    merged[k] = value
                self._values = merged
            else:
                self._values = _defaults()
                self._write_locked()
            self._started = dict(self._values)

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
            log.warning("config field %r failed to decrypt (%s); treating as unset", key, e)
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

                if key in managed_keys():
                    rejected.append(ConfigFieldError(key=key, message=MANAGED_BY))
                    continue

                if _is_unset(field, new_value):
                    new_value = None
                if new_value is None and field.default is not None:
                    self._values[key] = field.default
                else:
                    self._values[key] = new_value

                applied.append(key)
                # This PATCH's keys, while they differ from what the
                # process runs on: the contract's subset of `applied`. It
                # was every restart key saved since start.
                if field.requiresRestart and self._values.get(key) != self._started.get(key):
                    pending_restart.append(key)

            if applied:
                self._write_locked()

            return ConfigUpdateResult(
                applied=applied,
                rejected=rejected,
                requiresRestart=bool(pending_restart),
                pendingRestart=pending_restart,
            )

    def pending_restart(self) -> dict[str, Any]:
        """`requiresRestart` fields whose saved value is not the one this
        process runs on, with the value it runs on."""
        with self._lock:
            return {
                f.key: self._started.get(f.key)
                for f in FIELDS
                if f.requiresRestart and self._values.get(f.key) != self._started.get(f.key)
            }

    def started(self, key: str) -> Any:
        """The value this process started with -- what a restart field is
        in effect as."""
        with self._lock:
            return self._started.get(key)

    def values(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._values)

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
        # 0600 and replaced rather than rewritten: that plaintext window
        # above is an API key on disk, and a truncate-then-write killed
        # halfway is a driver that boots having forgotten it. See
        # `_private_files`.
        _private_files.write_private_text(
            self._path, yaml.safe_dump(on_disk, sort_keys=True, default_flow_style=False)
        )
