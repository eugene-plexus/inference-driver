"""Provider registry — what subscriptions / services this driver can wrap.

A *provider* is the user-facing identity ("OpenAI", "xAI", "Claude
(Pro/Max)", "OpenRouter", "Local — Ollama"). An *engine* is the
protocol-level adapter (HTTP, CLI subprocess) — see `engines/`.
Many providers share one engine; this registry is where they're paired
up.

Adding a new OpenAI-compatible provider = one entry below. Adding a
new wire protocol = one new engine class + entries here that point at
it. Neither requires touching `config.py` or `app.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ._generated.models import BackendKind, ConfigField
from .engines.claude_code_cli import ClaudeCodeCliEngine
from .engines.codex_cli import CodexCliEngine
from .engines.openai_compat_http import (
    OPENAI_DENY_PATTERN,
    OpenAiCompatibleHttpEngine,
)


@dataclass(frozen=True)
class Provider:
    """One row of the registry — what shows up in the user's `provider` dropdown."""

    key: str
    """Stored as the `provider` config value. Stable, machine-friendly."""

    label: str
    """Shown in the UI dropdown — operator-facing language."""

    engine_class: Any
    """Which engine class talks to this provider's backend. Typed as
    `Any` rather than `type[BackendEngine]` because Python Protocol
    classes are invariant in `type[]` and concrete engine classes
    don't unify under a single Protocol subtype — runtime correctness
    is enforced by the engines all implementing the protocol shape."""

    engine_kwargs: dict[str, Any] = field(default_factory=dict)
    """Provider-specific kwargs forwarded to `engine_class.from_config(get, **engine_kwargs)`."""

    extra_field_specs: list[ConfigField] = field(default_factory=list)
    """Extra config fields specific to this single provider (e.g. the custom
    provider needs a user-supplied `baseUrl` — most providers don't)."""


# `baseUrl` is shown only for the custom provider — for named providers,
# the URL is implicit in the registry entry. Defined here so multiple
# providers (none today, but room for future "BYO compat URL" variants)
# could reuse the same field declaration if needed.
def _custom_base_url_field() -> list[ConfigField]:
    from ._generated.models import ConfigFieldShowWhen, ConfigValueType

    return [
        ConfigField(
            key="baseUrl",
            label="Base URL",
            description=(
                "HTTP base of your OpenAI-compatible endpoint (e.g. "
                "`https://my-vllm.example.com`). The driver appends "
                "`/v1/chat/completions` automatically. Only used by the "
                "Custom OpenAI-compatible provider — for named providers "
                "(OpenAI, xAI, OpenRouter, …) the URL is built in."
            ),
            category="adapter",
            valueType=ConfigValueType.url,
            requiresRestart=True,
            required=True,
            showWhen=ConfigFieldShowWhen(key="provider", equals="openai_compat_custom"),
        ),
    ]


# Future xAI deny pattern lives here for symmetry — Grok models that
# reject temperature would land here. Empty for now since Grok-2 / 3 / 4
# all support temperature; revisit if xAI ships a reasoning-only variant
# that doesn't.
_XAI_DENY_PATTERN: re.Pattern[str] | None = None

# Provider registry. The dict order is the order entries appear in the
# UI dropdown — group personal-subscription engines first, then OpenAI
# proper, then OpenAI-compatible third-party providers, then local
# options, then the BYO escape hatch.
PROVIDERS: dict[str, Provider] = {
    "claude_subscription": Provider(
        key="claude_subscription",
        label="Claude (Pro/Max subscription via Claude Code CLI)",
        engine_class=ClaudeCodeCliEngine,
    ),
    "chatgpt_subscription": Provider(
        key="chatgpt_subscription",
        label="ChatGPT (subscription via Codex CLI)",
        engine_class=CodexCliEngine,
    ),
    "openai": Provider(
        key="openai",
        label="OpenAI API",
        engine_class=OpenAiCompatibleHttpEngine,
        engine_kwargs={
            "default_base_url": "https://api.openai.com",
            "deny_pattern": OPENAI_DENY_PATTERN,
            "backend_kind": BackendKind.openai_api,
        },
    ),
    "xai": Provider(
        key="xai",
        label="xAI (Grok)",
        engine_class=OpenAiCompatibleHttpEngine,
        engine_kwargs={
            "default_base_url": "https://api.x.ai",
            "deny_pattern": _XAI_DENY_PATTERN,
            "backend_kind": BackendKind.openai_compat_http,
        },
    ),
    "openrouter": Provider(
        key="openrouter",
        label="OpenRouter",
        engine_class=OpenAiCompatibleHttpEngine,
        engine_kwargs={
            "default_base_url": "https://openrouter.ai/api",
            # OpenRouter proxies many providers — we don't try to
            # second-guess which models are temp-tunable here. If a
            # specific underlying model rejects temperature, the
            # request just 400s and the new error-propagation surfaces
            # the upstream message clearly.
            "deny_pattern": None,
            "backend_kind": BackendKind.openai_compat_http,
        },
    ),
    "minimax": Provider(
        key="minimax",
        label="MiniMax",
        engine_class=OpenAiCompatibleHttpEngine,
        engine_kwargs={
            "default_base_url": "https://api.minimax.io",
            "deny_pattern": None,
            "backend_kind": BackendKind.openai_compat_http,
        },
    ),
    "ollama_local": Provider(
        key="ollama_local",
        label="Local — Ollama",
        engine_class=OpenAiCompatibleHttpEngine,
        engine_kwargs={
            "default_base_url": "http://127.0.0.1:11434",
            "deny_pattern": None,
            "backend_kind": BackendKind.openai_compat_http,
            # Ollama doesn't require auth on its local endpoint; the
            # apiKey field exists but is optional. Without this flag,
            # engine construction fails on a blank key even though the
            # request would have worked.
            "auth_required": False,
            # Ollama only lists what the operator pulled. Disable the
            # chat-prefix heuristic — model ids like
            # `huihui_ai/dolphin3-abliterated:...` don't match it and
            # would be invisibly hidden from the dropdown.
            "filter_models": False,
        },
    ),
    "lmstudio_local": Provider(
        key="lmstudio_local",
        label="Local — LM Studio",
        engine_class=OpenAiCompatibleHttpEngine,
        engine_kwargs={
            "default_base_url": "http://127.0.0.1:1234",
            "deny_pattern": None,
            "backend_kind": BackendKind.openai_compat_http,
            "auth_required": False,
            "filter_models": False,
        },
    ),
    "openai_compat_custom": Provider(
        key="openai_compat_custom",
        label="Custom OpenAI-compatible URL",
        engine_class=OpenAiCompatibleHttpEngine,
        engine_kwargs={
            # No default — user must supply baseUrl.
            "default_base_url": None,
            "deny_pattern": None,
            "backend_kind": BackendKind.openai_compat_http,
        },
        extra_field_specs=_custom_base_url_field(),
    ),
}


def providers_using(engine_class: Any) -> list[str]:
    """Return the provider keys that route to a given engine class.

    Used by the schema builder to compute `showWhen` predicates — an
    engine's config fields are visible only when the selected provider
    is in this list.
    """
    return [p.key for p in PROVIDERS.values() if p.engine_class is engine_class]


def collect_extra_field_specs() -> list[ConfigField]:
    """Provider-specific config fields (the union across all providers).
    Each carries its own `showWhen` so the UI hides what's irrelevant."""
    out: list[ConfigField] = []
    for provider in PROVIDERS.values():
        out.extend(provider.extra_field_specs)
    return out


def get_provider(key: str) -> Provider:
    """Look up a provider by key, raising KeyError with a helpful list
    on unknown keys."""
    try:
        return PROVIDERS[key]
    except KeyError as e:
        valid = ", ".join(PROVIDERS.keys())
        raise KeyError(f"unknown provider {key!r}; valid: {valid}") from e
