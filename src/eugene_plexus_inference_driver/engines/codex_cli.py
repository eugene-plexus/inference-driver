"""Adapter that wraps the OpenAI Codex CLI as a subprocess.

Invocation pattern, verified by hand against `codex-cli` v0.130.0:

    codex exec --json --skip-git-repo-check --ephemeral
               --sandbox read-only [--model <id>] "<prompt>"

The CLI emits a JSONL stream on stdout:

    {"type":"thread.started","thread_id":"..."}
    {"type":"turn.started"}
    {"type":"item.completed","item":{"type":"agent_message","text":"<reply>"}}
    {"type":"turn.completed","usage":{"input_tokens":...,"output_tokens":...}}

We collect every `item.completed` whose `item.type == "agent_message"` for
text content, and pull the final `usage` from `turn.completed`.

Sandbox is `read-only` and `--ephemeral` so codex doesn't try to mutate the
working tree or persist session state. `--skip-git-repo-check` allows the
inference-driver process to run outside a repo.

## Known limitations vs the Claude Code adapter

Codex CLI does not expose an equivalent of Claude Code's
`--system-prompt` flag, so we have no way to suppress its built-in
system prompt or its cwd-injection. Practical consequences:

- **Persona override**: any system messages in the request are folded
  into the user prompt by `messages_to_prompt`, then prefixed with the
  CLI's own (coding-agent-flavored) system prompt. The Eugene persona
  reads as user content rather than a directive, which the smoke test
  (2026-05-09) showed Codex tends to ignore.
- **cwd identity leak**: Codex includes the cwd in its prompt context.
  Run the driver from a neutral directory if running this adapter is
  important for blind-bicameral integrity.

For self-hosted / personal-subscription deployments, the recommended
workaround is to use the `openai_api` adapter for the OpenAI-side
hemisphere instead — it gives full persona + cwd control. The Codex CLI
adapter is retained for the case where the operator's only OpenAI
access is via a Codex-eligible subscription.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from .._generated.models import (
    BackendKind,
    ConfigField,
    ConfigFieldShowWhen,
    ConfigValueType,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    Usage,
)
from ._prompt import messages_to_prompt
from ._subprocess import CliError, run_cli, stream_cli_lines
from ._thinking import apply_thinking_mode, strip_thinking_blocks
from .base import Chunk

log = logging.getLogger(__name__)

# Models that Codex CLI is known to surface to the user. The CLI itself
# decides which model to call based on its own config and the active
# ChatGPT subscription tier — our `modelId` field is informational only,
# not directly bound to the backend call. Keep the list short and
# deliberately exclude reasoning models that wouldn't pass our
# temperature-controllability bar (see openai_api).
_KNOWN_CODEX_MODELS: list[str] = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4-turbo",
]


class CodexCliEngine:
    backend_kind = BackendKind.codex_cli

    #: Codex's JSONL is live, but its answer arrives as one whole
    #: `agent_message` -- incremental delivery, not per-token. See
    #: `stream` for why this is False rather than optimistic.
    supports_streaming = False
    supports_tool_calling = False
    """Codex CLI takes a prompt and returns prose; it exposes no
    OpenAI-shaped tool-call surface to carry.

    Declared rather than inherited, because the default being
    False is a safety net and not a statement. A request carrying
    `tools` is refused with a reason; see `routes/generate.py`."""

    def __init__(
        self,
        *,
        binary_path: str = "codex",
        model_id: str | None = None,
        timeout_seconds: float = 120.0,
        thinking_mode: str = "auto",
    ) -> None:
        self._binary_path = binary_path
        self._model_id = model_id
        self._timeout_seconds = timeout_seconds
        self._thinking_mode = thinking_mode or "auto"

    @classmethod
    def field_specs(cls, *, applicable_providers: list[str]) -> list[ConfigField]:
        show_when = ConfigFieldShowWhen(key="provider", equals=applicable_providers)
        return [
            ConfigField(
                key="codexCliPath",
                label="Codex CLI binary",
                description=(
                    "Where to find the `codex` command. Just `codex` "
                    "works if the binary is on your `PATH`; otherwise "
                    "give the full path."
                ),
                category="adapter",
                valueType=ConfigValueType.file_path,
                default="codex",
                requiresRestart=True,
                showWhen=show_when,
            ),
        ]

    @classmethod
    def from_config(cls, get: Any) -> CodexCliEngine:
        return cls(
            binary_path=str(get("codexCliPath") or "codex"),
            model_id=get("modelId") or None,
            timeout_seconds=float(get("requestTimeoutSeconds") or 120),
            thinking_mode=str(get("thinkingMode") or "auto"),
        )

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        # Codex CLI's own system prompt overrides ours (see module
        # docstring's Persona-override note), but applying the
        # thinkingMode directive still helps in practice — Codex
        # follows user-content instructions and reasoning-tag models
        # surface as Codex backends from time to time.
        messages = apply_thinking_mode(list(request.messages), self._thinking_mode)
        flattened_prompt = messages_to_prompt(messages)
        argv = self._build_argv(flattened_prompt)

        # DEBUG-level full-payload trace. CLI adapters flatten the
        # gateway's structured message list into a single labeled
        # transcript string before sending — the operator's copy-trace
        # shows the pre-flattening shape, this shows what actually
        # reaches the model.
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "codex_cli → argv:\n%s\n--- flattened prompt ---\n%s",
                argv,
                flattened_prompt,
            )

        result = await run_cli(argv, timeout_seconds=self._timeout_seconds)

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "codex_cli ← exit=%d (%dms)\n--- stdout (JSONL) ---\n%s\n--- stderr ---\n%s",
                result.returncode,
                result.elapsed_ms,
                result.stdout.decode(errors="replace"),
                result.stderr.decode(errors="replace") or "(empty)",
            )

        if result.returncode != 0:
            raise CliError(
                f"codex exited {result.returncode}: "
                f"{result.stderr.decode(errors='replace').strip() or '<no stderr>'}"
            )

        text_parts: list[str] = []
        usage_event: dict[str, Any] | None = None
        saw_any_event = False

        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            saw_any_event = True
            event_type = event.get("type")
            if event_type == "item.completed":
                item = event.get("item") or {}
                if isinstance(item, dict) and item.get("type") == "agent_message":
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            elif event_type == "turn.completed":
                usage_event = event.get("usage")

        if not saw_any_event:
            raise CliError(
                f"codex emitted no parseable events. stdout head: {result.stdout[:200]!r}"
            )

        if not text_parts:
            raise CliError("codex completed without producing an agent_message")

        content = "".join(text_parts)
        if self._thinking_mode == "off":
            content = strip_thinking_blocks(content)

        return GenerateResponse(
            content=content,
            finishReason=FinishReason.stop,
            usage=_usage_from_codex(usage_event) if usage_event else None,
            requestId=request.requestId,
            backend=BackendKind.codex_cli,
            modelId=self._model_id,
            latencyMs=result.elapsed_ms,
        )

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[Chunk, None]:
        """Incremental, but at message granularity rather than per token.

        **`supports_streaming` is False for this engine and that is the
        honest answer.** Codex's `--json` really is a live JSONL feed,
        and reading it with `stream_cli_lines` instead of buffering the
        whole subprocess is a genuine improvement: an answer reaches the
        client when Codex emits it rather than when the process exits.
        But the event that carries the answer is `item.completed` with
        `item.type == "agent_message"`, and its `text` is the **whole**
        message. There is no per-token delta on this wire to forward.

        The stream contract allows exactly this — *"backends that don't
        support native streaming MAY emit the entire response as a
        single `token` event followed by `done`"* — so the endpoint
        works and the capability flag stays truthful. Claiming True here
        would make the flag useless for the one thing it exists for:
        telling a UI whether to expect progressive output.

        Upgrade path, unverified: if Codex gains a delta event, forward
        it here and flip the flag. It could not be checked while writing
        this, because this box's Codex auth is stale (`refresh_token_reused`)
        and every request 401s.
        """
        messages = apply_thinking_mode(list(request.messages), self._thinking_mode)
        flattened_prompt = messages_to_prompt(messages)
        argv = self._build_argv(flattened_prompt)

        if log.isEnabledFor(logging.DEBUG):
            log.debug("codex_cli → (stream) argv:\n%s", argv)

        started = time.perf_counter()
        text_parts: list[str] = []
        usage_event: dict[str, Any] | None = None
        saw_any_event = False

        async for raw_line in stream_cli_lines(argv, timeout_seconds=self._timeout_seconds):
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Codex interleaves non-JSON log lines on stdout when its
                # auth is unhappy; skip rather than fail the turn.
                continue
            if not isinstance(event, dict):
                continue
            saw_any_event = True
            event_type = event.get("type")
            if event_type == "item.completed":
                item = event.get("item") or {}
                if isinstance(item, dict) and item.get("type") == "agent_message":
                    text = item.get("text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
                        yield Chunk(text=text)
            elif event_type == "turn.completed":
                usage_event = event.get("usage")

        if not saw_any_event:
            raise CliError("codex emitted no parseable events")
        if not text_parts:
            raise CliError("codex produced no agent_message item")

        content = "".join(text_parts)
        if self._thinking_mode == "off":
            content = strip_thinking_blocks(content)

        yield Chunk(
            done=True,
            result=GenerateResponse(
                content=content,
                finishReason=FinishReason.stop,
                usage=_usage_from_codex(usage_event) if usage_event else None,
                requestId=request.requestId,
                backend=BackendKind.codex_cli,
                modelId=self._model_id,
                latencyMs=int((time.perf_counter() - started) * 1000),
            ),
        )

    async def list_models(self) -> list[str]:
        # Codex CLI doesn't expose a list endpoint. Return a hardcoded
        # set; note that `modelId` is informational here — the CLI's
        # actual model selection is driven by Codex's own config.
        return list(_KNOWN_CODEX_MODELS)

    def _build_argv(self, prompt: str) -> list[str]:
        argv = [
            self._binary_path,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--ephemeral",
            "--sandbox",
            "read-only",
        ]
        if self._model_id:
            argv += ["--model", self._model_id]
        argv.append(prompt)
        return argv


def _usage_from_codex(usage: dict[str, Any] | None) -> Usage | None:
    if not usage:
        return None
    prompt = (usage.get("input_tokens") or 0) + (usage.get("cached_input_tokens") or 0)
    completion = (usage.get("output_tokens") or 0) + (usage.get("reasoning_output_tokens") or 0)
    if prompt == 0 and completion == 0:
        return None
    return Usage(
        promptTokens=prompt,
        completionTokens=completion,
        totalTokens=prompt + completion,
    )
