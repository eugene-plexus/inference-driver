"""Adapter that wraps the Anthropic Claude Code CLI as a subprocess.

Invocation pattern, verified by hand against `claude` v2.1.138:

    claude --print --output-format json [--model <id>] "<prompt>"

The CLI emits a single JSON envelope on stdout:

    {
      "type": "result",
      "subtype": "success" | "error_*",
      "is_error": false,
      "result": "<assistant response text>",
      "stop_reason": "end_turn" | "stop_sequence" | "max_tokens" | ...,
      "duration_ms": 1942,
      "duration_api_ms": 2622,
      "usage": { "input_tokens": ..., "output_tokens": ..., ... },
      ...
    }

Auth is handled by the CLI itself (OAuth / keychain / ANTHROPIC_API_KEY).
Do NOT pass `--bare`: it disables OAuth + keychain and forces API-key auth,
which breaks personal-subscription deployments — Eugene Plexus's primary
production mode.

Internally Claude Code is an *agent* (it routes through haiku for cheap
classification before sending to the user-facing model). The reported
`usage` aggregates tokens across all internal model calls. We surface
those totals as-is.
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
    EmbedResponse,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    Role,
    Usage,
)
from ._prompt import messages_to_prompt, text_content
from ._subprocess import CliError, run_cli, stream_cli_lines
from ._thinking import apply_thinking_mode, strip_thinking_blocks
from .base import DEFAULT_REQUEST_TIMEOUT_SECONDS, Chunk, warn_dropped_sampling

log = logging.getLogger(__name__)

_STOP_REASON_MAP = {
    "end_turn": FinishReason.stop,
    "stop_sequence": FinishReason.stop_sequence,
    "max_tokens": FinishReason.length,
}

# Hardcoded list of known-good Claude models supported by the Claude
# Code CLI. The CLI doesn't expose a `--list-models` so we can't
# discover this live the way openai_api can. Update by hand when
# Anthropic ships new chat models. Extended-thinking variants are
# excluded — Eugene Plexus IS the synthesis layer (see o-series
# rejection in openai_api).
_KNOWN_CLAUDE_MODELS: list[str] = [
    "claude-opus-4-7",
    "claude-sonnet-4-7",
    "claude-haiku-4-5",
    "claude-opus-4",
    "claude-sonnet-4",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
]


class ClaudeCodeCliEngine:
    backend_kind = BackendKind.claude_code_cli

    #: `--include-partial-messages` yields real `text_delta` events,
    #: verified against the CLI at v2.1.207.
    supports_streaming = True
    supports_tool_calling = False
    #: Declared, not inherited: `BackendEngine` is a Protocol, so a
    #: default on it reaches nothing. A CLI subscription has no
    #: embeddings surface at all -- there is no flag that makes the
    #: harness on the other side of the pipe return a vector.
    supports_embeddings = False
    """Claude Code CLI takes a prompt and returns prose; its own tool use
    is internal to the subprocess and is not exposed as OpenAI tool
    calls we could carry.

    Declared rather than inherited, because the default being
    False is a safety net and not a statement. A request carrying
    `tools` is refused with a reason; see `routes/generate.py`."""

    def __init__(
        self,
        *,
        binary_path: str = "claude",
        model_id: str | None = None,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        thinking_mode: str = "auto",
    ) -> None:
        self._binary_path = binary_path
        self._model_id = model_id
        self._timeout_seconds = timeout_seconds
        self._thinking_mode = thinking_mode or "auto"
        self._warned_sampling: set[str] = set()

    @classmethod
    def field_specs(cls, *, applicable_providers: list[str]) -> list[ConfigField]:
        show_when = ConfigFieldShowWhen(key="provider", equals=applicable_providers)
        return [
            ConfigField(
                key="claudeCodeCliPath",
                label="Claude Code CLI binary",
                description=(
                    "Where to find the `claude` command. Just `claude` "
                    "works if the binary is on your `PATH`; otherwise "
                    "give the full path (e.g. `/usr/local/bin/claude` "
                    "or `C:\\Users\\you\\AppData\\Local\\claude\\claude.exe`)."
                ),
                category="adapter",
                valueType=ConfigValueType.file_path,
                default="claude",
                requiresRestart=True,
                showWhen=show_when,
            ),
        ]

    @classmethod
    def from_config(cls, get: Any) -> ClaudeCodeCliEngine:
        return cls(
            binary_path=str(get("claudeCodeCliPath") or "claude"),
            model_id=get("modelId") or None,
            timeout_seconds=float(get("requestTimeoutSeconds") or DEFAULT_REQUEST_TIMEOUT_SECONDS),
            thinking_mode=str(get("thinkingMode") or "auto"),
        )

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        warn_dropped_sampling(
            request,
            engine="claude_code_cli",
            model_id=self._model_id or "the CLI's own default model",
            warned=self._warned_sampling,
        )
        # Apply the operator's thinkingMode by mutating the system
        # message before splitting / serialization. Claude Code itself
        # doesn't emit <think> tags inline (its thinking goes via the
        # native Messages API extended-thinking field), but routing
        # the directive through the system prompt is consistent with
        # the other engines and operators may want to suppress
        # reasoning emit in special test setups.
        messages = apply_thinking_mode(list(request.messages), self._thinking_mode)
        # Split system messages from the rest. Claude Code's --system-prompt
        # *replaces* its default system prompt entirely (which also disables
        # the "current working directory: ..." injection per the CLI's own
        # docs), giving us closer-to-raw-LLM behavior than passing system
        # text inside the user-message argv.
        system_messages = [m for m in messages if m.role == Role.system]
        other_messages = [m for m in messages if m.role != Role.system]
        system_prompt = "\n\n".join(text_content(m.content) for m in system_messages).strip()
        user_prompt = messages_to_prompt(other_messages)

        # The user-prompt transcript can contain newlines (paragraph breaks
        # in prior assistant messages, multi-line user input, etc). On
        # Windows, putting that on argv breaks: cmd.exe — which wraps
        # `claude.cmd` — treats a literal newline inside a quoted arg as a
        # command separator. Pipe via stdin instead. Claude Code reads
        # stdin under --print when no positional prompt is given.
        argv = self._build_argv(system_prompt=system_prompt)

        # DEBUG-level full-payload trace. CLI adapters flatten the
        # gateway's structured message list into a single labeled
        # transcript string before sending — the operator's copy-trace
        # shows the pre-flattening shape, this shows what actually
        # reaches the model.
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "claude_code_cli → argv:\n%s\n--- system prompt ---\n%s\n"
                "--- user prompt (stdin) ---\n%s",
                argv,
                system_prompt or "(empty)",
                user_prompt,
            )

        result = await run_cli(
            argv,
            timeout_seconds=self._timeout_seconds,
            stdin_input=user_prompt.encode("utf-8"),
        )

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "claude_code_cli ← exit=%d (%dms)\n--- stdout ---\n%s\n--- stderr ---\n%s",
                result.returncode,
                result.elapsed_ms,
                result.stdout.decode(errors="replace"),
                result.stderr.decode(errors="replace") or "(empty)",
            )

        if result.returncode != 0:
            raise CliError(
                f"claude exited {result.returncode}: "
                f"{result.stderr.decode(errors='replace').strip() or '<no stderr>'}"
            )

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise CliError(f"claude stdout was not valid JSON: {result.stdout[:200]!r}") from e

        if not isinstance(data, dict):
            raise CliError(f"claude JSON envelope was not an object: {data!r}")

        if data.get("is_error"):
            raise CliError(f"claude reported error: {data.get('result')!r}")

        content = data.get("result")
        if not isinstance(content, str):
            raise CliError(f"claude JSON missing string `result`: {data!r}")
        if self._thinking_mode == "off":
            content = strip_thinking_blocks(content)

        return GenerateResponse(
            content=content,
            finishReason=_STOP_REASON_MAP.get(
                str(data.get("stop_reason") or ""), FinishReason.stop
            ),
            usage=_usage_from_envelope(data.get("usage") or {}),
            requestId=request.requestId,
            backend=BackendKind.claude_code_cli,
            modelId=self._model_id,
            latencyMs=result.elapsed_ms,
        )

    async def stream(self, request: GenerateRequest) -> AsyncGenerator[Chunk, None]:
        """Token-by-token, over Claude Code's `stream-json` output.

        **The event shapes here were captured from the real CLI
        (v2.1.207), not remembered.** `--output-format stream-json
        --include-partial-messages --verbose` emits one JSON object per
        line, and the ones that matter are:

          * `{"type":"stream_event","event":{"type":"content_block_delta",
            "delta":{"type":"text_delta","text":"..."}}}` — the answer,
            arriving a fragment at a time. This is what we forward.
          * the same wrapper with `"delta":{"type":"thinking_delta",
            "thinking":"..."}` — extended reasoning, which must **not**
            be forwarded as content. Claude Code carries thinking in its
            own field rather than inline `<think>` tags, so the
            `ThinkingFilter` the HTTP engine needs has nothing to do
            here: dropping a delta by type is exact where a text filter
            would be a guess.
          * `{"type":"result","subtype":"success","result":"...",...}` —
            the terminal envelope, identical to the one `--output-format
            json` produces, so the final chunk is built by the same code
            path as `generate()` rather than a second parser that could
            disagree with it.

        `--verbose` is not optional: Claude Code refuses
        `--output-format stream-json` under `--print` without it.
        """
        warn_dropped_sampling(
            request,
            engine="claude_code_cli",
            model_id=self._model_id or "the CLI's own default model",
            warned=self._warned_sampling,
        )

        messages = apply_thinking_mode(list(request.messages), self._thinking_mode)
        system_messages = [m for m in messages if m.role == Role.system]
        other_messages = [m for m in messages if m.role != Role.system]
        system_prompt = "\n\n".join(text_content(m.content) for m in system_messages).strip()
        user_prompt = messages_to_prompt(other_messages)

        argv = self._build_argv(system_prompt=system_prompt, stream=True)
        if log.isEnabledFor(logging.DEBUG):
            log.debug("claude_code_cli → (stream) argv:\n%s", argv)

        started = time.perf_counter()
        emitted: list[str] = []
        envelope: dict[str, Any] | None = None

        async for line in stream_cli_lines(
            argv,
            timeout_seconds=self._timeout_seconds,
            stdin_input=user_prompt.encode("utf-8"),
        ):
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # Claude Code writes progress lines that are not JSON when
                # a terminal is attached; ignore rather than fail a
                # half-delivered answer.
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("type")
            if kind == "result":
                envelope = event
                continue
            if kind != "stream_event":
                continue
            inner = event.get("event") or {}
            if inner.get("type") != "content_block_delta":
                continue
            delta = inner.get("delta") or {}
            # Only text. `thinking_delta` and `signature_delta` are the
            # reasoning block and its attestation, neither of which is
            # the answer.
            if delta.get("type") != "text_delta":
                continue
            text = delta.get("text")
            if isinstance(text, str) and text:
                emitted.append(text)
                yield Chunk(text=text)

        if envelope is None:
            raise CliError("claude stream ended without a `result` envelope")
        if envelope.get("is_error"):
            raise CliError(f"claude reported error: {envelope.get('result')!r}")

        # Prefer the envelope's own `result` over our reassembly: it is
        # what `generate()` returns, so the two paths cannot disagree
        # about the final content. Fall back to what we streamed if the
        # envelope somehow carries none.
        content = envelope.get("result")
        if not isinstance(content, str):
            content = "".join(emitted)
        if self._thinking_mode == "off":
            content = strip_thinking_blocks(content)

        yield Chunk(
            done=True,
            result=GenerateResponse(
                content=content,
                finishReason=_STOP_REASON_MAP.get(
                    str(envelope.get("stop_reason") or ""), FinishReason.stop
                ),
                usage=_usage_from_envelope(envelope.get("usage") or {}),
                requestId=request.requestId,
                backend=BackendKind.claude_code_cli,
                modelId=self._model_id,
                latencyMs=int((time.perf_counter() - started) * 1000),
            ),
        )

    async def embed(self, inputs: list[str]) -> EmbedResponse:
        """Refused. A CLI subscription exposes no embeddings surface at
        all -- there is no flag that makes the harness on the other side
        of the pipe return a vector. `supports_embeddings` stays False,
        so the route rejects before reaching here; this exists so the
        engine satisfies the protocol rather than failing at import."""
        raise CliError(
            f"the {self.backend_kind.value} backend has no embeddings surface; "
            "point an inference-driver at a local engine or an OpenAI-compatible "
            "provider to serve embeddings."
        )

    async def context_window(self) -> int | None:
        """Unknown, and honestly so.

        A CLI subscription has no context window of its own to report:
        the harness on the other side of the pipe owns the window, picks
        the model, and manages its own compaction. Any number here would
        be a guess about a moving target, and the gateway treats an
        absent window as "do not promise one" -- which is the truth.
        """
        return None

    async def list_models(self) -> list[str]:
        # Claude Code CLI doesn't expose a list endpoint — return a
        # hardcoded set of currently-shipping chat models. All listed
        # models support tunable temperature.
        return list(_KNOWN_CLAUDE_MODELS)

    def _build_argv(self, *, system_prompt: str, stream: bool = False) -> list[str]:
        argv = [
            self._binary_path,
            "--print",
            "--output-format",
            "stream-json" if stream else "json",
        ]
        if stream:
            # `--verbose` is required: Claude Code refuses stream-json
            # under --print without it. `--include-partial-messages` is
            # what turns whole-message events into text deltas.
            argv += ["--include-partial-messages", "--verbose"]
        if system_prompt:
            argv += ["--system-prompt", system_prompt]
        if self._model_id:
            argv += ["--model", self._model_id]
        # No positional prompt; user prompt is piped via stdin.
        return argv


def _usage_from_envelope(usage: dict[str, Any]) -> Usage | None:
    """Best-effort mapping of claude's usage block to our Usage schema."""
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    cache_creation = usage.get("cache_creation_input_tokens", 0) or 0

    if input_tokens is None and output_tokens is None:
        return None

    prompt = (input_tokens or 0) + cache_read + cache_creation
    completion = output_tokens or 0
    return Usage(
        promptTokens=prompt,
        completionTokens=completion,
        totalTokens=prompt + completion,
    )
