"""Adapter that wraps the Anthropic Claude Code CLI as a subprocess.

Invocation pattern, verified by hand against `claude` v2.1.138:

    claude --print --output-format json [--model <id>] "<prompt>"

and since 2026-10-03 always with `_NO_TOOLS` (no tools, `dontAsk`, no MCP
servers, no saved session) and without the routing variables in
`_ROUTING_ENV`: it is a text backend, never an agent on the driver host.

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

import contextlib
import json
import logging
import os
import tempfile
import time
from collections.abc import AsyncGenerator, AsyncIterator, Iterator
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
    Stage,
    StreamProgress,
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

# Suggestions for the Model field, nothing more: the CLI has no
# `--list-models`, and `--model` takes any full name. Update by hand when
# Anthropic ships new models. Checked 2026-10-03 against Claude Code
# 2.1.288's changelog (Opus 5.5, Sonnet 5.5 and Fable 5.1 are each the
# default of their family) and 2.1.283's binary, which knows every id
# below but Sonnet 5.5 (added in 2.1.284 and accepted by name all the
# same, as any full name is).
_KNOWN_CLAUDE_MODELS: list[str] = [
    "claude-opus-5-5",
    "claude-sonnet-5-5",
    "claude-fable-5-1",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-haiku-4-5",
]


#: **A text backend runs no tools.** Claude Code here answers prompts that
#: any client key can send, on the driver's host, as the driver's account.
#: Before 2026-10-03 it was given its default tools and no permission
#: mode, and Claude Code 2.1.285 starts `-p` in auto mode "on third-party
#: providers, or with telemetry off" -- so a prompt could have run Bash or
#: Edit here, approved by a classifier rather than a person. Each flag is
#: accepted by 2.1.283 and 2.1.288 (help, changelog and the binary):
#:
#: * `--tools ""` -- no built-in tool at all ("Use "" to disable all
#:   tools"). Bash, Edit, WebFetch and the rest are not offered.
#: * `--permission-mode dontAsk` -- anything not pre-approved is denied
#:   without asking, and auto mode is never chosen for us.
#: * `--strict-mcp-config` with no `--mcp-config` -- no MCP server from the
#:   operator's own configuration, whose tools `--tools` does not cover.
#: * `--no-session-persistence` -- a client's conversation is not saved as
#:   a resumable session on the driver host.
#:
#: The cost is the "uses a tool" progress the stream used to report: there
#: is no tool left to report.
_NO_TOOLS: tuple[str, ...] = (
    "--tools",
    "",
    "--permission-mode",
    "dontAsk",
    "--strict-mcp-config",
    "--no-session-persistence",
)

#: **Where Claude Code is told to send its requests, as opposed to who it
#: is.** Stripped from the child's environment, because these are exactly
#: what Eugene's own Claude Code recipe writes (`ANTHROPIC_BASE_URL`,
#: `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_MODEL`, and the two
#: `CLAUDE_CODE_MAX_*_TOKENS` sized for a local model), and a driver host
#: that is also somebody's workstation has them set: the backend would loop
#: back into Eugene, with a client's key, instead of reaching Anthropic.
#: The model is the driver's to choose (`modelId` / `upstreamModelId`, sent
#: as `--model`), and output size is the gateway's (it owns every
#: output-affecting setting).
#:
#: **Kept, deliberately:** the CLI's own credentials -- `ANTHROPIC_API_KEY`,
#: `CLAUDE_CODE_OAUTH_TOKEN` and the keychain the CLI reads for a
#: subscription -- and the Bedrock/Vertex/Foundry variables, which name the
#: operator's own provider rather than Eugene.
_ROUTING_ENV = frozenset(
    {
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_CUSTOM_HEADERS",
        "ANTHROPIC_UNIX_SOCKET",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_FABLE_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
        "ANTHROPIC_CUSTOM_MODEL_OPTION",
        "CLAUDE_CODE_SUBAGENT_MODEL",
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    }
)


@contextlib.contextmanager
def _system_prompt_file(text: str) -> Iterator[str | None]:
    """The system prompt in a file of its own for one call, or None for none.

    **Never on the command line** (upstream drift audit, 2026-10-03). npm
    installs `claude` on Windows as a `.cmd` shim, so its argv passes
    through cmd.exe: a newline ends the command, the line stops at 8191
    characters, and -- measured here with a shim like npm's -- a `"` in the
    text closes cmd.exe's quoting, after which `&` ran a command of the
    client's choosing on the driver host. `--system-prompt-file` exists in
    every Claude Code this was checked against (2.1.283's binary registers
    it; the changelog has it since print mode's early days).

    `mkstemp` makes the file readable by this account only (0600 on POSIX;
    on Windows the user's own temp directory), and it is removed when the
    call ends however it ends. Written as bytes: no newline translation.
    """
    if not text:
        yield None
        return
    fd, path = tempfile.mkstemp(prefix="eugene-plexus-system-", suffix=".txt")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(text.encode("utf-8"))
        yield path
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


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
        upstream_model_id: str | None = None,
        timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        thinking_mode: str = "auto",
    ) -> None:
        self._binary_path = binary_path
        self._model_id = model_id
        # What the CLI is asked for, when the public alias is not the
        # provider's own model name. Wire boundary only: responses and
        # logs keep the public `_model_id`.
        self._upstream_model_id = upstream_model_id or model_id
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
            upstream_model_id=get("upstreamModelId") or None,
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
        # stdin under --print when no positional prompt is given. The
        # system prompt goes in a file, for the same reason and a worse
        # one (see `_system_prompt_file`).
        with _system_prompt_file(system_prompt) as prompt_file:
            argv = self._build_argv(system_prompt_file=prompt_file)

            # DEBUG-level full-payload trace. CLI adapters flatten the
            # gateway's structured message list into a single labeled
            # transcript string before sending — the operator's copy-trace
            # shows the pre-flattening shape, this shows what actually
            # reaches the model.
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "claude_code_cli → argv:\n%s\n--- system prompt (file) ---\n%s\n"
                    "--- user prompt (stdin) ---\n%s",
                    argv,
                    system_prompt or "(empty)",
                    user_prompt,
                )

            result = await run_cli(
                argv,
                timeout_seconds=self._timeout_seconds,
                stdin_input=user_prompt.encode("utf-8"),
                drop_env=_ROUTING_ENV,
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

        started = time.perf_counter()
        emitted: list[str] = []
        thoughts: list[str] = []
        envelope: dict[str, Any] | None = None
        report = bool(request.reportProgress)
        show_reasoning = self._thinking_mode != "off"

        async for line in self._stream_lines(system_prompt, user_prompt):
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
            if kind == "system":
                # `init` when the CLI is up, `status: requesting` each
                # time it sends the conversation to Anthropic -- again
                # after every tool it runs. Captured from 2.1.207.
                if report and event.get("subtype") in ("init", "status"):
                    yield Chunk(progress=StreamProgress(stage=Stage.working))
                continue
            if kind != "stream_event":
                continue
            inner = event.get("event") or {}
            if inner.get("type") == "content_block_start":
                block = inner.get("content_block") or {}
                if report and block.get("type") == "tool_use":
                    # The agent starting one of its own tools: the thing
                    # it can spend a minute on before it writes a word.
                    # The name only; the arguments are the operator's
                    # files and commands and are not carried.
                    name = block.get("name")
                    yield Chunk(
                        progress=StreamProgress(
                            stage=Stage.tool, tool=name if isinstance(name, str) else None
                        )
                    )
                continue
            if inner.get("type") != "content_block_delta":
                continue
            delta = inner.get("delta") or {}
            if delta.get("type") == "thinking_delta":
                # Claude's own reasoning, on its own channel -- forwarded
                # as reasoning, as the HTTP engine forwards llama.cpp's.
                # Until 2026-09-27 it was read past, so a Claude that
                # thought for a minute showed nothing for that minute.
                thought = delta.get("thinking")
                if show_reasoning and isinstance(thought, str) and thought:
                    thoughts.append(thought)
                    yield Chunk(reasoning=thought)
                continue
            # Otherwise only text. `signature_delta` is the reasoning
            # block's attestation, which is not the answer either.
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
                reasoning="".join(thoughts) or None,
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

    async def _stream_lines(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        """The CLI's stdout lines, with the system prompt's file alive for
        exactly as long as the child: closed explicitly, so a stream that
        ends early kills the child first and removes the file after."""
        with _system_prompt_file(system_prompt) as prompt_file:
            argv = self._build_argv(system_prompt_file=prompt_file, stream=True)
            if log.isEnabledFor(logging.DEBUG):
                log.debug("claude_code_cli → (stream) argv:\n%s", argv)
            lines = stream_cli_lines(
                argv,
                timeout_seconds=self._timeout_seconds,
                stdin_input=user_prompt.encode("utf-8"),
                drop_env=_ROUTING_ENV,
            )
            async with contextlib.aclosing(lines):
                async for line in lines:
                    yield line

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

    def _build_argv(self, *, system_prompt_file: str | None, stream: bool = False) -> list[str]:
        argv = [
            self._binary_path,
            "--print",
            "--output-format",
            "stream-json" if stream else "json",
            *_NO_TOOLS,
        ]
        if stream:
            # `--verbose` is required: Claude Code refuses stream-json
            # under --print without it. `--include-partial-messages` is
            # what turns whole-message events into text deltas.
            argv += ["--include-partial-messages", "--verbose"]
        if system_prompt_file:
            # Replaces Claude Code's own system prompt, as `--system-prompt`
            # did; read from a file since 2026-10-03.
            argv += ["--system-prompt-file", system_prompt_file]
        if self._upstream_model_id:
            argv += ["--model", self._upstream_model_id]
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
