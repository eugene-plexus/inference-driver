"""Unit tests for the two CLI adapters.

Subprocess invocation is mocked at the `run_cli` boundary so these tests
don't require `claude` or `codex` to be installed. End-to-end verification
against real binaries is gated behind an env var (see the bottom of this
file) and skipped by default.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from eugene_plexus_inference_driver._generated.models import (
    BackendKind,
    FinishReason,
    GenerateRequest,
    Message,
    Role,
)
from eugene_plexus_inference_driver.engines import _subprocess
from eugene_plexus_inference_driver.engines._subprocess import CliError, CliResult
from eugene_plexus_inference_driver.engines.claude_code_cli import ClaudeCodeCliEngine
from eugene_plexus_inference_driver.engines.codex_cli import CodexCliEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _request(prompt: str = "say PING") -> GenerateRequest:
    return GenerateRequest(
        messages=[
            Message(role=Role.user, content=prompt),
        ],
    )


def _patch_run_cli(
    monkeypatch: pytest.MonkeyPatch,
    fn: Callable[[list[str]], Awaitable[CliResult] | CliResult],
) -> dict[str, Any]:
    """Replace adapters._subprocess.run_cli with `fn` and capture argv calls."""
    captured: dict[str, Any] = {"argv": None, "timeout": None, "stdin_input": None}

    async def _fake_run_cli(
        argv: list[str], *, timeout_seconds: float, stdin_input: bytes | None = None
    ) -> CliResult:
        captured["argv"] = argv
        captured["timeout"] = timeout_seconds
        captured["stdin_input"] = stdin_input
        result = fn(argv)
        if hasattr(result, "__await__"):
            return await result
        return result

    # Patch on every importing module's symbol table because we use
    # `from ._subprocess import run_cli` style imports.
    monkeypatch.setattr(_subprocess, "run_cli", _fake_run_cli)
    monkeypatch.setattr(
        "eugene_plexus_inference_driver.engines.claude_code_cli.run_cli",
        _fake_run_cli,
    )
    monkeypatch.setattr(
        "eugene_plexus_inference_driver.engines.codex_cli.run_cli",
        _fake_run_cli,
    )
    return captured


# ---------------------------------------------------------------------------
# Claude Code CLI
# ---------------------------------------------------------------------------

CLAUDE_OK_ENVELOPE = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "PING",
    "stop_reason": "end_turn",
    "duration_ms": 1942,
    "usage": {
        "input_tokens": 6,
        "output_tokens": 6,
        "cache_read_input_tokens": 33349,
        "cache_creation_input_tokens": 0,
    },
}


async def test_claude_adapter_parses_success_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_run_cli(
        monkeypatch,
        lambda argv: CliResult(
            stdout=json.dumps(CLAUDE_OK_ENVELOPE).encode(),
            stderr=b"",
            returncode=0,
            elapsed_ms=2000,
        ),
    )
    adapter = ClaudeCodeCliEngine(model_id="claude-opus-4-7", timeout_seconds=30.0)

    response = await adapter.generate(_request())

    assert response.content == "PING"
    assert response.finishReason == FinishReason.stop
    assert response.backend == BackendKind.claude_code_cli
    assert response.modelId == "claude-opus-4-7"
    assert response.latencyMs == 2000
    assert response.usage is not None
    assert response.usage.completionTokens == 6
    # Reflects cache reads as part of prompt accounting.
    assert response.usage.promptTokens == 6 + 33349

    # argv shape: claude --print --output-format json --model <id>
    # User prompt now goes via stdin (argv-newline issue on Windows).
    assert captured["argv"] == [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--model",
        "claude-opus-4-7",
    ]
    assert captured["stdin_input"] is not None
    assert captured["stdin_input"].startswith(b"[USER]")
    assert captured["timeout"] == 30.0


async def test_claude_adapter_omits_model_when_not_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_run_cli(
        monkeypatch,
        lambda argv: CliResult(
            stdout=json.dumps(CLAUDE_OK_ENVELOPE).encode(),
            stderr=b"",
            returncode=0,
            elapsed_ms=1000,
        ),
    )
    adapter = ClaudeCodeCliEngine()  # model_id=None
    await adapter.generate(_request())
    assert "--model" not in captured["argv"]


async def test_claude_adapter_handles_multiline_history_via_stdin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: prior assistant messages with newlines must not break argv.

    On Windows, cmd.exe (which wraps `claude.cmd`) treats a literal newline
    inside a quoted argv item as a command separator, so a multi-line user
    prompt would corrupt the command line. Piping the user prompt through
    stdin sidesteps this entirely.
    """
    captured = _patch_run_cli(
        monkeypatch,
        lambda argv: CliResult(
            stdout=json.dumps(CLAUDE_OK_ENVELOPE).encode(),
            stderr=b"",
            returncode=0,
            elapsed_ms=1000,
        ),
    )
    adapter = ClaudeCodeCliEngine()
    await adapter.generate(
        GenerateRequest(
            messages=[
                Message(role=Role.system, content="You are Eugene."),
                Message(role=Role.user, content="First question?"),
                Message(
                    role=Role.assistant,
                    content="A multi-paragraph reply.\n\nSecond paragraph here.\n\nThird.",
                ),
                Message(role=Role.user, content="Follow-up\nwith\nnewlines"),
            ]
        )
    )
    # Argv must NOT contain the prompt content. Only flags + values.
    for item in captured["argv"]:
        assert "First question" not in item
        assert "Second paragraph" not in item
        assert "Follow-up" not in item
    # Prompt content must be on stdin instead.
    stdin = captured["stdin_input"]
    assert stdin is not None
    decoded = stdin.decode("utf-8")
    assert "First question?" in decoded
    assert "Second paragraph here." in decoded
    assert "Follow-up\nwith\nnewlines" in decoded


async def test_claude_adapter_passes_system_prompt_when_system_messages_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persona / cwd-leak fix: when the request includes system messages,
    we send them as `--system-prompt`. Per Claude Code's docs, that flag
    *replaces* the default system prompt, which transitively disables the
    automatic cwd-injection that was leaking hemisphere identity in the
    smoke test.
    """
    captured = _patch_run_cli(
        monkeypatch,
        lambda argv: CliResult(
            stdout=json.dumps(CLAUDE_OK_ENVELOPE).encode(),
            stderr=b"",
            returncode=0,
            elapsed_ms=1000,
        ),
    )
    adapter = ClaudeCodeCliEngine()
    await adapter.generate(
        GenerateRequest(
            messages=[
                Message(role=Role.system, content="You are Eugene."),
                Message(role=Role.user, content="hi"),
            ]
        )
    )
    argv = captured["argv"]
    assert "--system-prompt" in argv
    idx = argv.index("--system-prompt")
    assert argv[idx + 1] == "You are Eugene."


async def test_claude_adapter_omits_system_prompt_when_no_system_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No system messages -> no --system-prompt. Claude Code's default
    system prompt (with its cwd injection) takes over in that case.
    Documenting the contract explicitly so the persona/cwd behavior is
    pinned to the request shape, not implicit."""
    captured = _patch_run_cli(
        monkeypatch,
        lambda argv: CliResult(
            stdout=json.dumps(CLAUDE_OK_ENVELOPE).encode(),
            stderr=b"",
            returncode=0,
            elapsed_ms=1000,
        ),
    )
    adapter = ClaudeCodeCliEngine()
    await adapter.generate(_request())  # user-only message
    assert "--system-prompt" not in captured["argv"]


async def test_claude_adapter_preserves_utf8_em_dash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: an em-dash (U+2014) returned by Claude Code must come
    through as a single character, not as the `â€"` triple-byte mojibake
    seen in the smoke test on 2026-05-09. The fix is environment-level
    (PYTHONUTF8 etc. on the subprocess); this test asserts the pipeline
    *parses* UTF-8 bytes correctly, regardless of how they got there.
    """
    em_dash = "—"
    envelope = {**CLAUDE_OK_ENVELOPE, "result": f"Two paragraphs{em_dash}separated."}
    payload = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
    # Sanity: the em-dash is encoded as the canonical 3-byte UTF-8 sequence.
    assert b"\xe2\x80\x94" in payload

    _patch_run_cli(
        monkeypatch,
        lambda argv: CliResult(stdout=payload, stderr=b"", returncode=0, elapsed_ms=10),
    )
    adapter = ClaudeCodeCliEngine()
    response = await adapter.generate(_request())
    assert em_dash in response.content
    assert "â€" not in response.content


async def test_run_cli_passes_utf8_env_to_subprocess() -> None:
    """The real `run_cli` sets PYTHONUTF8=1 + PYTHONIOENCODING=utf-8 on the
    child's environment so a Python subprocess can't fall back to the
    Windows system codepage. We assert by inspecting the env-builder
    helper directly — spawning a real subprocess from the test suite is
    out of scope here.
    """
    env = _subprocess._utf8_subprocess_env()
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["LC_ALL"] == "C.UTF-8"
    assert env["LANG"] == "C.UTF-8"


async def test_claude_adapter_raises_on_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = {**CLAUDE_OK_ENVELOPE, "is_error": True, "result": "Not logged in"}
    _patch_run_cli(
        monkeypatch,
        lambda argv: CliResult(
            stdout=json.dumps(envelope).encode(),
            stderr=b"",
            returncode=0,
            elapsed_ms=100,
        ),
    )
    adapter = ClaudeCodeCliEngine()
    with pytest.raises(CliError, match="Not logged in"):
        await adapter.generate(_request())


async def test_claude_adapter_raises_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_run_cli(
        monkeypatch,
        lambda argv: CliResult(
            stdout=b"",
            stderr=b"some error",
            returncode=2,
            elapsed_ms=50,
        ),
    )
    adapter = ClaudeCodeCliEngine()
    with pytest.raises(CliError, match="exited 2"):
        await adapter.generate(_request())


# ---------------------------------------------------------------------------
# Codex CLI
# ---------------------------------------------------------------------------

CODEX_OK_STREAM = b"""
{"type":"thread.started","thread_id":"abc"}
{"type":"turn.started"}
{"type":"item.completed","item":{"type":"reasoning","summary":"thinking"}}
{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"PING"}}
{"type":"turn.completed","usage":{"input_tokens":26820,"cached_input_tokens":6528,"output_tokens":21,"reasoning_output_tokens":14}}
""".strip()


async def test_codex_adapter_parses_jsonl_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_run_cli(
        monkeypatch,
        lambda argv: CliResult(stdout=CODEX_OK_STREAM, stderr=b"", returncode=0, elapsed_ms=1500),
    )
    adapter = CodexCliEngine(model_id="gpt-5", timeout_seconds=60.0)

    response = await adapter.generate(_request())

    assert response.content == "PING"
    assert response.finishReason == FinishReason.stop
    assert response.backend == BackendKind.codex_cli
    assert response.modelId == "gpt-5"
    assert response.latencyMs == 1500
    assert response.usage is not None
    assert response.usage.promptTokens == 26820 + 6528
    assert response.usage.completionTokens == 21 + 14

    argv = captured["argv"]
    assert argv[0] == "codex"
    assert argv[1] == "exec"
    assert "--json" in argv
    assert "--ephemeral" in argv
    assert "--skip-git-repo-check" in argv
    assert "read-only" in argv
    assert "--model" in argv and argv[argv.index("--model") + 1] == "gpt-5"


async def test_codex_adapter_raises_when_no_agent_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = b'{"type":"turn.started"}\n{"type":"turn.completed","usage":{}}\n'
    _patch_run_cli(
        monkeypatch,
        lambda argv: CliResult(stdout=stream, stderr=b"", returncode=0, elapsed_ms=10),
    )
    adapter = CodexCliEngine()
    with pytest.raises(CliError, match="without producing an agent_message"):
        await adapter.generate(_request())


async def test_codex_adapter_concatenates_multi_message_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = (
        b'{"type":"item.completed","item":{"type":"agent_message","text":"Hello"}}\n'
        b'{"type":"item.completed","item":{"type":"agent_message","text":" world"}}\n'
        b'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":2}}\n'
    )
    _patch_run_cli(
        monkeypatch,
        lambda argv: CliResult(stdout=stream, stderr=b"", returncode=0, elapsed_ms=10),
    )
    adapter = CodexCliEngine()
    response = await adapter.generate(_request())
    assert response.content == "Hello world"


async def test_claude_adapter_list_models_returns_known_chat_models() -> None:
    """The CLI doesn't expose a list endpoint, so we hardcode known
    chat-tier Claude models. Pin the contract: list is non-empty,
    every entry starts with `claude-`, and it includes the current
    flagship model."""
    adapter = ClaudeCodeCliEngine()
    models = await adapter.list_models()
    assert models  # non-empty
    assert all(m.startswith("claude-") for m in models)
    assert "claude-opus-4-7" in models


async def test_codex_adapter_list_models_excludes_temperature_uncontrollable() -> None:
    """Codex CLI hardcodes a list too; verify it deliberately excludes
    o-series and gpt-5 family — Eugene Plexus's hemisphere policy
    applies regardless of which adapter is delivering the model."""
    adapter = CodexCliEngine()
    models = await adapter.list_models()
    assert models
    assert "gpt-4o" in models
    for forbidden in ("o1", "o1-mini", "o3", "gpt-5", "gpt-5-mini"):
        assert forbidden not in models


# ---------------------------------------------------------------------------
# End-to-end against real binaries — opt-in via env var
# ---------------------------------------------------------------------------

LIVE = os.environ.get("EUGENE_PLEXUS_DRIVER_LIVE_CLI") == "1"


@pytest.mark.skipif(not LIVE, reason="set EUGENE_PLEXUS_DRIVER_LIVE_CLI=1 to run live")
async def test_claude_live_call() -> None:
    adapter = ClaudeCodeCliEngine(timeout_seconds=180.0)
    response = await adapter.generate(_request("Reply with exactly the four characters: PING"))
    assert "PING" in response.content


@pytest.mark.skipif(not LIVE, reason="set EUGENE_PLEXUS_DRIVER_LIVE_CLI=1 to run live")
async def test_codex_live_call() -> None:
    adapter = CodexCliEngine(timeout_seconds=180.0)
    response = await adapter.generate(_request("Reply with exactly the four characters: PING"))
    assert "PING" in response.content
