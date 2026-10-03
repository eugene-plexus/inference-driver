"""A subscription CLI is a text backend, and nothing a client sends may make it more.

Upstream drift audit, 2026-10-03, finding 3: `claude -p` was run with no
`--permission-mode` and no `--tools`, and its child inherited the whole
environment but `EUGENE_PLEXUS_*`. Claude Code 2.1.285 starts `-p` in auto
mode "on third-party providers, or with telemetry off", so an inherited
`ANTHROPIC_BASE_URL` -- the variable Eugene's own Claude Code recipe sets --
would have let a prompt from any client key run Bash or Edit on the driver
host, approved by a classifier rather than a person. The same recipe's
`ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_MODEL` would have pointed the backend
back into Eugene.

Codex has the same loop by another road: Eugene's Codex recipe writes
`model_provider = "eugene"` into `~/.codex/config.toml` and keys it from
`EUGENE_API_KEY`, which no `EUGENE_PLEXUS_` rule strips; and Codex's
headless `never` approval policy falls back to the config file's own when
that file chooses the automatic reviewer.

These tests run the real subprocess helpers against a fake `claude` /
`codex` -- a `.cmd` shim on Windows, exactly as npm installs the real ones,
and a shell script elsewhere -- that reports its argv, stdin, environment
names and system-prompt file back as the answer.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from eugene_plexus_inference_driver._generated.models import GenerateRequest, Message, Role
from eugene_plexus_inference_driver.engines._subprocess import _utf8_subprocess_env
from eugene_plexus_inference_driver.engines.claude_code_cli import ClaudeCodeCliEngine
from eugene_plexus_inference_driver.engines.codex_cli import CodexCliEngine

_FAKE_CLI = r"""
import json, os, sys

args = sys.argv[1:]
report = {"argv": args, "stdin": sys.stdin.read(), "env": sorted(os.environ)}
if "--system-prompt-file" in args:
    path = args[args.index("--system-prompt-file") + 1]
    report["system_prompt_path"] = path
    with open(path, encoding="utf-8") as handle:
        report["system_prompt"] = handle.read()
text = json.dumps(report)
usage = {"input_tokens": 3, "output_tokens": 2}
if args and args[0] == "exec":
    print(json.dumps({"type": "turn.started"}))
    item = {"type": "agent_message", "text": text}
    print(json.dumps({"type": "item.completed", "item": item}))
    print(json.dumps({"type": "turn.completed", "usage": usage}))
else:
    envelope = {"type": "result", "subtype": "success", "is_error": False,
                "result": text, "stop_reason": "end_turn", "usage": usage}
    if "stream-json" in args:
        delta = {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}
        print(json.dumps({"type": "stream_event", "event": delta}))
    print(json.dumps(envelope))
"""


def _fake(tmp_path: Path, name: str) -> str:
    """A fake CLI that answers with what it was given."""
    script = tmp_path / "fake_cli.py"
    script.write_text(_FAKE_CLI, encoding="utf-8")
    if os.name == "nt":
        shim = tmp_path / f"{name}.cmd"
        shim.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        shim = tmp_path / name
        shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        shim.chmod(shim.stat().st_mode | stat.S_IEXEC)
    return str(shim)


def _request(*messages: tuple[Role, str]) -> GenerateRequest:
    return GenerateRequest(
        messages=[Message(role=role, content=text) for role, text in messages]
        or [Message(role=Role.user, content="say PING")]
    )


async def _report(engine: Any, request: GenerateRequest, *, stream: bool) -> dict[str, Any]:
    if not stream:
        return json.loads((await engine.generate(request)).content)
    final = None
    async for chunk in engine.stream(request):
        if chunk.done:
            final = chunk.result
    assert final is not None
    return json.loads(final.content)


def _after(argv: list[str], flag: str) -> str:
    assert flag in argv, argv
    return argv[argv.index(flag) + 1]


# --------------------------------------------------------------------------- #
# Claude Code: no tools, nothing that may ask, nowhere but its own provider
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stream", [False, True])
async def test_claude_is_given_no_tools_and_no_mode_that_runs_one(
    tmp_path: Path, stream: bool
) -> None:
    engine = ClaudeCodeCliEngine(binary_path=_fake(tmp_path, "claude"), timeout_seconds=30)
    argv = (await _report(engine, _request(), stream=stream))["argv"]
    # `""` survives the `.cmd` shim as an empty argument: no tools at all.
    assert _after(argv, "--tools") == ""
    # Pinned, so a Claude Code that would start `-p` in auto mode does not,
    # and anything not pre-approved is denied without asking anyone.
    assert _after(argv, "--permission-mode") == "dontAsk"
    # No MCP server from the operator's own configuration, and no
    # transcript of a client's conversation left on the driver host.
    assert "--strict-mcp-config" in argv
    assert "--no-session-persistence" in argv
    assert "--dangerously-skip-permissions" not in argv


ROUTING = {
    # What Eugene's own Claude Code recipe writes (ui/src/lib/clientKeys.ts).
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8080",
    "ANTHROPIC_AUTH_TOKEN": "eugene-client-key",
    "ANTHROPIC_MODEL": "qwen3-14b",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "32768",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "8192",
    # The rest of the family that chooses where requests go or which model.
    "ANTHROPIC_CUSTOM_HEADERS": "x-eugene: 1",
    "ANTHROPIC_UNIX_SOCKET": "/tmp/eugene.sock",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "qwen3-14b",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "qwen3-14b",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "qwen3-14b",
    "ANTHROPIC_DEFAULT_FABLE_MODEL": "qwen3-14b",
    "ANTHROPIC_SMALL_FAST_MODEL": "qwen3-14b",
    "CLAUDE_CODE_SUBAGENT_MODEL": "qwen3-14b",
    # Eugene's Codex recipe's key variable.
    "EUGENE_API_KEY": "eugene-client-key",
}
OWN_CREDENTIALS = {
    "ANTHROPIC_API_KEY": "operator-anthropic-key",
    "CLAUDE_CODE_OAUTH_TOKEN": "operator-oauth-token",
    "HTTPS_PROXY": "http://operator-proxy:8000",
}


@pytest.mark.parametrize("stream", [False, True])
async def test_claude_never_sees_a_route_back_into_eugene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    for name, value in {**ROUTING, **OWN_CREDENTIALS}.items():
        monkeypatch.setenv(name, value)
    engine = ClaudeCodeCliEngine(binary_path=_fake(tmp_path, "claude"), timeout_seconds=30)
    names = {n.upper() for n in (await _report(engine, _request(), stream=stream))["env"]}
    leaked = sorted(set(ROUTING) & names)
    assert not leaked, f"the backend CLI could be pointed back at Eugene by {leaked}"
    # Its own subscription and API-key auth, and the operator's proxy, stay.
    assert set(OWN_CREDENTIALS) <= names


def test_eugene_client_keys_reach_no_cli_whatever_the_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EUGENE_API_KEY", "eugene-client-key")
    assert "EUGENE_API_KEY" not in {k.upper() for k in _utf8_subprocess_env()}


# --------------------------------------------------------------------------- #
# Codex: the operator's config cannot choose Eugene or an automatic reviewer
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stream", [False, True])
async def test_codex_config_cannot_choose_eugene_or_a_reviewer(
    tmp_path: Path, stream: bool
) -> None:
    engine = CodexCliEngine(binary_path=_fake(tmp_path, "codex"), timeout_seconds=30)
    argv = (await _report(engine, _request(), stream=stream))["argv"]
    assert _after(argv, "--sandbox") == "read-only"
    overrides = [argv[i + 1] for i, a in enumerate(argv) if a == "-c"]
    # `~/.codex/config.toml` is where Eugene's own recipe puts
    # `model_provider = "eugene"`: the subscription backend is Codex's
    # built-in OpenAI provider, whatever that file says.
    assert "model_provider=openai" in overrides
    # Headless `exec` already means `never`, unless a config chooses the
    # automatic reviewer -- then it falls back to the config's own policy.
    assert "approval_policy=never" in overrides
    assert "approvals_reviewer=user" in overrides
    # Not the whole file: it also says where the subscription's
    # credentials are kept (`cli_auth_credentials_store`), and ignoring it
    # would log a keyring user out.
    assert "--ignore-user-config" not in argv
    assert "--dangerously-bypass-approvals-and-sandbox" not in argv


@pytest.mark.parametrize("stream", [False, True])
async def test_codex_never_sees_a_route_back_into_eugene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    monkeypatch.setenv("EUGENE_API_KEY", "eugene-client-key")
    # Read by older Codex as the built-in provider's address.
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("CODEX_API_KEY", "operator-codex-key")
    monkeypatch.setenv("OPENAI_API_KEY", "operator-openai-key")
    engine = CodexCliEngine(binary_path=_fake(tmp_path, "codex"), timeout_seconds=30)
    names = {n.upper() for n in (await _report(engine, _request(), stream=stream))["env"]}
    assert "EUGENE_API_KEY" not in names
    assert "OPENAI_BASE_URL" not in names
    assert {"CODEX_API_KEY", "OPENAI_API_KEY"} <= names
