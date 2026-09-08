"""Tiny shared helper for invoking a CLI subprocess from an adapter.

Centralizes the asyncio plumbing so each adapter only has to write its argv
builder and its output parser.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from dataclasses import dataclass

# Forced-UTF-8 environment for child processes. The smoke test on
# 2026-05-09 surfaced em-dashes coming back as `â€"` triple-byte
# mojibake on Windows — UTF-8 bytes from the child reinterpreted via
# the system codepage somewhere in the pipeline. Setting these vars
# makes Python children unconditionally use UTF-8 for stdio. Node-based
# tools (claude.cmd, codex.cmd) generally already write UTF-8 to stdout,
# so this primarily protects any future Python-based adapters; harmless
# elsewhere. We also set LC_ALL/LANG so native tools that consult the
# locale (less common on Windows but normal on Linux) get a UTF-8 hint.
_UTF8_ENV: dict[str, str] = {
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
    "LC_ALL": "C.UTF-8",
    "LANG": "C.UTF-8",
}


def _utf8_subprocess_env() -> dict[str, str]:
    """Return os.environ overlaid with UTF-8 hints for the child."""
    env = os.environ.copy()
    env.update(_UTF8_ENV)
    return env


class CliError(RuntimeError):
    """Raised when a CLI invocation exits non-zero or times out."""


@dataclass
class CliResult:
    stdout: bytes
    stderr: bytes
    returncode: int
    elapsed_ms: int


async def run_cli(
    argv: list[str],
    *,
    timeout_seconds: float,
    stdin_input: bytes | None = None,
) -> CliResult:
    """Run argv as a subprocess. Returns stdout/stderr/returncode + elapsed time.

    When `stdin_input` is None, stdin is closed (DEVNULL) so the CLI never
    blocks. When provided, stdin_input is written to the child's stdin and
    closed — useful for prompt content that contains newlines, which can't
    safely round-trip through argv on Windows (cmd.exe interprets a literal
    newline inside a quoted argv item as a command separator). Raises
    `CliError` on timeout; non-zero exit is returned to the caller for
    inspection.
    """
    if not argv:
        raise ValueError("argv is empty")

    # Resolve the binary up-front so the error message is clearer than
    # asyncio's default "FileNotFoundError" if it isn't on PATH.
    binary = argv[0]
    resolved = shutil.which(binary)
    if resolved is None:
        raise CliError(f"binary {binary!r} not found on PATH")
    argv = [resolved, *argv[1:]]

    start = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin_input is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_utf8_subprocess_env(),
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=stdin_input),
            timeout=timeout_seconds,
        )
    except TimeoutError as e:
        proc.kill()
        await proc.wait()
        raise CliError(f"CLI {argv[0]!r} did not respond within {timeout_seconds}s; killed") from e

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return CliResult(
        stdout=stdout, stderr=stderr, returncode=proc.returncode or 0, elapsed_ms=elapsed_ms
    )
