"""Tiny shared helper for invoking a CLI subprocess from an adapter.

Centralizes the asyncio plumbing so each adapter only has to write its argv
builder and its output parser.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import time
from collections.abc import AsyncIterator
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
    """Raised when a backend invocation fails.

    Named for the CLI engines it was written for, and since M0 also the
    way the HTTP engine reports an upstream failure -- so it carries the
    upstream status when there was one.

    `upstream_status` is what makes the difference between a refusal the
    next backend would repeat and one worth trying elsewhere, and the
    route turns it into the driver's own 400 or 502. Without it every
    backend failure looked the same from one layer up, and llama.cpp's
    exact "your prompt is 20597 tokens and the window is 512" arrived as
    a retryable 502 that cascaded through every replica in the slot.
    `None` for a subprocess backend or a transport failure, where there
    is no status to carry and 502 is the honest answer.
    """

    def __init__(self, *args: object, upstream_status: int | None = None) -> None:
        super().__init__(*args)
        self.upstream_status = upstream_status


@dataclass
class CliResult:
    stdout: bytes
    stderr: bytes
    returncode: int
    elapsed_ms: int


async def stream_cli_lines(
    argv: list[str],
    *,
    timeout_seconds: float,
    stdin_input: bytes | None = None,
) -> AsyncIterator[str]:
    """Run argv and yield its stdout a line at a time, as it arrives.

    The streaming counterpart to `run_cli`, which buffers to completion
    and therefore cannot be used for token delivery -- a CLI backend
    that emits JSONL is already producing tokens, and `communicate()`
    hides them until the process exits.

    Three things this owes the caller that a naive read loop does not:

      * **The child dies with the iterator.** A consumer that stops
        early -- which is what a client disconnecting mid-stream looks
        like from here -- runs the `finally`, which kills the process
        rather than leaving a model generating into a closed pipe.
      * **The timeout covers the whole stream**, not one read, because a
        CLI that emits one line and then wedges is the case worth
        bounding.
      * **stderr is drained concurrently.** A child that fills the
        stderr pipe while we only read stdout deadlocks, and the CLI
        backends are chatty on stderr -- codex writes an auth-refresh
        error per event.

    Decoding is UTF-8 with replacement, matching `run_cli`: a mangled
    character must not end a half-delivered answer.
    """
    if not argv:
        raise ValueError("argv is empty")
    binary = argv[0]
    resolved = shutil.which(binary)
    if resolved is None:
        raise CliError(f"binary {binary!r} not found on PATH")
    argv = [resolved, *argv[1:]]

    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin_input is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_utf8_subprocess_env(),
    )
    assert proc.stdout is not None

    async def _drain_stderr() -> bytes:
        if proc.stderr is None:
            return b""
        return await proc.stderr.read()

    stderr_task = asyncio.create_task(_drain_stderr())
    deadline = time.perf_counter() + timeout_seconds

    try:
        if stdin_input is not None and proc.stdin is not None:
            proc.stdin.write(stdin_input)
            await proc.stdin.drain()
            proc.stdin.close()
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise CliError(f"CLI {binary!r} did not finish within {timeout_seconds}s; killed")
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=remaining)
            except TimeoutError as e:
                raise CliError(
                    f"CLI {binary!r} did not finish within {timeout_seconds}s; killed"
                ) from e
            if not line:
                break
            yield line.decode("utf-8", "replace").rstrip()
        await asyncio.wait_for(proc.wait(), timeout=max(0.1, deadline - time.perf_counter()))
        if proc.returncode:
            stderr = (await stderr_task).decode("utf-8", "replace").strip()
            raise CliError(f"{binary} exited {proc.returncode}: {stderr or '<no stderr>'}")
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
        stderr_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await stderr_task


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
