"""Files only this account can read, written so a crash leaves the old one.

**This driver's config file holds its backend's API key.** Sealed under
the master key when the agent has handed one over, and in plaintext
when it has not -- a standalone driver, or the window before the
operator unlocks (see `ConfigStore._write_locked`). Until 2026-09-22 the
file was written with `open("w")`, so it landed at the process umask:
`0644` on a stock Linux, and a cloud provider's key readable by every
account on the host.

So the config is written through here, and both halves are the point:

  * **Private from the first byte.** `os.open(..., 0o600)` sets the mode
    at creation. Creating the file and `chmod`-ing it afterwards leaves a
    window in which another account can open it, and an open descriptor
    outlives the `chmod`.
  * **Replaced, not rewritten.** A temp file beside the target, flushed
    and fsync'd, then `os.replace`. `open("w")` truncates before it
    writes, so a kill between the two leaves an empty config -- and a
    driver that boots on defaults, having forgotten its key. The temp is
    created `O_EXCL`, so a name somebody placed there in advance is
    refused rather than written through.

A **symlinked** target is written through: the real path is resolved
first and the temp lives beside *it*, so an operator who linked the file
somewhere keeps the link. `os.replace` on the link itself would swap the
link for a regular file.

On Windows the mode argument sets only the read-only bit; access there
is the ACL inherited from the directory, which for a per-user install is
the user's own profile. The code is the same on both, which is what lets
a test on this box assert the mode was *asked for*.

Deliberately duplicated in each component rather than shared: components
share schemas, not code (see CLAUDE.md, "no shared `core` library").
"""

from __future__ import annotations

import contextlib
import os
import uuid
from pathlib import Path

__all__ = ["PRIVATE_MODE", "write_private_text"]

PRIVATE_MODE = 0o600
"""Owner read and write, nobody else anything."""

# Without it a Windows descriptor is in the C runtime's text mode and
# every "\n" we write becomes "\r\n". Zero, and so a no-op, elsewhere.
_O_BINARY = getattr(os, "O_BINARY", 0)


def write_private_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Replace `path` with `text`, readable by this account only.

    Raises what the filesystem raises, after removing the temp file, so
    no `.tmp` is left beside the target to be mistaken for something.
    """
    target = Path(os.path.realpath(path))
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    data = text.encode(encoding)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_BINARY, PRIVATE_MODE)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
