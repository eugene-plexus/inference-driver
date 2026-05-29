"""Auth state for the hemisphere-driver's verify-only role.

Built once at startup from the env vars the watchdog threads in when
it spawns this child:

  * `EUGENE_PLEXUS_HD_AUTH_SIGNING_KEY` — base64 of the 32-byte HMAC
    key used to validate inbound bearer tokens.
  * `EUGENE_PLEXUS_HD_SERVICE_TOKEN` — long-lived JWT (`aud:
    service:hemisphere-driver`). The hemisphere-driver currently
    makes no outbound calls to peer components (LLM backend calls are
    a separate trust boundary), so this is captured but unused in
    v0.2. Reserved for v0.3 if hemispheres start reading memory.
  * `EUGENE_PLEXUS_HD_MASTER_KEY` — base64 of the 32-byte secretbox
    key. Used by Phase 6 to decrypt the at-rest envelope around
    `apiKey` config values; not consumed in this phase.

If `AUTH_SIGNING_KEY` is unset, the driver runs in `auth_disabled=True`
mode: route dependencies short-circuit and let everything through.
That's the dev/standalone path. Production via the watchdog always
supplies the env var.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthState:
    """Process-wide auth posture. Immutable for the driver's lifetime —
    rotating the signing key requires a restart, which is by design
    (per-restart key rotation is the v0.2 revocation story)."""

    signing_key: bytes | None
    """32-byte HMAC key, or None when auth is disabled."""

    service_token: str | None
    """Outbound bearer token. Not consumed in v0.2 — the driver doesn't
    call peer components. Captured for v0.3 use."""

    master_key: bytes | None
    """At-rest secretbox key. Only set when the operator has logged in
    at the watchdog. Phase 6 uses this to decrypt config-stored apiKey
    values; Phase 4 leaves it untouched."""

    @property
    def auth_disabled(self) -> bool:
        return self.signing_key is None


def _decode_b64_key(value: str | None, *, expected_len: int, label: str) -> bytes | None:
    if not value:
        return None
    try:
        raw = base64.b64decode(value, validate=True)
    except Exception as e:
        raise ValueError(f"{label}: not valid base64 ({e})") from e
    if len(raw) != expected_len:
        raise ValueError(
            f"{label}: expected {expected_len} bytes after base64-decode, got {len(raw)}"
        )
    return raw


def load_auth_state(
    *,
    signing_key_b64: str | None,
    service_token: str | None,
    master_key_b64: str | None,
) -> AuthState:
    """Build an `AuthState` from the three env-var inputs.

    Returns auth-disabled state when no signing key is supplied (with
    a one-shot warning so dev runs are obvious). Raises on
    inconsistent partial-auth configurations — a missing piece is
    almost always a wiring bug we want loud rather than silently 401-y.
    """
    signing_key = _decode_b64_key(signing_key_b64, expected_len=32, label="AUTH_SIGNING_KEY")
    master_key = _decode_b64_key(master_key_b64, expected_len=32, label="MASTER_KEY")

    if signing_key is None:
        if service_token or master_key:
            raise ValueError(
                "auth env vars inconsistent: SERVICE_TOKEN or MASTER_KEY is set but "
                "AUTH_SIGNING_KEY is not — refusing to start in a partially-auth state"
            )
        log.warning(
            "EUGENE_PLEXUS_HD_AUTH_SIGNING_KEY not set — running unauthenticated "
            "(dev/standalone mode). Production spawns via watchdog always supply this."
        )
        return AuthState(signing_key=None, service_token=None, master_key=None)

    return AuthState(
        signing_key=signing_key,
        service_token=service_token or None,
        master_key=master_key,
    )
