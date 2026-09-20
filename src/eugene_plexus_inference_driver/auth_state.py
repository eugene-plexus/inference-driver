"""Verifier bootstrap: AUTH_VERIFY_KEY is base64 public Ed25519 PEM.
AUTH_SIGNING_KEY accepts only a legacy 32-byte HS256 key during upgrade.
Never supply both. SERVICE_TOKEN is minted by the supervising agent;
MASTER_KEY is a separate at-rest encryption credential. Missing both
verification inputs retains the existing standalone/dev mode.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass

from . import security

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthState:
    """Process-wide auth posture. Immutable for the driver's lifetime —
    rotating the signing key requires a restart, which is by design
    (per-restart key rotation is the v0.2 revocation story)."""

    signing_key: bytes | None
    """Public Ed25519 PEM or legacy HMAC key; None when auth is disabled."""

    service_token: str | None
    """Outbound bearer token. Not consumed in v0.2 — the driver doesn't
    call peer components. Captured for v0.3 use."""

    master_key: bytes | None
    """At-rest secretbox key. Only set when the operator has logged in
    at the agent. Phase 6 uses this to decrypt config-stored apiKey
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
    verify_key_b64: str | None = None,
) -> AuthState:
    """Build an `AuthState` from the three env-var inputs.

    Returns auth-disabled state when no signing key is supplied (with
    a one-shot warning so dev runs are obvious). Raises on
    inconsistent partial-auth configurations — a missing piece is
    almost always a wiring bug we want loud rather than silently 401-y.
    """
    signing_key: bytes | None
    if verify_key_b64 is not None:
        if signing_key_b64 is not None:
            raise ValueError("both AUTH_VERIFY_KEY and legacy AUTH_SIGNING_KEY are set")
        try:
            raw = base64.b64decode(verify_key_b64, validate=True)
            if not raw.startswith(b"-----BEGIN PUBLIC KEY-----"):
                raise ValueError("public PEM required")
            signing_key = security.verification_key(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError("AUTH_VERIFY_KEY must contain base64 Ed25519 public PEM") from exc
    else:
        signing_key = _decode_b64_key(signing_key_b64, expected_len=32, label="AUTH_SIGNING_KEY")
    master_key = _decode_b64_key(master_key_b64, expected_len=32, label="MASTER_KEY")

    if signing_key is None:
        if service_token or master_key:
            raise ValueError(
                "auth env vars inconsistent: SERVICE_TOKEN or MASTER_KEY is set but "
                "neither verification key is set — refusing a partially-auth state"
            )
        log.warning(
            "EUGENE_PLEXUS_DRIVER_AUTH_SIGNING_KEY not set — running unauthenticated "
            "(dev/standalone mode). Production spawns via agent always supply this."
        )
        return AuthState(signing_key=None, service_token=None, master_key=None)

    return AuthState(
        signing_key=signing_key,
        service_token=service_token or None,
        master_key=master_key,
    )
