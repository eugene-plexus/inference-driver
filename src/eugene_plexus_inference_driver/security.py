"""v0.2 security primitives.

The inference-driver is never the trust root. The watchdog generates
the per-restart HMAC signing key and the install-wide master key
(libsodium secretbox), and distributes both via env vars
(`EUGENE_PLEXUS_DRIVER_AUTH_SIGNING_KEY`, `EUGENE_PLEXUS_DRIVER_MASTER_KEY`).
This module exposes:

  * JWT decode (verify-only) for inbound bearer token validation —
    mirror of `eugene_plexus_orchestrator.security` and
    `eugene_plexus_memory.security`.
  * Secretbox envelope `seal` / `open_envelope` for at-rest encryption
    of sensitive config fields (Phase 6). The wire shape is identical
    to the watchdog's `MasterKeyEnvelope` in common.yaml so envelopes
    written by one component can be opened by another given the same
    master key — useful for future per-component key rotation flows.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import jwt
import nacl.exceptions
import nacl.secret
import nacl.utils

_JWT_ALG = "HS256"

AUDIENCE_OPERATOR = "operator"
SERVICE_AUDIENCE_PREFIX = "service:"

# Envelope `alg` constant — matches MasterKeyEnvelope.alg in common.yaml.
ENVELOPE_ALG = "secretbox-xsalsa20poly1305"


@dataclass(frozen=True)
class TokenPayload:
    """Decoded JWT claims. `iat` / `exp` are unix seconds."""

    sub: str
    aud: str
    iat: int
    exp: int


def decode_token(
    *,
    token: str,
    signing_key: bytes,
    accept_operator: bool = True,
    accept_any_service: bool = True,
) -> TokenPayload:
    """Verify a bearer token's signature + expiry and return its claims.

    `accept_operator` / `accept_any_service` together decide which
    audiences are acceptable. Used by:

      * `require_authorized` (operator + service:*) — /v1/info,
        /v1/generate, /v1/config GETs.
      * `require_operator` (operator only) — PATCH /v1/config,
        /v1/admin/restart.

    Raises `jwt.InvalidTokenError` (or its subclass
    `InvalidAudienceError`) on any failure so the dependency can
    collapse all auth-rejection paths into one except branch.
    """
    if not (accept_operator or accept_any_service):
        raise ValueError("must accept at least one audience class")

    options: Any = {
        "require": ["sub", "aud", "iat", "exp"],
        "verify_aud": False,
    }
    claims = jwt.decode(token, key=signing_key, algorithms=[_JWT_ALG], options=options)

    aud = str(claims["aud"])
    is_operator = accept_operator and aud == AUDIENCE_OPERATOR
    is_service = accept_any_service and aud.startswith(SERVICE_AUDIENCE_PREFIX)
    if not (is_operator or is_service):
        raise jwt.InvalidAudienceError(f"audience {aud!r} not accepted")

    return TokenPayload(
        sub=str(claims["sub"]),
        aud=aud,
        iat=int(claims["iat"]),
        exp=int(claims["exp"]),
    )


# --------------------------------------------------------------------------- #
# At-rest envelope encryption (libsodium secretbox)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Envelope:
    """Canonical shape of an at-rest encrypted secret. Wire-identical
    to the watchdog's `MasterKeyEnvelope` schema in common.yaml so
    envelopes round-trip across components if both hold the same
    master key."""

    alg: str
    nonce: str  # base64
    ciphertext: str  # base64

    def to_dict(self) -> dict[str, str]:
        return {"alg": self.alg, "nonce": self.nonce, "ciphertext": self.ciphertext}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Envelope:
        alg = raw.get("alg")
        nonce = raw.get("nonce")
        ciphertext = raw.get("ciphertext")
        if alg != ENVELOPE_ALG:
            raise ValueError(f"unsupported envelope alg: {alg!r}")
        if not isinstance(nonce, str) or not isinstance(ciphertext, str):
            raise ValueError("envelope nonce/ciphertext must be base64 strings")
        return cls(alg=alg, nonce=nonce, ciphertext=ciphertext)


def is_envelope(value: Any) -> bool:
    """Quick discriminator — is this a dict shaped like an Envelope?

    Used by the config loader to decide "decrypt this vs. take as
    plaintext" on every field on disk. The check is structural (alg
    + nonce + ciphertext keys present) so an operator's hand-edited
    plaintext can't accidentally look like an envelope.
    """
    return (
        isinstance(value, dict)
        and value.get("alg") == ENVELOPE_ALG
        and "nonce" in value
        and "ciphertext" in value
    )


def seal(plaintext: str, master_key: bytes) -> Envelope:
    """Encrypt a plaintext config value. Fresh 24-byte nonce per call.

    Raises ValueError on a misshaped master key — the only way that
    can happen at runtime is a wiring bug, so prefer loud over
    silent."""
    if len(master_key) != 32:
        raise ValueError("master key must be 32 bytes")
    box = nacl.secret.SecretBox(master_key)
    nonce = nacl.utils.random(nacl.secret.SecretBox.NONCE_SIZE)
    ciphertext = box.encrypt(plaintext.encode("utf-8"), nonce).ciphertext
    return Envelope(
        alg=ENVELOPE_ALG,
        nonce=base64.b64encode(nonce).decode("ascii"),
        ciphertext=base64.b64encode(ciphertext).decode("ascii"),
    )


def open_envelope(envelope: Envelope, master_key: bytes) -> str:
    """Decrypt back to plaintext. Raises ValueError on bad key /
    tampered ciphertext / malformed envelope fields."""
    if len(master_key) != 32:
        raise ValueError("master key must be 32 bytes")
    try:
        nonce = base64.b64decode(envelope.nonce, validate=True)
        ciphertext = base64.b64decode(envelope.ciphertext, validate=True)
    except Exception as e:
        raise ValueError(f"envelope decoding failed: {e}") from e
    box = nacl.secret.SecretBox(master_key)
    try:
        return box.decrypt(ciphertext, nonce).decode("utf-8")
    except nacl.exceptions.CryptoError as e:
        raise ValueError(f"envelope decryption failed: {e}") from e
