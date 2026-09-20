"""Token verification with public Ed25519 PEM.
Legacy 32-byte HS256 verification is supported only while the install
retains its old key. No algorithm is selected from the token header.
This component receives no private signing key in Ed25519 mode.
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from typing import Any

import jwt
import nacl.exceptions
import nacl.secret
import nacl.utils
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

log = logging.getLogger(__name__)


def verification_key(key: bytes) -> bytes:
    """Validate public Ed25519 PEM, or an explicitly legacy 32-byte HMAC key."""
    if len(key) == 32:
        return key
    parsed = serialization.load_pem_public_key(key)
    if not isinstance(parsed, Ed25519PublicKey):
        raise ValueError("token verification requires an Ed25519 public key")
    return key


def verification_algorithm(key: bytes) -> str:
    """Select from trusted key material, never an untrusted JWT header."""
    return "HS256" if len(key) == 32 else "EdDSA"


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


CLOCK_SKEW_LEEWAY_SECONDS = 300
"""How far apart two hosts' clocks may drift before a token is refused.

Five minutes: Kerberos's `MaxClockSkew`, and the window Entra and most
OAuth validators apply to `iat`, `nbf` and `exp`. **It was zero until
2026-09-15.** On the live two-machine install the control root's clock
ran half a second ahead of a worker whose Windows Time service had
stopped, and every token the root minted in the first half of each
second was refused by that worker as "not yet valid (iat)" a few
milliseconds later. Long-lived tokens (the gateway's, the operator's
session) passed, every health check said ok, and the root listed the
node `down` with no reason -- so it read as a key or enrollment fault
and was neither. A skew large enough to matter for security is a
broken clock; a broken clock is *reported* (`_note_clock_skew`), not
enforced by refusing traffic between two healthy hosts.
"""

_SKEW_WARN_AFTER_SECONDS = 2.0
_SKEW_WARN_INTERVAL_SECONDS = 60.0
_last_skew_warning = 0.0


def _note_clock_skew(iat: int, *, now: float | None = None) -> None:
    """Warn, at most once a minute, when a token was issued in this host's future.

    Accepted within `CLOCK_SKEW_LEEWAY_SECONDS`, so nothing breaks. Logged
    so a wrong clock on either host is visible long before the skew grows
    past the leeway and starts refusing traffic.
    """
    global _last_skew_warning
    current = time.time() if now is None else now
    ahead = iat - current
    if ahead <= _SKEW_WARN_AFTER_SECONDS:
        return
    if current - _last_skew_warning < _SKEW_WARN_INTERVAL_SECONDS:
        return
    _last_skew_warning = current
    log.warning(
        "accepted a token issued %.1f s in this host's future: the issuer's clock or "
        "this host's is wrong (tolerated up to %d s, then tokens are refused)",
        ahead,
        CLOCK_SKEW_LEEWAY_SECONDS,
    )


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
    claims = jwt.decode(
        token,
        key=verification_key(signing_key),
        algorithms=[verification_algorithm(signing_key)],
        options=options,
        leeway=CLOCK_SKEW_LEEWAY_SECONDS,
    )
    _note_clock_skew(int(claims["iat"]))

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
    to the agent's `MasterKeyEnvelope` schema in common.yaml so
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
