"""At-rest envelope encryption for sensitive config values.

Token verification moved to `tokens.py` and `auth_state.py` with the
per-node token keys (2026-09-25): this component holds no key and checks
every bearer against the trust bundle its agent keeps.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import nacl.exceptions
import nacl.secret
import nacl.utils

ENVELOPE_ALG = "secretbox-xsalsa20poly1305"


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
