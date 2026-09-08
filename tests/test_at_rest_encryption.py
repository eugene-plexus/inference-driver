"""Tests for Phase 6 — at-rest encryption of sensitive config fields.

Exercises ConfigStore directly (the encryption boundary) plus the
round-trip through the file system. The watchdog provides the master
key in production; tests construct one and pass it in directly.

Three scenarios matter and are covered:

  * No master key — store behaves like v0.1: plaintext on disk,
    plaintext on read. Back-compat with existing dev installs and
    the pre-login window before the operator has unlocked.
  * Master key present — sensitive fields are sealed into envelopes
    on disk; in-memory `get()` always returns plaintext.
  * Plaintext v0.1 config on disk + master key now available —
    loads fine (treats plaintext as plaintext) and auto-upgrades to
    an envelope on the next save.

`Envelope`/`seal`/`open_envelope` get their own focused tests so a
crypto regression points at the exact primitive rather than at
ConfigStore.
"""

from __future__ import annotations

import secrets
from pathlib import Path

import pytest
import yaml

from eugene_plexus_inference_driver import security
from eugene_plexus_inference_driver._generated.models import ConfigUpdateRequest
from eugene_plexus_inference_driver.config import ConfigStore

# --------------------------------------------------------------------------- #
# Primitive: Envelope round-trip
# --------------------------------------------------------------------------- #


def test_envelope_round_trip() -> None:
    key = secrets.token_bytes(32)
    plaintext = "sk-test-abc123"
    envelope = security.seal(plaintext, key)
    assert envelope.alg == security.ENVELOPE_ALG
    assert envelope.nonce != envelope.ciphertext  # both base64; sanity
    assert security.open_envelope(envelope, key) == plaintext


def test_envelope_rejects_wrong_key() -> None:
    a = secrets.token_bytes(32)
    b = secrets.token_bytes(32)
    envelope = security.seal("secret", a)
    with pytest.raises(ValueError, match="decryption failed"):
        security.open_envelope(envelope, b)


def test_envelope_rejects_wrong_size_key() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        security.seal("x", b"\x00" * 16)


def test_envelope_to_from_dict_round_trip() -> None:
    key = secrets.token_bytes(32)
    e1 = security.seal("hello", key)
    e2 = security.Envelope.from_dict(e1.to_dict())
    assert e1 == e2


def test_envelope_from_dict_rejects_wrong_alg() -> None:
    with pytest.raises(ValueError, match="unsupported envelope alg"):
        security.Envelope.from_dict({"alg": "aes-gcm", "nonce": "AA==", "ciphertext": "AA=="})


def test_is_envelope_discriminator() -> None:
    """A plain string and a non-envelope dict should NOT be flagged
    as envelopes — false positives here would mean ConfigStore tries
    to decrypt operator-typed plaintext and silently zeroes it."""
    assert security.is_envelope({"alg": security.ENVELOPE_ALG, "nonce": "n", "ciphertext": "c"})
    assert not security.is_envelope("sk-plain")
    assert not security.is_envelope({"alg": "other", "nonce": "n", "ciphertext": "c"})
    assert not security.is_envelope({"alg": security.ENVELOPE_ALG, "nonce": "n"})


# --------------------------------------------------------------------------- #
# ConfigStore: no master key — v0.1 behavior preserved
# --------------------------------------------------------------------------- #


def _patch(d: dict[str, object]) -> ConfigUpdateRequest:
    return ConfigUpdateRequest.model_validate(d)


def test_no_master_key_writes_apikey_plaintext_on_disk(tmp_path: Path) -> None:
    store = ConfigStore(tmp_path / "config.yaml", master_key=None)
    store.load()
    result = store.apply_patch(_patch({"provider": "openai", "apiKey": "sk-plain"}))
    assert result.rejected == []

    raw = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert raw["apiKey"] == "sk-plain"  # plaintext, no envelope


# --------------------------------------------------------------------------- #
# ConfigStore: master key present — sensitive fields encrypt on save
# --------------------------------------------------------------------------- #


def test_master_key_writes_apikey_as_envelope_on_disk(tmp_path: Path) -> None:
    key = secrets.token_bytes(32)
    store = ConfigStore(tmp_path / "config.yaml", master_key=key)
    store.load()
    store.apply_patch(_patch({"provider": "openai", "apiKey": "sk-secret"}))

    raw = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert security.is_envelope(raw["apiKey"]), f"apiKey not encrypted on disk: {raw['apiKey']!r}"
    # The ciphertext must not contain the plaintext.
    assert "sk-secret" not in str(raw["apiKey"])


def test_non_sensitive_fields_not_encrypted(tmp_path: Path) -> None:
    """logLevel isn't sensitive — must NOT be wrapped in an envelope."""
    key = secrets.token_bytes(32)
    store = ConfigStore(tmp_path / "config.yaml", master_key=key)
    store.load()
    store.apply_patch(_patch({"logLevel": "DEBUG"}))

    raw = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert raw["logLevel"] == "DEBUG"


def test_master_key_get_returns_plaintext_after_save(tmp_path: Path) -> None:
    """In-memory `get(apiKey)` must return plaintext so the engine can
    set `Authorization: Bearer ...` — the encryption is purely an
    on-disk concern."""
    key = secrets.token_bytes(32)
    store = ConfigStore(tmp_path / "config.yaml", master_key=key)
    store.load()
    store.apply_patch(_patch({"provider": "openai", "apiKey": "sk-secret"}))
    assert store.get("apiKey") == "sk-secret"


def test_envelope_round_trips_through_disk(tmp_path: Path) -> None:
    """Save with master key, construct a fresh store with the same key
    pointing at the same file, load — plaintext must come back."""
    key = secrets.token_bytes(32)
    path = tmp_path / "config.yaml"

    writer = ConfigStore(path, master_key=key)
    writer.load()
    writer.apply_patch(_patch({"provider": "openai", "apiKey": "sk-secret"}))

    reader = ConfigStore(path, master_key=key)
    reader.load()
    assert reader.get("apiKey") == "sk-secret"


# --------------------------------------------------------------------------- #
# ConfigStore: graceful degradation when key is missing or wrong
# --------------------------------------------------------------------------- #


def test_load_envelope_without_key_drops_to_none(tmp_path: Path) -> None:
    """If the on-disk config has envelopes but the operator hasn't
    unlocked (no master key), the field loads as None rather than
    raising. The engine then fails with the existing "no API key"
    error — a clean degraded-mode story."""
    key = secrets.token_bytes(32)
    path = tmp_path / "config.yaml"

    writer = ConfigStore(path, master_key=key)
    writer.load()
    writer.apply_patch(_patch({"provider": "openai", "apiKey": "sk-secret"}))

    no_key_reader = ConfigStore(path, master_key=None)
    no_key_reader.load()
    assert no_key_reader.get("apiKey") is None
    # Non-sensitive fields stay readable so /v1/config still works.
    assert no_key_reader.get("provider") == "openai"


def test_load_envelope_with_wrong_key_drops_to_none(tmp_path: Path) -> None:
    a = secrets.token_bytes(32)
    b = secrets.token_bytes(32)
    path = tmp_path / "config.yaml"

    writer = ConfigStore(path, master_key=a)
    writer.load()
    writer.apply_patch(_patch({"provider": "openai", "apiKey": "sk-secret"}))

    wrong_reader = ConfigStore(path, master_key=b)
    wrong_reader.load()
    assert wrong_reader.get("apiKey") is None


# --------------------------------------------------------------------------- #
# ConfigStore: v0.1 plaintext on disk → still loads, upgrades on save
# --------------------------------------------------------------------------- #


def test_plaintext_v01_config_loads_and_upgrades(tmp_path: Path) -> None:
    """An existing v0.1 install has plaintext apiKey on disk. The new
    store must read it as plaintext, and on the next save (master key
    now in play) write it back as an envelope."""
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"provider": "openai", "apiKey": "sk-legacy"}),
        encoding="utf-8",
    )

    key = secrets.token_bytes(32)
    store = ConfigStore(path, master_key=key)
    store.load()
    assert store.get("apiKey") == "sk-legacy"  # plaintext read OK

    # Any save now re-serializes with encryption applied to apiKey.
    store.apply_patch(_patch({"logLevel": "WARNING"}))

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert security.is_envelope(raw["apiKey"]), (
        "apiKey didn't auto-upgrade to envelope after first encrypted save"
    )
    # And it still decrypts to the original.
    envelope = security.Envelope.from_dict(raw["apiKey"])
    assert security.open_envelope(envelope, key) == "sk-legacy"
