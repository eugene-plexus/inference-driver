"""Startup-time settings, sourced from environment variables.

Distinct from the runtime *config* (see `config.py`), which is editable via
`PATCH /v1/config` at runtime. These settings only control bootstrap:
where to find the config file, which port to bind, etc. Once the config
file is loaded, runtime config takes precedence for everything it covers.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EUGENE_PLEXUS_DRIVER_",
        env_file=None,
        case_sensitive=False,
    )

    config_file: Path = Path("config.yaml")
    """Where the runtime config is persisted. PATCH /v1/config writes here."""

    bind_host: str = "127.0.0.1"
    """Network interface to bind. Override to 0.0.0.0 for tailnet exposure."""

    safe_mode: bool = False
    """If true, skip loading the persisted config file at startup and run on
    built-in defaults. Set by the watchdog via EUGENE_PLEXUS_DRIVER_SAFE_MODE=1
    when a previous boot failed because the config was broken; lets the
    operator reach /v1/config to fix it. PATCH /v1/config still writes to
    `config_file` normally, so the next non-safe-mode boot picks up the
    repair. Per the safe-mode contract in specs/openapi/inference-driver.yaml."""

    auth_signing_key: str | None = None
    """Base64-encoded 32-byte HMAC signing key, supplied by the watchdog at
    spawn time (EUGENE_PLEXUS_DRIVER_AUTH_SIGNING_KEY). When absent the driver
    runs unauthenticated — dev / standalone path only; production via the
    watchdog always supplies this."""

    service_token: str | None = None
    """Long-lived service JWT (EUGENE_PLEXUS_DRIVER_SERVICE_TOKEN). Not consumed
    by the driver in v0.2 — captured for v0.3 when hemispheres may read
    from memory. The watchdog supplies it for symmetry with other kinds."""

    master_key: str | None = None
    """Base64-encoded 32-byte secretbox key for at-rest decryption
    (EUGENE_PLEXUS_DRIVER_MASTER_KEY). Reserved for Phase 6 (encrypted apiKey
    in adapter config); Phase 4 does not consume it."""


def load_settings() -> Settings:
    return Settings()
