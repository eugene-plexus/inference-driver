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
    built-in defaults. Set by the agent via EUGENE_PLEXUS_DRIVER_SAFE_MODE=1
    when a previous boot failed because the config was broken; lets the
    operator reach /v1/config to fix it. PATCH /v1/config still writes to
    `config_file` normally, so the next non-safe-mode boot picks up the
    repair. Per the safe-mode contract in specs/openapi/inference-driver.yaml."""

    auth_verify_key: str | None = None
    """Base64 public Ed25519 PEM from the agent. Exclusive with auth_signing_key."""

    auth_signing_key: str | None = None
    """Legacy base64 32-byte HS256 key; used only until install rotation."""

    service_token: str | None = None
    """Long-lived service JWT (EUGENE_PLEXUS_DRIVER_SERVICE_TOKEN). Presented to
    the agent when a `runtimeName` is resolved to a URL — reads on the
    agent's `/v1/runtimes` accept any service token, the same rule that
    lets the gateway resolve topology. The agent supplies it at spawn."""

    agent_url: str = "http://127.0.0.1:8079"
    """Agent endpoint a `runtimeName` is resolved against
    (EUGENE_PLEXUS_DRIVER_AGENT_URL). Bootstrap-only, like the gateway's
    identical setting: the agent supervises only its own host, so the
    driver it spawned can always reach it on loopback at the default
    port. Override when the agent binds elsewhere."""

    master_key: str | None = None
    """Base64-encoded 32-byte secretbox key for at-rest decryption
    (EUGENE_PLEXUS_DRIVER_MASTER_KEY). Reserved for Phase 6 (encrypted apiKey
    in adapter config); Phase 4 does not consume it."""


def load_settings() -> Settings:
    return Settings()
