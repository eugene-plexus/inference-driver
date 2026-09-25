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

    trust_bundle_file: str | None = None
    """The trust bundle the agent keeps beside `node.yaml`, reloaded when it changes."""

    trust_authority: str | None = None
    """The public key that bundle must be signed by (base64url Ed25519)."""

    auth_recipient: str | None = None
    """This machine as a token's audience names it: `node:<name>`."""

    service_token: str | None = None
    """This driver's own token (EUGENE_PLEXUS_DRIVER_SERVICE_TOKEN), addressed
    to this machine alone. Presented to this machine's agent when a
    `runtimeName` is resolved to a URL -- the agent's reads accept its own
    children's tokens. Worth nothing on any other machine."""

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
