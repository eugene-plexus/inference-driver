"""Entrypoint: `python -m eugene_plexus_inference_driver`."""

from __future__ import annotations

import logging
import os

import uvicorn

from .app import create_app
from .config import ConfigStore
from .settings import load_settings

# Default bind port when neither the agent (via env var) nor the
# operator (via standalone launch) overrides it. Matches the smoke-test
# convention for the canonical bicameral pair.
_DEFAULT_PORT = 8081


def _resolve_port(bootstrap_store: ConfigStore) -> int:
    """Resolution order, highest precedence first:

    1. `EUGENE_PLEXUS_DRIVER_BIND_PORT` env var — the agent sets this when
       it spawns the driver, parsed from the topology's component URL.
       Agent-supervised installs always hit this branch.
    2. Built-in default (8081). Used when running the driver standalone
       outside the agent. The `port` field used to live in the
       per-driver config file; it's gone now (one source of truth: the
       agent topology owns ports).
    """
    env_port = os.environ.get("EUGENE_PLEXUS_DRIVER_BIND_PORT")
    if env_port:
        return int(env_port)
    return _DEFAULT_PORT


def main() -> None:
    settings = load_settings()

    # Bootstrap the config store just to discover log_level. Ports are
    # owned by the agent now (or the default for standalone launch).
    bootstrap_store = ConfigStore(settings.config_file)
    if not settings.safe_mode:
        bootstrap_store.load()

    port = _resolve_port(bootstrap_store)

    # Wire the application logger to the same level as uvicorn so
    # `log.warning(...)` calls in our code (the apiKey-decryption
    # warning is the noisy example) emit with timestamp + level +
    # logger name. uvicorn only configures its own loggers; the root
    # logger has no handler by default, so without basicConfig our
    # warnings arrive bare-message and untimestamped. `force=True`
    # overrides any prior basicConfig (uvicorn touches logging
    # before we get here).
    log_level = str(bootstrap_store.get("logLevel") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.bind_host,
        port=port,
        log_level=log_level.lower(),
    )


if __name__ == "__main__":
    main()
