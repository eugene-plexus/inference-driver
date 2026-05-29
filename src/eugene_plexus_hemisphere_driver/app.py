"""FastAPI app factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI

from . import __version__
from .auth_state import load_auth_state
from .config import ConfigStore
from .dependencies import require_authorized, require_operator
from .engines.base import HemisphereEngine
from .providers import get_provider
from .routes import admin as admin_routes
from .routes import config as config_routes
from .routes import generate as generate_routes
from .routes import health as health_routes
from .routes import info as info_routes
from .settings import Settings, load_settings

log = logging.getLogger(__name__)


def build_engine_with(get: Callable[[str], Any]) -> HemisphereEngine:
    """Construct an engine from a key->value getter.

    Reads `provider` from the getter, looks up its registry entry, and
    asks the entry's engine class to construct itself from the same
    getter (with provider-specific kwargs forwarded). Used directly by
    `/v1/config/test` to build a temporary engine from saved config +
    transient overrides without touching the persisted store.
    """
    provider_key = str(get("provider") or "").strip()
    if not provider_key:
        raise ValueError("config has no `provider` set; pick one in the UI / config file")
    provider = get_provider(provider_key)
    # `engine_class` is `Any` in the registry (Protocol classes are
    # invariant in `type[]`), but every registered class implements
    # `HemisphereEngine` — annotate the return through a local cast.
    engine: HemisphereEngine = provider.engine_class.from_config(get, **provider.engine_kwargs)
    return engine


def build_engine(store: ConfigStore) -> HemisphereEngine:
    """Construct the configured engine from the runtime config store."""
    return build_engine_with(store.get)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    # v0.2 auth state has to be built BEFORE the ConfigStore so its
    # master key can be threaded into the store for at-rest envelope
    # decryption of sensitive fields (apiKey, etc). Tests can pre-
    # populate `app.state.auth_state`; production reads it from env.
    if not hasattr(app.state, "auth_state"):
        app.state.auth_state = load_auth_state(
            signing_key_b64=settings.auth_signing_key,
            service_token=settings.service_token,
            master_key_b64=settings.master_key,
        )

    store = ConfigStore(settings.config_file, master_key=app.state.auth_state.master_key)
    if settings.safe_mode:
        # Safe mode: skip the on-disk config entirely, leaving the store
        # populated with built-in defaults. PATCH /v1/config still writes
        # to disk, so the operator's repair survives the next boot. No
        # engine is constructed — /v1/generate reports degraded.
        log.warning(
            "starting in SAFE MODE (EUGENE_PLEXUS_HD_SAFE_MODE=1); ignoring "
            "%s and running on defaults. Fix config via /v1/config, then "
            "restart without the env var.",
            settings.config_file,
        )
    else:
        store.load()
    app.state.config_store = store
    app.state.safe_mode = settings.safe_mode

    # Engine construction can fail (missing API key, unknown provider,
    # bad binary path, etc). The driver MUST come up anyway so its
    # /v1/config endpoints stay reachable — otherwise a broken config
    # locks operators out of fixing it through the UI, exactly the
    # OpenClaw failure mode this project exists to avoid. We record the
    # error on app.state and let /v1/generate surface it as a 503 until
    # the config is fixed and the driver restarted.
    if settings.safe_mode:
        # No engine in safe mode — defaults have no provider set, so
        # `build_engine` would raise "no provider". Skip cleanly with
        # an explicit safe-mode marker on app.state.
        app.state.adapter = None
        app.state.adapter_error = "running in safe mode"
    else:
        try:
            engine = build_engine(store)
            app.state.adapter = engine  # historical name; routes still read `app.state.adapter`
            app.state.adapter_error = None
            log.info("engine ready: backend=%s", engine.backend_kind.value)
        except Exception as e:
            app.state.adapter = None
            app.state.adapter_error = str(e)
            log.error(
                "engine initialization failed (%s); driver running in degraded "
                "mode — fix config via /v1/config and restart",
                e,
            )

    # Discover the engine's available models for the modelId dropdown
    # in the UI. Best-effort: an unreachable backend leaves the list
    # empty and the schema falls back to free-text input. Failure here
    # is NEVER fatal — the driver itself is otherwise up.
    app.state.available_models = []
    if app.state.adapter is not None:
        try:
            models = await app.state.adapter.list_models()
            app.state.available_models = list(models)
            log.info(
                "discovered %d models from %s",
                len(app.state.available_models),
                app.state.adapter.backend_kind.value,
            )
        except Exception as e:
            log.warning(
                "list_models failed for %s: %s",
                app.state.adapter.backend_kind.value,
                e,
            )

    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a FastAPI app with all routers mounted."""
    settings = settings or load_settings()

    app = FastAPI(
        title="Eugene Plexus — hemisphere-driver",
        description="One half of an Eugene Plexus bicameral pair.",
        version=__version__,
        lifespan=_lifespan,
    )
    app.state.settings = settings

    # Health stays unauthenticated — supervisors and load balancers need
    # to probe it without holding credentials.
    app.include_router(health_routes.router)

    # Mixed surfaces: /v1/info backs UI model dropdowns (operator) and
    # orchestrator health checks (service); /v1/generate is called by
    # the orchestrator (service:orchestrator) but operators can hit it
    # too for one-off testing.
    authorized = [Depends(require_authorized)]
    app.include_router(info_routes.router, dependencies=authorized)
    app.include_router(generate_routes.router, dependencies=authorized)

    # Operator-only surfaces: config edits and the restart trigger ride
    # on the UI's session token. Service tokens are rejected so a
    # compromised peer can't reconfigure the driver.
    operator_only = [Depends(require_operator)]
    app.include_router(config_routes.router, dependencies=operator_only)
    app.include_router(admin_routes.router, dependencies=operator_only)

    return app
