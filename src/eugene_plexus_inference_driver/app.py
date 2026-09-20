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
from .engines.base import BackendEngine
from .providers import get_provider
from .routes import admin as admin_routes
from .routes import config as config_routes
from .routes import generate as generate_routes
from .routes import health as health_routes
from .routes import info as info_routes
from .runtime_lookup import resolve_runtime_url
from .settings import Settings, load_settings

log = logging.getLogger(__name__)

#: Turns a `runtimeName` into the URL that runtime listens on. Built once
#: per app from its settings and auth state — see `runtime_resolver_for`.
RuntimeResolver = Callable[[str], str]


def runtime_resolver_for(app: FastAPI) -> RuntimeResolver:
    """The resolver this app uses: the agent from settings, the service
    token the agent handed us at spawn."""
    settings: Settings = app.state.settings
    service_token: str | None = getattr(app.state.auth_state, "service_token", None)

    def resolve(name: str) -> str:
        return resolve_runtime_url(name, agent_url=settings.agent_url, service_token=service_token)

    return resolve


def build_engine_with(
    get: Callable[[str], Any],
    *,
    resolve_runtime: RuntimeResolver | None = None,
) -> BackendEngine:
    """Construct an engine from a key->value getter.

    Reads `provider` from the getter, looks up its registry entry, and
    asks the entry's engine class to construct itself from the same
    getter (with provider-specific kwargs forwarded). Used directly by
    `/v1/config/test` to build a temporary engine from saved config +
    transient overrides without touching the persisted store.

    When the config names a `runtimeName` and the provider's engine can
    follow one, the name is resolved to a URL through `resolve_runtime`
    first — that is the whole of the M2 routing-gap fix on the driver's
    side. A resolution failure raises, so the driver comes up degraded
    with the reason on `/healthz` and the config endpoints reachable.
    """
    provider_key = str(get("provider") or "").strip()
    if not provider_key:
        raise ValueError("config has no `provider` set; pick one in the UI / config file")
    provider = get_provider(provider_key)

    kwargs: dict[str, Any] = dict(provider.engine_kwargs)
    runtime_name = str(get("runtimeName") or "").strip()
    if runtime_name:
        if getattr(provider.engine_class, "follows_runtimes", False):
            if resolve_runtime is None:
                raise ValueError(
                    f"`runtimeName` is {runtime_name!r} but this driver has no way to reach "
                    f"the agent to resolve it"
                )
            kwargs["runtime_url"] = resolve_runtime(runtime_name)
            kwargs["runtime_name"] = runtime_name
        else:
            # A stale value left behind by a provider switch. The field is
            # only shown for the custom provider, so say so rather than
            # fail a subscription-backed driver over it.
            log.warning(
                "`runtimeName` %r is set but provider %r does not front a supervised "
                "runtime; ignoring it",
                runtime_name,
                provider_key,
            )

    # `engine_class` is `Any` in the registry (Protocol classes are
    # invariant in `type[]`), but every registered class implements
    # `BackendEngine` — annotate the return through a local cast.
    engine: BackendEngine = provider.engine_class.from_config(get, **kwargs)
    return engine


def build_engine(
    store: ConfigStore, *, resolve_runtime: RuntimeResolver | None = None
) -> BackendEngine:
    """Construct the configured engine from the runtime config store."""
    return build_engine_with(store.get, resolve_runtime=resolve_runtime)


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
            verify_key_b64=settings.auth_verify_key,
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
            "starting in SAFE MODE (EUGENE_PLEXUS_DRIVER_SAFE_MODE=1); ignoring "
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
            engine = build_engine(store, resolve_runtime=runtime_resolver_for(app))
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

    try:
        yield
    finally:
        # The engine owns one HTTP client for the life of the process
        # (see `OpenAiCompatibleHttpEngine._client`). Release its
        # connection pool on the way out rather than leaving sockets to
        # the garbage collector, which on Windows is what produces the
        # "address already in use" a restart then trips over.
        closer = getattr(app.state.adapter, "aclose", None)
        if closer is not None:
            try:
                await closer()
            except Exception as e:  # never fail a shutdown over cleanup
                log.debug("engine aclose failed: %s", e)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a FastAPI app with all routers mounted."""
    settings = settings or load_settings()

    app = FastAPI(
        title="Eugene Plexus — inference-driver",
        description="A uniform HTTP surface over one model backend.",
        version=__version__,
        lifespan=_lifespan,
    )
    app.state.settings = settings

    # Health stays unauthenticated — supervisors and load balancers need
    # to probe it without holding credentials.
    app.include_router(health_routes.router)

    # Mixed surfaces: /v1/info backs UI model dropdowns (operator) and
    # gateway health checks (service); /v1/generate is called by
    # the gateway (service:gateway) but operators can hit it
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
