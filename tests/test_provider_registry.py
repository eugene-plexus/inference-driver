"""Every provider in the registry can build its engine.

A provider's `engine_kwargs` are forwarded to its engine class's
`from_config`; an argument that method requires and nobody passes is a
driver that comes up degraded on every start. `strata_local` was one from
the day it was added: the agent configures every Strata runtime's driver with
it, and no Strata driver ever answered, so nothing reached Strata through the
gateway (2026-10-10, Troy's live install). The tests built the engine
directly, never through the registry.
"""

from __future__ import annotations

import inspect

import pytest

from eugene_plexus_inference_driver.app import build_engine_with
from eugene_plexus_inference_driver.engines.strata_http import StrataHttpEngine
from eugene_plexus_inference_driver.providers import PROVIDERS

#: What `build_engine_with` itself passes, beside a provider's own kwargs.
BUILDER_PASSES = {"provider", "catalogue_path", "runtime_url", "runtime_name"}


@pytest.mark.parametrize("key", sorted(PROVIDERS))
def test_every_argument_from_config_requires_is_given(key: str) -> None:
    provider = PROVIDERS[key]
    signature = inspect.signature(provider.engine_class.from_config)
    required = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        and parameter.default is inspect.Parameter.empty
    }
    missing = required - set(provider.engine_kwargs) - BUILDER_PASSES
    assert not missing, f"{key}: {provider.engine_class.__name__}.from_config needs {missing}"
    unknown = set(provider.engine_kwargs) - set(signature.parameters)
    assert not unknown, f"{key}: {provider.engine_class.__name__}.from_config takes no {unknown}"


def test_a_strata_driver_builds_as_the_agent_configures_it() -> None:
    """The agent's companion for a Strata runtime: `strata_local`, the
    runtime by name, no slot pinning (agent `StrataAdapter.companion_overrides`)."""
    config = {
        "provider": "strata_local",
        "runtimeName": "qwen3-8-flash-next-iq2-xs",
        "slotPinning": False,
    }
    engine = build_engine_with(config.get, resolve_runtime=lambda _name: "http://127.0.0.1:8092")
    assert isinstance(engine, StrataHttpEngine)
    assert engine._base_url == "http://127.0.0.1:8092"
    assert engine.runtime == "qwen3-8-flash-next-iq2-xs"
