"""A transport address cannot silently change an explicitly selected protocol."""

import pytest

from eugene_plexus_inference_driver.engines import dialects
from eugene_plexus_inference_driver.engines.openai_compat_http import OpenAiCompatibleHttpEngine
from eugene_plexus_inference_driver.providers import PROVIDERS


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:9999/openai.com",
        "https://openai.com.example.test",
        "https://example.test/?upstream=api.openai.com",
        "https://api.openai.com@example.test",
    ],
)
def test_auto_does_not_infer_provider_from_substrings(url):
    assert dialects.select(None, base_url=url).name == "compatible"


def test_named_openai_keeps_its_protocol_behind_a_proxy():
    provider = PROVIDERS["openai"]
    settings = {"baseUrl": "https://proxy.example.test", "apiKey": "test", "modelId": "gpt-4o"}
    engine = provider.engine_class.from_config(settings.get, **provider.engine_kwargs)
    assert engine._dialect.openai_parameters
    assert engine._dialect.max_tokens_field == "max_completion_tokens"


def test_custom_proxy_can_select_its_wire_protocol():
    provider = PROVIDERS["openai_compat_custom"]
    settings = {"baseUrl": "https://proxy.example.test", "wireDialect": "openrouter"}
    engine = provider.engine_class.from_config(settings.get, **provider.engine_kwargs)
    assert engine._dialect.catalogue == "openrouter"


def test_explicit_compatible_does_not_change_with_host_name():
    engine = OpenAiCompatibleHttpEngine(api_key="test", dialect="compatible")
    assert engine._dialect.max_tokens_field == "max_tokens"
    assert engine._dialect.send_reasoning
