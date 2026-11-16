"""Row 40: the provider's price for each model, in exact micro-dollars."""

import datetime as dt

from eugene_plexus_inference_driver.engines._catalogue import from_openrouter
from eugene_plexus_inference_driver.model_prices import openrouter_price


def test_an_openrouter_listing_is_priced_exactly():
    (model,) = from_openrouter(
        {
            "data": [
                {
                    "id": "anthropic/claude-sonnet-4.5",
                    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                    # Dollars a token, as OpenRouter lists them: $3 and $15 a million.
                    "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                }
            ]
        }
    )
    assert model.price is not None
    assert (model.price.inputPerMillionMicros, model.price.outputPerMillionMicros) == (
        3_000_000,
        15_000_000,
    )
    assert model.price.source.value == "openrouter"
    assert model.price.asOf == dt.datetime.now(dt.UTC).date()


def test_a_price_finer_than_a_pico_dollar_rounds_up_and_a_float_is_no_price():
    fine = openrouter_price({"pricing": {"prompt": "0.0000000000001", "completion": "0"}})
    assert fine is not None and fine.inputPerMillionMicros == 1
    assert openrouter_price({"pricing": {"prompt": 0.000003, "completion": "0"}}) is None


def test_a_variable_or_missing_price_is_none_never_free():
    assert openrouter_price({"pricing": {"prompt": "-1", "completion": "-1"}}) is None
    assert openrouter_price({"pricing": {"prompt": "0.1"}}) is None
    assert openrouter_price({"id": "x"}) is None
    (model,) = from_openrouter({"data": [{"id": "openrouter/auto"}]})
    assert model.price is None


def test_no_built_in_table_prices_a_provider_that_reports_none():
    """Prices come only from a provider's own listing: an OpenAI account's models,
    whose listing carries no price, have none (unpriced, never a remembered price)."""
    from eugene_plexus_inference_driver import model_prices
    from eugene_plexus_inference_driver.engines._catalogue import EngineDefaults, from_openai_list

    assert not hasattr(model_prices, "table_price")
    listed = from_openai_list(
        {"data": [{"id": "gpt-4o-mini", "object": "model"}]},
        EngineDefaults(supported_settings=[], streaming=True, tool_calling=True),
        classify=None,
    )
    assert [m.price for m in listed] == [None]
