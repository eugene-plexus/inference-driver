"""Row 40: the provider's price for each model, in exact micro-dollars."""

import datetime as dt

from eugene_plexus_inference_driver._generated.models import DriverModel
from eugene_plexus_inference_driver.engines._catalogue import from_openrouter
from eugene_plexus_inference_driver.model_prices import (
    openrouter_price,
    priced,
    table_price,
)


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


def test_the_table_prices_an_openai_account_and_leaves_others_alone():
    price = table_price("openai", "gpt-4o-mini")
    assert price is not None
    assert (price.inputPerMillionMicros, price.outputPerMillionMicros) == (150_000, 600_000)
    assert price.source.value == "table" and price.asOf is not None
    assert table_price("openai", "not-a-listed-model") is None
    assert table_price("ollama_local", "gpt-4o-mini") is None
    models = [DriverModel(id="gpt-4o-mini", surfaces=["chat"]), DriverModel(id="x", surfaces=[])]
    first, second = priced(models, "openai")
    assert first.price is not None and second.price is None
    # A local engine's models never take a table price.
    assert priced(models, "ollama_local")[0].price is None
