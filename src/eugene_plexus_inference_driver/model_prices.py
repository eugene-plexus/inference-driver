"""Row 40: the price a model's provider publishes, reported per model.

The control root counts what each key, person and group spends, at the owner's
rate for a model when the owner set one, else at the provider's price this
driver reports here (`DriverModel.price`). Two sources:

* an OpenRouter account reads it from its own listing: `pricing.prompt` and
  `pricing.completion`, US dollars a token, as decimal strings;
* an OpenAI, Anthropic or Gemini account reads Eugene's table
  (`model_prices.json`, copied from `specs/docs/data/model-prices.json`),
  US dollars a million tokens.

A price is said in millionths of a dollar for a million tokens, which is one
pico-dollar a token, so every sum the root makes is an exact integer. A price
finer than that is rounded up. Nothing here is a float. A model the driver has
no price for has none: the root counts it as unpriced, never as free.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from functools import cache
from importlib import resources
from typing import Any

from ._generated.models import DriverModel, DriverModelPrice, DriverPriceSource

log = logging.getLogger(__name__)

#: A provider's own name for its models in the table, by driver provider key.
TABLE_PROVIDERS = {"openai": "openai", "gemini": "gemini", "anthropic": "anthropic"}
#: A price above a million dollars a million tokens is not one.
MAX_MICROS = 10**12


def _micros(amount: Decimal) -> int | None:
    """Dollars a million tokens as whole micro-dollars, rounded up."""
    if not amount.is_finite() or amount < 0:
        return None
    micros = int((amount * 1_000_000).to_integral_value(rounding=ROUND_CEILING))
    return micros if micros <= MAX_MICROS else None


def _decimal(raw: Any) -> Decimal | None:
    if isinstance(raw, bool) or not isinstance(raw, str | int):
        # A float is refused, not converted: OpenRouter sends strings, and
        # the table is strings, so a float is a listing that changed shape.
        return None
    try:
        return Decimal(str(raw).strip())
    except InvalidOperation:
        return None


def openrouter_price(entry: Any, as_of: str | None = None) -> DriverModelPrice | None:
    """OpenRouter's `pricing` for a listed model: dollars a token.

    `-1` is OpenRouter's word for a variable price (its `openrouter/auto`
    router), and a missing or malformed field is no price: both are None.
    """
    pricing = entry.get("pricing") if isinstance(entry, dict) else None
    if not isinstance(pricing, dict):
        return None
    prompt, completion = _decimal(pricing.get("prompt")), _decimal(pricing.get("completion"))
    if prompt is None or completion is None:
        return None
    # Dollars a token times a million tokens is dollars a million tokens.
    inward, outward = _micros(prompt * 1_000_000), _micros(completion * 1_000_000)
    if inward is None or outward is None:
        return None
    return DriverModelPrice(
        inputPerMillionMicros=inward,
        outputPerMillionMicros=outward,
        source=DriverPriceSource.openrouter,
        asOf=_day(as_of),
    )


def _day(raw: Any) -> _dt.date | None:
    try:
        return _dt.date.fromisoformat(raw) if isinstance(raw, str) else None
    except ValueError:
        return None


@cache
def _table() -> dict[str, Any]:
    try:
        text = resources.files(__package__).joinpath("model_prices.json").read_text("utf-8")
        table = json.loads(text)
    except (OSError, ValueError) as exc:
        log.warning("The provider price table could not be read (%s); no table prices.", exc)
        return {}
    return table if isinstance(table, dict) else {}


def table_price(provider: str | None, model_id: str) -> DriverModelPrice | None:
    """The table's price for a model of an OpenAI, Anthropic or Gemini account."""
    table = _table()
    name = TABLE_PROVIDERS.get(provider or "")
    listed = ((table.get("providers") or {}).get(name) or {}) if name else {}
    entry = listed.get(model_id) if isinstance(listed, dict) else None
    if not isinstance(entry, dict):
        return None
    inward, outward = _decimal(entry.get("input")), _decimal(entry.get("output"))
    if inward is None or outward is None:
        return None
    micros_in, micros_out = _micros(inward), _micros(outward)
    if micros_in is None or micros_out is None:
        return None
    as_of = table.get("asOf")
    return DriverModelPrice(
        inputPerMillionMicros=micros_in,
        outputPerMillionMicros=micros_out,
        source=DriverPriceSource.table,
        asOf=_day(as_of),
    )


def priced(models: list[DriverModel], provider: str | None) -> list[DriverModel]:
    """Each model with the table's price, where it has none of its own."""
    if TABLE_PROVIDERS.get(provider or "") is None:
        return models
    out: list[DriverModel] = []
    for model in models:
        if model.price is None:
            found = table_price(provider, model.upstreamId or model.id)
            if found is not None:
                model = model.model_copy(update={"price": found})
        out.append(model)
    return out
