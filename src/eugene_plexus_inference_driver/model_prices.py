"""Row 40: the price a model's provider itself reports, per model.

The control root counts what each key, person and group spends, at the owner's
rate for a model when the owner set one, else at the provider's price this
driver reports here (`DriverModel.price`). The only source is a provider's own
listing: an OpenRouter account reads `pricing.prompt` and `pricing.completion`,
US dollars a token, as decimal strings.

Eugene keeps no price table of its own. A list written down once shows
yesterday's prices as facts (settings never lie); a model whose provider
reports no price has none, and the root counts it as unpriced, never as free.

A price is said in millionths of a dollar for a million tokens, which is one
pico-dollar a token, so every sum the root makes is an exact integer. A price
finer than that is rounded up. Nothing here is a float.
"""

from __future__ import annotations

import datetime as _dt
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from typing import Any

from ._generated.models import DriverModelPrice, DriverPriceSource

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
        # A float is refused, not converted: OpenRouter sends strings, so a
        # float is a listing that changed shape.
        return None
    try:
        return Decimal(str(raw).strip())
    except InvalidOperation:
        return None


def _day(raw: Any) -> _dt.date | None:
    try:
        return _dt.date.fromisoformat(raw) if isinstance(raw, str) else None
    except ValueError:
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
