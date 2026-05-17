"""
Pure concentrated-liquidity mathematics.
"""
from typing import Tuple

import numpy as np

from .types import PriceRange


EPSILON = 1e-12


def sqrt_price(price: float) -> float:
    """Return sqrt(price) after validating positive price."""
    if price <= 0:
        raise ValueError("Price must be positive")
    return float(np.sqrt(price))


def liquidity_from_value_asset1(
    value_asset1: float,
    price: float,
    price_range: PriceRange,
    epsilon: float = EPSILON,
) -> float:
    """Calculate CL liquidity from a total value denominated in asset1."""
    if value_asset1 < 0:
        raise ValueError("Value must be non-negative")
    if not price_range.contains(price):
        raise ValueError("Initial price must be within the liquidity range")

    sp = sqrt_price(price)
    sp_a = sqrt_price(price_range.lower)
    sp_b = sqrt_price(price_range.upper)
    denominator = price * (1 / sp - 1 / sp_b) + (sp - sp_a)
    if denominator <= epsilon:
        raise ValueError("Invalid range configuration")
    return value_asset1 / denominator


def amounts_from_liquidity(
    liquidity: float,
    price: float,
    price_range: PriceRange,
) -> Tuple[float, float]:
    """Calculate token0 and token1 amounts for liquidity at price."""
    if liquidity <= 0:
        return 0.0, 0.0

    sp = sqrt_price(price)
    sp_a = sqrt_price(price_range.lower)
    sp_b = sqrt_price(price_range.upper)

    if price <= price_range.lower:
        return liquidity * (1 / sp_a - 1 / sp_b), 0.0
    if price >= price_range.upper:
        return 0.0, liquidity * (sp_b - sp_a)
    return liquidity * (1 / sp - 1 / sp_b), liquidity * (sp - sp_a)


def liquidity_from_amounts(
    amount0: float,
    amount1: float,
    price: float,
    price_range: PriceRange,
    epsilon: float = EPSILON,
) -> float:
    """Calculate usable CL liquidity from token amounts."""
    if amount0 < 0 or amount1 < 0:
        raise ValueError("Token amounts must be non-negative")

    sp = sqrt_price(price)
    sp_a = sqrt_price(price_range.lower)
    sp_b = sqrt_price(price_range.upper)

    if price <= price_range.lower:
        return amount0 / (1 / sp_a - 1 / sp_b) if amount0 > epsilon else 0.0
    if price >= price_range.upper:
        return amount1 / (sp_b - sp_a) if amount1 > epsilon else 0.0

    candidates = []
    if amount0 > epsilon:
        candidates.append(amount0 / (1 / sp - 1 / sp_b))
    if amount1 > epsilon:
        candidates.append(amount1 / (sp - sp_a))
    return min(candidates) if candidates else 0.0


def value_asset1(amount0: float, amount1: float, price: float) -> float:
    """Value token amounts in asset1 terms."""
    return amount0 * price + amount1
