"""Concentrated-liquidity math (Uniswap v3 / Aerodrome Slipstream).

These are the only functions that touch sqrt-price geometry. They are pure
and side-effect free so the rest of the package can stay protocol-agnostic.

Reference: Uniswap v3 Whitepaper, sections 6.1-6.3.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .types import PriceRange

EPS = 1e-12


def sqrt_price(price: float) -> float:
    if price <= 0:
        raise ValueError("price must be positive")
    return math.sqrt(price)


def liquidity_from_value_asset1(
    value_asset1: float,
    price: float,
    price_range: PriceRange,
) -> float:
    """Liquidity L sized so the position is worth `value_asset1` at `price`.

    Derived from V(L, P) = L * [P * (1/√P - 1/√Pb) + (√P - √Pa)] for in-range P.
    """
    if value_asset1 < 0:
        raise ValueError("value_asset1 must be non-negative")
    if not price_range.contains(price):
        raise ValueError("price must lie within price_range")

    sp = sqrt_price(price)
    sp_a = sqrt_price(price_range.lower)
    sp_b = sqrt_price(price_range.upper)
    denom = price * (1.0 / sp - 1.0 / sp_b) + (sp - sp_a)
    if denom <= EPS:
        raise ValueError("degenerate range produces zero denominator")
    return value_asset1 / denom


def amounts_from_liquidity(
    liquidity: float,
    price: float,
    price_range: PriceRange,
) -> Tuple[float, float]:
    """Token amounts (asset0, asset1) implied by a given L at `price`."""
    if liquidity <= 0:
        return 0.0, 0.0

    sp = sqrt_price(price)
    sp_a = sqrt_price(price_range.lower)
    sp_b = sqrt_price(price_range.upper)

    if price <= price_range.lower:
        return liquidity * (1.0 / sp_a - 1.0 / sp_b), 0.0
    if price >= price_range.upper:
        return 0.0, liquidity * (sp_b - sp_a)
    return liquidity * (1.0 / sp - 1.0 / sp_b), liquidity * (sp - sp_a)


def liquidity_from_amounts(
    amount0: float,
    amount1: float,
    price: float,
    price_range: PriceRange,
) -> float:
    """Maximum usable liquidity given (amount0, amount1) at `price`."""
    if amount0 < 0 or amount1 < 0:
        raise ValueError("amount0/amount1 must be non-negative")

    sp = sqrt_price(price)
    sp_a = sqrt_price(price_range.lower)
    sp_b = sqrt_price(price_range.upper)

    if price <= price_range.lower:
        return amount0 / (1.0 / sp_a - 1.0 / sp_b) if amount0 > EPS else 0.0
    if price >= price_range.upper:
        return amount1 / (sp_b - sp_a) if amount1 > EPS else 0.0

    candidates = []
    if amount0 > EPS:
        candidates.append(amount0 / (1.0 / sp - 1.0 / sp_b))
    if amount1 > EPS:
        candidates.append(amount1 / (sp - sp_a))
    return min(candidates) if candidates else 0.0


def amounts_path_from_liquidity(
    liquidity: float,
    prices: np.ndarray,
    price_range: PriceRange,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised version of `amounts_from_liquidity` over a price array."""
    if liquidity <= 0:
        zero = np.zeros_like(prices, dtype=float)
        return zero, zero.copy()

    sp = np.sqrt(prices)
    sp_a = math.sqrt(price_range.lower)
    sp_b = math.sqrt(price_range.upper)
    inv_sp_a = 1.0 / sp_a
    inv_sp_b = 1.0 / sp_b

    a0 = np.where(
        prices <= price_range.lower,
        liquidity * (inv_sp_a - inv_sp_b),
        np.where(prices >= price_range.upper, 0.0, liquidity * (1.0 / sp - inv_sp_b)),
    )
    a1 = np.where(
        prices <= price_range.lower,
        0.0,
        np.where(prices >= price_range.upper, liquidity * (sp_b - sp_a), liquidity * (sp - sp_a)),
    )
    return a0.astype(float, copy=False), a1.astype(float, copy=False)


def value_asset1(amount0: float, amount1: float, price: float) -> float:
    """Position value denominated in asset1 (quote)."""
    return amount0 * price + amount1
