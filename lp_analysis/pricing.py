"""Pure numeraire conversion - no API calls, no external data fetching.

All inputs are arrays/scalars supplied by the caller as part of the scenario.
Given a pool price `P = price_asset1_per_asset0` and (optionally) USD prices
per symbol, this module converts a balance into any numeraire (asset0,
asset1, USD, or any other symbol that has a USD price available).

The triangulation rule:
    value_usd(symbol)         = balance(symbol) * usd_prices[symbol]
    value_asset1(symbol)      = balance(symbol) * (P if symbol == asset0 else 1)
    value_in_X(balance_usd)   = balance_usd / usd_prices[X]

If the user supplies USD prices for any symbol referenced by a report, that
symbol becomes a usable numeraire. Stablecoins like USDC simply have a path
of ones (or a fixed depeg series) - exactly as the user prefers to model.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def to_usd(
    symbol_balances: Mapping[str, np.ndarray | Sequence[float] | float],
    usd_prices: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Sum dollar value across multiple symbol balances.

    Every symbol in `symbol_balances` must have a USD price array in
    `usd_prices` of matching length.
    """
    total: np.ndarray | None = None
    for sym, bal in symbol_balances.items():
        if sym not in usd_prices:
            raise KeyError(f"usd_prices missing symbol {sym!r}")
        bal_arr = np.asarray(bal, dtype=float)
        usd_arr = np.asarray(usd_prices[sym], dtype=float)
        if bal_arr.shape != usd_arr.shape:
            raise ValueError(
                f"shape mismatch for {sym!r}: balance {bal_arr.shape} vs usd {usd_arr.shape}"
            )
        contrib = bal_arr * usd_arr
        total = contrib if total is None else total + contrib
    if total is None:
        raise ValueError("symbol_balances is empty")
    return total


def convert_usd(value_usd: np.ndarray, numeraire_usd_price: np.ndarray) -> np.ndarray:
    """Convert a USD-denominated value path into another numeraire by its USD price."""
    value_usd = np.asarray(value_usd, dtype=float)
    px = np.asarray(numeraire_usd_price, dtype=float)
    if value_usd.shape != px.shape:
        raise ValueError("value_usd and numeraire_usd_price must share shape")
    return value_usd / px


def value_in_asset1(amount0: np.ndarray, amount1: np.ndarray, price: np.ndarray) -> np.ndarray:
    """Convert (amount0, amount1) into asset1 (quote) units."""
    amount0 = np.asarray(amount0, dtype=float)
    amount1 = np.asarray(amount1, dtype=float)
    price = np.asarray(price, dtype=float)
    return amount0 * price + amount1


def value_in_asset0(amount0: np.ndarray, amount1: np.ndarray, price: np.ndarray) -> np.ndarray:
    """Convert (amount0, amount1) into asset0 (base) units (1/price * amount1 + amount0)."""
    amount0 = np.asarray(amount0, dtype=float)
    amount1 = np.asarray(amount1, dtype=float)
    price = np.asarray(price, dtype=float)
    return amount0 + amount1 / price
