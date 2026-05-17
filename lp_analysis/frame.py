"""Build a polars DataFrame from a SimulationResult, with multi-numeraire columns.

Columns are added in three layers:
    1. Always present  - step, t_days, price, amounts, debt, asset1-denominated values
    2. Margin layer    - Arcadia-style collateral/liquidation/used margin per step
    3. Numeraire layer - for every USD price path supplied in the scenario, the
                         frame gains *_usd_<symbol> columns for position, debt,
                         equity values converted via that symbol's USD path.
                         A reserved "USD" key yields raw *_usd columns.

This keeps the API tiny while supporting the user's main requirement: see
the same strategy expressed in BTC, ETH, USDC and USD simultaneously.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import polars as pl

from .analytics import compute_exposure, margin_path
from .strategy import SimulationResult


def _maybe_collect_usd_columns(
    result: SimulationResult,
) -> dict[str, np.ndarray]:
    """Compute USD-denominated columns when the scenario carries the needed paths."""
    pair = result.config.pool.pair
    usd_paths = result.scenario.usd_prices
    if pair.asset0 not in usd_paths or pair.asset1 not in usd_paths:
        return {}

    p0_usd = usd_paths[pair.asset0]
    p1_usd = usd_paths[pair.asset1]
    position_usd = result.amount0 * p0_usd + result.amount1 * p1_usd

    debt_token = result.initial.debt_token
    if debt_token is None:
        debt_usd = np.zeros_like(position_usd)
    elif debt_token in usd_paths:
        debt_usd = result.debt_amount * usd_paths[debt_token]
    elif debt_token == pair.asset0:
        debt_usd = result.debt_amount * p0_usd
    elif debt_token == pair.asset1:
        debt_usd = result.debt_amount * p1_usd
    else:
        # Should not happen given StrategyConfig validation; default to zero.
        debt_usd = np.zeros_like(position_usd)

    return {
        f"asset0_usd_{pair.asset0}": p0_usd,
        f"asset1_usd_{pair.asset1}": p1_usd,
        "position_value_usd": position_usd,
        "debt_value_usd": debt_usd,
        "equity_usd": position_usd - debt_usd,
    }


def _numeraire_columns(
    result: SimulationResult,
    base_usd: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """For every USD-priced symbol in the scenario, add *_in_<symbol> columns."""
    if not base_usd:
        return {}
    position_usd = base_usd["position_value_usd"]
    debt_usd = base_usd["debt_value_usd"]
    equity_usd = base_usd["equity_usd"]
    out: dict[str, np.ndarray] = {}
    for sym, usd_path in result.scenario.usd_prices.items():
        out[f"position_value_in_{sym}"] = position_usd / usd_path
        out[f"debt_value_in_{sym}"] = debt_usd / usd_path
        out[f"equity_in_{sym}"] = equity_usd / usd_path
    return out


def to_frame(
    result: SimulationResult,
    include_margin: bool = True,
) -> pl.DataFrame:
    """Convert a SimulationResult into a tidy polars DataFrame.

    Always-present columns:
        step, t_days, price,
        amount_<asset0>, amount_<asset1>,
        debt_token, debt_amount, debt_value_asset1,
        position_value_asset1, equity_asset1,
        asset0_net_value_asset1, asset1_net_value_asset1   (LP - debt)

    Margin columns (when include_margin is True):
        margin_collateral_value, margin_liquidation_value,
        margin_used, margin_free, margin_health_ratio, margin_state

    USD/numeraire columns are appended when the scenario supplies the
    required USD price paths. See module docstring for details.
    """
    cfg = result.config
    pair = cfg.pool.pair
    n = result.n_steps

    exposure = compute_exposure(result)

    data: dict[str, object] = {
        "step": np.arange(n, dtype=np.int64),
        "t_days": result.scenario.t_days,
        "price": result.scenario.price_path,
        f"amount_{pair.asset0}": result.amount0,
        f"amount_{pair.asset1}": result.amount1,
        "debt_token": [result.initial.debt_token or ""] * n,
        "debt_amount": result.debt_amount,
        "debt_value_asset1": result.debt_value_asset1,
        "position_value_asset1": result.position_value_asset1,
        "equity_asset1": result.equity_asset1,
        "asset0_net_value_asset1": exposure.asset0_net_value_asset1,
        "asset1_net_value_asset1": exposure.asset1_net_value_asset1,
    }

    if include_margin:
        rows = margin_path(result)
        data["margin_collateral_value"] = np.array([r["collateral_value"] for r in rows])
        data["margin_liquidation_value"] = np.array([r["liquidation_value"] for r in rows])
        data["margin_used"] = np.array([r["used_margin"] for r in rows])
        data["margin_free"] = np.array([r["free_margin"] for r in rows])
        data["margin_health_ratio"] = np.array([r["health_ratio"] for r in rows])
        data["margin_state"] = [r["state"] for r in rows]

    usd_cols = _maybe_collect_usd_columns(result)
    data.update(usd_cols)
    data.update(_numeraire_columns(result, usd_cols))

    return pl.DataFrame(data)
