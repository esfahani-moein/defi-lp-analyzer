"""
Tabular valuation outputs for LP strategy simulations.
"""
from typing import Optional, Sequence, Union

import numpy as np
import polars as pl

from ..lp_calc.types import DebtAsset, LeveragedLPConfig, LeveragedPosition, SimulationResult


def _path_or_none(
    values: Optional[Union[float, Sequence[float]]],
    size: int,
    name: str,
) -> Optional[np.ndarray]:
    if values is None:
        return None
    if isinstance(values, (int, float)):
        return np.full(size, float(values), dtype=float)
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size != size:
        raise ValueError(f"{name} must be scalar or match simulation length")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def simulation_to_frame(
    result: SimulationResult,
    asset0_usd: Optional[Union[float, Sequence[float]]] = None,
    asset1_usd: Optional[Union[float, Sequence[float]]] = None,
    debt_usd: Optional[Union[float, Sequence[float]]] = None,
) -> pl.DataFrame:
    """Convert a simulation result to a Polars valuation frame."""
    positions = result.positions
    size = len(positions)
    asset0_usd_path = _path_or_none(asset0_usd, size, "asset0_usd")
    asset1_usd_path = _path_or_none(asset1_usd, size, "asset1_usd")
    debt_usd_path = _path_or_none(debt_usd, size, "debt_usd")

    amount0 = np.array([position.amount0 for position in positions], dtype=float)
    amount1 = np.array([position.amount1 for position in positions], dtype=float)
    price = np.array(result.prices, dtype=float)
    position_value_asset1 = amount0 * price + amount1

    data: dict[str, object] = {
        'step': np.arange(size),
        'price_asset1_per_asset0': price,
        'amount_asset0': amount0,
        'amount_asset1': amount1,
        'position_value_asset1': position_value_asset1,
    }

    if isinstance(positions[0], LeveragedPosition):
        leveraged_positions = positions
        debt_amount0 = np.array([position.debt_amount0 for position in leveraged_positions], dtype=float)
        debt_amount1 = np.array([position.debt_amount1 for position in leveraged_positions], dtype=float)
        debt_amount_external = np.array(
            [position.debt_amount_external for position in leveraged_positions],
            dtype=float,
        )
        debt_price_asset1_path = np.array(
            [position.debt_price_asset1 for position in leveraged_positions],
            dtype=float,
        )
        debt_value_asset1 = np.array(
            [position.debt_value_asset1 for position in leveraged_positions],
            dtype=float,
        )
        data.update({
            'debt_amount_asset0': debt_amount0,
            'debt_amount_asset1': debt_amount1,
            'debt_amount_external': debt_amount_external,
            'debt_price_asset1': debt_price_asset1_path,
            'debt_value_asset1': debt_value_asset1,
            'equity_asset1': position_value_asset1 - debt_value_asset1,
            'ltv': np.array([position.ltv for position in leveraged_positions], dtype=float),
            'collateral_value_asset1': np.array(
                [position.collateral_value_asset1 for position in leveraged_positions],
                dtype=float,
            ),
            'liquidation_value_asset1': np.array(
                [position.liquidation_value_asset1 for position in leveraged_positions],
                dtype=float,
            ),
            'used_margin_asset1': np.array(
                [position.used_margin_asset1 for position in leveraged_positions],
                dtype=float,
            ),
            'free_margin_asset1': np.array(
                [position.free_margin_asset1 for position in leveraged_positions],
                dtype=float,
            ),
            'margin_health_ratio': np.array(
                [position.margin_health_ratio for position in leveraged_positions],
                dtype=float,
            ),
            'margin_state': [position.margin_state for position in leveraged_positions],
        })

    if asset0_usd_path is not None and asset1_usd_path is not None:
        position_value_usd = amount0 * asset0_usd_path + amount1 * asset1_usd_path
        data.update({
            'asset0_usd': asset0_usd_path,
            'asset1_usd': asset1_usd_path,
            'position_value_usd': position_value_usd,
        })
        if isinstance(positions[0], LeveragedPosition):
            config = result.config
            if debt_usd_path is None and isinstance(config, LeveragedLPConfig):
                if config.debt_asset == DebtAsset.ASSET0:
                    debt_usd_path = asset0_usd_path
                elif config.debt_asset == DebtAsset.ASSET1:
                    debt_usd_path = asset1_usd_path
                elif config.debt_asset == DebtAsset.EXTERNAL:
                    debt_usd_path = np.array(data['debt_price_asset1'], dtype=float) * asset1_usd_path
            if debt_usd_path is not None:
                debt_value_usd = (
                    np.array(data['debt_amount_asset0'], dtype=float) * asset0_usd_path
                    + np.array(data['debt_amount_asset1'], dtype=float) * asset1_usd_path
                    + np.array(data['debt_amount_external'], dtype=float) * debt_usd_path
                )
                data.update({
                    'debt_usd': debt_usd_path,
                    'debt_value_usd': debt_value_usd,
                    'equity_usd': position_value_usd - debt_value_usd,
                })

    return pl.DataFrame(data)
