"""
Leveraged LP strategy construction and path simulation.
"""
from typing import Optional, Sequence, Union

import numpy as np

from .cl_math import amounts_from_liquidity, liquidity_from_value_asset1
from .types import DebtAsset, LPConfig, LPPosition, LeveragedLPConfig, LeveragedPosition, SimulationResult


def accrue_debt_amount(amount: float, apr_percent: float, days_elapsed: float) -> float:
    """Continuously compound a debt token amount over elapsed days."""
    if days_elapsed < 0:
        raise ValueError("days_elapsed must be non-negative")
    if amount == 0:
        return 0.0
    annual_rate = apr_percent / 100.0
    return float(amount * np.exp(annual_rate * days_elapsed / 365.0))


def debt_price_asset1(
    config: LeveragedLPConfig,
    price_asset1_per_asset0: float,
    external_debt_price_asset1: Optional[float] = None,
) -> float:
    """Return debt-token price denominated in asset1."""
    if config.debt_asset == DebtAsset.ASSET0:
        return price_asset1_per_asset0
    if config.debt_asset == DebtAsset.ASSET1:
        return 1.0
    if config.debt_asset == DebtAsset.EXTERNAL:
        price = external_debt_price_asset1
        if price is None:
            price = config.debt_price_initial_asset1
        if price is None or price <= 0:
            raise ValueError("External debt requires a positive debt price in asset1")
        return price
    return 0.0


def initial_debt_amounts(config: LeveragedLPConfig) -> tuple[float, float, float, float]:
    """Calculate initial debt token amounts and debt price in asset1."""
    borrowed_value_asset1 = config.capital_asset1 * (config.leverage - 1.0)
    debt_price = debt_price_asset1(config, config.price_initial, config.debt_price_initial_asset1)

    if borrowed_value_asset1 == 0:
        return 0.0, 0.0, 0.0, debt_price
    if config.debt_asset == DebtAsset.ASSET0:
        return borrowed_value_asset1 / config.price_initial, 0.0, 0.0, debt_price
    if config.debt_asset == DebtAsset.ASSET1:
        return 0.0, borrowed_value_asset1, 0.0, debt_price
    if config.debt_asset == DebtAsset.EXTERNAL:
        return 0.0, 0.0, borrowed_value_asset1 / debt_price, debt_price
    return 0.0, 0.0, 0.0, debt_price


def create_initial_position(config: LPConfig) -> LPPosition:
    """Create a non-leveraged CL position from asset1-denominated capital."""
    liquidity = liquidity_from_value_asset1(
        config.capital_asset1,
        config.price_initial,
        config.price_range,
    )
    amount0, amount1 = amounts_from_liquidity(liquidity, config.price_initial, config.price_range)
    return LPPosition(
        amount0=amount0,
        amount1=amount1,
        liquidity=liquidity,
        current_price=config.price_initial,
        config=config,
    )


def create_initial_leveraged_position(config: LeveragedLPConfig) -> LeveragedPosition:
    """Create an initial leveraged CL position with explicit debt token amounts."""
    total_value_asset1 = config.capital_asset1 * config.leverage
    liquidity = liquidity_from_value_asset1(
        total_value_asset1,
        config.price_initial,
        config.price_range,
    )
    amount0, amount1 = amounts_from_liquidity(liquidity, config.price_initial, config.price_range)
    debt_amount0, debt_amount1, debt_amount_external, debt_price = initial_debt_amounts(config)
    return LeveragedPosition(
        amount0=amount0,
        amount1=amount1,
        liquidity=liquidity,
        current_price=config.price_initial,
        debt_amount0=debt_amount0,
        debt_amount1=debt_amount1,
        config=config,
        debt_amount_external=debt_amount_external,
        debt_price_asset1=debt_price,
    )


def update_position_at_price(position: LPPosition, new_price: float) -> LPPosition:
    """Calculate a non-leveraged position at a new price."""
    amount0, amount1 = amounts_from_liquidity(
        position.liquidity,
        new_price,
        position.config.price_range,
    )
    return LPPosition(
        amount0=amount0,
        amount1=amount1,
        liquidity=position.liquidity,
        current_price=new_price,
        config=position.config,
    )


def update_leveraged_position_at_price(
    position: LeveragedPosition,
    new_price: float,
    days_elapsed: float = 0.0,
    external_debt_price_asset1: Optional[float] = None,
) -> LeveragedPosition:
    """Calculate a leveraged position at a new price and elapsed time."""
    amount0, amount1 = amounts_from_liquidity(
        position.liquidity,
        new_price,
        position.config.price_range,
    )
    debt_price = debt_price_asset1(position.config, new_price, external_debt_price_asset1)
    return LeveragedPosition(
        amount0=amount0,
        amount1=amount1,
        liquidity=position.liquidity,
        current_price=new_price,
        debt_amount0=accrue_debt_amount(position.debt_amount0, position.config.borrow_apr, days_elapsed),
        debt_amount1=accrue_debt_amount(position.debt_amount1, position.config.borrow_apr, days_elapsed),
        config=position.config,
        debt_amount_external=accrue_debt_amount(
            position.debt_amount_external,
            position.config.borrow_apr,
            days_elapsed,
        ),
        debt_price_asset1=debt_price,
    )


def _as_float_array(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _optional_path(
    values: Optional[Union[float, Sequence[float]]],
    size: int,
    name: str,
) -> np.ndarray:
    if values is None:
        return np.zeros(size, dtype=float)
    if isinstance(values, (int, float)):
        return np.full(size, float(values), dtype=float)
    array = _as_float_array(values, name)
    if array.size != size:
        raise ValueError(f"{name} length must match prices length")
    return array


def simulate_price_path(
    config: Union[LPConfig, LeveragedLPConfig],
    prices: Sequence[float],
    days_elapsed: Optional[Union[float, Sequence[float]]] = None,
    external_debt_prices_asset1: Optional[Union[float, Sequence[float]]] = None,
) -> SimulationResult:
    """Simulate an LP or leveraged LP over an explicit price/time path."""
    price_path = _as_float_array(prices, "prices")
    if np.any(price_path <= 0):
        raise ValueError("prices must be positive")

    day_path = _optional_path(days_elapsed, price_path.size, "days_elapsed")

    if isinstance(config, LeveragedLPConfig):
        if external_debt_prices_asset1 is None:
            debt_price_path = np.full(
                price_path.size,
                debt_price_asset1(config, config.price_initial, config.debt_price_initial_asset1),
                dtype=float,
            )
        else:
            debt_price_path = _optional_path(
                external_debt_prices_asset1,
                price_path.size,
                "external_debt_prices_asset1",
            )
        initial_position = create_initial_leveraged_position(config)
        positions = [
            update_leveraged_position_at_price(
                initial_position,
                float(price),
                float(days),
                float(debt_price),
            )
            for price, days, debt_price in zip(price_path, day_path, debt_price_path)
        ]
    else:
        initial_position = create_initial_position(config)
        positions = [
            update_position_at_price(initial_position, float(price))
            for price in price_path
        ]

    return SimulationResult(
        prices=price_path.tolist(),
        positions=positions,
        metrics={},
        config=config,
    )
