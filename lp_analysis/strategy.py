"""Build and simulate leveraged concentrated-liquidity strategies.

The engine deliberately ignores any specific protocol's contract layout and
operates on the generic StrategyConfig / LendingPolicy abstractions. Any
debt provider with per-asset CF/LF and a borrow APR fits.

Lifecycle of a position:
    1. open_position(config)  - sizes liquidity from leveraged notional,
                                 derives the debt token amount so that
                                 borrowed value at t=0 equals
                                 capital_asset1 * (leverage - 1).
    2. step_position(state, price, days)  - re-evaluates LP token amounts
                                 at the new pool price and accrues debt.
    3. simulate(config, scenario) - applies step_position across the entire
                                 ScenarioPath and returns batched arrays of
                                 per-step state (efficient for big paths).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import cl_math
from .lending import accrue_continuous, accrue_continuous_path
from .types import PositionState, ScenarioPath, StrategyConfig


# ----- Initial sizing ---------------------------------------------------- #


def _initial_debt_amount(config: StrategyConfig) -> float:
    """Debt token quantity needed so its value at price_initial == leveraged borrow."""
    if config.leverage <= 1.0 or config.debt_symbol is None:
        return 0.0
    borrowed_value_asset1 = config.capital_asset1 * (config.leverage - 1.0)
    if config.debt_symbol == config.pool.pair.asset0:
        return borrowed_value_asset1 / config.pool.price_initial
    if config.debt_symbol == config.pool.pair.asset1:
        return borrowed_value_asset1
    # Pool-leg restriction is enforced by StrategyConfig.__post_init__.
    raise AssertionError("unreachable: debt_symbol not a pool leg")


def open_position(config: StrategyConfig) -> PositionState:
    """Build a PositionState at t=0 with the configured leverage and debt."""
    total_value = config.capital_asset1 * config.leverage
    liquidity = cl_math.liquidity_from_value_asset1(
        total_value,
        config.pool.price_initial,
        config.pool.price_range,
    )
    amount0, amount1 = cl_math.amounts_from_liquidity(
        liquidity, config.pool.price_initial, config.pool.price_range
    )
    debt_amount = _initial_debt_amount(config)
    return PositionState(
        t_days=0.0,
        price=config.pool.price_initial,
        amount0=amount0,
        amount1=amount1,
        liquidity=liquidity,
        debt_token=config.debt_symbol,
        debt_amount=debt_amount,
        config=config,
    )


# ----- Single-step evolution -------------------------------------------- #


def step_position(state: PositionState, new_price: float, t_days: float) -> PositionState:
    """Move an existing position to (new_price, t_days). Liquidity L is constant."""
    amount0, amount1 = cl_math.amounts_from_liquidity(
        state.liquidity, new_price, state.config.pool.price_range
    )
    if state.debt_token is None:
        new_debt = 0.0
    else:
        apr_pct = state.config.lending.apr(state.debt_token)
        elapsed = max(0.0, t_days - state.t_days)
        new_debt = accrue_continuous(state.debt_amount, apr_pct, elapsed)
    return PositionState(
        t_days=t_days,
        price=new_price,
        amount0=amount0,
        amount1=amount1,
        liquidity=state.liquidity,
        debt_token=state.debt_token,
        debt_amount=new_debt,
        config=state.config,
    )


# ----- Batched simulation ----------------------------------------------- #


@dataclass(frozen=True)
class SimulationResult:
    """Batched per-step arrays produced by `simulate`."""

    config: StrategyConfig
    scenario: ScenarioPath
    amount0: np.ndarray
    amount1: np.ndarray
    debt_amount: np.ndarray           # debt-token units (post-interest)
    debt_value_asset1: np.ndarray     # debt value in pool quote
    position_value_asset1: np.ndarray  # MtM value in pool quote (no fees)
    equity_asset1: np.ndarray         # position - debt, in pool quote
    initial: PositionState

    @property
    def liquidity(self) -> float:
        return self.initial.liquidity

    @property
    def n_steps(self) -> int:
        return int(self.scenario.n_steps)


def simulate(config: StrategyConfig, scenario: ScenarioPath) -> SimulationResult:
    """Apply the strategy to every step of `scenario` in a vectorised way."""
    initial = open_position(config)
    prices = scenario.price_path
    days = scenario.t_days

    amount0, amount1 = cl_math.amounts_path_from_liquidity(
        initial.liquidity, prices, config.pool.price_range
    )

    if initial.debt_token is None:
        debt_amount = np.zeros_like(prices)
    else:
        apr_pct = config.lending.apr(initial.debt_token)
        debt_amount = accrue_continuous_path(initial.debt_amount, apr_pct, days)

    if initial.debt_token == config.pool.pair.asset0:
        debt_value_asset1 = debt_amount * prices
    elif initial.debt_token == config.pool.pair.asset1:
        debt_value_asset1 = debt_amount.copy()
    else:
        debt_value_asset1 = np.zeros_like(prices)

    position_value_asset1 = amount0 * prices + amount1
    equity_asset1 = position_value_asset1 - debt_value_asset1

    return SimulationResult(
        config=config,
        scenario=scenario,
        amount0=amount0,
        amount1=amount1,
        debt_amount=debt_amount,
        debt_value_asset1=debt_value_asset1,
        position_value_asset1=position_value_asset1,
        equity_asset1=equity_asset1,
        initial=initial,
    )


# ----- Convenience builders --------------------------------------------- #


def linear_price_sweep(
    config: StrategyConfig,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    n_points: int = 300,
    days_elapsed: float = 0.0,
) -> ScenarioPath:
    """Build a price-only sweep around the LP range (zero or constant time)."""
    if price_min is None:
        price_min = config.pool.price_range.lower * 0.8
    if price_max is None:
        price_max = config.pool.price_range.upper * 1.2
    if price_min <= 0 or price_max <= price_min:
        raise ValueError("require 0 < price_min < price_max")
    prices = np.linspace(price_min, price_max, n_points)
    days = np.full(n_points, float(days_elapsed))
    return ScenarioPath(price_path=prices, t_days=days)
