"""Core data types for leveraged-LP strategy analysis.

Three-layer config separation:
    PoolConfig      - CL pool parameters (pair, range, initial price, fee)
    LendingPolicy   - Debt-provider parameters (CF, LF, APR, min margin)
    StrategyConfig  - User-level spec (capital, leverage, debt symbol)

This separation lets the same code model Arcadia, Aave, Moonwell or any other
debt provider that exposes asset-level haircut factors plus an interest rate.

ScenarioPath bundles the time-indexed market inputs that drive a simulation:
the pool price path, elapsed days at each step, and optional USD price paths
per symbol (used to produce values in numeraires beyond the pool's own legs).

PositionState is a per-step snapshot produced by the strategy engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np


# ----- Pool --------------------------------------------------------------- #


@dataclass(frozen=True)
class AssetPair:
    """Concentrated-liquidity pair. Pool price is asset1 per 1 asset0."""

    asset0: str
    asset1: str

    def __str__(self) -> str:
        return f"{self.asset0}/{self.asset1}"


@dataclass(frozen=True)
class PriceRange:
    """Inclusive concentrated-liquidity range in pool-price units."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        if not (0.0 < self.lower < self.upper):
            raise ValueError("PriceRange requires 0 < lower < upper")

    def contains(self, price: float) -> bool:
        return self.lower <= price <= self.upper

    def width_pct(self, ref_price: float) -> float:
        return (self.upper - self.lower) / ref_price * 100.0


@dataclass(frozen=True)
class PoolConfig:
    """Static pool parameters for a single CL position."""

    pair: AssetPair
    price_initial: float       # asset1 per 1 asset0
    price_range: PriceRange
    fee_tier_pct: float = 0.30  # display-only; trading fees applied externally

    def __post_init__(self) -> None:
        if self.price_initial <= 0:
            raise ValueError("price_initial must be positive")
        if not self.price_range.contains(self.price_initial):
            raise ValueError("price_initial must lie within price_range")


# ----- Lending ----------------------------------------------------------- #


@dataclass(frozen=True)
class LendingPolicy:
    """Generic debt-provider policy applied per asset symbol.

    Conventions follow the Arcadia whitepaper:
        * collateral_factor (CF) and liquidation_factor (LF) are in [0, 1].
        * available_margin = Σ CF_i · value_i  (a.k.a. "collateral value").
        * liquidation_value = Σ LF_i · value_i.
        * Arcadia's CF = 1 / (1 + Initial Margin); LF = 1 / (1 + Maintenance Margin).
        * Aave: CF == LTV; LF == Liquidation Threshold (same math, different name).
        * Moonwell: CF acts as the per-asset collateral cap.

    `lp_protocol_factor` is the protocol-level haircut for derived assets
    (LP NFTs in Arcadia speak). The Arcadia paper expresses an LP risk factor
    as CF_LP = protocol_factor · min(CF_asset0, CF_asset1). Here we apply it
    multiplicatively at the position level, matching the prototype convention.

    `borrow_apr_by_symbol` lets each borrowable token have its own APR (e.g.
    USDC at 6%, ETH at 2%, WBTC at 1.5%). A missing symbol defaults to 0%.
    """

    collateral_factors: Mapping[str, float]
    liquidation_factors: Mapping[str, float]
    borrow_apr_by_symbol: Mapping[str, float] = field(default_factory=dict)
    lp_protocol_factor: float = 1.0
    minimum_margin_usd: float = 0.0

    def __post_init__(self) -> None:
        for sym, cf in self.collateral_factors.items():
            lf = self.liquidation_factors.get(sym)
            if lf is None:
                raise ValueError(f"liquidation_factors missing symbol {sym!r}")
            if not (0.0 <= cf <= lf <= 1.0):
                raise ValueError(f"need 0 <= CF[{sym}]={cf} <= LF[{sym}]={lf} <= 1")
        if not (0.0 < self.lp_protocol_factor <= 1.0):
            raise ValueError("lp_protocol_factor must be in (0, 1]")
        if self.minimum_margin_usd < 0:
            raise ValueError("minimum_margin_usd must be non-negative")

    def cf(self, symbol: str) -> float:
        try:
            return self.collateral_factors[symbol]
        except KeyError as exc:
            raise KeyError(f"No collateral factor for {symbol!r}") from exc

    def lf(self, symbol: str) -> float:
        try:
            return self.liquidation_factors[symbol]
        except KeyError as exc:
            raise KeyError(f"No liquidation factor for {symbol!r}") from exc

    def apr(self, symbol: str) -> float:
        return float(self.borrow_apr_by_symbol.get(symbol, 0.0))


# ----- Strategy ---------------------------------------------------------- #


@dataclass(frozen=True)
class StrategyConfig:
    """A leveraged LP position spec.

    `capital_asset1` is initial equity denominated in the pool's quote asset.
    The total notional deployed into the LP equals capital_asset1 * leverage,
    and the debt token amount is sized so its value equals
    capital_asset1 * (leverage - 1) at the initial price.
    """

    name: str
    pool: PoolConfig
    lending: LendingPolicy
    capital_asset1: float
    leverage: float = 1.0
    debt_symbol: Optional[str] = None  # required when leverage > 1

    def __post_init__(self) -> None:
        if self.capital_asset1 <= 0:
            raise ValueError("capital_asset1 must be positive")
        if self.leverage < 1.0:
            raise ValueError("leverage must be >= 1.0")
        if self.leverage > 1.0:
            if self.debt_symbol is None:
                raise ValueError("debt_symbol required when leverage > 1")
            if self.debt_symbol not in self.lending.collateral_factors:
                raise ValueError(
                    f"debt_symbol {self.debt_symbol!r} missing from "
                    "LendingPolicy.collateral_factors"
                )
            pool_syms = {self.pool.pair.asset0, self.pool.pair.asset1}
            if self.debt_symbol not in pool_syms:
                raise ValueError(
                    f"debt_symbol {self.debt_symbol!r} must be one of "
                    f"pool legs {pool_syms} (external-debt not yet supported)"
                )


# ----- Scenario / state -------------------------------------------------- #


@dataclass(frozen=True)
class ScenarioPath:
    """Time-indexed market inputs for a simulation.

    `price_path` is the pool price (asset1 per asset0) at each step.
    `t_days` is the elapsed days since strategy opening at each step
    (monotonic, may equal zero for a pure price sweep).

    `usd_prices` optionally provides USD-per-token paths for any symbol
    referenced by the report (e.g. {"ETH": ..., "USDC": ..., "WBTC": ...}).
    Conversions into USD-or-arbitrary-numeraire reports require enough of
    these paths to triangulate; see `lp_analysis.pricing`.
    """

    price_path: np.ndarray              # shape (T,)
    t_days: np.ndarray                  # shape (T,)
    usd_prices: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pp = np.asarray(self.price_path, dtype=float)
        td = np.asarray(self.t_days, dtype=float)
        if pp.ndim != 1 or pp.size == 0:
            raise ValueError("price_path must be a non-empty 1-D array")
        if td.shape != pp.shape:
            raise ValueError("t_days must match price_path length")
        if not np.all(np.isfinite(pp)) or np.any(pp <= 0):
            raise ValueError("price_path must be positive and finite")
        if not np.all(np.isfinite(td)) or np.any(td < 0):
            raise ValueError("t_days must be non-negative and finite")
        for sym, arr in self.usd_prices.items():
            a = np.asarray(arr, dtype=float)
            if a.shape != pp.shape:
                raise ValueError(f"usd_prices[{sym!r}] must match price_path length")
            if not np.all(np.isfinite(a)) or np.any(a <= 0):
                raise ValueError(f"usd_prices[{sym!r}] must be positive and finite")
        # Re-store as float arrays without violating frozen=True
        object.__setattr__(self, "price_path", pp)
        object.__setattr__(self, "t_days", td)
        object.__setattr__(
            self,
            "usd_prices",
            {sym: np.asarray(arr, dtype=float) for sym, arr in self.usd_prices.items()},
        )

    @property
    def n_steps(self) -> int:
        return int(self.price_path.size)


@dataclass(frozen=True)
class PositionState:
    """Snapshot of a strategy position at one step."""

    t_days: float
    price: float                 # asset1 per asset0
    amount0: float
    amount1: float
    liquidity: float
    debt_token: Optional[str]    # symbol of borrowed token, or None
    debt_amount: float           # debt in units of debt_token (after interest)
    config: StrategyConfig
