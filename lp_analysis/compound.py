"""Compound (outer-loop) strategies: lending leg + one-or-more LP positions.

Many real Arcadia strategies are funded by a lower-protocol loan (Moonwell,
Aave, ...) whose collateral is USDC and whose debt is the very tokens that
feed the LP. This module captures that *outer* leg with its own continuous
interest accrual, then aggregates it with one or more inner `SimulationResult`
objects produced by `lp_analysis.strategy.simulate`.

The split mirrors the codebase's separation of concerns:
    * `strategy.py`  - inner CL LP position (one per `StrategyConfig`)
    * `compound.py`  - outer collateralised loan plus portfolio aggregation

Everything in this module values positions in USD using the per-symbol USD
price paths carried by `ScenarioPath.usd_prices`. The pool price still drives
the inner LP math through the existing engine; USD prices add a numeraire
on top so the whole composite stays in one currency for reporting.

LP fees
-------
The inner engine does not model trading fees (they depend on volume which is
exogenous). `lp_fees_path` integrates a user-supplied APY against the LP
notional over time, accruing only while the pool price is inside the LP range.
This is the standard concentrated-liquidity convention: an out-of-range LP
earns nothing because its liquidity sits on one side of the book.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .lending import accrue_continuous_path
from .strategy import SimulationResult


# ----- Outer lending leg ------------------------------------------------- #


@dataclass(frozen=True)
class OuterLending:
    """A single-collateral, multi-debt lending position (Moonwell / Aave shape).

    All token quantities are in their native unit (e.g. USDC tokens, ETH,
    BTC). APRs are annual percentages applied with continuous compounding
    (matching `lending.accrue_continuous`). Collateral and liquidation
    factors are the lender's per-asset haircut on the **collateral** side
    (debt is always counted at full USD value).
    """

    name: str
    collateral_symbol: str
    collateral_amount: float
    supply_apr_pct: float
    debt_tokens: Mapping[str, float]
    borrow_apr_pct: Mapping[str, float]
    collateral_factor: float = 1.0
    liquidation_factor: float = 1.0
    protocol_name: str = "Moonwell"

    def __post_init__(self) -> None:
        if self.collateral_amount < 0:
            raise ValueError("collateral_amount must be non-negative")
        if not (0.0 < self.collateral_factor <= self.liquidation_factor <= 1.0):
            raise ValueError(
                "need 0 < collateral_factor <= liquidation_factor <= 1"
            )
        for sym in self.debt_tokens:
            if sym not in self.borrow_apr_pct:
                raise ValueError(f"borrow_apr_pct missing symbol {sym!r}")
            if self.debt_tokens[sym] < 0:
                raise ValueError(f"debt_tokens[{sym!r}] must be non-negative")


@dataclass(frozen=True)
class OuterLendingPath:
    """Time-evolution of an `OuterLending` position over a `ScenarioPath`."""

    position: OuterLending
    t_days: np.ndarray
    collateral_amount_path: np.ndarray            # tokens after supply accrual
    collateral_value_usd: np.ndarray              # tokens * usd_price
    debt_token_paths: Mapping[str, np.ndarray]    # symbol -> tokens after interest
    debt_value_usd_by_symbol: Mapping[str, np.ndarray]
    debt_value_usd: np.ndarray                    # sum across debt symbols
    equity_usd: np.ndarray                        # collateral_usd - debt_usd
    haircut_collateral_usd: np.ndarray            # collateral_factor * collateral_usd
    liquidation_value_usd: np.ndarray             # liquidation_factor * collateral_usd
    health_factor: np.ndarray                     # liquidation_value / debt_usd
    free_credit_usd: np.ndarray                   # haircut_collateral - debt
    is_liquidatable: np.ndarray                   # debt >= liquidation_value


def simulate_outer(
    pos: OuterLending,
    t_days: np.ndarray,
    usd_prices: Mapping[str, np.ndarray],
) -> OuterLendingPath:
    """Roll the outer lending leg over `t_days` with given USD price paths.

    `usd_prices` must contain a path for `pos.collateral_symbol` and for each
    debt symbol. All arrays share the same length as `t_days`.
    """
    t_days = np.asarray(t_days, dtype=float)
    if pos.collateral_symbol not in usd_prices:
        raise KeyError(
            f"usd_prices missing collateral symbol {pos.collateral_symbol!r}"
        )

    coll_price = np.asarray(usd_prices[pos.collateral_symbol], dtype=float)
    if coll_price.shape != t_days.shape:
        raise ValueError("usd_prices[collateral] must match t_days length")

    coll_tokens = accrue_continuous_path(
        pos.collateral_amount, pos.supply_apr_pct, t_days
    )
    coll_usd = coll_tokens * coll_price

    debt_token_paths: dict[str, np.ndarray] = {}
    debt_usd_by_symbol: dict[str, np.ndarray] = {}
    debt_total = np.zeros_like(t_days)
    for sym, amt0 in pos.debt_tokens.items():
        if sym not in usd_prices:
            raise KeyError(f"usd_prices missing debt symbol {sym!r}")
        price_path = np.asarray(usd_prices[sym], dtype=float)
        if price_path.shape != t_days.shape:
            raise ValueError(f"usd_prices[{sym!r}] must match t_days length")
        tokens = accrue_continuous_path(amt0, pos.borrow_apr_pct[sym], t_days)
        debt_token_paths[sym] = tokens
        usd = tokens * price_path
        debt_usd_by_symbol[sym] = usd
        debt_total = debt_total + usd

    haircut_coll = pos.collateral_factor * coll_usd
    liq_value = pos.liquidation_factor * coll_usd
    equity_usd = coll_usd - debt_total
    free_credit = haircut_coll - debt_total
    with np.errstate(divide="ignore", invalid="ignore"):
        hf = np.where(debt_total > 0, liq_value / debt_total, np.inf)
    is_liq = debt_total >= liq_value

    return OuterLendingPath(
        position=pos,
        t_days=t_days,
        collateral_amount_path=coll_tokens,
        collateral_value_usd=coll_usd,
        debt_token_paths=debt_token_paths,
        debt_value_usd_by_symbol=debt_usd_by_symbol,
        debt_value_usd=debt_total,
        equity_usd=equity_usd,
        haircut_collateral_usd=haircut_coll,
        liquidation_value_usd=liq_value,
        health_factor=hf,
        free_credit_usd=free_credit,
        is_liquidatable=is_liq,
    )


# ----- LP fee accrual ---------------------------------------------------- #


def in_range_mask(prices: np.ndarray, lower: float, upper: float) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    return (prices >= lower) & (prices <= upper)


def lp_fees_path(
    notional_usd: np.ndarray,
    apy_pct: float,
    t_days: np.ndarray,
    in_range: np.ndarray | None = None,
) -> np.ndarray:
    """Cumulative LP trading-fee revenue in USD.

    Fees accrue linearly:  dfee/dt = notional_usd(t) * apy / 365.
    When `in_range` is supplied, fee growth is zeroed on steps where the LP
    is out of range (no liquidity is active there). This is a coarse but
    standard approximation - the true distribution depends on tick liquidity
    competition which we treat as captured inside the user-supplied APY.
    """
    notional_usd = np.asarray(notional_usd, dtype=float)
    t_days = np.asarray(t_days, dtype=float)
    if notional_usd.shape != t_days.shape:
        raise ValueError("notional_usd and t_days must share shape")
    if t_days.size == 0:
        return np.zeros(0, dtype=float)

    daily_rate = apy_pct / 100.0 / 365.0
    dt = np.diff(t_days, prepend=t_days[0])
    dt = np.maximum(dt, 0.0)
    if in_range is None:
        active = np.ones_like(notional_usd, dtype=float)
    else:
        active = np.asarray(in_range, dtype=float)
        if active.shape != notional_usd.shape:
            raise ValueError("in_range must match notional_usd shape")
    flow = notional_usd * daily_rate * dt * active
    return np.cumsum(flow)


# ----- Portfolio aggregation -------------------------------------------- #


@dataclass(frozen=True)
class CompoundPortfolio:
    """Outer lending leg + N inner LP positions, all USD-valued.

    `lp_results` must each carry USD prices in their scenarios that cover
    the LP pair's two legs (so `to_frame` style USD valuation works). The
    `lp_fee_paths` array (one per LP) is what comes out of `lp_fees_path`.
    """

    outer: OuterLendingPath
    lp_results: Sequence[SimulationResult]
    lp_fee_paths: Sequence[np.ndarray]
    initial_user_capital_usd: float

    def __post_init__(self) -> None:
        n = self.outer.t_days.size
        if len(self.lp_results) != len(self.lp_fee_paths):
            raise ValueError("lp_results and lp_fee_paths must have equal length")
        for r in self.lp_results:
            if r.n_steps != n:
                raise ValueError("LP result step count does not match outer path")
        for fees in self.lp_fee_paths:
            if np.asarray(fees).shape != self.outer.t_days.shape:
                raise ValueError("fee path shape mismatch with outer path")
        if self.initial_user_capital_usd <= 0:
            raise ValueError("initial_user_capital_usd must be positive")

    # --- per-step USD aggregates ---------------------------------------- #

    def lp_position_value_usd(self, idx: int) -> np.ndarray:
        return _lp_usd(self.lp_results[idx], "position")

    def lp_debt_value_usd(self, idx: int) -> np.ndarray:
        return _lp_usd(self.lp_results[idx], "debt")

    def lp_equity_value_usd(self, idx: int) -> np.ndarray:
        return self.lp_position_value_usd(idx) - self.lp_debt_value_usd(idx)

    def total_lp_position_usd(self) -> np.ndarray:
        return np.sum([self.lp_position_value_usd(i) for i in range(len(self.lp_results))], axis=0)

    def total_lp_debt_usd(self) -> np.ndarray:
        return np.sum([self.lp_debt_value_usd(i) for i in range(len(self.lp_results))], axis=0)

    def total_lp_fees_usd(self) -> np.ndarray:
        return np.sum(self.lp_fee_paths, axis=0)

    def portfolio_equity_usd(self) -> np.ndarray:
        """User equity in USD: outer-equity + sum(LP equity) + cumulative fees.

        At t=0 this equals `initial_user_capital_usd` modulo the tiny pool/USD
        oracle mismatch inherent to leveraged CL positions where pool price
        does not exactly equal USD_price_ratio.
        """
        outer_eq = self.outer.equity_usd
        # Sum of LP equity counts the LP-token value minus LP-side debt.
        # The borrowed tokens funding the LP equity *are* the outer debt, so
        # the user's net equity = (USDC collateral) - (outer debt)
        #                       + (LP value) - (LP-side debt)
        #                       + (cumulative LP fees).
        # The outer debt remains on the books in `outer_eq`, the LP-side
        # debt is the Arcadia-borrowed amount tracked in lp_results.
        lp_eq = np.sum(
            [self.lp_equity_value_usd(i) for i in range(len(self.lp_results))],
            axis=0,
        )
        fees = self.total_lp_fees_usd()
        # Subtract the "double count": the borrowed tokens used to seed each
        # LP's equity show up on BOTH sides:
        #   - as outer debt (negative)
        #   - as LP equity (positive, because user deposited them)
        # which is correct - the user did fund LP equity from the outer loan.
        # No subtraction needed; both legs sum cleanly.
        return outer_eq + lp_eq + fees

    def portfolio_pnl_usd(self) -> np.ndarray:
        return self.portfolio_equity_usd() - self.initial_user_capital_usd

    def portfolio_pnl_pct(self) -> np.ndarray:
        return self.portfolio_pnl_usd() / self.initial_user_capital_usd * 100.0


# ----- Helpers ---------------------------------------------------------- #


def _lp_usd(result: SimulationResult, which: str) -> np.ndarray:
    """Recover USD-valued position or debt from an inner SimulationResult.

    Uses the USD price paths attached to the scenario; raises if the scenario
    lacks the required entries (cannot be silently inferred).
    """
    pair = result.config.pool.pair
    usd = result.scenario.usd_prices
    if pair.asset0 not in usd or pair.asset1 not in usd:
        raise KeyError(
            f"scenario.usd_prices must contain {pair.asset0!r} and "
            f"{pair.asset1!r} to value LP results in USD"
        )
    p0 = usd[pair.asset0]
    p1 = usd[pair.asset1]
    if which == "position":
        return result.amount0 * p0 + result.amount1 * p1
    if which == "debt":
        token = result.initial.debt_token
        if token is None:
            return np.zeros_like(p0)
        if token == pair.asset0:
            return result.debt_amount * p0
        if token == pair.asset1:
            return result.debt_amount * p1
        if token in usd:
            return result.debt_amount * usd[token]
        raise KeyError(f"USD price missing for debt token {token!r}")
    raise ValueError("which must be 'position' or 'debt'")
