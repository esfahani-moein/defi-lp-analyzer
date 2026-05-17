"""Strategy analytics: PnL, impermanent loss, Greeks, VaR/CVaR, liquidation prices.

These functions operate on a `SimulationResult` (numpy-array fields) and return
either scalars, numpy arrays, or small dicts. Everything is pure - no plotting
and no protocol-specific terminology bleeding in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import cl_math
from .lending import margin_metrics
from .strategy import SimulationResult


# ----- PnL and impermanent loss ----------------------------------------- #


@dataclass(frozen=True)
class PnLReport:
    pnl_asset1: np.ndarray         # equity - initial capital, per step
    pnl_pct: np.ndarray            # PnL as percent of initial capital
    max_gain: float
    max_loss: float
    max_gain_pct: float
    max_loss_pct: float


def compute_pnl(result: SimulationResult) -> PnLReport:
    cap = result.config.capital_asset1
    pnl = result.equity_asset1 - cap
    pnl_pct = pnl / cap * 100.0
    return PnLReport(
        pnl_asset1=pnl,
        pnl_pct=pnl_pct,
        max_gain=float(np.max(pnl)),
        max_loss=float(np.min(pnl)),
        max_gain_pct=float(np.max(pnl_pct)),
        max_loss_pct=float(np.min(pnl_pct)),
    )


def impermanent_loss(result: SimulationResult) -> np.ndarray:
    """IL vs holding the initial token mix, in percent (negative == LP underperforms hold).

    Hold benchmark uses the strategy's *leveraged* notional split at t=0, so a
    leveraged LP is compared against simply holding the same leveraged exposure
    (debt is ignored on both sides - IL is a structural LP property).
    """
    init = result.initial
    hold_value = init.amount0 * result.scenario.price_path + init.amount1
    lp_value = result.position_value_asset1
    with np.errstate(divide="ignore", invalid="ignore"):
        il = np.where(hold_value > 0, (lp_value - hold_value) / hold_value * 100.0, 0.0)
    return il


# ----- Greeks (numerical) ----------------------------------------------- #


@dataclass(frozen=True)
class Greeks:
    delta: np.ndarray
    gamma: np.ndarray
    max_abs_delta: float
    max_abs_gamma: float


def compute_greeks(result: SimulationResult) -> Greeks:
    """Numerical delta/gamma of equity_asset1 against pool price.

    Both Greeks are taken from np.gradient (central differences); the scenario
    is assumed monotonic in price for these to be meaningful.
    """
    prices = result.scenario.price_path
    if prices.size < 2:
        zero = np.zeros_like(prices)
        return Greeks(delta=zero, gamma=zero.copy(), max_abs_delta=0.0, max_abs_gamma=0.0)
    # Sort by price to make gradient meaningful even for non-monotonic scenarios.
    order = np.argsort(prices)
    p = prices[order]
    v = result.equity_asset1[order]
    d = np.gradient(v, p)
    g = np.gradient(d, p)
    inv = np.argsort(order)
    delta = d[inv]
    gamma = g[inv]
    return Greeks(
        delta=delta,
        gamma=gamma,
        max_abs_delta=float(np.max(np.abs(delta))),
        max_abs_gamma=float(np.max(np.abs(gamma))),
    )


# ----- Net asset exposure (LP holdings - debt) -------------------------- #


@dataclass(frozen=True)
class ExposureReport:
    asset0_net: np.ndarray         # token units of asset0 held net of debt
    asset1_net: np.ndarray
    asset0_net_value_asset1: np.ndarray  # net asset0 expressed in asset1
    asset1_net_value_asset1: np.ndarray
    total_net_value_asset1: np.ndarray


def compute_exposure(result: SimulationResult) -> ExposureReport:
    """LP holdings minus debt, per asset, both in tokens and asset1 value."""
    debt0 = np.zeros_like(result.amount0)
    debt1 = np.zeros_like(result.amount1)
    cfg = result.config
    if result.initial.debt_token == cfg.pool.pair.asset0:
        debt0 = result.debt_amount
    elif result.initial.debt_token == cfg.pool.pair.asset1:
        debt1 = result.debt_amount

    net0 = result.amount0 - debt0
    net1 = result.amount1 - debt1
    p = result.scenario.price_path
    return ExposureReport(
        asset0_net=net0,
        asset1_net=net1,
        asset0_net_value_asset1=net0 * p,
        asset1_net_value_asset1=net1,
        total_net_value_asset1=net0 * p + net1,
    )


# ----- Risk: liquidation prices, VaR/CVaR ------------------------------- #


def margin_path(result: SimulationResult) -> list[dict]:
    """Compute MarginMetrics at every step (as dicts) using the config's policy."""
    cfg = result.config
    pair = cfg.pool.pair
    policy = cfg.lending
    out = []
    for i in range(result.n_steps):
        asset_values = {
            pair.asset0: float(result.amount0[i] * result.scenario.price_path[i]),
            pair.asset1: float(result.amount1[i]),
        }
        m = margin_metrics(
            asset_values,
            float(result.debt_value_asset1[i]),
            policy,
            minimum_margin=policy.minimum_margin_usd,  # numeraire mismatch tolerated when min=0
        )
        out.append({
            "spot_value": m.spot_value,
            "open_position": m.open_position,
            "minimum_margin": m.minimum_margin,
            "used_margin": m.used_margin,
            "collateral_value": m.collateral_value,
            "liquidation_value": m.liquidation_value,
            "free_margin": m.free_margin,
            "health_ratio": m.health_ratio,
            "state": m.state,
            "is_liquidatable": m.is_liquidatable,
        })
    return out


def liquidation_price_bounds(result: SimulationResult) -> dict[str, Optional[float]]:
    """Lowest and highest prices at which the position would be liquidatable."""
    states = [row["state"] for row in margin_path(result)]
    flags = np.array([s == "liquidatable" for s in states])
    prices = result.scenario.price_path
    if not flags.any():
        return {"lower": None, "upper": None}
    return {"lower": float(prices[flags].min()), "upper": float(prices[flags].max())}


def value_at_risk(pnl: np.ndarray, confidence: float = 0.95) -> float:
    pnl = np.asarray(pnl, dtype=float)
    return float(np.percentile(pnl, (1.0 - confidence) * 100.0))


def expected_shortfall(pnl: np.ndarray, confidence: float = 0.95) -> float:
    var = value_at_risk(pnl, confidence)
    losses = pnl[pnl < var]
    return float(losses.mean()) if losses.size else 0.0


# ----- Liquidation-bound finder (analytic, for diagnostics) ------------- #


def solve_liquidation_price(
    result: SimulationResult,
    side: str,                       # "lower" or "upper"
    n_search: int = 4096,
) -> Optional[float]:
    """Refined search for the liquidation price boundary via dense sweep.

    Useful when the scenario doesn't already include the boundary - e.g. when
    quoting a "liquidation distance" outside of the simulated price range.
    """
    cfg = result.config
    pair = cfg.pool.pair
    policy = cfg.lending
    debt_token = result.initial.debt_token
    debt_amt0 = result.initial.debt_amount if debt_token == pair.asset0 else 0.0
    debt_amt1 = result.initial.debt_amount if debt_token == pair.asset1 else 0.0

    lo = cfg.pool.price_range.lower * 0.01
    hi = cfg.pool.price_range.upper * 100.0
    prices = np.geomspace(lo, hi, n_search)
    a0, a1 = cl_math.amounts_path_from_liquidity(result.initial.liquidity, prices, cfg.pool.price_range)
    a0_val = a0 * prices
    debt_value = debt_amt0 * prices + debt_amt1
    cv = policy.cf(pair.asset0) * a0_val + policy.cf(pair.asset1) * a1
    cv *= policy.lp_protocol_factor
    lv = policy.lf(pair.asset0) * a0_val + policy.lf(pair.asset1) * a1
    lv *= policy.lp_protocol_factor
    used = debt_value + policy.minimum_margin_usd
    liq = used >= lv
    if not liq.any():
        return None
    init_idx = np.searchsorted(prices, cfg.pool.price_initial)
    if side == "lower":
        candidates = np.where(liq[:init_idx])[0]
        return float(prices[candidates.max()]) if candidates.size else None
    if side == "upper":
        candidates = np.where(liq[init_idx:])[0]
        return float(prices[candidates.min() + init_idx]) if candidates.size else None
    raise ValueError("side must be 'lower' or 'upper'")
