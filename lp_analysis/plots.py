"""Compact matplotlib plotting for SimulationResult.

All plots are asset-symbol agnostic - axes are labelled directly from the
pool's AssetPair. Plot functions accept an `ax=` and return it so callers
can compose multi-panel layouts without internal coupling.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .analytics import compute_exposure, compute_greeks, compute_pnl, impermanent_loss, margin_path
from .strategy import SimulationResult


# ----- Single-position plots -------------------------------------------- #


def plot_pnl(result: SimulationResult, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    pair = result.config.pool.pair
    p = result.scenario.price_path
    pnl = compute_pnl(result).pnl_asset1
    ax.plot(p, pnl, linewidth=2.0, color="steelblue", label="PnL")
    ax.fill_between(p, 0, pnl, where=pnl >= 0, color="green", alpha=0.2)
    ax.fill_between(p, 0, pnl, where=pnl < 0, color="red", alpha=0.2)
    _mark_range(ax, result)
    ax.set_xlabel(f"{pair.asset1} per {pair.asset0}")
    ax.set_ylabel(f"PnL ({pair.asset1})")
    ax.set_title(f"PnL - {result.config.name} ({pair})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    return ax


def plot_value_breakdown(result: SimulationResult, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    pair = result.config.pool.pair
    p = result.scenario.price_path
    ax.plot(p, result.position_value_asset1, color="green", linewidth=2.0, label="Position value")
    ax.plot(p, result.debt_value_asset1, color="red", linewidth=2.0, linestyle="--", label="Debt")
    ax.plot(p, result.equity_asset1, color="navy", linewidth=2.2, label="Equity")
    _mark_range(ax, result)
    ax.set_xlabel(f"{pair.asset1} per {pair.asset0}")
    ax.set_ylabel(f"Value ({pair.asset1})")
    ax.set_title("Position / Debt / Equity")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    return ax


def plot_composition(result: SimulationResult, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    pair = result.config.pool.pair
    p = result.scenario.price_path
    ax2 = ax.twinx()
    ax.plot(p, result.amount0, color="purple", linewidth=2.0, label=pair.asset0)
    ax2.plot(p, result.amount1, color="orange", linewidth=2.0, label=pair.asset1)
    _mark_range(ax, result)
    ax.set_xlabel(f"{pair.asset1} per {pair.asset0}")
    ax.set_ylabel(f"{pair.asset0} amount", color="purple")
    ax2.set_ylabel(f"{pair.asset1} amount", color="orange")
    ax.tick_params(axis="y", labelcolor="purple")
    ax2.tick_params(axis="y", labelcolor="orange")
    ax.set_title("LP Composition")
    ax.grid(alpha=0.3)
    return ax


def plot_margin(result: SimulationResult, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    pair = result.config.pool.pair
    p = result.scenario.price_path
    rows = margin_path(result)
    cv = np.array([r["collateral_value"] for r in rows])
    lv = np.array([r["liquidation_value"] for r in rows])
    used = np.array([r["used_margin"] for r in rows])
    ax.plot(p, cv, color="forestgreen", linewidth=2.0, label="Collateral value")
    ax.plot(p, lv, color="darkorange", linewidth=2.0, label="Liquidation value")
    ax.plot(p, used, color="red", linewidth=2.0, linestyle="--", label="Used margin")
    _mark_range(ax, result)
    ax.set_xlabel(f"{pair.asset1} per {pair.asset0}")
    ax.set_ylabel(f"Value ({pair.asset1})")
    ax.set_title("Arcadia margin readout")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    return ax


def plot_exposure(result: SimulationResult, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    pair = result.config.pool.pair
    p = result.scenario.price_path
    exp = compute_exposure(result)
    ax.plot(p, exp.asset0_net, color="purple", linewidth=2.2, label=f"Net {pair.asset0} (tokens)")
    ax.axhline(0, color="black", linewidth=1, alpha=0.4)
    ax.fill_between(p, 0, exp.asset0_net, where=exp.asset0_net >= 0, color="green", alpha=0.15)
    ax.fill_between(p, 0, exp.asset0_net, where=exp.asset0_net < 0, color="red", alpha=0.15)
    _mark_range(ax, result)
    ax.set_xlabel(f"{pair.asset1} per {pair.asset0}")
    ax.set_ylabel(f"Net {pair.asset0} (LP - debt)")
    ax.set_title("Net asset exposure")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    return ax


def plot_greeks(result: SimulationResult, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    pair = result.config.pool.pair
    p = result.scenario.price_path
    g = compute_greeks(result)
    ax.plot(p, g.delta, color="navy", linewidth=2.0, label="Δ (∂equity/∂price)")
    ax2 = ax.twinx()
    ax2.plot(p, g.gamma, color="firebrick", linewidth=1.5, label="Γ", alpha=0.7)
    _mark_range(ax, result)
    ax.set_xlabel(f"{pair.asset1} per {pair.asset0}")
    ax.set_ylabel("Delta")
    ax2.set_ylabel("Gamma", color="firebrick")
    ax2.tick_params(axis="y", labelcolor="firebrick")
    ax.set_title("Equity Greeks")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    return ax


def dashboard(result: SimulationResult, figsize: tuple = (16, 11)) -> plt.Figure:
    """4-panel single-strategy dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plot_pnl(result, ax=axes[0, 0])
    plot_value_breakdown(result, ax=axes[0, 1])
    plot_margin(result, ax=axes[1, 0])
    plot_exposure(result, ax=axes[1, 1])
    fig.suptitle(
        f"{result.config.name}  |  {result.config.pool.pair}  |  "
        f"leverage {result.config.leverage:.2f}x  |  debt: {result.initial.debt_token or 'none'}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ----- Multi-strategy comparison ---------------------------------------- #


def plot_combined_equity(
    results: Sequence[SimulationResult],
    labels: Optional[Iterable[str]] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot equity_asset1 of multiple strategies and their portfolio sum.

    Strategies must share an identical price_path (same scenario lengths and
    pool numeraire) for the sum to be well-defined.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))
    if not results:
        raise ValueError("results must contain at least one SimulationResult")
    base_prices = results[0].scenario.price_path
    for r in results:
        if r.scenario.price_path.shape != base_prices.shape:
            raise ValueError("all strategies must share the same scenario length")
    labels = list(labels) if labels is not None else [r.config.name for r in results]

    total = np.zeros_like(base_prices)
    for r, name in zip(results, labels):
        ax.plot(r.scenario.price_path, r.equity_asset1, alpha=0.7, linewidth=1.8, label=name)
        total = total + r.equity_asset1
    ax.plot(base_prices, total, color="black", linewidth=2.6, label="Portfolio Σ")
    ax.axhline(0, color="grey", linewidth=1, alpha=0.4)
    pair = results[0].config.pool.pair
    ax.set_xlabel(f"{pair.asset1} per {pair.asset0}")
    ax.set_ylabel(f"Equity ({pair.asset1})")
    ax.set_title("Combined equity across strategies")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    return ax


# ----- Helpers ---------------------------------------------------------- #


def _mark_range(ax: plt.Axes, result: SimulationResult) -> None:
    cfg = result.config
    ax.axvline(cfg.pool.price_initial, color="green", linestyle="--", alpha=0.6, linewidth=1)
    ax.axvline(cfg.pool.price_range.lower, color="orange", linestyle=":", alpha=0.5)
    ax.axvline(cfg.pool.price_range.upper, color="orange", linestyle=":", alpha=0.5)
