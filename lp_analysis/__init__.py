"""lp_analysis - Leveraged concentrated-liquidity strategy analyser.

Multi-protocol (Arcadia, Aave, Moonwell, ...) and multi-pool (Uniswap v3,
Aerodrome Slipstream, ...) modelling kit for quantitative research.

The public surface is intentionally small. Build a strategy in three layers
(pool / lending / strategy), supply a price scenario, simulate, then convert
the result into a polars frame or matplotlib plots.

    >>> from lp_analysis import (
    ...     AssetPair, PriceRange, PoolConfig, LendingPolicy, StrategyConfig,
    ...     ScenarioPath, simulate, to_frame, linear_price_sweep,
    ... )
"""
from .analytics import (
    ExposureReport,
    Greeks,
    PnLReport,
    compute_exposure,
    compute_greeks,
    compute_pnl,
    expected_shortfall,
    impermanent_loss,
    liquidation_price_bounds,
    margin_path,
    solve_liquidation_price,
    value_at_risk,
)
from .cl_math import (
    amounts_from_liquidity,
    amounts_path_from_liquidity,
    liquidity_from_amounts,
    liquidity_from_value_asset1,
    value_asset1,
)
from .frame import to_frame
from .lending import (
    MarginMetrics,
    aave_view,
    accrue_continuous,
    accrue_continuous_path,
    arcadia_view,
    classify_state,
    margin_metrics,
    moonwell_view,
)
from .plots import (
    dashboard,
    plot_combined_equity,
    plot_composition,
    plot_exposure,
    plot_greeks,
    plot_margin,
    plot_pnl,
    plot_value_breakdown,
)
from .pricing import convert_usd, to_usd, value_in_asset0, value_in_asset1
from .strategy import (
    SimulationResult,
    linear_price_sweep,
    open_position,
    simulate,
    step_position,
)
from .types import (
    AssetPair,
    LendingPolicy,
    PoolConfig,
    PositionState,
    PriceRange,
    ScenarioPath,
    StrategyConfig,
)

__version__ = "3.1.0"

__all__ = [
    # Core types
    "AssetPair", "PriceRange", "PoolConfig", "LendingPolicy",
    "StrategyConfig", "ScenarioPath", "PositionState",
    # CL math
    "amounts_from_liquidity", "amounts_path_from_liquidity",
    "liquidity_from_amounts", "liquidity_from_value_asset1", "value_asset1",
    # Pricing / numeraire
    "to_usd", "convert_usd", "value_in_asset0", "value_in_asset1",
    # Lending
    "MarginMetrics", "accrue_continuous", "accrue_continuous_path",
    "classify_state", "margin_metrics", "arcadia_view", "aave_view", "moonwell_view",
    # Strategy engine
    "SimulationResult", "simulate", "open_position", "step_position",
    "linear_price_sweep",
    # Analytics
    "PnLReport", "Greeks", "ExposureReport",
    "compute_pnl", "compute_greeks", "compute_exposure",
    "impermanent_loss", "margin_path",
    "liquidation_price_bounds", "solve_liquidation_price",
    "value_at_risk", "expected_shortfall",
    # Reporting
    "to_frame",
    # Plots
    "dashboard", "plot_pnl", "plot_value_breakdown", "plot_composition",
    "plot_margin", "plot_exposure", "plot_greeks", "plot_combined_equity",
]
