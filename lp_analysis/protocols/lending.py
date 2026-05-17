"""
Lending and margin metrics for Arcadia-style and peer-to-pool protocols.
"""
from typing import Dict

from ..lp_calc.types import LeveragedPosition


def margin_state(used_margin: float, collateral_value: float, liquidation_value: float) -> str:
    """Classify an Arcadia margin account state."""
    if used_margin < collateral_value:
        return "healthy"
    if used_margin < liquidation_value:
        return "unhealthy"
    return "liquidatable"


def arcadia_margin_metrics(position: LeveragedPosition) -> Dict[str, float | str | bool]:
    """Calculate Arcadia whitepaper margin quantities for one position."""
    used_margin = position.used_margin_asset1
    collateral_value = position.collateral_value_asset1
    liquidation_value = position.liquidation_value_asset1
    state = margin_state(used_margin, collateral_value, liquidation_value)
    return {
        'spot_value_asset1': position.position_value_asset1,
        'open_position_asset1': position.debt_value_asset1,
        'minimum_margin_asset1': position.config.minimum_margin_asset1,
        'available_margin_asset1': collateral_value,
        'collateral_value_asset1': collateral_value,
        'liquidation_value_asset1': liquidation_value,
        'used_margin_asset1': used_margin,
        'free_margin_asset1': collateral_value - used_margin,
        'margin_health_ratio': position.margin_health_ratio,
        'margin_state': state,
        'is_liquidatable': state == "liquidatable",
    }


def aave_health_factor(position: LeveragedPosition) -> float:
    """Calculate an Aave-style health factor from liquidation value and debt."""
    debt = position.debt_value_asset1
    return position.liquidation_value_asset1 / debt if debt > 0 else float('inf')


def moonwell_credit_metrics(position: LeveragedPosition) -> Dict[str, float | bool]:
    """Calculate Moonwell-style credit limit and credit remaining metrics."""
    credit_limit = position.collateral_value_asset1
    borrowed = position.used_margin_asset1
    credit_remaining = credit_limit - borrowed
    return {
        'credit_limit_asset1': credit_limit,
        'borrowed_asset1': borrowed,
        'credit_remaining_asset1': credit_remaining,
        'credit_remaining_pct': credit_remaining / credit_limit * 100 if credit_limit > 0 else 0.0,
        'is_liquidatable': position.margin_state == "liquidatable",
    }
