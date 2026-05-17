"""Lending mechanics shared by Arcadia, Aave, Moonwell (and anything similar).

This module is intentionally protocol-agnostic. The protocol-specific helpers
at the bottom are thin labelling wrappers around the same generic margin math.

Definitions follow the Arcadia whitepaper (Section 4 - Arcadia Accounts):
    available_margin  = collateral_value
                      = Σ CF_i · value_i  · lp_protocol_factor
    open_position     = total debt value
    used_margin       = open_position + minimum_margin
    free_margin       = available_margin - used_margin
    liquidation_value = Σ LF_i · value_i  · lp_protocol_factor

Health states (whitepaper Figure 5):
    healthy       iff used_margin < collateral_value
    unhealthy     iff collateral_value ≤ used_margin < liquidation_value
    liquidatable  iff used_margin ≥ liquidation_value

All values returned here are in pool-asset1 units. Use `pricing.convert`
to obtain values in any other numeraire (USD, ETH, BTC, USDC, ...).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .types import LendingPolicy


# ----- Interest accrual --------------------------------------------------- #


def accrue_continuous(amount: float, apr_pct: float, days: float) -> float:
    """Continuously-compounded interest accrual matching the Arcadia whitepaper.

        a(t) = a(0) · exp(r · t / 365),   r = apr_pct / 100
    """
    if days < 0:
        raise ValueError("days must be non-negative")
    if amount == 0.0 or apr_pct == 0.0 or days == 0.0:
        return float(amount)
    return float(amount * np.exp((apr_pct / 100.0) * (days / 365.0)))


def accrue_continuous_path(amount: float, apr_pct: float, days: np.ndarray) -> np.ndarray:
    """Vectorised continuous compounding over a days-elapsed array."""
    days = np.asarray(days, dtype=float)
    return amount * np.exp((apr_pct / 100.0) * (days / 365.0))


# ----- Margin metrics ----------------------------------------------------- #


@dataclass(frozen=True)
class MarginMetrics:
    """Per-step margin readout in pool-asset1 units."""

    spot_value: float            # MtM portfolio value
    open_position: float         # total debt value
    minimum_margin: float        # creditor-specific floor
    used_margin: float           # open_position + minimum_margin
    collateral_value: float      # Σ CF_i · value_i (haircut by lp factor)
    liquidation_value: float     # Σ LF_i · value_i (haircut by lp factor)
    free_margin: float           # collateral_value - used_margin
    state: str                   # "healthy" | "unhealthy" | "liquidatable"

    @property
    def is_liquidatable(self) -> bool:
        return self.state == "liquidatable"

    @property
    def health_ratio(self) -> float:
        """liquidation_value / used_margin (Arcadia health proxy)."""
        return self.liquidation_value / self.used_margin if self.used_margin > 0 else float("inf")


def classify_state(used_margin: float, collateral_value: float, liquidation_value: float) -> str:
    if used_margin < collateral_value:
        return "healthy"
    if used_margin < liquidation_value:
        return "unhealthy"
    return "liquidatable"


def margin_metrics(
    asset_values: Mapping[str, float],
    debt_value: float,
    policy: LendingPolicy,
    minimum_margin: float = 0.0,
) -> MarginMetrics:
    """Compute margin readout from a dict of symbol → MtM value.

    `asset_values` should contain the position's holdings already converted
    into a common numeraire (typically pool-asset1). `debt_value` is the
    outstanding debt in the same numeraire.
    """
    cv = sum(policy.cf(sym) * val for sym, val in asset_values.items()) * policy.lp_protocol_factor
    lv = sum(policy.lf(sym) * val for sym, val in asset_values.items()) * policy.lp_protocol_factor
    spot = sum(asset_values.values())
    used = debt_value + minimum_margin
    free = cv - used
    state = classify_state(used, cv, lv)
    return MarginMetrics(
        spot_value=spot,
        open_position=debt_value,
        minimum_margin=minimum_margin,
        used_margin=used,
        collateral_value=cv,
        liquidation_value=lv,
        free_margin=free,
        state=state,
    )


# ----- Protocol views ----------------------------------------------------- #
#
# Different protocols name the same quantities differently. These helpers
# return dicts in the language of each protocol so reports stay legible.


def arcadia_view(m: MarginMetrics) -> dict:
    """Arcadia margin-account readout (whitepaper Section 4)."""
    return {
        "spot_value": m.spot_value,
        "open_position": m.open_position,
        "minimum_margin": m.minimum_margin,
        "available_margin": m.collateral_value,
        "collateral_value": m.collateral_value,
        "liquidation_value": m.liquidation_value,
        "used_margin": m.used_margin,
        "free_margin": m.free_margin,
        "health_ratio": m.health_ratio,
        "state": m.state,
        "is_liquidatable": m.is_liquidatable,
    }


def aave_view(m: MarginMetrics) -> dict:
    """Aave-style health factor.

        HF = Σ(collateral_i · liquidation_threshold_i) / total_debt
           = liquidation_value / debt

    With the convention LF == liquidation-threshold, the Arcadia liquidation
    value is exactly Aave's numerator. HF >= 1 == healthy.
    """
    debt = m.open_position
    return {
        "total_collateral": m.spot_value,
        "total_debt": debt,
        "ltv": (debt / m.spot_value) if m.spot_value > 0 else float("inf"),
        "health_factor": (m.liquidation_value / debt) if debt > 0 else float("inf"),
        "is_liquidatable": m.is_liquidatable,
    }


def moonwell_view(m: MarginMetrics) -> dict:
    """Moonwell-style credit-limit readout (CF is the per-asset collateral cap)."""
    credit_limit = m.collateral_value
    borrowed = m.used_margin
    remaining = credit_limit - borrowed
    return {
        "credit_limit": credit_limit,
        "borrowed": borrowed,
        "credit_remaining": remaining,
        "credit_remaining_pct": (remaining / credit_limit * 100.0) if credit_limit > 0 else 0.0,
        "is_liquidatable": m.is_liquidatable,
    }
