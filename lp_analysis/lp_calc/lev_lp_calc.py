"""
Leveraged LP calculations with debt management.
Extends base LP calculator with leverage mechanics.
"""
from typing import Optional, Sequence, Union
import numpy as np
from .lp_calc import LPCalculator
from .types import (
    LPConfig, LeveragedLPConfig, LeveragedPosition, SimulationResult
)
from .strategy import (
    create_initial_leveraged_position,
    simulate_price_path,
    update_leveraged_position_at_price,
)


class LeveragedLPCalculator(LPCalculator):
    """
    Calculator for leveraged concentrated liquidity positions.
    Handles debt in either asset0 or asset1.
    """
    
    def calculate_initial_leveraged_position(
        self,
        config: LeveragedLPConfig
    ) -> LeveragedPosition:
        """
        Create initial leveraged LP position.
        
        Steps:
        1. Calculate total position value with leverage
        2. Determine borrowed amounts based on debt asset
        3. Create LP position with leveraged capital
        
        Returns:
            LeveragedPosition with debt tracking
        """
        return create_initial_leveraged_position(config)
    
    def update_leveraged_position_at_price(
        self,
        position: LeveragedPosition,
        new_price: float,
        days_elapsed: float = 0.0,
        external_debt_price_asset1: Optional[float] = None
    ) -> LeveragedPosition:
        """
        Update leveraged position at new price.
        LP composition changes, debt remains constant (in token terms).
        
        Returns:
            Updated LeveragedPosition
        """
        return update_leveraged_position_at_price(
            position,
            new_price,
            days_elapsed,
            external_debt_price_asset1,
        )
    
    def simulate_price_range(
        self,
        config: Union[LPConfig, LeveragedLPConfig],
        price_min: float = None,
        price_max: float = None,
        num_points: int = 300,
        days_elapsed: Optional[Union[float, Sequence[float]]] = None,
        external_debt_prices_asset1: Optional[Union[float, Sequence[float]]] = None
    ) -> SimulationResult:
        """
        Simulate position across price range.
        Works with both leveraged and non-leveraged configs.
        
        Args:
            config: LP or Leveraged LP configuration
            price_min: Min price (default: 80% of lower bound)
            price_max: Max price (default: 120% of upper bound)
            num_points: Number of price points to simulate
        
        Returns:
            SimulationResult with positions at each price
        """
        # Default price range
        if price_min is None:
            price_min = config.price_range.lower * 0.8
        if price_max is None:
            price_max = config.price_range.upper * 1.2
        
        prices = np.linspace(price_min, price_max, num_points)
        return simulate_price_path(
            config,
            prices,
            days_elapsed=days_elapsed,
            external_debt_prices_asset1=external_debt_prices_asset1,
        )