"""
Leveraged LP calculations with debt management.
Extends base LP calculator with leverage mechanics.
"""
from typing import List, Union
import numpy as np
from .lp_calc import LPCalculator
from .types import (
    LPConfig, LeveragedLPConfig, LPPosition, LeveragedPosition, DebtAsset,
    SimulationResult
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
        # Total capital after leverage
        borrowed_value = config.capital_asset1 * (config.leverage - 1)
        total_value = config.capital_asset1 * config.leverage
        
        # Calculate LP composition
        initial_pos = super().calculate_initial_position(
            config  # LeveragedLPConfig extends LPConfig
        )
        
        # Override total value for leveraged position
        sp = np.sqrt(config.price_initial)
        sp_a = np.sqrt(config.price_range.lower)
        sp_b = np.sqrt(config.price_range.upper)
        
        denominator = config.price_initial * (1/sp - 1/sp_b) + (sp - sp_a)
        liquidity = total_value / denominator
        
        amount0, amount1 = self.calculate_amounts_from_liquidity(
            liquidity, config.price_initial, config.price_range
        )
        
        # Determine debt
        if config.debt_asset == DebtAsset.ASSET0:
            debt_amount0 = borrowed_value / config.price_initial
            debt_amount1 = 0.0
        elif config.debt_asset == DebtAsset.ASSET1:
            debt_amount0 = 0.0
            debt_amount1 = borrowed_value
        else:
            debt_amount0 = 0.0
            debt_amount1 = 0.0
        
        return LeveragedPosition(
            amount0=amount0,
            amount1=amount1,
            liquidity=liquidity,
            current_price=config.price_initial,
            debt_amount0=debt_amount0,
            debt_amount1=debt_amount1,
            config=config
        )
    
    def update_leveraged_position_at_price(
        self,
        position: LeveragedPosition,
        new_price: float
    ) -> LeveragedPosition:
        """
        Update leveraged position at new price.
        LP composition changes, debt remains constant (in token terms).
        
        Returns:
            Updated LeveragedPosition
        """
        amount0, amount1 = self.calculate_amounts_from_liquidity(
            position.liquidity,
            new_price,
            position.config.price_range
        )
        
        return LeveragedPosition(
            amount0=amount0,
            amount1=amount1,
            liquidity=position.liquidity,
            current_price=new_price,
            debt_amount0=position.debt_amount0,
            debt_amount1=position.debt_amount1,
            config=position.config
        )
    
    def simulate_price_range(
        self,
        config: Union[LPConfig, LeveragedLPConfig],
        price_min: float = None,
        price_max: float = None,
        num_points: int = 300
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
        
        # Check if leveraged or not
        is_leveraged = isinstance(config, LeveragedLPConfig)
        
        if is_leveraged:
            # Initial leveraged position
            initial_pos = self.calculate_initial_leveraged_position(config)
            
            # Simulate at each price
            positions = []
            for price in prices:
                pos = self.update_leveraged_position_at_price(initial_pos, price)
                positions.append(pos)
        else:
            # Non-leveraged position
            initial_pos = self.calculate_initial_position(config)
            
            # Simulate at each price
            positions = []
            for price in prices:
                pos = self.update_position_at_price(initial_pos, price)
                positions.append(pos)
        
        return SimulationResult(
            prices=prices.tolist(),
            positions=positions,
            metrics={},  # Will be filled by analysis module
            config=config
        )