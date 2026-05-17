"""
Core Uniswap V3 concentrated liquidity mathematics.
No leverage - pure LP calculations.
"""
from typing import Tuple
from .types import LPConfig, LPPosition, PriceRange
from .cl_math import amounts_from_liquidity, liquidity_from_amounts, value_asset1
from .strategy import create_initial_position, update_position_at_price


class LPCalculator:
    """
    Uniswap V3 math for concentrated liquidity.
    Reference: Uniswap V3 Whitepaper
    """
    
    def __init__(self, epsilon: float = 1e-10):
        """
        Args:
            epsilon: Numerical stability threshold
        """
        self.epsilon = epsilon
    
    # ==================== Formulas ====================
    
    def calculate_liquidity_from_amounts(
        self,
        amount0: float,
        amount1: float,
        price: float,
        price_range: PriceRange
    ) -> float:
        """
        Calculate liquidity L from asset amounts.
        
        Formula:
        - If P < P_a: L = Δx / (1/√P_a - 1/√P_b)
        - If P > P_b: L = Δy / (√P_b - √P_a)
        - If P in range: L is constrained by the limiting token side
        
        Returns:
            L: Liquidity constant
        """
        return liquidity_from_amounts(amount0, amount1, price, price_range, self.epsilon)
    
    def calculate_amounts_from_liquidity(
        self,
        liquidity: float,
        price: float,
        price_range: PriceRange
    ) -> Tuple[float, float]:
        """
        Calculate asset amounts from liquidity L.
        
        Formula:
        - x = L * (1/√P - 1/√P_b)  when P in range
        - y = L * (√P - √P_a)      when P in range
        
        Returns:
            (amount0, amount1)
        """
        return amounts_from_liquidity(liquidity, price, price_range)
    
    def calculate_initial_position(
        self,
        config: LPConfig
    ) -> LPPosition:
        """
        Create initial LP position from configuration.
        
        Given total value in asset1, split optimally between assets.
        
        Returns:
            LPPosition with amounts and liquidity
        """
        return create_initial_position(config)
    
    def update_position_at_price(
        self,
        position: LPPosition,
        new_price: float
    ) -> LPPosition:
        """
        Calculate position state at a new price.
        Liquidity L remains constant.
        
        Returns:
            New LPPosition at updated price
        """
        return update_position_at_price(position, new_price)
    
    def calculate_value_asset1(self, position: LPPosition) -> float:
        """Calculate total position value in asset1 terms."""
        return value_asset1(position.amount0, position.amount1, position.current_price)