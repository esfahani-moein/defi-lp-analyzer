"""
Core Uniswap V3 concentrated liquidity mathematics.
No leverage - pure LP calculations.
"""
import numpy as np
from typing import Tuple
from .types import LPConfig, LPPosition, PriceRange


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
        - If P in range: L calculated from both, averaged
        
        Returns:
            L: Liquidity constant
        """
        sp = np.sqrt(price)
        sp_a = np.sqrt(price_range.lower)
        sp_b = np.sqrt(price_range.upper)
        
        if price <= price_range.lower:
            return amount0 / (1/sp_a - 1/sp_b) if amount0 > self.epsilon else 0.0
        
        elif price >= price_range.upper:
            return amount1 / (sp_b - sp_a) if amount1 > self.epsilon else 0.0
        
        else:
            # In range - use both formulas for stability
            L0 = amount0 / (1/sp - 1/sp_b) if amount0 > self.epsilon else 0.0
            L1 = amount1 / (sp - sp_a) if amount1 > self.epsilon else 0.0
            
            if L0 > 0 and L1 > 0:
                return (L0 + L1) / 2.0
            return L0 if L0 > 0 else L1
    
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
        if liquidity <= 0:
            return 0.0, 0.0
        
        sp = np.sqrt(price)
        sp_a = np.sqrt(price_range.lower)
        sp_b = np.sqrt(price_range.upper)
        
        if price <= price_range.lower:
            return liquidity * (1/sp_a - 1/sp_b), 0.0
        
        elif price >= price_range.upper:
            return 0.0, liquidity * (sp_b - sp_a)
        
        else:
            amount0 = liquidity * (1/sp - 1/sp_b)
            amount1 = liquidity * (sp - sp_a)
            return amount0, amount1
    
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
        sp = np.sqrt(config.price_initial)
        sp_a = np.sqrt(config.price_range.lower)
        sp_b = np.sqrt(config.price_range.upper)
        
        # Solve: V = amount0 * P + amount1
        # Where: amount0 = L(1/√P - 1/√P_b), amount1 = L(√P - √P_a)
        # Result: L = V / [P(1/√P - 1/√P_b) + (√P - √P_a)]
        
        denominator = config.price_initial * (1/sp - 1/sp_b) + (sp - sp_a)
        
        if denominator <= self.epsilon:
            raise ValueError("Invalid range configuration")
        
        liquidity = config.capital_asset1 / denominator
        amount0, amount1 = self.calculate_amounts_from_liquidity(
            liquidity, config.price_initial, config.price_range
        )
        
        return LPPosition(
            amount0=amount0,
            amount1=amount1,
            liquidity=liquidity,
            current_price=config.price_initial,
            config=config
        )
    
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
        amount0, amount1 = self.calculate_amounts_from_liquidity(
            position.liquidity,
            new_price,
            position.config.price_range
        )
        
        return LPPosition(
            amount0=amount0,
            amount1=amount1,
            liquidity=position.liquidity,
            current_price=new_price,
            config=position.config
        )
    
    def calculate_value_asset1(self, position: LPPosition) -> float:
        """Calculate total position value in asset1 terms."""
        return position.amount0 * position.current_price + position.amount1