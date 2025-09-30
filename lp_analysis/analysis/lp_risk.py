"""
Risk management: liquidation, VaR, utilization.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from ..lp_calc.types import LeveragedPosition, SimulationResult


class RiskAnalyzer:
    """
    Risk analysis for leveraged LP positions.
    """
    
    @staticmethod
    def calculate_liquidation_risk(
        positions: List[LeveragedPosition]
    ) -> List[bool]:
        """
        Determine liquidation risk at each position.
        
        Liquidation occurs when LTV >= liquidation_threshold.
        
        Returns:
            List of boolean flags (True = at risk)
        """
        return [pos.ltv >= pos.config.liquidation_threshold for pos in positions]
    
    @staticmethod
    def find_liquidation_prices(
        result: SimulationResult
    ) -> Dict[str, Optional[float]]:
        """
        Find exact liquidation price boundaries.
        
        Returns:
            Dict with lower and upper liquidation prices
        """
        if not isinstance(result.positions[0], LeveragedPosition):
            return {'lower': None, 'upper': None}
        
        liq_flags = RiskAnalyzer.calculate_liquidation_risk(result.positions)
        liq_indices = [i for i, flag in enumerate(liq_flags) if flag]
        
        if not liq_indices:
            return {'lower': None, 'upper': None}
        
        liq_prices = [result.prices[i] for i in liq_indices]
        
        return {
            'lower': min(liq_prices),
            'upper': max(liq_prices),
            'distance_lower_pct': ((result.config.price_initial - min(liq_prices)) / 
                                  result.config.price_initial * 100),
            'distance_upper_pct': ((max(liq_prices) - result.config.price_initial) / 
                                  result.config.price_initial * 100)
        }
    
    @staticmethod
    def calculate_value_at_risk(
        pnl_distribution: List[float],
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            pnl_distribution: List of PnL values
            confidence_level: Confidence level (0.95 = 95%)
        
        Returns:
            VaR value (negative = potential loss)
        """
        return float(np.percentile(pnl_distribution, (1 - confidence_level) * 100))
    
    @staticmethod
    def calculate_expected_shortfall(
        pnl_distribution: List[float],
        confidence_level: float = 0.95
    ) -> float:
        """
        Calculate Expected Shortfall (CVaR).
        Average loss beyond VaR.
        
        Returns:
            CVaR value
        """
        var = RiskAnalyzer.calculate_value_at_risk(pnl_distribution, confidence_level)
        losses = [pnl for pnl in pnl_distribution if pnl < var]
        return float(np.mean(losses)) if losses else 0.0
    
    @staticmethod
    def calculate_utilization_ratios(
        positions: List[LeveragedPosition]
    ) -> List[float]:
        """
        Calculate debt/collateral utilization ratio at each position.
        
        Returns:
            List of utilization ratios (0-1+)
        """
        return [pos.ltv for pos in positions]
    
    @staticmethod
    def calculate_daily_borrow_cost(
        position: LeveragedPosition
    ) -> float:
        """
        Calculate daily borrowing cost in asset1 terms.
        
        Returns:
            Daily cost in asset1
        """
        return position.debt_value_asset1 * (position.config.borrow_apr / 100) / 365
    
    @staticmethod
    def analyze_risk(result: SimulationResult) -> Dict:
        """
        Comprehensive risk analysis.
        
        Returns:
            Dict with all risk metrics
        """
        if not isinstance(result.positions[0], LeveragedPosition):
            return {'message': 'No leverage - no liquidation risk'}
        
        positions = result.positions
        pnl = result.metrics.get('pnl', {}).get('pnl_asset1', [])
        
        return {
            'liquidation_prices': RiskAnalyzer.find_liquidation_prices(result),
            'liquidation_flags': RiskAnalyzer.calculate_liquidation_risk(positions),
            'utilization_ratios': RiskAnalyzer.calculate_utilization_ratios(positions),
            'var_95': RiskAnalyzer.calculate_value_at_risk(pnl, 0.95) if pnl else None,
            'cvar_95': RiskAnalyzer.calculate_expected_shortfall(pnl, 0.95) if pnl else None,
            'daily_cost': RiskAnalyzer.calculate_daily_borrow_cost(positions[0])
        }