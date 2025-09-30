"""
Advanced LP analysis: exposure, Greeks, performance metrics.
"""
import numpy as np
from typing import Dict, List
from ..lp_calc.types import LPPosition, LeveragedPosition, SimulationResult


class LPAnalyzer:
    """
    Comprehensive analysis for LP positions.
    Calculates exposure, Greeks, and performance metrics.
    """
    
    @staticmethod
    def calculate_asset_exposure(position: LPPosition) -> Dict[str, float]:
        """
        Calculate exposure to each asset.
        
        Returns:
            Dict with asset0/asset1 exposure in asset1 terms
        """
        total_value = position.amount0 * position.current_price + position.amount1
        
        return {
            'asset0_value': position.amount0 * position.current_price,
            'asset1_value': position.amount1,
            'total_value': total_value,
            'asset0_percent': (position.amount0 * position.current_price / total_value * 100) if total_value > 0 else 0,
            'asset1_percent': (position.amount1 / total_value * 100) if total_value > 0 else 0
        }
    
    @staticmethod
    def calculate_greeks(result: SimulationResult) -> Dict[str, List[float]]:
        """
        Calculate position Greeks (delta, gamma).
        
        Delta: ∂V/∂P (price sensitivity)
        Gamma: ∂²V/∂P² (convexity)
        
        Returns:
            Dict with delta and gamma arrays
        """
        prices = np.array(result.prices)
        
        # Calculate values
        if isinstance(result.positions[0], LeveragedPosition):
            values = np.array([p.equity_asset1 for p in result.positions])
        else:
            values = np.array([p.amount0 * p.current_price + p.amount1 
                             for p in result.positions])
        
        # Numerical derivatives
        delta = np.gradient(values, prices)
        gamma = np.gradient(delta, prices)
        
        return {
            'delta': delta.tolist(),
            'gamma': gamma.tolist(),
            'max_abs_delta': float(np.max(np.abs(delta))),
            'max_abs_gamma': float(np.max(np.abs(gamma)))
        }
    
    @staticmethod
    def calculate_pnl(result: SimulationResult) -> Dict[str, List[float]]:
        """
        Calculate PnL across price range.
        
        Returns:
            Dict with PnL in absolute and percentage terms
        """
        initial_capital = result.config.capital_asset1
        
        pnl_list = []
        pnl_pct_list = []
        
        for pos in result.positions:
            if isinstance(pos, LeveragedPosition):
                equity = pos.equity_asset1
            else:
                equity = pos.amount0 * pos.current_price + pos.amount1
            
            pnl = equity - initial_capital
            pnl_pct = (pnl / initial_capital) * 100
            
            pnl_list.append(pnl)
            pnl_pct_list.append(pnl_pct)
        
        return {
            'pnl_asset1': pnl_list,
            'pnl_percent': pnl_pct_list,
            'max_gain': max(pnl_list),
            'max_loss': min(pnl_list),
            'max_gain_pct': max(pnl_pct_list),
            'max_loss_pct': min(pnl_pct_list)
        }
    
    @staticmethod
    def calculate_impermanent_loss(result: SimulationResult) -> List[float]:
        """
        Calculate impermanent loss vs. holding.
        
        IL = (LP_value - Hold_value) / Hold_value
        
        Returns:
            List of IL percentages at each price
        """
        initial_pos = result.positions[0]
        initial_price = result.config.price_initial
        
        # Hold strategy value at each price
        hold_value_initial = initial_pos.amount0 * initial_price + initial_pos.amount1
        
        il_list = []
        for pos in result.positions:
            lp_value = pos.amount0 * pos.current_price + pos.amount1
            hold_value = initial_pos.amount0 * pos.current_price + initial_pos.amount1
            
            il = ((lp_value - hold_value) / hold_value * 100) if hold_value > 0 else 0
            il_list.append(il)
        
        return il_list
    
    @classmethod
    def analyze_simulation(cls, result: SimulationResult) -> SimulationResult:
        """
        Run full analysis on simulation result.
        Populates the metrics dictionary.
        
        Returns:
            SimulationResult with filled metrics
        """
        result.metrics['exposure'] = [
            cls.calculate_asset_exposure(pos) for pos in result.positions
        ]
        result.metrics['greeks'] = cls.calculate_greeks(result)
        result.metrics['pnl'] = cls.calculate_pnl(result)
        result.metrics['impermanent_loss'] = cls.calculate_impermanent_loss(result)
        
        return result
