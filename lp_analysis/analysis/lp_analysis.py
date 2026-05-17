"""
Advanced LP analysis: exposure, Greeks, performance metrics.
"""
import numpy as np
from typing import Dict, List
from ..lp_calc.types import LPPosition, LeveragedLPConfig, LeveragedPosition, SimulationResult
from ..lp_calc.strategy import create_initial_leveraged_position, create_initial_position


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
        if isinstance(result.config, LeveragedLPConfig):
            initial_pos = create_initial_leveraged_position(result.config)
        else:
            initial_pos = create_initial_position(result.config)
        initial_price = result.config.price_initial
        
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


class ExposureAnalyzer:
    """
    Advanced exposure analysis for LP positions.
    Calculates net exposure to volatile assets considering debt.
    """
    
    @staticmethod
    def calculate_net_exposure(position: LeveragedPosition) -> Dict[str, float]:
        """
        Calculate net exposure to each asset accounting for debt.
        
        Net Exposure = LP Holdings - Debt
        
        For volatile asset (asset0) exposure optimization:
        - Long exposure: holding asset0 in LP
        - Short exposure: owing asset0 as debt
        
        Returns:
            Dict with net exposure metrics
        """
        # LP holdings
        lp_asset0 = position.amount0
        lp_asset1 = position.amount1
        
        # Debt
        debt_asset0 = position.debt_amount0
        debt_asset1 = position.debt_amount1
        
        # Net exposure (in token terms)
        net_asset0 = lp_asset0 - debt_asset0
        net_asset1 = lp_asset1 - debt_asset1
        
        # Net exposure in asset1 value terms
        net_asset0_value = net_asset0 * position.current_price
        net_asset1_value = net_asset1
        
        # Total net value
        total_net_value = net_asset0_value + net_asset1_value
        
        # Exposure percentages
        asset0_exposure_pct = (net_asset0_value / total_net_value * 100) if total_net_value != 0 else 0
        asset1_exposure_pct = (net_asset1_value / total_net_value * 100) if total_net_value != 0 else 0
        
        return {
            'net_asset0_tokens': net_asset0,
            'net_asset1_tokens': net_asset1,
            'net_asset0_value': net_asset0_value,
            'net_asset1_value': net_asset1_value,
            'total_net_value': total_net_value,
            'asset0_exposure_pct': asset0_exposure_pct,
            'asset1_exposure_pct': asset1_exposure_pct,
            'is_long_asset0': net_asset0 > 0,
            'is_short_asset0': net_asset0 < 0
        }
    
    @staticmethod
    def calculate_exposure_profile(result: SimulationResult) -> Dict:
        """
        Calculate comprehensive exposure profile across price range.
        
        Returns:
            Dict with exposure metrics at each price point
        """
        if not isinstance(result.positions[0], LeveragedPosition):
            # For non-leveraged, exposure = holdings
            asset0_exposure = [p.amount0 for p in result.positions]
            asset1_exposure = [p.amount1 for p in result.positions]
            
            return {
                'prices': result.prices,
                'asset0_net_tokens': asset0_exposure,
                'asset1_net_tokens': asset1_exposure,
                'asset0_net_value': [a0 * p for a0, p in zip(asset0_exposure, result.prices)],
                'asset1_net_value': asset1_exposure,
                'asset0_exposure_pct': [],
                'asset1_exposure_pct': [],
                'is_leveraged': False
            }
        
        # Leveraged positions
        net_exposures = [ExposureAnalyzer.calculate_net_exposure(pos) 
                        for pos in result.positions]
        
        return {
            'prices': result.prices,
            'asset0_net_tokens': [exp['net_asset0_tokens'] for exp in net_exposures],
            'asset1_net_tokens': [exp['net_asset1_tokens'] for exp in net_exposures],
            'asset0_net_value': [exp['net_asset0_value'] for exp in net_exposures],
            'asset1_net_value': [exp['net_asset1_value'] for exp in net_exposures],
            'asset0_exposure_pct': [exp['asset0_exposure_pct'] for exp in net_exposures],
            'asset1_exposure_pct': [exp['asset1_exposure_pct'] for exp in net_exposures],
            'total_net_value': [exp['total_net_value'] for exp in net_exposures],
            'is_leveraged': True
        }
    
    @staticmethod
    def calculate_exposure_delta(result: SimulationResult) -> Dict:
        """
        Calculate exposure delta (sensitivity of exposure to price changes).
        
        This shows how net exposure changes as price moves.
        Critical for understanding rebalancing needs.
        
        Returns:
            Dict with exposure delta metrics
        """
        exposure_profile = ExposureAnalyzer.calculate_exposure_profile(result)
        
        if not exposure_profile['is_leveraged']:
            return {'message': 'Exposure delta only relevant for leveraged positions'}
        
        prices = np.array(exposure_profile['prices'])
        asset0_net_tokens = np.array(exposure_profile['asset0_net_tokens'])
        asset0_net_value = np.array(exposure_profile['asset0_net_value'])
        
        # Delta of net asset0 tokens vs price (should be relatively flat due to debt)
        token_delta = np.gradient(asset0_net_tokens, prices)
        
        # Delta of net asset0 value vs price
        value_delta = np.gradient(asset0_net_value, prices)
        
        return {
            'token_delta': token_delta.tolist(),
            'value_delta': value_delta.tolist(),
            'max_token_delta': float(np.max(np.abs(token_delta))),
            'max_value_delta': float(np.max(np.abs(value_delta)))
        }
    
    @classmethod
    def analyze_exposure(cls, result: SimulationResult) -> SimulationResult:
        """
        Add comprehensive exposure analysis to simulation result.
        
        Returns:
            SimulationResult with exposure metrics added
        """
        result.metrics['exposure_profile'] = cls.calculate_exposure_profile(result)
        result.metrics['exposure_delta'] = cls.calculate_exposure_delta(result)
        
        return result
    
    @staticmethod
    def compare_exposure_strategies(results_list: List[SimulationResult]) -> Dict:
        """
        Compare exposure profiles across multiple strategies.
        
        Useful for optimizing exposure by comparing:
        - ASSET0 debt vs ASSET1 debt
        - Different leverage levels
        - Different price ranges
        
        Args:
            results_list: List of SimulationResult objects to compare
            
        Returns:
            Dict with comparison metrics
        """
        comparison = {
            'strategies': [],
            'initial_price': results_list[0].config.price_initial
        }
        
        for i, result in enumerate(results_list):
            exposure_profile = ExposureAnalyzer.calculate_exposure_profile(result)
            
            # Find exposure at initial price
            initial_idx = min(range(len(result.prices)), 
                            key=lambda i: abs(result.prices[i] - result.config.price_initial))
            
            config = result.config
            strategy_info = {
                'name': f"{config.assets.asset0_symbol}/{config.assets.asset1_symbol}",
                'leverage': getattr(config, 'leverage', 1.0),
                'debt_asset': getattr(config, 'debt_asset', 'NONE'),
                'initial_asset0_exposure_pct': exposure_profile['asset0_exposure_pct'][initial_idx] if exposure_profile['is_leveraged'] else 0,
                'initial_asset1_exposure_pct': exposure_profile['asset1_exposure_pct'][initial_idx] if exposure_profile['is_leveraged'] else 0,
                'avg_asset0_exposure_pct': np.mean(exposure_profile['asset0_exposure_pct']) if exposure_profile['is_leveraged'] else 0,
                'exposure_stability': np.std(exposure_profile['asset0_exposure_pct']) if exposure_profile['is_leveraged'] else 0
            }
            
            comparison['strategies'].append(strategy_info)
        
        return comparison
    
    @staticmethod
    def calculate_market_exposure(result: SimulationResult) -> Dict:
        """
        Calculate market exposure metrics including delta and gamma.
        
        This method provides delta (price sensitivity) and gamma (convexity)
        of the position's exposure, which are critical for understanding
        hedging effectiveness in delta-neutral strategies.
        
        Returns:
            Dict with delta_exposure and gamma_exposure arrays
        """
        prices = np.array(result.prices)
        
        # Get exposure profile
        exposure_profile = ExposureAnalyzer.calculate_exposure_profile(result)
        
        if not exposure_profile['is_leveraged']:
            # For non-leveraged positions, delta is simpler
            asset0_net_tokens = np.array(exposure_profile['asset0_net_tokens'])
            delta_exposure = asset0_net_tokens  # Delta ≈ ETH holdings
            gamma_exposure = np.gradient(delta_exposure, prices)
            
            return {
                'delta_exposure': delta_exposure.tolist(),
                'gamma_exposure': gamma_exposure.tolist(),
                'avg_abs_delta': float(np.mean(np.abs(delta_exposure))),
                'max_abs_delta': float(np.max(np.abs(delta_exposure))),
                'avg_abs_gamma': float(np.mean(np.abs(gamma_exposure))),
                'max_abs_gamma': float(np.max(np.abs(gamma_exposure)))
            }
        
        values = np.array([p.equity_asset1 for p in result.positions])
        delta_exposure = np.gradient(values, prices)
        gamma_exposure = np.gradient(delta_exposure, prices)
        
        return {
            'delta_exposure': delta_exposure.tolist(),
            'gamma_exposure': gamma_exposure.tolist(),
            'net_asset0_tokens': exposure_profile['asset0_net_tokens'],
            'net_asset0_value': exposure_profile['asset0_net_value'],
            'avg_abs_delta': float(np.mean(np.abs(delta_exposure))),
            'max_abs_delta': float(np.max(np.abs(delta_exposure))),
            'avg_abs_gamma': float(np.mean(np.abs(gamma_exposure))),
            'max_abs_gamma': float(np.max(np.abs(gamma_exposure))),
            'delta_at_current_price': float(delta_exposure[len(delta_exposure) // 2]),
            'gamma_at_current_price': float(gamma_exposure[len(gamma_exposure) // 2])
        }

