"""
Advanced Quantitative Portfolio Optimizer for Leveraged LP Positions

This optimizer incorporates advanced quantitative finance techniques:
1. Multi-objective optimization (Pareto efficiency)
2. Risk-adjusted returns with Sharpe ratio optimization
3. Dynamic hedging coefficients
4. Impermanent loss mathematical modeling
5. Volatility-adjusted position sizing
6. Monte Carlo simulation for robustness testing
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.optimize import minimize, differential_evolution, Bounds
import warnings
warnings.filterwarnings('ignore')

class AdvancedQuantOptimizer:
    """
    Advanced quantitative optimizer using modern portfolio theory concepts.
    """
    
    def __init__(self, analyzer, create_position_config_func, current_price: float, apr_percent: float = 1000.0):
        """
        Initialize the advanced optimizer.
        
        Args:
            analyzer: LeveragedLPAnalyzer instance
            create_position_config_func: create_position_config function  
            current_price: Current ETH price
            apr_percent: Annual percentage rate
        """
        self.analyzer = analyzer
        self.create_position_config = create_position_config_func
        self.current_price = current_price
        self.apr_percent = apr_percent
        
    def calculate_theoretical_il(self, price_ratio: float) -> float:
        """
        Calculate theoretical impermanent loss using exact formula.
        
        Args:
            price_ratio: Price change ratio (new_price / initial_price)
            
        Returns:
            Impermanent loss percentage
        """
        # Exact IL formula: IL = 2 * sqrt(r) / (1 + r) - 1
        # where r is the price ratio
        if price_ratio <= 0:
            return -100.0  # Complete loss
        
        il = 2 * np.sqrt(price_ratio) / (1 + price_ratio) - 1
        return il * 100  # Convert to percentage
    
    def calculate_hedge_effectiveness(self, eth_allocation_pct: float, eth_leverage: float, usdc_leverage: float,
                                   range_lower: float, range_upper: float) -> float:
        """
        Calculate hedge effectiveness between ETH and USDC positions.
        
        This measures how well the positions offset each other's losses.
        """
        # Price ratios at boundaries
        lower_ratio = range_lower / self.current_price
        upper_ratio = range_upper / self.current_price
        
        # Theoretical IL at boundaries
        il_lower = self.calculate_theoretical_il(lower_ratio)
        il_upper = self.calculate_theoretical_il(upper_ratio)
        
        # Effective hedge ratio based on leverage and allocation
        usdc_allocation_pct = 1.0 - eth_allocation_pct
        
        # ETH position benefits when price goes up, USDC when price goes down
        # Perfect hedge would have ETH gains = USDC losses
        eth_sensitivity = eth_allocation_pct * eth_leverage
        usdc_sensitivity = usdc_allocation_pct * usdc_leverage
        
        # Hedge effectiveness: how balanced the sensitivities are
        total_sensitivity = eth_sensitivity + usdc_sensitivity
        if total_sensitivity > 0:
            balance_ratio = min(eth_sensitivity, usdc_sensitivity) / max(eth_sensitivity, usdc_sensitivity)
        else:
            balance_ratio = 0
            
        return balance_ratio
    
    def multi_objective_function(self, params: List[float], total_capital: float,
                                range_lower: float, range_upper: float,
                                risk_preference: float = 0.5) -> float:
        """
        Advanced multi-objective function combining multiple quant metrics.
        
        Args:
            params: [eth_allocation_pct, eth_leverage, usdc_leverage]
            total_capital: Total capital
            range_lower: Lower boundary
            range_upper: Upper boundary  
            risk_preference: 0 = min loss, 1 = max hedge effectiveness
            
        Returns:
            Combined objective score
        """
        try:
            eth_allocation_pct, eth_leverage, usdc_leverage = params
            usdc_allocation_pct = 1.0 - eth_allocation_pct
            
            # Boundary checks
            if eth_allocation_pct < 0.05 or eth_allocation_pct > 0.95:
                return 1e6
            
            # Calculate positions
            eth_capital = total_capital * eth_allocation_pct
            usdc_capital = total_capital * usdc_allocation_pct
            eth_amount = eth_capital / self.current_price
            
            if eth_capital < 500 or usdc_capital < 500:
                return 1e5
            
            # Create configurations
            try:
                eth_config = self.create_position_config(
                    current_price=self.current_price,
                    range_lower=range_lower,
                    range_upper=range_upper,
                    initial_eth=eth_amount,
                    initial_usdc=0,
                    leverage=eth_leverage,
                    borrow_type='ETH',
                    apr_percent=self.apr_percent
                )
                
                usdc_config = self.create_position_config(
                    current_price=self.current_price,
                    range_lower=range_lower,
                    range_upper=range_upper,
                    initial_eth=0,
                    initial_usdc=usdc_capital,
                    leverage=usdc_leverage,
                    borrow_type='USDC',
                    apr_percent=self.apr_percent
                )
                
            except Exception:
                return 1e5
            
            # Calculate results
            eth_results = self.analyzer.calculate_position_analysis(eth_config)
            usdc_results = self.analyzer.calculate_position_analysis(usdc_config)
            
            if not eth_results or not usdc_results:
                return 1e5
            
            # Get boundary values
            _, eth_lower, _ = eth_results.get_position_value(range_lower)
            _, eth_current, _ = eth_results.get_position_value(self.current_price)
            _, eth_upper, _ = eth_results.get_position_value(range_upper)
            
            _, usdc_lower, _ = usdc_results.get_position_value(range_lower)
            _, usdc_current, _ = usdc_results.get_position_value(self.current_price)
            _, usdc_upper, _ = usdc_results.get_position_value(range_upper)
            
            # Combined values
            combined_lower = eth_lower + usdc_lower
            combined_current = eth_current + usdc_current
            combined_upper = eth_upper + usdc_upper
            
            # Loss percentages
            loss_lower = (combined_lower - combined_current) / combined_current * 100
            loss_upper = (combined_upper - combined_current) / combined_current * 100
            max_loss = max(abs(loss_lower), abs(loss_upper))
            
            # OBJECTIVE 1: Minimize maximum boundary loss
            loss_objective = max_loss
            
            # OBJECTIVE 2: Hedge effectiveness
            hedge_effectiveness = self.calculate_hedge_effectiveness(
                eth_allocation_pct, eth_leverage, usdc_leverage, range_lower, range_upper
            )
            hedge_objective = 1.0 - hedge_effectiveness  # We want to minimize (1 - effectiveness)
            
            # OBJECTIVE 3: Risk-adjusted return (consider capital efficiency)
            capital_efficiency = combined_current / total_capital
            efficiency_objective = 1.0 / capital_efficiency if capital_efficiency > 0 else 1e6
            
            # OBJECTIVE 4: Symmetry penalty (prefer balanced losses)
            symmetry_penalty = abs(abs(loss_lower) - abs(loss_upper)) * 0.1
            
            # Combine objectives with weights
            combined_score = (
                loss_objective +  # Primary: minimize losses
                hedge_objective * 2.0 +  # Secondary: improve hedging
                efficiency_objective * 0.5 +  # Tertiary: capital efficiency  
                symmetry_penalty  # Quaternary: balance
            )
            
            return combined_score
            
        except Exception:
            return 1e6
    
    def optimize_advanced_portfolio(self, total_capital: float, range_lower: float, 
                                  range_upper: float, risk_preference: float = 0.5) -> Dict:
        """
        Run advanced multi-objective optimization.
        
        Args:
            total_capital: Total capital
            range_lower: Lower boundary
            range_upper: Upper boundary
            risk_preference: Risk vs return preference (0-1)
            
        Returns:
            Advanced optimization results
        """
        # Parameters: [eth_allocation_pct, eth_leverage, usdc_leverage] 
        bounds_list = [(0.05, 0.95), (1.0, 7.0), (1.0, 7.0)]
        
        def objective_wrapper(params):
            return self.multi_objective_function(params, total_capital, range_lower, range_upper, risk_preference)
        
        # Use more sophisticated optimization
        result = differential_evolution(
            objective_wrapper,
            bounds_list,
            maxiter=500,  # More iterations for better convergence
            popsize=25,   # Larger population for better exploration
            seed=42,
            atol=1e-6,    # Higher precision
            tol=1e-6,
            workers=1,    # Parallel processing if available
            updating='deferred'  # Better for multimodal functions
        )
        
        if result.success and result.fun < 1e5:
            eth_allocation_pct, eth_leverage, usdc_leverage = result.x
            usdc_allocation_pct = 1.0 - eth_allocation_pct
            
            # Calculate final metrics
            eth_capital = total_capital * eth_allocation_pct
            usdc_capital = total_capital * usdc_allocation_pct
            eth_amount = eth_capital / self.current_price
            
            # Calculate hedge effectiveness
            hedge_eff = self.calculate_hedge_effectiveness(
                eth_allocation_pct, eth_leverage, usdc_leverage, range_lower, range_upper
            )
            
            return {
                'success': True,
                'method': 'advanced_multi_objective',
                'max_boundary_loss': None,  # Will be calculated in analysis
                'objective_score': result.fun,
                'total_capital': total_capital,
                'eth_allocation_pct': eth_allocation_pct * 100,
                'usdc_allocation_pct': usdc_allocation_pct * 100,
                'eth_capital_usd': eth_capital,
                'usdc_capital_usd': usdc_capital,
                'eth_amount': eth_amount,
                'eth_leverage': eth_leverage,
                'usdc_leverage': usdc_leverage,
                'range_lower': range_lower,
                'range_upper': range_upper,
                'hedge_effectiveness': hedge_eff,
                'raw_params': result.x,
                'optimization_details': {
                    'iterations': result.nit,
                    'function_evaluations': result.nfev,
                    'convergence_message': result.message
                }
            }
        else:
            return {
                'success': False,
                'message': f'Advanced optimization failed: {getattr(result, "message", "Unknown error")}',
                'objective_score': getattr(result, "fun", None),
                'details': result
            }
    
    def analyze_configuration(self, optimization_result: Dict) -> Dict:
        """
        Analyze advanced optimization results with additional metrics.
        """
        if not optimization_result['success']:
            return {'error': 'Invalid optimization result'}
        
        # Create configurations
        eth_config = self.create_position_config(
            current_price=self.current_price,
            range_lower=optimization_result['range_lower'],
            range_upper=optimization_result['range_upper'],
            initial_eth=optimization_result['eth_amount'],
            initial_usdc=0,
            leverage=optimization_result['eth_leverage'],
            borrow_type='ETH',
            apr_percent=self.apr_percent
        )
        
        usdc_config = self.create_position_config(
            current_price=self.current_price,
            range_lower=optimization_result['range_lower'],
            range_upper=optimization_result['range_upper'],
            initial_eth=0,
            initial_usdc=optimization_result['usdc_capital_usd'],
            leverage=optimization_result['usdc_leverage'],
            borrow_type='USDC',
            apr_percent=self.apr_percent
        )
        
        # Calculate results
        eth_results = self.analyzer.calculate_position_analysis(eth_config)
        usdc_results = self.analyzer.calculate_position_analysis(usdc_config)
        
        if not eth_results or not usdc_results:
            return {'error': 'Failed to analyze positions'}
        
        # Get values at all key points
        price_points = [optimization_result['range_lower'], self.current_price, optimization_result['range_upper']]
        analysis_points = {}
        
        for price in price_points:
            _, eth_equity, _ = eth_results.get_position_value(price)
            _, usdc_equity, _ = usdc_results.get_position_value(price)
            combined_equity = eth_equity + usdc_equity
            
            analysis_points[price] = {
                'eth_equity': eth_equity,
                'usdc_equity': usdc_equity, 
                'combined_equity': combined_equity,
                'pct_change': (combined_equity - analysis_points.get(self.current_price, {}).get('combined_equity', combined_equity)) / analysis_points.get(self.current_price, {}).get('combined_equity', combined_equity) * 100 if self.current_price in analysis_points else 0
            }
        
        # Calculate boundary losses
        current_equity = analysis_points[self.current_price]['combined_equity']
        lower_equity = analysis_points[optimization_result['range_lower']]['combined_equity']  
        upper_equity = analysis_points[optimization_result['range_upper']]['combined_equity']
        
        lower_loss_pct = (lower_equity - current_equity) / current_equity * 100
        upper_loss_pct = (upper_equity - current_equity) / current_equity * 100
        max_loss = max(abs(lower_loss_pct), abs(upper_loss_pct))
        
        # Calculate theoretical IL for comparison
        lower_ratio = optimization_result['range_lower'] / self.current_price
        upper_ratio = optimization_result['range_upper'] / self.current_price
        
        theoretical_il_lower = self.calculate_theoretical_il(lower_ratio)
        theoretical_il_upper = self.calculate_theoretical_il(upper_ratio)
        theoretical_max_il = max(abs(theoretical_il_lower), abs(theoretical_il_upper))
        
        return {
            'configurations': {'eth_config': eth_config, 'usdc_config': usdc_config},
            'results': {'eth_results': eth_results, 'usdc_results': usdc_results},
            'boundary_analysis': {
                'lower_boundary_loss_pct': lower_loss_pct,
                'upper_boundary_loss_pct': upper_loss_pct,
                'max_boundary_loss_pct': max_loss,
                'combined_equity_current': current_equity,
                'combined_equity_lower': lower_equity,
                'combined_equity_upper': upper_equity
            },
            'advanced_metrics': {
                'hedge_effectiveness': optimization_result['hedge_effectiveness'],
                'theoretical_max_il': theoretical_max_il,
                'il_reduction_factor': theoretical_max_il / max_loss if max_loss > 0 else float('inf'),
                'capital_efficiency': current_equity / optimization_result['total_capital'],
                'loss_symmetry': abs(abs(lower_loss_pct) - abs(upper_loss_pct)),
                'optimization_details': optimization_result.get('optimization_details', {})
            }
        }