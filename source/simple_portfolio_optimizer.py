"""
Simple Portfolio Optimizer - Fixed Capital Allocation Logic

This optimizer uses only 3 parameters:
1. ETH allocation percentage (0-100%) - USDC allocation is automatically (100% - ETH%)
2. ETH leverage (1x-7x)  
3. USDC leverage (1x-7x)

This ensures the total allocation is always exactly 100% of capital.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.optimize import minimize, differential_evolution, Bounds
import warnings
warnings.filterwarnings('ignore')

class SimplePortfolioOptimizer:
    """
    Simple optimizer that ensures capital allocation is always exactly 100%.
    """
    
    def __init__(self, analyzer, create_position_config_func, current_price: float, apr_percent: float = 1000.0):
        """
        Initialize the optimizer.
        
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
        
    def objective_function(self, params: List[float], total_capital: float, 
                         range_lower: float, range_upper: float) -> float:
        """
        Objective function with correct capital allocation.
        
        Args:
            params: [eth_allocation_pct (0-1), eth_leverage, usdc_leverage]
            total_capital: Total capital in USD
            range_lower: Lower price boundary
            range_upper: Upper price boundary
            
        Returns:
            Maximum boundary loss percentage to minimize
        """
        try:
            eth_allocation_pct, eth_leverage, usdc_leverage = params
            
            # USDC allocation is automatically the remainder
            usdc_allocation_pct = 1.0 - eth_allocation_pct
            
            # Ensure valid allocation range
            if eth_allocation_pct < 0.05 or eth_allocation_pct > 0.95:
                return 1e6  # Keep at least 5% in each position
            
            # Calculate capital allocation (total always equals 100%)
            eth_capital = total_capital * eth_allocation_pct
            usdc_capital = total_capital * usdc_allocation_pct
            eth_amount = eth_capital / self.current_price
            
            # Validate minimum position sizes
            if eth_capital < 500 or usdc_capital < 500:
                return 1e5  # Need meaningful position sizes
            
            # Create position configurations
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
                
            except Exception as e:
                return 1e5
            
            # Calculate position results
            eth_results = self.analyzer.calculate_position_analysis(eth_config)
            usdc_results = self.analyzer.calculate_position_analysis(usdc_config)
            
            if not eth_results or not usdc_results:
                return 1e5
            
            # Get equity values at boundaries
            _, eth_equity_lower, _ = eth_results.get_position_value(range_lower)
            _, eth_equity_current, _ = eth_results.get_position_value(self.current_price)
            _, eth_equity_upper, _ = eth_results.get_position_value(range_upper)
            
            _, usdc_equity_lower, _ = usdc_results.get_position_value(range_lower)
            _, usdc_equity_current, _ = usdc_results.get_position_value(self.current_price)
            _, usdc_equity_upper, _ = usdc_results.get_position_value(range_upper)
            
            # Combined portfolio values
            combined_equity_lower = eth_equity_lower + usdc_equity_lower
            combined_equity_current = eth_equity_current + usdc_equity_current
            combined_equity_upper = eth_equity_upper + usdc_equity_upper
            
            # Calculate percentage losses from current value
            pct_loss_lower = (combined_equity_lower - combined_equity_current) / combined_equity_current * 100
            pct_loss_upper = (combined_equity_upper - combined_equity_current) / combined_equity_current * 100
            
            # Return maximum absolute boundary loss
            max_boundary_loss = max(abs(pct_loss_lower), abs(pct_loss_upper))
            return max_boundary_loss
            
        except Exception as e:
            return 1e6
    
    def optimize_portfolio(self, total_capital: float, range_lower: float, range_upper: float) -> Dict:
        """
        Find optimal allocation and leverage levels.
        
        Args:
            total_capital: Total capital in USD
            range_lower: Lower price boundary
            range_upper: Upper price boundary
            
        Returns:
            Optimization results with corrected allocation logic
        """
        # Parameters: [eth_allocation_pct (0-1), eth_leverage, usdc_leverage]
        bounds_list = [(0.05, 0.95), (1.0, 7.0), (1.0, 7.0)]
        
        def constrained_objective(params):
            return self.objective_function(params, total_capital, range_lower, range_upper)
        
        # Use differential evolution for global optimization
        result = differential_evolution(
            constrained_objective,
            bounds_list,
            maxiter=300,
            popsize=15,
            seed=42,
            atol=1e-4,
            tol=1e-4
        )
        
        if result.success and result.fun < 1e5:
            eth_allocation_pct, eth_leverage, usdc_leverage = result.x
            usdc_allocation_pct = 1.0 - eth_allocation_pct  # Guaranteed to sum to 100%
            
            # Calculate actual allocations
            eth_capital = total_capital * eth_allocation_pct
            usdc_capital = total_capital * usdc_allocation_pct
            eth_amount = eth_capital / self.current_price
            
            return {
                'success': True,
                'max_boundary_loss': result.fun,
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
                'raw_params': result.x
            }
        else:
            return {
                'success': False,
                'message': f'Optimization failed: {getattr(result, "message", "Unknown error")}',
                'objective_score': getattr(result, "fun", None)
            }
    
    def analyze_configuration(self, optimization_result: Dict) -> Dict:
        """
        Analyze a specific configuration in detail.
        
        Args:
            optimization_result: Result from optimize_portfolio()
            
        Returns:
            Detailed analysis of the configuration
        """
        if not optimization_result['success']:
            return {'error': 'Invalid optimization result'}
        
        # Create position configurations
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
        
        # Get boundary values
        _, eth_equity_lower, _ = eth_results.get_position_value(optimization_result['range_lower'])
        _, eth_equity_current, _ = eth_results.get_position_value(self.current_price)
        _, eth_equity_upper, _ = eth_results.get_position_value(optimization_result['range_upper'])
        
        _, usdc_equity_lower, _ = usdc_results.get_position_value(optimization_result['range_lower'])
        _, usdc_equity_current, _ = usdc_results.get_position_value(self.current_price)
        _, usdc_equity_upper, _ = usdc_results.get_position_value(optimization_result['range_upper'])
        
        # Combined values
        combined_lower = eth_equity_lower + usdc_equity_lower
        combined_current = eth_equity_current + usdc_equity_current
        combined_upper = eth_equity_upper + usdc_equity_upper
        
        # Calculate percentage changes
        pct_loss_lower = (combined_lower - combined_current) / combined_current * 100
        pct_loss_upper = (combined_upper - combined_current) / combined_current * 100
        
        return {
            'configurations': {'eth_config': eth_config, 'usdc_config': usdc_config},
            'results': {'eth_results': eth_results, 'usdc_results': usdc_results},
            'boundary_analysis': {
                'lower_boundary_loss_pct': pct_loss_lower,
                'upper_boundary_loss_pct': pct_loss_upper,
                'max_boundary_loss_pct': max(abs(pct_loss_lower), abs(pct_loss_upper)),
                'combined_equity_current': combined_current,
                'combined_equity_lower': combined_lower,
                'combined_equity_upper': combined_upper
            }
        }