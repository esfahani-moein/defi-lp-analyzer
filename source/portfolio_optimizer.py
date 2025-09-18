"""
Portfolio Optimization Module for Leveraged LP Positions

This module provides optimization algorithms to find optimal capital allocation,
leverage levels, and range parameters to minimize boundary losses.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from scipy.optimize import minimize, differential_evolution, Bounds
import warnings
warnings.filterwarnings('ignore')

# Import the position creation function
from .leveraged_lp_analysis import create_position_config

class PortfolioOptimizer:
    """
    Optimizer for leveraged LP portfolios to minimize boundary losses
    and maximize risk-adjusted returns.
    """
    
    def __init__(self, analyzer, current_price: float = 4500.0, apr_percent: float = 1000.0):
        """
        Initialize the portfolio optimizer.
        
        Args:
            analyzer: LeveragedLPAnalyzer instance
            current_price: Current ETH price
            apr_percent: Annual percentage rate for fees
        """
        self.analyzer = analyzer
        self.current_price = current_price
        self.apr_percent = apr_percent
        
    def objective_function(self, params: List[float], total_capital: float, 
                         range_lower: float, range_upper: float,
                         optimization_type: str = 'minimize_max_boundary_loss') -> float:
        """
        Objective function for portfolio optimization.
        
        Args:
            params: [eth_allocation_pct, usdc_allocation_pct, eth_leverage, usdc_leverage]
            total_capital: Total available capital in USD
            range_lower: Lower price boundary
            range_upper: Upper price boundary
            optimization_type: Type of optimization strategy
            
        Returns:
            Objective score to minimize
        """
        try:
            eth_allocation_pct, usdc_allocation_pct, eth_leverage, usdc_leverage = params
            
            # Ensure allocations sum to 1 (100%)
            if abs(eth_allocation_pct + usdc_allocation_pct - 1.0) > 0.01:
                return 1e6  # Heavy penalty for invalid allocation
            
            # Calculate capital allocation
            eth_capital = total_capital * eth_allocation_pct
            usdc_capital = total_capital * usdc_allocation_pct
            
            # Convert ETH capital to ETH amount
            eth_amount = eth_capital / self.current_price
            
            # Skip if allocations are too small
            if eth_capital < 100 or usdc_capital < 100:
                return 1e5
            
            # Create position configurations
            try:
                eth_config = create_position_config(
                    current_price=self.current_price,
                    range_lower=range_lower,
                    range_upper=range_upper,
                    initial_eth=eth_amount,
                    initial_usdc=0,
                    leverage=eth_leverage,
                    borrow_type='ETH',
                    apr_percent=self.apr_percent
                )
                
                usdc_config = create_position_config(
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
            
            # Calculate percentage losses
            pct_loss_lower = (combined_equity_lower - combined_equity_current) / combined_equity_current * 100
            pct_loss_upper = (combined_equity_upper - combined_equity_current) / combined_equity_current * 100
            
            # Different optimization objectives
            if optimization_type == 'minimize_max_boundary_loss':
                # Minimize the maximum absolute loss at boundaries
                return max(abs(pct_loss_lower), abs(pct_loss_upper))
            
            elif optimization_type == 'minimize_total_boundary_loss':
                # Minimize sum of absolute boundary losses
                return abs(pct_loss_lower) + abs(pct_loss_upper)
            
            elif optimization_type == 'balanced_boundary_loss':
                # Minimize max loss with penalty for imbalance
                max_loss = max(abs(pct_loss_lower), abs(pct_loss_upper))
                imbalance_penalty = abs(abs(pct_loss_lower) - abs(pct_loss_upper)) * 0.3
                return max_loss + imbalance_penalty
            
            elif optimization_type == 'risk_adjusted':
                # Risk-adjusted objective considering capital efficiency
                max_loss = max(abs(pct_loss_lower), abs(pct_loss_upper))
                capital_efficiency = combined_equity_current / total_capital
                return max_loss / capital_efficiency
            
            else:
                return max(abs(pct_loss_lower), abs(pct_loss_upper))
            
        except Exception as e:
            return 1e6
    
    def optimize_portfolio(self, total_capital: float, range_lower: float, range_upper: float,
                          optimization_type: str = 'minimize_max_boundary_loss',
                          method: str = 'differential_evolution') -> Dict:
        """
        Optimize portfolio parameters to minimize boundary losses.
        
        Args:
            total_capital: Total capital to allocate (USD)
            range_lower: Lower price boundary
            range_upper: Upper price boundary
            optimization_type: Optimization strategy
            method: Optimization algorithm
            
        Returns:
            Dictionary with optimization results
        """
        if method == 'differential_evolution':
            # For differential evolution, we'll handle the constraint by penalizing invalid allocations
            # in the objective function instead of using scipy constraints
            bounds_list = [(0.1, 0.9), (0.1, 0.9), (1.5, 12.0), (1.5, 12.0)]
            
            def constrained_objective(x):
                # Normalize allocations to sum to 1
                eth_alloc, usdc_alloc, eth_lev, usdc_lev = x
                total_alloc = eth_alloc + usdc_alloc
                
                if total_alloc <= 0:
                    return 1e6
                
                # Normalize allocations
                normalized_params = [
                    eth_alloc / total_alloc,
                    usdc_alloc / total_alloc,
                    eth_lev,
                    usdc_lev
                ]
                
                return self.objective_function(normalized_params, total_capital, range_lower, range_upper, optimization_type)
            
            result = differential_evolution(
                constrained_objective,
                bounds_list,
                seed=42,
                maxiter=200,
                popsize=20,
                tol=1e-8,
                atol=1e-8
            )
            
            # Normalize the result
            if result.success:
                eth_alloc, usdc_alloc, eth_lev, usdc_lev = result.x
                total_alloc = eth_alloc + usdc_alloc
                result.x = [eth_alloc / total_alloc, usdc_alloc / total_alloc, eth_lev, usdc_lev]
        else:
            # Local optimization with proper constraint handling
            # Parameter bounds: [eth_allocation_pct, usdc_allocation_pct, eth_leverage, usdc_leverage]
            bounds = Bounds(
                lb=[0.1, 0.1, 1.5, 1.5],  # Lower bounds
                ub=[0.9, 0.9, 12.0, 12.0]  # Upper bounds
            )
            
            # Constraint: allocations must sum to 1
            def allocation_constraint(x):
                return x[0] + x[1] - 1.0
            
            constraints = [{'type': 'eq', 'fun': allocation_constraint}]
            
            x0 = [0.4, 0.6, 6.0, 6.0]  # Initial guess
            
            result = minimize(
                lambda x: self.objective_function(x, total_capital, range_lower, range_upper, optimization_type),
                x0,
                bounds=bounds,
                constraints=constraints,
                method='SLSQP',
                options={'ftol': 1e-9, 'disp': False}
            )
        
        # Package results
        if result.success and result.fun < 1e5:
            eth_alloc_pct, usdc_alloc_pct, eth_lev, usdc_lev = result.x
            
            # Calculate actual allocations
            eth_capital = total_capital * eth_alloc_pct
            usdc_capital = total_capital * usdc_alloc_pct
            eth_amount = eth_capital / self.current_price
            
            return {
                'success': True,
                'optimization_type': optimization_type,
                'method': method,
                'objective_score': result.fun,
                'total_capital': total_capital,
                'allocations': {
                    'eth_allocation_pct': eth_alloc_pct * 100,
                    'usdc_allocation_pct': usdc_alloc_pct * 100,
                    'eth_capital_usd': eth_capital,
                    'usdc_capital_usd': usdc_capital,
                    'eth_amount': eth_amount
                },
                'leverages': {
                    'eth_leverage': eth_lev,
                    'usdc_leverage': usdc_lev
                },
                'range': {
                    'lower': range_lower,
                    'upper': range_upper,
                    'width': range_upper - range_lower
                },
                'raw_params': result.x,
                'optimization_result': result
            }
        else:
            return {
                'success': False,
                'message': f'Optimization failed: {result.message if hasattr(result, "message") else "Unknown error"}',
                'objective_score': result.fun if hasattr(result, "fun") else None
            }
    
    def analyze_optimal_position(self, optimization_result: Dict) -> Dict:
        """
        Analyze the performance of an optimized position.
        
        Args:
            optimization_result: Result from optimize_portfolio()
            
        Returns:
            Dictionary with detailed analysis
        """
        if not optimization_result['success']:
            return {'error': 'Invalid optimization result'}
        
        # Extract parameters
        alloc = optimization_result['allocations']
        lev = optimization_result['leverages']
        range_info = optimization_result['range']
        
        # Create position configurations
        from .leveraged_lp_analysis import create_position_config
        
        eth_config = create_position_config(
            current_price=self.current_price,
            range_lower=range_info['lower'],
            range_upper=range_info['upper'],
            initial_eth=alloc['eth_amount'],
            initial_usdc=0,
            leverage=lev['eth_leverage'],
            borrow_type='ETH',
            apr_percent=self.apr_percent
        )
        
        usdc_config = create_position_config(
            current_price=self.current_price,
            range_lower=range_info['lower'],
            range_upper=range_info['upper'],
            initial_eth=0,
            initial_usdc=alloc['usdc_capital_usd'],
            leverage=lev['usdc_leverage'],
            borrow_type='USDC',
            apr_percent=self.apr_percent
        )
        
        # Calculate position results
        eth_results = self.analyzer.calculate_position_analysis(eth_config)
        usdc_results = self.analyzer.calculate_position_analysis(usdc_config)
        
        if not eth_results or not usdc_results:
            return {'error': 'Failed to analyze optimal position'}
        
        # Calculate performance at key points
        price_points = {
            'lower_boundary': range_info['lower'],
            'current_price': self.current_price,
            'upper_boundary': range_info['upper'],
            'stress_low': self.current_price * 0.85,
            'stress_high': self.current_price * 1.15
        }
        
        performance = {}
        for name, price in price_points.items():
            _, eth_equity, _ = eth_results.get_position_value(price)
            _, usdc_equity, _ = usdc_results.get_position_value(price)
            
            combined_equity = eth_equity + usdc_equity
            current_combined = eth_results.get_position_value(self.current_price)[1] + \
                             usdc_results.get_position_value(self.current_price)[1]
            
            pct_change = (combined_equity - current_combined) / current_combined * 100
            
            performance[name] = {
                'price': price,
                'combined_equity': combined_equity,
                'pct_change': pct_change,
                'eth_equity': eth_equity,
                'usdc_equity': usdc_equity
            }
        
        return {
            'configurations': {
                'eth_config': eth_config,
                'usdc_config': usdc_config
            },
            'results': {
                'eth_results': eth_results,
                'usdc_results': usdc_results
            },
            'performance': performance,
            'boundary_losses': {
                'lower_loss_pct': performance['lower_boundary']['pct_change'],
                'upper_loss_pct': performance['upper_boundary']['pct_change'],
                'max_boundary_loss': max(
                    abs(performance['lower_boundary']['pct_change']),
                    abs(performance['upper_boundary']['pct_change'])
                )
            }
        }
    
    def compare_strategies(self, total_capital: float, range_lower: float, range_upper: float) -> Dict:
        """
        Compare different optimization strategies.
        
        Args:
            total_capital: Total capital to allocate
            range_lower: Lower price boundary
            range_upper: Upper price boundary
            
        Returns:
            Dictionary comparing all strategies
        """
        strategies = [
            'minimize_max_boundary_loss',
            'minimize_total_boundary_loss', 
            'balanced_boundary_loss',
            'risk_adjusted'
        ]
        
        results = {}
        
        for strategy in strategies:
            print(f"🔄 Optimizing strategy: {strategy}...")
            
            opt_result = self.optimize_portfolio(
                total_capital=total_capital,
                range_lower=range_lower,
                range_upper=range_upper,
                optimization_type=strategy,
                method='differential_evolution'
            )
            
            if opt_result['success']:
                analysis = self.analyze_optimal_position(opt_result)
                
                results[strategy] = {
                    'optimization': opt_result,
                    'analysis': analysis,
                    'summary': {
                        'max_boundary_loss': analysis['boundary_losses']['max_boundary_loss'],
                        'lower_loss': analysis['boundary_losses']['lower_loss_pct'],
                        'upper_loss': analysis['boundary_losses']['upper_loss_pct'],
                        'eth_allocation': opt_result['allocations']['eth_allocation_pct'],
                        'usdc_allocation': opt_result['allocations']['usdc_allocation_pct'],
                        'eth_leverage': opt_result['leverages']['eth_leverage'],
                        'usdc_leverage': opt_result['leverages']['usdc_leverage']
                    }
                }
            else:
                results[strategy] = {'success': False, 'error': opt_result.get('message', 'Unknown error')}
        
        return results