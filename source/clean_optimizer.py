"""
Clean Portfolio Optimizer with Heatmap Analysis

This is the ONLY optimizer file you need. It combines:
1. Proper capital allocation (ETH% + USDC% = 100%)
2. Optimization with 1x-7x leverage
3. Heatmap generation for leverage and allocation effects
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List, Optional
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

class CleanPortfolioOptimizer:
    """
    The ONLY optimizer you need - clean and focused.
    """
    
    def __init__(self, analyzer, create_position_config_func, current_price: float, apr_percent: float = 1000.0):
        self.analyzer = analyzer
        self.create_position_config = create_position_config_func
        self.current_price = current_price
        self.apr_percent = apr_percent
        
    def calculate_boundary_loss(self, eth_allocation_pct: float, eth_leverage: float, 
                              usdc_leverage: float, total_capital: float, 
                              range_lower: float, range_upper: float) -> float:
        """
        Calculate maximum boundary loss for given parameters.
        
        Returns:
            Maximum boundary loss percentage (or high penalty if invalid)
        """
        try:
            # USDC allocation is remainder (guaranteed 100% total)
            usdc_allocation_pct = 1.0 - eth_allocation_pct
            
            # Boundary checks
            if eth_allocation_pct < 0.05 or eth_allocation_pct > 0.95:
                return 999.0  # Invalid allocation
            
            if not (1.0 <= eth_leverage <= 7.0) or not (1.0 <= usdc_leverage <= 7.0):
                return 999.0  # Invalid leverage
                
            # Calculate capital allocation
            eth_capital = total_capital * eth_allocation_pct
            usdc_capital = total_capital * usdc_allocation_pct
            eth_amount = eth_capital / self.current_price
            
            if eth_capital < 500 or usdc_capital < 500:
                return 999.0  # Too small positions
                
            # Create position configurations
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
            
            # Calculate results
            eth_results = self.analyzer.calculate_position_analysis(eth_config)
            usdc_results = self.analyzer.calculate_position_analysis(usdc_config)
            
            if not eth_results or not usdc_results:
                return 999.0
                
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
            
            return max_loss
            
        except Exception:
            return 999.0
    
    def optimize_portfolio(self, total_capital: float, range_lower: float, range_upper: float) -> Dict:
        """
        Find optimal allocation and leverage levels.
        """
        def objective(params):
            eth_allocation_pct, eth_leverage, usdc_leverage = params
            return self.calculate_boundary_loss(
                eth_allocation_pct, eth_leverage, usdc_leverage,
                total_capital, range_lower, range_upper
            )
        
        # Parameters: [eth_allocation_pct (0-1), eth_leverage, usdc_leverage]
        bounds = [(0.05, 0.95), (1.0, 7.0), (1.0, 7.0)]
        
        result = differential_evolution(
            objective, bounds, maxiter=300, popsize=20, seed=42
        )
        
        if result.success and result.fun < 999.0:
            eth_alloc_pct, eth_lev, usdc_lev = result.x
            usdc_alloc_pct = 1.0 - eth_alloc_pct
            
            eth_capital = total_capital * eth_alloc_pct
            usdc_capital = total_capital * usdc_alloc_pct
            eth_amount = eth_capital / self.current_price
            
            return {
                'success': True,
                'max_boundary_loss': result.fun,
                'eth_allocation_pct': eth_alloc_pct * 100,
                'usdc_allocation_pct': usdc_alloc_pct * 100,
                'eth_capital_usd': eth_capital,
                'usdc_capital_usd': usdc_capital,
                'eth_amount': eth_amount,
                'eth_leverage': eth_lev,
                'usdc_leverage': usdc_lev,
                'total_capital': total_capital,
                'range_lower': range_lower,
                'range_upper': range_upper
            }
        else:
            return {'success': False, 'message': 'Optimization failed'}
    
    def generate_allocation_heatmap(self, total_capital: float, range_lower: float, 
                                  range_upper: float, eth_leverage: float = 3.0, 
                                  usdc_leverage: float = 3.0):
        """
        Generate heatmap showing effect of ETH allocation percentage.
        
        Args:
            eth_leverage, usdc_leverage: Fixed leverage levels for this analysis
        """
        print(f"🔥 Generating Allocation Heatmap (Fixed: ETH {eth_leverage:.1f}x, USDC {usdc_leverage:.1f}x)")
        
        # ETH allocation range (5% to 95%)
        eth_allocations = np.linspace(0.05, 0.95, 20)
        losses = []
        
        for eth_alloc in eth_allocations:
            loss = self.calculate_boundary_loss(
                eth_alloc, eth_leverage, usdc_leverage, 
                total_capital, range_lower, range_upper
            )
            losses.append(loss if loss < 999.0 else np.nan)
        
        # Create plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot as line chart for allocation
        eth_alloc_pct = eth_allocations * 100
        ax.plot(eth_alloc_pct, losses, 'b-', linewidth=3, alpha=0.8)
        ax.fill_between(eth_alloc_pct, losses, alpha=0.3)
        
        # Find minimum
        valid_losses = np.array(losses)
        if not np.all(np.isnan(valid_losses)):
            min_idx = np.nanargmin(valid_losses)
            min_alloc = eth_alloc_pct[min_idx]
            min_loss = valid_losses[min_idx]
            
            ax.plot(min_alloc, min_loss, 'ro', markersize=10, label=f'Optimal: {min_alloc:.0f}% ETH')
            ax.annotate(f'Min Loss: {min_loss:.2f}%\\nETH: {min_alloc:.0f}%', 
                       xy=(min_alloc, min_loss), xytext=(min_alloc+10, min_loss+0.2),
                       arrowprops=dict(arrowstyle='->', color='red'), fontsize=12, fontweight='bold')
        
        ax.set_xlabel('ETH Allocation (%)', fontsize=14)
        ax.set_ylabel('Maximum Boundary Loss (%)', fontsize=14)
        ax.set_title(f'Effect of ETH/USDC Allocation on Boundary Losses\\n(ETH {eth_leverage:.1f}x, USDC {usdc_leverage:.1f}x)', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        return eth_allocations, losses
    
    def generate_leverage_heatmap(self, total_capital: float, range_lower: float, 
                                range_upper: float, eth_allocation_pct: float = 0.5):
        """
        Generate 2D heatmap showing effect of ETH and USDC leverage levels.
        
        Args:
            eth_allocation_pct: Fixed allocation (50% default)
        """
        print(f"🔥 Generating Leverage Heatmap (Fixed: {eth_allocation_pct*100:.0f}% ETH allocation)")
        
        # Leverage ranges
        eth_leverages = np.linspace(1.0, 7.0, 15)
        usdc_leverages = np.linspace(1.0, 7.0, 15)
        
        # Create grid
        loss_matrix = np.zeros((len(usdc_leverages), len(eth_leverages)))
        
        for i, usdc_lev in enumerate(usdc_leverages):
            for j, eth_lev in enumerate(eth_leverages):
                loss = self.calculate_boundary_loss(
                    eth_allocation_pct, eth_lev, usdc_lev,
                    total_capital, range_lower, range_upper
                )
                loss_matrix[i, j] = loss if loss < 999.0 else np.nan
        
        # Create heatmap
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Use seaborn for nice heatmap
        sns.heatmap(loss_matrix, 
                   xticklabels=[f'{x:.1f}x' for x in eth_leverages[::2]], 
                   yticklabels=[f'{y:.1f}x' for y in usdc_leverages[::2]],
                   annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   cbar_kws={'label': 'Max Boundary Loss (%)'}, ax=ax)
        
        ax.set_xlabel('ETH Leverage', fontsize=14)
        ax.set_ylabel('USDC Leverage', fontsize=14) 
        ax.set_title(f'Leverage Heatmap: Effect on Boundary Losses\\n(Fixed: {eth_allocation_pct*100:.0f}% ETH, {(1-eth_allocation_pct)*100:.0f}% USDC)', fontsize=16)
        
        # Find and mark minimum
        min_idx = np.unravel_index(np.nanargmin(loss_matrix), loss_matrix.shape)
        if not np.isnan(loss_matrix[min_idx]):
            min_usdc_lev = usdc_leverages[min_idx[0]]
            min_eth_lev = eth_leverages[min_idx[1]]
            min_loss = loss_matrix[min_idx]
            
            print(f"🎯 Optimal Leverage Combination:")
            print(f"   ETH Leverage: {min_eth_lev:.1f}x")
            print(f"   USDC Leverage: {min_usdc_lev:.1f}x") 
            print(f"   Minimum Loss: {min_loss:.2f}%")
        
        plt.tight_layout()
        plt.show()
        
        return eth_leverages, usdc_leverages, loss_matrix
    
    def generate_combined_heatmap(self, total_capital: float, range_lower: float, range_upper: float):
        """
        Generate combined heatmap: ETH allocation vs ETH leverage (USDC leverage auto-optimized).
        """
        print("🔥 Generating Combined Allocation vs Leverage Heatmap")
        
        # Ranges
        eth_allocations = np.linspace(0.1, 0.9, 12)
        eth_leverages = np.linspace(1.0, 7.0, 12)
        
        loss_matrix = np.zeros((len(eth_leverages), len(eth_allocations)))
        
        for i, eth_lev in enumerate(eth_leverages):
            for j, eth_alloc in enumerate(eth_allocations):
                # Auto-optimize USDC leverage for each combination
                best_loss = 999.0
                for usdc_lev in np.linspace(1.0, 7.0, 10):
                    loss = self.calculate_boundary_loss(
                        eth_alloc, eth_lev, usdc_lev,
                        total_capital, range_lower, range_upper
                    )
                    if loss < best_loss:
                        best_loss = loss
                
                loss_matrix[i, j] = best_loss if best_loss < 999.0 else np.nan
        
        # Create heatmap
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        sns.heatmap(loss_matrix,
                   xticklabels=[f'{x:.0f}%' for x in eth_allocations * 100],
                   yticklabels=[f'{y:.1f}x' for y in eth_leverages],
                   annot=True, fmt='.2f', cmap='RdYlBu_r',
                   cbar_kws={'label': 'Min Boundary Loss (%)'}, ax=ax)
        
        ax.set_xlabel('ETH Allocation (%)', fontsize=14)
        ax.set_ylabel('ETH Leverage', fontsize=14)
        ax.set_title('Combined Heatmap: ETH Allocation vs Leverage\\n(USDC leverage auto-optimized)', fontsize=16)
        
        plt.tight_layout()
        plt.show()
        
        return eth_allocations, eth_leverages, loss_matrix