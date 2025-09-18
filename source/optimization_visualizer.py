"""
Visualization utilities for portfolio optimization results.

This module provides plotting functions for optimization results,
strategy comparisons, and sensitivity analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple
import seaborn as sns

class OptimizationVisualizer:
    """
    Visualization tools for portfolio optimization results.
    """
    
    def __init__(self):
        """Initialize the visualizer with default styling."""
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'profit': '#2E8B57',
            'loss': '#DC143C', 
            'neutral': '#4682B4',
            'eth': '#627EEA',
            'usdc': '#26A17B',
            'combined': '#1f1f1f'
        }
    
    def plot_strategy_comparison(self, strategy_results: Dict, total_capital: float):
        """
        Plot comparison of different optimization strategies.
        
        Args:
            strategy_results: Results from compare_strategies()
            total_capital: Total capital used
        """
        # Filter successful strategies
        successful_strategies = {k: v for k, v in strategy_results.items() 
                               if v.get('success', True) and 'summary' in v}
        
        if not successful_strategies:
            print("❌ No successful optimization strategies to compare")
            return
        
        strategy_names = list(successful_strategies.keys())
        n_strategies = len(strategy_names)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Portfolio Optimization Strategy Comparison\nTotal Capital: ${total_capital:,.0f}', 
                     fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        max_losses = [successful_strategies[s]['summary']['max_boundary_loss'] for s in strategy_names]
        lower_losses = [successful_strategies[s]['summary']['lower_loss'] for s in strategy_names]
        upper_losses = [successful_strategies[s]['summary']['upper_loss'] for s in strategy_names]
        eth_allocations = [successful_strategies[s]['summary']['eth_allocation'] for s in strategy_names]
        usdc_allocations = [successful_strategies[s]['summary']['usdc_allocation'] for s in strategy_names]
        eth_leverages = [successful_strategies[s]['summary']['eth_leverage'] for s in strategy_names]
        usdc_leverages = [successful_strategies[s]['summary']['usdc_leverage'] for s in strategy_names]
        
        # Clean strategy names for display
        display_names = [s.replace('_', ' ').title() for s in strategy_names]
        
        # 1. Maximum Boundary Losses
        ax1 = axes[0, 0]
        bars1 = ax1.bar(display_names, max_losses, color=self.colors['loss'], alpha=0.7)
        ax1.set_title('Maximum Boundary Loss (%)', fontweight='bold')
        ax1.set_ylabel('Loss Percentage')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, max_losses):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Boundary Loss Breakdown
        ax2 = axes[0, 1]
        x = np.arange(len(strategy_names))
        width = 0.35
        
        bars2a = ax2.bar(x - width/2, [abs(l) for l in lower_losses], width, 
                        label='Lower Boundary Loss', color=self.colors['eth'], alpha=0.7)
        bars2b = ax2.bar(x + width/2, [abs(l) for l in upper_losses], width,
                        label='Upper Boundary Loss', color=self.colors['usdc'], alpha=0.7)
        
        ax2.set_title('Boundary Loss Breakdown', fontweight='bold')
        ax2.set_ylabel('Absolute Loss (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(display_names, rotation=45)
        ax2.legend()
        
        # 3. Capital Allocation
        ax3 = axes[1, 0]
        bottom = np.zeros(len(strategy_names))
        
        ax3.bar(display_names, eth_allocations, label='ETH Allocation', 
               color=self.colors['eth'], alpha=0.7)
        ax3.bar(display_names, usdc_allocations, bottom=eth_allocations,
               label='USDC Allocation', color=self.colors['usdc'], alpha=0.7)
        
        ax3.set_title('Capital Allocation (%)', fontweight='bold')
        ax3.set_ylabel('Allocation Percentage')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
        ax3.set_ylim(0, 100)
        
        # 4. Leverage Levels
        ax4 = axes[1, 1]
        x = np.arange(len(strategy_names))
        
        bars4a = ax4.bar(x - width/2, eth_leverages, width,
                        label='ETH Leverage', color=self.colors['eth'], alpha=0.7)
        bars4b = ax4.bar(x + width/2, usdc_leverages, width,
                        label='USDC Leverage', color=self.colors['usdc'], alpha=0.7)
        
        ax4.set_title('Leverage Levels', fontweight='bold')
        ax4.set_ylabel('Leverage Multiplier')
        ax4.set_xticks(x)
        ax4.set_xticklabels(display_names, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary table
        self._print_strategy_summary_table(successful_strategies, strategy_names, total_capital)
    
    def plot_optimized_position(self, analyzer, optimization_analysis: Dict, title_suffix: str = ""):
        """
        Plot the optimized position similar to the standard plot but with optimization context.
        
        Args:
            analyzer: LeveragedLPAnalyzer instance
            optimization_analysis: Analysis result from analyze_optimal_position()
            title_suffix: Additional text for plot title
        """
        if 'error' in optimization_analysis:
            print(f"❌ Cannot plot: {optimization_analysis['error']}")
            return
        
        configs = optimization_analysis['configurations']
        performance = optimization_analysis['performance']
        
        # Plot using the existing analyzer method
        analyzer.plot_combined_positions(
            configs['eth_config'], 
            configs['usdc_config']
        )
        
        # Add optimization context
        boundary_losses = optimization_analysis['boundary_losses']
        print(f"\n{'='*60}")
        print(f"OPTIMIZED POSITION PERFORMANCE{' - ' + title_suffix if title_suffix else ''}")
        print(f"{'='*60}")
        print(f"Lower Boundary Loss: {boundary_losses['lower_loss_pct']:+.2f}%")
        print(f"Upper Boundary Loss: {boundary_losses['upper_loss_pct']:+.2f}%")
        print(f"Maximum Boundary Loss: {boundary_losses['max_boundary_loss']:.2f}%")
        print(f"{'='*60}")
    
    def plot_parameter_sensitivity(self, optimizer, base_params: Dict, 
                                 total_capital: float, range_lower: float, range_upper: float):
        """
        Plot parameter sensitivity analysis.
        
        Args:
            optimizer: PortfolioOptimizer instance
            base_params: Base parameter set for sensitivity analysis
            total_capital: Total capital
            range_lower: Lower price boundary  
            range_upper: Upper price boundary
        """
        parameters = {
            'ETH Allocation %': (base_params[0] * 100, np.linspace(10, 90, 20)),
            'USDC Allocation %': (base_params[1] * 100, np.linspace(10, 90, 20)),  
            'ETH Leverage': (base_params[2], np.linspace(1.0, 7.0, 20)),  # Updated to 1x-7x range
            'USDC Leverage': (base_params[3], np.linspace(1.0, 7.0, 20))  # Updated to 1x-7x range
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        fig.suptitle('Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        for i, (param_name, (base_value, test_values)) in enumerate(parameters.items()):
            ax = axes[i]
            objective_scores = []
            
            for test_value in test_values:
                # Create test parameters
                test_params = base_params.copy()
                
                if 'Allocation' in param_name:
                    # Handle allocation constraints
                    if 'ETH' in param_name:
                        test_params[0] = test_value / 100
                        test_params[1] = 1 - test_params[0]  # Adjust USDC allocation
                    else:  # USDC allocation
                        test_params[1] = test_value / 100  
                        test_params[0] = 1 - test_params[1]  # Adjust ETH allocation
                elif 'ETH Leverage' in param_name:
                    test_params[2] = test_value
                else:  # USDC Leverage
                    test_params[3] = test_value
                
                # Calculate objective score
                score = optimizer.objective_function(
                    test_params, total_capital, range_lower, range_upper, 'minimize_max_boundary_loss'
                )
                objective_scores.append(score if score < 1e5 else np.nan)
            
            # Plot sensitivity
            ax.plot(test_values, objective_scores, 'b-', linewidth=2, alpha=0.7)
            ax.axvline(base_value, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Current')
            ax.set_xlabel(param_name)
            ax.set_ylabel('Objective Score (% Loss)')
            ax.set_title(f'Sensitivity: {param_name}')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Find and mark minimum
            valid_scores = np.array(objective_scores)
            valid_scores = valid_scores[~np.isnan(valid_scores)]
            if len(valid_scores) > 0:
                min_idx = np.nanargmin(objective_scores)
                ax.plot(test_values[min_idx], objective_scores[min_idx], 
                       'go', markersize=8, label='Optimal')
        
        plt.tight_layout()
        plt.show()
    
    def plot_capital_allocation_heatmap(self, optimizer, total_capital: float,
                                       range_lower: float, range_upper: float):
        """
        Plot heatmap showing objective scores for different capital allocations.
        
        Args:
            optimizer: PortfolioOptimizer instance
            total_capital: Total capital
            range_lower: Lower price boundary
            range_upper: Upper price boundary
        """
        # Create allocation grid
        eth_allocations = np.linspace(0.1, 0.9, 15)
        usdc_allocations = np.linspace(0.1, 0.9, 15)
        
        # Fixed leverage levels for this analysis (within new 1x-7x range)
        eth_leverage = 3.0
        usdc_leverage = 3.0
        
        # Calculate objective scores for each allocation combination
        score_matrix = np.zeros((len(eth_allocations), len(usdc_allocations)))
        
        print("🔄 Calculating allocation heatmap...")
        for i, eth_alloc in enumerate(eth_allocations):
            for j, usdc_alloc in enumerate(usdc_allocations):
                # Normalize allocations to sum to 1
                total_alloc = eth_alloc + usdc_alloc
                if total_alloc > 0:
                    normalized_eth = eth_alloc / total_alloc
                    normalized_usdc = usdc_alloc / total_alloc
                    
                    params = [normalized_eth, normalized_usdc, eth_leverage, usdc_leverage]
                    score = optimizer.objective_function(
                        params, total_capital, range_lower, range_upper, 'minimize_max_boundary_loss'
                    )
                    score_matrix[i, j] = score if score < 1e5 else np.nan
                else:
                    score_matrix[i, j] = np.nan
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Mask invalid values
        masked_scores = np.ma.masked_invalid(score_matrix)
        
        im = ax.imshow(masked_scores, cmap='RdYlBu_r', aspect='auto', origin='lower')
        
        # Set labels
        ax.set_xlabel('USDC Allocation (%)')
        ax.set_ylabel('ETH Allocation (%)')
        ax.set_title(f'Portfolio Allocation Optimization Heatmap\nFixed Leverages: ETH={eth_leverage}x, USDC={usdc_leverage}x')
        
        # Set ticks
        eth_tick_labels = [f'{x*100:.0f}' for x in eth_allocations[::3]]
        usdc_tick_labels = [f'{x*100:.0f}' for x in usdc_allocations[::3]]
        
        ax.set_xticks(range(0, len(usdc_allocations), 3))
        ax.set_xticklabels(usdc_tick_labels)
        ax.set_yticks(range(0, len(eth_allocations), 3))
        ax.set_yticklabels(eth_tick_labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Objective Score (% Max Loss)')
        
        # Mark optimal point
        min_indices = np.unravel_index(np.nanargmin(masked_scores), masked_scores.shape)
        ax.plot(min_indices[1], min_indices[0], 'w*', markersize=15, 
               markeredgecolor='black', markeredgewidth=2, label='Optimal')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print optimal allocation
        optimal_eth_pct = eth_allocations[min_indices[0]] * 100
        optimal_usdc_pct = usdc_allocations[min_indices[1]] * 100
        optimal_score = masked_scores[min_indices[0], min_indices[1]]
        
        print(f"\n🎯 OPTIMAL ALLOCATION (Fixed Leverages):")
        print(f"   ETH Allocation: {optimal_eth_pct:.1f}%")
        print(f"   USDC Allocation: {optimal_usdc_pct:.1f}%")
        print(f"   Max Boundary Loss: {optimal_score:.2f}%")
    
    def _print_strategy_summary_table(self, successful_strategies: Dict, strategy_names: List[str], total_capital: float):
        """Print a formatted summary table of strategy comparisons."""
        print(f"\n{'='*100}")
        print(f"STRATEGY COMPARISON SUMMARY - Total Capital: ${total_capital:,.0f}")
        print(f"{'='*100}")
        
        header = f"{'Strategy':<25} {'Max Loss':<10} {'Lower':<8} {'Upper':<8} {'ETH %':<8} {'USDC %':<9} {'ETH Lev':<8} {'USDC Lev':<8}"
        print(header)
        print(f"{'-'*100}")
        
        for strategy in strategy_names:
            summary = successful_strategies[strategy]['summary']
            display_name = strategy.replace('_', ' ').title()[:24]
            
            row = (f"{display_name:<25} "
                  f"{summary['max_boundary_loss']:>7.2f}% "
                  f"{summary['lower_loss']:>6.2f}% "
                  f"{summary['upper_loss']:>6.2f}% "
                  f"{summary['eth_allocation']:>6.1f}% "
                  f"{summary['usdc_allocation']:>7.1f}% "
                  f"{summary['eth_leverage']:>6.2f}x "
                  f"{summary['usdc_leverage']:>7.2f}x")
            print(row)
        
        print(f"{'='*100}")
        
        # Find and highlight best strategy
        best_strategy = min(strategy_names, 
                          key=lambda x: successful_strategies[x]['summary']['max_boundary_loss'])
        best_loss = successful_strategies[best_strategy]['summary']['max_boundary_loss']
        
        print(f"🏆 BEST STRATEGY: {best_strategy.replace('_', ' ').title()}")
        print(f"   Minimum Max Boundary Loss: {best_loss:.2f}%")
        print(f"{'='*100}")