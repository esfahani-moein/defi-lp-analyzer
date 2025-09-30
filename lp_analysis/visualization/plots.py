"""
Comprehensive visualization for LP analysis.
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from ..lp_calc.types import SimulationResult, LeveragedPosition


class LPVisualizer:
    """
    Plotting functions for LP analysis.
    """
    
    @staticmethod
    def plot_pnl_analysis(
        result: SimulationResult,
        ax: Optional[plt.Axes] = None,
        show: bool = True
    ) -> plt.Axes:
        """Plot PnL across price range."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        prices = result.prices
        pnl = result.metrics['pnl']['pnl_asset1']
        config = result.config
        
        ax.plot(prices, pnl, 'b-', linewidth=2, label='PnL')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(config.price_initial, color='green', linestyle='--', 
                  alpha=0.7, label='Initial Price')
        ax.axvline(config.price_range.lower, color='orange', linestyle='--', 
                  alpha=0.5, label='Range')
        ax.axvline(config.price_range.upper, color='orange', linestyle='--', alpha=0.5)
        
        # Fill profit/loss areas
        ax.fill_between(prices, 0, pnl, where=[p > 0 for p in pnl], 
                       alpha=0.3, color='green')
        ax.fill_between(prices, 0, pnl, where=[p < 0 for p in pnl], 
                       alpha=0.3, color='red')
        
        ax.set_xlabel(f'Price ({config.assets.asset1_symbol}/{config.assets.asset0_symbol})')
        ax.set_ylabel(f'PnL ({config.assets.asset1_symbol})')
        ax.set_title(f'Profit & Loss: {config.assets}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if show:
            plt.tight_layout()
            plt.show()
        
        return ax
    
    @staticmethod
    def plot_asset_composition(
        result: SimulationResult,
        ax: Optional[plt.Axes] = None,
        show: bool = True
    ) -> plt.Axes:
        """Plot asset amounts across price range."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        prices = result.prices
        amount0 = [p.amount0 for p in result.positions]
        amount1 = [p.amount1 for p in result.positions]
        config = result.config
        
        ax_twin = ax.twinx()
        
        ax.plot(prices, amount0, 'purple', linewidth=2, 
               label=config.assets.asset0_symbol)
        ax_twin.plot(prices, amount1, 'orange', linewidth=2, 
                    label=config.assets.asset1_symbol)
        
        ax.axvline(config.price_initial, color='green', linestyle='--', alpha=0.7)
        ax.axvline(config.price_range.lower, color='orange', linestyle='--', alpha=0.5)
        ax.axvline(config.price_range.upper, color='orange', linestyle='--', alpha=0.5)
        
        ax.set_xlabel(f'Price ({config.assets.asset1_symbol}/{config.assets.asset0_symbol})')
        ax.set_ylabel(f'{config.assets.asset0_symbol} Amount', color='purple')
        ax_twin.set_ylabel(f'{config.assets.asset1_symbol} Amount', color='orange')
        ax.set_title(f'Asset Composition: {config.assets}')
        ax.tick_params(axis='y', labelcolor='purple')
        ax_twin.tick_params(axis='y', labelcolor='orange')
        ax.grid(True, alpha=0.3)
        
        if show:
            plt.tight_layout()
            plt.show()
        
        return ax
    
    @staticmethod
    def plot_liquidation_map(
        result: SimulationResult,
        ax: Optional[plt.Axes] = None,
        show: bool = True
    ) -> plt.Axes:
        """Plot liquidation zones for leveraged positions."""
        if not isinstance(result.positions[0], LeveragedPosition):
            raise ValueError("Liquidation map only for leveraged positions")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        prices = result.prices
        pnl_pct = result.metrics['pnl']['pnl_percent']
        liq_flags = result.metrics['risk']['liquidation_flags']
        config = result.config
        
        colors = ['green' if not liq else 'red' for liq in liq_flags]
        ax.scatter(prices, pnl_pct, c=colors, s=10, alpha=0.6)
        
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(config.price_initial, color='green', linestyle='--', alpha=0.7)
        
        ax.set_xlabel(f'Price ({config.assets.asset1_symbol}/{config.assets.asset0_symbol})')
        ax.set_ylabel('Return (%)')
        ax.set_title(f'Liquidation Map: {config.assets}')
        ax.grid(True, alpha=0.3)
        
        if show:
            plt.tight_layout()
            plt.show()
        
        return ax
    
    @staticmethod
    def plot_comprehensive_dashboard(
        result: SimulationResult,
        figsize: tuple = (18, 12)
    ):
        """
        Create comprehensive dashboard with all key plots.
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: PnL
        ax1 = fig.add_subplot(gs[0, :2])
        LPVisualizer.plot_pnl_analysis(result, ax=ax1, show=False)
        
        # Plot 2: PnL %
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(result.prices, result.metrics['pnl']['pnl_percent'], 'purple', linewidth=2)
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel(f'Price')
        ax2.set_ylabel('Return (%)')
        ax2.set_title('Percentage Returns')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Position Value
        ax3 = fig.add_subplot(gs[1, :2])
        if isinstance(result.positions[0], LeveragedPosition):
            pos_val = [p.position_value_asset1 for p in result.positions]
            debt_val = [p.debt_value_asset1 for p in result.positions]
            equity = [p.equity_asset1 for p in result.positions]
            
            ax3.plot(result.prices, pos_val, 'g-', linewidth=2, label='Position Value')
            ax3.plot(result.prices, debt_val, 'r--', linewidth=2, label='Debt Value')
            ax3.plot(result.prices, equity, 'b-', linewidth=2, label='Equity')
            ax3.legend()
        else:
            values = [p.amount0 * p.current_price + p.amount1 for p in result.positions]
            ax3.plot(result.prices, values, 'g-', linewidth=2, label='Position Value')
        
        ax3.set_xlabel('Price')
        ax3.set_ylabel('Value')
        ax3.set_title('Position Breakdown')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Asset Composition
        ax4 = fig.add_subplot(gs[1, 2])
        LPVisualizer.plot_asset_composition(result, ax=ax4, show=False)
        
        # Plot 5: Impermanent Loss
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(result.prices, result.metrics['impermanent_loss'], 'brown', linewidth=2)
        ax5.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax5.set_xlabel('Price')
        ax5.set_ylabel('IL (%)')
        ax5.set_title('Impermanent Loss')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Greeks
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(result.prices, result.metrics['greeks']['delta'], 'blue', linewidth=2)
        ax6.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Price')
        ax6.set_ylabel('Delta')
        ax6.set_title('Position Delta')
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Liquidation or Utilization
        ax7 = fig.add_subplot(gs[2, 2])
        if isinstance(result.positions[0], LeveragedPosition):
            LPVisualizer.plot_liquidation_map(result, ax=ax7, show=False)
        else:
            ax7.text(0.5, 0.5, 'No Leverage\nNo Liquidation Risk', 
                    ha='center', va='center', fontsize=14)
            ax7.set_xlim(0, 1)
            ax7.set_ylim(0, 1)
            ax7.axis('off')
        
        plt.suptitle(f'{result.config.assets} LP Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()