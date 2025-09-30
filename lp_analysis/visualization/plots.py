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


class ExposureVisualizer:
    """
    Specialized plotting functions for asset exposure analysis.
    """
    
    @staticmethod
    def plot_net_exposure(
        result: SimulationResult,
        ax: Optional[plt.Axes] = None,
        show: bool = True
    ) -> plt.Axes:
        """
        Plot net exposure to each asset across price range.
        
        Shows how net holdings (LP - debt) change with price.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        if 'exposure_profile' not in result.metrics:
            from ..analysis.lp_analysis import ExposureAnalyzer
            result = ExposureAnalyzer.analyze_exposure(result)
        
        exposure = result.metrics['exposure_profile']
        config = result.config
        
        if not exposure['is_leveraged']:
            ax.text(0.5, 0.5, 'Exposure analysis only for leveraged positions', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return ax
        
        prices = exposure['prices']
        
        # Plot net token exposure
        ax.plot(prices, exposure['asset0_net_tokens'], 'purple', linewidth=2.5, 
               label=f'Net {config.assets.asset0_symbol} Tokens', marker='o', markersize=2)
        
        # Add zero line
        ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Mark initial price
        ax.axvline(config.price_initial, color='green', linestyle='--', 
                  alpha=0.7, label='Initial Price', linewidth=2)
        
        # Mark range boundaries
        ax.axvline(config.price_range.lower, color='orange', linestyle='--', 
                  alpha=0.5, linewidth=1.5)
        ax.axvline(config.price_range.upper, color='orange', linestyle='--', 
                  alpha=0.5, linewidth=1.5, label='LP Range')
        
        # Fill positive/negative exposure areas
        ax.fill_between(prices, 0, exposure['asset0_net_tokens'],
                       where=[exp > 0 for exp in exposure['asset0_net_tokens']],
                       alpha=0.2, color='green', label='Long Exposure')
        ax.fill_between(prices, 0, exposure['asset0_net_tokens'],
                       where=[exp < 0 for exp in exposure['asset0_net_tokens']],
                       alpha=0.2, color='red', label='Short Exposure')
        
        ax.set_xlabel(f'Price ({config.assets.asset1_symbol}/{config.assets.asset0_symbol})', fontsize=12)
        ax.set_ylabel(f'Net {config.assets.asset0_symbol} Exposure (tokens)', fontsize=12)
        ax.set_title(f'Net Asset Exposure: {config.assets} - Debt in {getattr(config, "debt_asset", "N/A")}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if show:
            plt.tight_layout()
            plt.show()
        
        return ax
    
    @staticmethod
    def plot_exposure_percentage(
        result: SimulationResult,
        ax: Optional[plt.Axes] = None,
        show: bool = True
    ) -> plt.Axes:
        """
        Plot exposure as percentage of total portfolio value.
        
        Critical for understanding risk concentration.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        if 'exposure_profile' not in result.metrics:
            from ..analysis.lp_analysis import ExposureAnalyzer
            result = ExposureAnalyzer.analyze_exposure(result)
        
        exposure = result.metrics['exposure_profile']
        config = result.config
        
        if not exposure['is_leveraged']:
            ax.text(0.5, 0.5, 'Exposure % analysis only for leveraged positions', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return ax
        
        prices = exposure['prices']
        
        # Plot exposure percentages
        ax.plot(prices, exposure['asset0_exposure_pct'], 'purple', linewidth=2.5, 
               label=f'{config.assets.asset0_symbol} Exposure', marker='o', markersize=2)
        ax.plot(prices, exposure['asset1_exposure_pct'], 'orange', linewidth=2.5, 
               label=f'{config.assets.asset1_symbol} Exposure', marker='s', markersize=2)
        
        # Add reference lines
        ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        ax.axhline(50, color='gray', linestyle=':', alpha=0.5, label='50% Balanced')
        
        # Mark initial price
        ax.axvline(config.price_initial, color='green', linestyle='--', 
                  alpha=0.7, label='Initial Price', linewidth=2)
        
        # Mark range boundaries
        ax.axvline(config.price_range.lower, color='orange', linestyle='--', alpha=0.5)
        ax.axvline(config.price_range.upper, color='orange', linestyle='--', alpha=0.5)
        
        ax.set_xlabel(f'Price ({config.assets.asset1_symbol}/{config.assets.asset0_symbol})', fontsize=12)
        ax.set_ylabel('Exposure (% of Net Value)', fontsize=12)
        ax.set_title(f'Exposure Distribution: {config.assets} @ {getattr(config, "leverage", 1)}x', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-20, 120])
        
        if show:
            plt.tight_layout()
            plt.show()
        
        return ax
    
    @staticmethod
    def plot_exposure_value(
        result: SimulationResult,
        ax: Optional[plt.Axes] = None,
        show: bool = True
    ) -> plt.Axes:
        """
        Plot net exposure value in asset1 terms.
        
        Shows dollar value of exposure to volatile asset.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        if 'exposure_profile' not in result.metrics:
            from ..analysis.lp_analysis import ExposureAnalyzer
            result = ExposureAnalyzer.analyze_exposure(result)
        
        exposure = result.metrics['exposure_profile']
        config = result.config
        
        if not exposure['is_leveraged']:
            ax.text(0.5, 0.5, 'Value exposure analysis only for leveraged positions', 
                   ha='center', va='center', fontsize=14)
            ax.axis('off')
            return ax
        
        prices = exposure['prices']
        
        # Plot value exposure
        ax.plot(prices, exposure['asset0_net_value'], 'purple', linewidth=2.5, 
               label=f'{config.assets.asset0_symbol} Value', marker='o', markersize=2)
        ax.plot(prices, exposure['asset1_net_value'], 'orange', linewidth=2.5, 
               label=f'{config.assets.asset1_symbol} Value', marker='s', markersize=2)
        ax.plot(prices, exposure['total_net_value'], 'blue', linewidth=2.5, 
               label='Total Net Value', linestyle='--', marker='d', markersize=2)
        
        # Add zero line
        ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Mark initial price
        ax.axvline(config.price_initial, color='green', linestyle='--', 
                  alpha=0.7, label='Initial Price', linewidth=2)
        
        # Mark range boundaries
        ax.axvline(config.price_range.lower, color='orange', linestyle='--', alpha=0.5)
        ax.axvline(config.price_range.upper, color='orange', linestyle='--', alpha=0.5)
        
        ax.set_xlabel(f'Price ({config.assets.asset1_symbol}/{config.assets.asset0_symbol})', fontsize=12)
        ax.set_ylabel(f'Exposure Value ({config.assets.asset1_symbol})', fontsize=12)
        ax.set_title(f'Net Value Exposure: {config.assets}', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        if show:
            plt.tight_layout()
            plt.show()
        
        return ax
    
    @staticmethod
    def plot_exposure_dashboard(
        result: SimulationResult,
        figsize: tuple = (18, 10)
    ):
        """
        Create comprehensive exposure analysis dashboard.
        
        Shows:
        1. Net token exposure
        2. Exposure percentages
        3. Value exposure
        4. Exposure comparison with PnL
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Ensure exposure metrics are calculated
        if 'exposure_profile' not in result.metrics:
            from ..analysis.lp_analysis import ExposureAnalyzer
            result = ExposureAnalyzer.analyze_exposure(result)
        
        exposure = result.metrics['exposure_profile']
        config = result.config
        
        if not exposure['is_leveraged']:
            fig.text(0.5, 0.5, 'Exposure Dashboard Only for Leveraged Positions', 
                    ha='center', va='center', fontsize=16, fontweight='bold')
            plt.show()
            return
        
        # Plot 1: Net Token Exposure
        ax1 = fig.add_subplot(gs[0, 0])
        ExposureVisualizer.plot_net_exposure(result, ax=ax1, show=False)
        
        # Plot 2: Exposure Percentages
        ax2 = fig.add_subplot(gs[0, 1])
        ExposureVisualizer.plot_exposure_percentage(result, ax=ax2, show=False)
        
        # Plot 3: Value Exposure
        ax3 = fig.add_subplot(gs[1, 0])
        ExposureVisualizer.plot_exposure_value(result, ax=ax3, show=False)
        
        # Plot 4: Exposure vs PnL
        ax4 = fig.add_subplot(gs[1, 1])
        prices = exposure['prices']
        pnl = result.metrics['pnl']['pnl_asset1']
        asset0_exp_pct = exposure['asset0_exposure_pct']
        
        ax4_twin = ax4.twinx()
        
        ax4.plot(prices, pnl, 'b-', linewidth=2, label='PnL')
        ax4_twin.plot(prices, asset0_exp_pct, 'purple', linewidth=2, 
                     label=f'{config.assets.asset0_symbol} Exposure %', alpha=0.7)
        
        ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax4.axvline(config.price_initial, color='green', linestyle='--', alpha=0.7)
        
        ax4.set_xlabel(f'Price ({config.assets.asset1_symbol}/{config.assets.asset0_symbol})')
        ax4.set_ylabel(f'PnL ({config.assets.asset1_symbol})', color='blue')
        ax4_twin.set_ylabel(f'{config.assets.asset0_symbol} Exposure %', color='purple')
        ax4.set_title('PnL vs Exposure')
        ax4.tick_params(axis='y', labelcolor='blue')
        ax4_twin.tick_params(axis='y', labelcolor='purple')
        ax4.grid(True, alpha=0.3)
        
        # Add legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.suptitle(f'Exposure Analysis Dashboard: {config.assets} @ {getattr(config, "leverage", 1)}x - '
                    f'Debt: {getattr(config, "debt_asset", "N/A")}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def compare_exposure_strategies(
        results_list: list,
        labels: list = None,
        figsize: tuple = (14, 8)
    ):
        """
        Compare exposure profiles across multiple strategies.
        
        Args:
            results_list: List of SimulationResult objects
            labels: Optional labels for each strategy
            figsize: Figure size
        """
        if labels is None:
            labels = [f"Strategy {i+1}" for i in range(len(results_list))]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))
        
        for i, (result, label) in enumerate(zip(results_list, labels)):
            # Ensure exposure metrics are calculated
            if 'exposure_profile' not in result.metrics:
                from ..analysis.lp_analysis import ExposureAnalyzer
                result = ExposureAnalyzer.analyze_exposure(result)
            
            exposure = result.metrics['exposure_profile']
            
            if not exposure['is_leveraged']:
                continue
            
            prices = exposure['prices']
            
            # Plot 1: Net token exposure comparison
            ax1.plot(prices, exposure['asset0_net_tokens'], 
                    color=colors[i], linewidth=2, label=label, alpha=0.7)
            
            # Plot 2: Exposure percentage comparison
            ax2.plot(prices, exposure['asset0_exposure_pct'], 
                    color=colors[i], linewidth=2, label=label, alpha=0.7)
        
        # Configure ax1
        ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax1.set_xlabel('Price')
        ax1.set_ylabel('Net Token Exposure')
        ax1.set_title('Token Exposure Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Configure ax2
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(50, color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Price')
        ax2.set_ylabel('Exposure (%)')
        ax2.set_title('Exposure % Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Strategy Exposure Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()