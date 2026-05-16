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
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()
    
    @staticmethod
    def plot_combined_positions_dashboard(
        result1: SimulationResult,
        result2: SimulationResult,
        position1_name: str = "Position 1",
        position2_name: str = "Position 2"
    ):
        """
        Plot comprehensive dashboard comparing and combining two LP positions.
        
        This is particularly useful for analyzing delta-neutral strategies where
        one position has USDC debt (long exposure) and another has ETH debt (short exposure).
        
        Args:
            result1: First simulation result (e.g., USDC debt / Long ETH)
            result2: Second simulation result (e.g., ETH debt / Short ETH)
            position1_name: Display name for first position
            position2_name: Display name for second position
        """
        # Validate inputs
        if not hasattr(result1, 'metrics') or not hasattr(result2, 'metrics'):
            raise ValueError("Both results must be analyzed first (call LPAnalyzer.analyze_simulation)")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # Color scheme
        color1 = '#2E86AB'  # Blue for position 1
        color2 = '#A23B72'  # Purple for position 2
        color_combined = '#F18F01'  # Orange for combined
        
        # Get exposure data if available
        has_exposure = ('exposure' in result1.metrics and 'exposure' in result2.metrics)
        
        if has_exposure:
            exposure1 = result1.metrics['exposure']
            exposure2 = result2.metrics['exposure']
        
        # Get position values
        if isinstance(result1.positions[0], LeveragedPosition):
            pos_values1 = np.array([p.equity_asset1 for p in result1.positions])
            pos_values2 = np.array([p.equity_asset1 for p in result2.positions])
        else:
            pos_values1 = np.array([p.amount0 * p.current_price + p.amount1 for p in result1.positions])
            pos_values2 = np.array([p.amount0 * p.current_price + p.amount1 for p in result2.positions])
        
        # 1. Combined Position Value
        ax1 = fig.add_subplot(gs[0, :])
        combined_value = pos_values1 + pos_values2
        
        ax1.plot(result1.prices, pos_values1, 
                label=position1_name, color=color1, linewidth=2, alpha=0.7)
        ax1.plot(result2.prices, pos_values2, 
                label=position2_name, color=color2, linewidth=2, alpha=0.7)
        ax1.plot(result1.prices, combined_value, 
                label='Combined Portfolio', color=color_combined, linewidth=3)
        
        # Add initial investment line
        total_capital = result1.config.capital_asset1 + result2.config.capital_asset1
        ax1.axhline(y=total_capital, color='gray', linestyle='--', 
                   label=f'Initial Capital: ${total_capital:,.0f}', alpha=0.5)
        
        # Mark current price
        current_idx = len(result1.prices) // 2
        ax1.axvline(x=result1.prices[current_idx], color='red', 
                   linestyle=':', alpha=0.5, label='Current Price')
        
        ax1.set_xlabel('ETH Price (USDC)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Position Value (USDC)', fontsize=12, fontweight='bold')
        ax1.set_title('Combined Position Value Across Price Range', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # 2. Combined PnL
        ax2 = fig.add_subplot(gs[1, 0])
        pnl1 = pos_values1 - result1.config.capital_asset1
        pnl2 = pos_values2 - result2.config.capital_asset1
        combined_pnl = pnl1 + pnl2
        
        ax2.plot(result1.prices, pnl1, label=position1_name, color=color1, alpha=0.7)
        ax2.plot(result2.prices, pnl2, label=position2_name, color=color2, alpha=0.7)
        ax2.plot(result1.prices, combined_pnl, label='Combined PnL', 
                color=color_combined, linewidth=2.5)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax2.axvline(x=result1.prices[current_idx], color='red', linestyle=':', alpha=0.3)
        
        # Fill profitable/unprofitable regions
        ax2.fill_between(result1.prices, 0, combined_pnl, 
                         where=(combined_pnl >= 0), alpha=0.2, 
                         color='green', label='Profit Zone')
        ax2.fill_between(result1.prices, 0, combined_pnl, 
                         where=(combined_pnl < 0), alpha=0.2, 
                         color='red', label='Loss Zone')
        
        ax2.set_xlabel('ETH Price (USDC)', fontsize=10)
        ax2.set_ylabel('PnL (USDC)', fontsize=10)
        ax2.set_title('Combined Profit & Loss', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Delta Exposure Comparison
        ax3 = fig.add_subplot(gs[1, 1])
        if has_exposure:
            delta1 = np.array(exposure1['delta_exposure'])
            delta2 = np.array(exposure2['delta_exposure'])
            combined_delta = delta1 + delta2
            
            ax3.plot(result1.prices, delta1, label=f'{position1_name} Δ', 
                    color=color1, alpha=0.7)
            ax3.plot(result2.prices, delta2, label=f'{position2_name} Δ', 
                    color=color2, alpha=0.7)
            ax3.plot(result1.prices, combined_delta, label='Combined Δ (Target: 0)', 
                    color=color_combined, linewidth=2.5)
            ax3.axhline(y=0, color='green', linestyle='--', linewidth=2, 
                       alpha=0.5, label='Delta Neutral')
            ax3.axvline(x=result1.prices[current_idx], color='red', linestyle=':', alpha=0.3)
            
            # Highlight delta-neutral zone
            ax3.fill_between(result1.prices, -0.1, 0.1, alpha=0.1, 
                            color='green', label='Near Neutral (±0.1)')
            
            ax3.set_xlabel('ETH Price (USDC)', fontsize=10)
            ax3.set_ylabel('Delta Exposure', fontsize=10)
            ax3.set_title('Delta Exposure - Hedge Effectiveness', fontsize=12, fontweight='bold')
            ax3.legend(loc='best', fontsize=8)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Exposure Analysis Not Available\nRun ExposureAnalyzer first', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=10)
            ax3.set_title('Delta Exposure', fontsize=12, fontweight='bold')
        
        # 4. Returns Distribution
        ax4 = fig.add_subplot(gs[1, 2])
        returns1 = (pnl1 / result1.config.capital_asset1) * 100
        returns2 = (pnl2 / result2.config.capital_asset1) * 100
        combined_returns = (combined_pnl / total_capital) * 100
        
        ax4.hist(returns1, bins=30, alpha=0.5, color=color1, label=position1_name)
        ax4.hist(returns2, bins=30, alpha=0.5, color=color2, label=position2_name)
        ax4.hist(combined_returns, bins=30, alpha=0.7, color=color_combined, 
                label='Combined', edgecolor='black', linewidth=1.5)
        ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
        
        ax4.set_xlabel('Return (%)', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.set_title('Returns Distribution', fontsize=12, fontweight='bold')
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Asset Holdings - ETH
        ax5 = fig.add_subplot(gs[2, 0])
        asset0_amounts1 = np.array([p.amount0 for p in result1.positions])
        asset0_amounts2 = np.array([p.amount0 for p in result2.positions])
        
        ax5.plot(result1.prices, asset0_amounts1, 
                label=f'{position1_name} ETH', color=color1, linestyle='-', alpha=0.7)
        ax5.plot(result2.prices, asset0_amounts2, 
                label=f'{position2_name} ETH', color=color2, linestyle='-', alpha=0.7)
        combined_eth = asset0_amounts1 + asset0_amounts2
        ax5.plot(result1.prices, combined_eth, 
                label='Combined ETH', color=color_combined, linewidth=2.5)
        ax5.axvline(x=result1.prices[current_idx], color='red', linestyle=':', alpha=0.3)
        
        ax5.set_xlabel('ETH Price (USDC)', fontsize=10)
        ax5.set_ylabel('ETH Amount', fontsize=10)
        ax5.set_title('ETH Holdings', fontsize=12, fontweight='bold')
        ax5.legend(loc='best', fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # 6. Asset Holdings - USDC
        ax6 = fig.add_subplot(gs[2, 1])
        asset1_amounts1 = np.array([p.amount1 for p in result1.positions])
        asset1_amounts2 = np.array([p.amount1 for p in result2.positions])
        
        ax6.plot(result1.prices, asset1_amounts1, 
                label=f'{position1_name} USDC', color=color1, linestyle='-', alpha=0.7)
        ax6.plot(result2.prices, asset1_amounts2, 
                label=f'{position2_name} USDC', color=color2, linestyle='-', alpha=0.7)
        combined_usdc = asset1_amounts1 + asset1_amounts2
        ax6.plot(result1.prices, combined_usdc, 
                label='Combined USDC', color=color_combined, linewidth=2.5)
        ax6.axvline(x=result1.prices[current_idx], color='red', linestyle=':', alpha=0.3)
        
        ax6.set_xlabel('ETH Price (USDC)', fontsize=10)
        ax6.set_ylabel('USDC Amount', fontsize=10)
        ax6.set_title('USDC Holdings', fontsize=12, fontweight='bold')
        ax6.legend(loc='best', fontsize=8)
        ax6.grid(True, alpha=0.3)
        ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # 7. Gamma Exposure
        ax7 = fig.add_subplot(gs[2, 2])
        if has_exposure:
            gamma1 = np.array(exposure1['gamma_exposure'])
            gamma2 = np.array(exposure2['gamma_exposure'])
            combined_gamma = gamma1 + gamma2
            
            ax7.plot(result1.prices, gamma1, label=f'{position1_name} Γ', 
                    color=color1, alpha=0.7)
            ax7.plot(result2.prices, gamma2, label=f'{position2_name} Γ', 
                    color=color2, alpha=0.7)
            ax7.plot(result1.prices, combined_gamma, label='Combined Γ', 
                    color=color_combined, linewidth=2.5)
            ax7.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax7.axvline(x=result1.prices[current_idx], color='red', linestyle=':', alpha=0.3)
            
            ax7.set_xlabel('ETH Price (USDC)', fontsize=10)
            ax7.set_ylabel('Gamma Exposure', fontsize=10)
            ax7.set_title('Gamma Exposure - Convexity', fontsize=12, fontweight='bold')
            ax7.legend(loc='best', fontsize=8)
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'Exposure Analysis Not Available', 
                    ha='center', va='center', transform=ax7.transAxes, fontsize=10)
            ax7.set_title('Gamma Exposure', fontsize=12, fontweight='bold')
        
        # 8. Key Metrics Table
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        # Calculate metrics
        current_val1 = pos_values1[current_idx]
        current_val2 = pos_values2[current_idx]
        current_combined = combined_value[current_idx]
        
        max_val1 = np.max(pos_values1)
        max_val2 = np.max(pos_values2)
        max_combined = np.max(combined_value)
        
        min_val1 = np.min(pos_values1)
        min_val2 = np.min(pos_values2)
        min_combined = np.min(combined_value)
        
        # Get leverage info
        lev1 = getattr(result1.config, 'leverage', 1.0)
        lev2 = getattr(result2.config, 'leverage', 1.0)
        
        # Get debt asset info
        debt1 = getattr(result1.config, 'debt_asset', None)
        debt2 = getattr(result2.config, 'debt_asset', None)
        debt1_str = debt1.value if debt1 else 'None'
        debt2_str = debt2.value if debt2 else 'None'
        
        if has_exposure:
            avg_delta1 = np.mean(np.abs(delta1))
            avg_delta2 = np.mean(np.abs(delta2))
            avg_delta_combined = np.mean(np.abs(combined_delta))
            max_delta1 = np.max(np.abs(delta1))
            max_delta2 = np.max(np.abs(delta2))
            max_delta_combined = np.max(np.abs(combined_delta))
        else:
            avg_delta1 = avg_delta2 = avg_delta_combined = 0
            max_delta1 = max_delta2 = max_delta_combined = 0
        
        # Create table data
        table_data = [
            ['Metric', position1_name, position2_name, 'Combined'],
            ['Capital Invested', f'${result1.config.capital_asset1:,.0f}', 
             f'${result2.config.capital_asset1:,.0f}', f'${total_capital:,.0f}'],
            ['Leverage', f'{lev1:.1f}x', f'{lev2:.1f}x', 
             f'{(result1.config.capital_asset1*lev1 + result2.config.capital_asset1*lev2)/total_capital:.1f}x avg'],
            ['Debt Asset', debt1_str, debt2_str, 'Mixed'],
            ['Current Value', f'${current_val1:,.0f}', 
             f'${current_val2:,.0f}', f'${current_combined:,.0f}'],
            ['Current PnL', f'${pnl1[current_idx]:,.0f}', 
             f'${pnl2[current_idx]:,.0f}', f'${combined_pnl[current_idx]:,.0f}'],
            ['Current Return', f'{returns1[current_idx]:.2f}%', 
             f'{returns2[current_idx]:.2f}%', f'{combined_returns[current_idx]:.2f}%'],
            ['Max Value', f'${max_val1:,.0f}', 
             f'${max_val2:,.0f}', f'${max_combined:,.0f}'],
            ['Min Value', f'${min_val1:,.0f}', 
             f'${min_val2:,.0f}', f'${min_combined:,.0f}'],
            ['Avg |Delta|', f'{avg_delta1:.4f}', 
             f'{avg_delta2:.4f}', f'{avg_delta_combined:.4f}'],
            ['Max |Delta|', f'{max_delta1:.4f}', 
             f'{max_delta2:.4f}', f'{max_delta_combined:.4f}'],
        ]
        
        # Create table
        table = ax8.table(cellText=table_data, cellLoc='center', loc='center',
                         colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor('#E8E8E8')
            table[(0, i)].set_text_props(weight='bold')
        
        # Color code PnL rows
        for i in [5, 6]:  # PnL and Return rows
            for j in range(1, 4):
                cell = table[(i, j)]
                value_str = table_data[i][j].replace('$', '').replace(',', '').replace('%', '')
                try:
                    value = float(value_str)
                    if value > 0:
                        cell.set_facecolor('#D4EDDA')  # Light green
                    elif value < 0:
                        cell.set_facecolor('#F8D7DA')  # Light red
                except:
                    pass
        
        # Highlight delta metrics
        if has_exposure:
            for i in [9, 10]:  # Delta rows
                cell = table[(i, 3)]
                if avg_delta_combined > 0.1:
                    cell.set_facecolor('#FFF3CD')  # Light yellow
                else:
                    cell.set_facecolor('#D4EDDA')  # Light green
        
        # Add title
        fig.suptitle(f'Combined LP Strategy Analysis: {position1_name} + {position2_name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        plt.show()
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"COMBINED POSITION SUMMARY")
        print(f"{'='*80}")
        print(f"\nPositions:")
        print(f"  • {position1_name}: ${result1.config.capital_asset1:,.0f} @ {lev1:.1f}x ({debt1_str} debt)")
        print(f"  • {position2_name}: ${result2.config.capital_asset1:,.0f} @ {lev2:.1f}x ({debt2_str} debt)")
        print(f"\nCombined Portfolio:")
        print(f"  Total Capital: ${total_capital:,.0f}")
        print(f"  Current Value: ${current_combined:,.0f}")
        print(f"  Current PnL: ${combined_pnl[current_idx]:,.0f} ({combined_returns[current_idx]:.2f}%)")
        
        if has_exposure:
            print(f"\nDelta Neutrality:")
            print(f"  Average |Δ|: {avg_delta_combined:.4f}")
            print(f"  Maximum |Δ|: {max_delta_combined:.4f}")
            if avg_delta_combined < 0.05:
                assessment = '✓ Well Hedged'
            elif avg_delta_combined < 0.2:
                assessment = '⚠ Needs Optimization'
            else:
                assessment = '✗ Poorly Hedged'
            print(f"  Assessment: {assessment}")
        
        print(f"{'='*80}\n")


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