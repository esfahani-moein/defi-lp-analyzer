"""
Leveraged Liquidity Position Analysis Module - CORRECTED VERSION

This module provides mathematically accurate analysis tools for leveraged concentrated 
liquidity positions in DeFi protocols like Uniswap v3.
"""

from typing import Dict, Tuple, Callable, Optional, List
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


@dataclass
class PositionConfig:
    """Configuration parameters for a leveraged LP position."""
    current_price: float
    range_lower: float
    range_upper: float
    initial_eth: float
    initial_usdc: float
    leverage: float
    borrow_type: str
    apr_percent: float


@dataclass
class AnalysisResults:
    """Results from leveraged LP position analysis."""
    initial_equity_usd: float
    total_position_usd: float
    borrow_type: str
    debt_eth: float
    debt_usdc: float
    price_lower: float
    price_upper: float
    current_price: float
    liquidity: float
    get_position_value: Callable[[float], Tuple[float, float, float]]


class LeveragedLPAnalyzer:
    """
    Mathematically accurate analyzer for leveraged concentrated liquidity positions.
    
    This class provides correct Uniswap v3 math implementation with proper 
    impermanent loss calculations.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self._results_cache: Dict[str, AnalysisResults] = {}
    
    def calculate_position_analysis(self, config: PositionConfig) -> Optional[AnalysisResults]:
        """
        Calculate the PnL profile of a leveraged concentrated liquidity position.
        
        Uses correct Uniswap v3 mathematics for accurate results.
        
        Args:
            config: Position configuration parameters
            
        Returns:
            AnalysisResults object containing all calculated values and functions
        """
        p_c, p_a, p_b = config.current_price, config.range_lower, config.range_upper
        
        # Validate inputs
        initial_equity_usd = config.initial_eth * p_c + config.initial_usdc
        if initial_equity_usd == 0:
            raise ValueError("Initial capital cannot be zero.")
        
        if config.borrow_type not in ['ETH', 'USDC']:
            raise ValueError("Borrow type must be 'ETH' or 'USDC'")
            
        # Check if price is within range
        if p_c <= p_a or p_c >= p_b:
            print(f"Warning: Current price {p_c} is outside range [{p_a}, {p_b}]")
            return None
        
        # Calculate leveraged position
        total_position_usd = initial_equity_usd * config.leverage
        borrowed_usd = total_position_usd - initial_equity_usd
        
        # Calculate debt amounts
        debt_eth = borrowed_usd / p_c if config.borrow_type == 'ETH' else 0
        debt_usdc = borrowed_usd if config.borrow_type == 'USDC' else 0
        
        # Calculate Uniswap v3 liquidity using correct formula
        sp_c, sp_a, sp_b = np.sqrt(p_c), np.sqrt(p_a), np.sqrt(p_b)
        
        # For total USD value V in concentrated liquidity:
        # V = L * [2*sqrt(P) - P/sqrt(Pb) - sqrt(Pa)]
        denominator = 2 * sp_c - p_c/sp_b - sp_a
        liquidity = total_position_usd / denominator
        
        def position_value_calculator(p_final: float) -> Tuple[float, float, float]:
            """Calculate position value at given price with correct math."""
            return self._calculate_position_value_correct(
                p_final, p_a, p_b, liquidity,
                config.initial_eth, config.initial_usdc, p_c,
                debt_eth, debt_usdc, config.borrow_type
            )
        
        return AnalysisResults(
            initial_equity_usd=initial_equity_usd,
            total_position_usd=total_position_usd,
            borrow_type=config.borrow_type,
            debt_eth=debt_eth,
            debt_usdc=debt_usdc,
            price_lower=p_a,
            price_upper=p_b,
            current_price=p_c,
            liquidity=liquidity,
            get_position_value=position_value_calculator
        )
    
    def _calculate_position_value_correct(self, p_final: float, p_a: float, p_b: float,
                                        L: float, original_eth: float, original_usdc: float,
                                        p_initial: float, debt_eth: float, debt_usdc: float,
                                        borrow_type: str) -> Tuple[float, float, float]:
        """
        Calculate position value with mathematically correct approach.
        
        This method properly handles the Uniswap v3 math and impermanent loss calculation.
        """
        sp_final = np.sqrt(p_final)
        sp_a = np.sqrt(p_a)
        sp_b = np.sqrt(p_b)
        
        # Calculate LP token amounts at the final price using Uniswap v3 formulas
        if sp_final <= sp_a:  # Price below range - all ETH
            amount_eth = L * (1/sp_a - 1/sp_b)
            amount_usdc = 0
        elif sp_final >= sp_b:  # Price above range - all USDC
            amount_eth = 0
            amount_usdc = L * (sp_b - sp_a)
        else:  # Price within range
            amount_eth = L * (1/sp_final - 1/sp_b)
            amount_usdc = L * (sp_final - sp_a)
        
        # Total LP position value
        lp_value = amount_eth * p_final + amount_usdc
        
        # Calculate final equity (LP value minus debt)
        debt_value_final = debt_usdc if borrow_type == 'USDC' else debt_eth * p_final
        final_equity = lp_value - debt_value_final
        
        # For impermanent loss calculation:
        # We compare LP strategy vs leveraged HODL strategy
        # Leveraged HODL = (original_assets + borrowed_assets) * price_change - debt
        if borrow_type == 'ETH':
            leveraged_hodl_eth = original_eth + debt_eth
            leveraged_hodl_usdc = original_usdc
        else:
            leveraged_hodl_eth = original_eth
            leveraged_hodl_usdc = original_usdc + debt_usdc
            
        leveraged_hodl_value = leveraged_hodl_eth * p_final + leveraged_hodl_usdc
        leveraged_hodl_equity = leveraged_hodl_value - debt_value_final
        
        # Impermanent Loss = LP equity - HODL equity
        impermanent_loss = final_equity - leveraged_hodl_equity
        
        return lp_value, final_equity, impermanent_loss
    
    def print_analysis_summary(self, config: PositionConfig, results: AnalysisResults) -> None:
        """Print comprehensive analysis summary with correct values."""
        print("--- Leveraged LP Position Analysis ---")
        print(f"Initial Equity: ${results.initial_equity_usd:,.2f} "
              f"({config.initial_eth} ETH + ${config.initial_usdc} USDC)")
        print(f"Leverage: {config.leverage}x")
        print(f"Total Position Value: ${results.total_position_usd:,.2f}")
        print("-" * 50)
        
        # Calculate and show LP composition at current price
        sp_c = np.sqrt(config.current_price)
        sp_a = np.sqrt(config.range_lower)  
        sp_b = np.sqrt(config.range_upper)
        
        current_eth_in_lp = results.liquidity * (1/sp_c - 1/sp_b)
        current_usdc_in_lp = results.liquidity * (sp_c - sp_a)
        
        print(f"LP Position at Current Price (${config.current_price:,.0f}):")
        print(f"  - ETH: {current_eth_in_lp:.6f} (${current_eth_in_lp * config.current_price:,.2f})")
        print(f"  - USDC: ${current_usdc_in_lp:,.2f}")
        print(f"  - Total: ${current_eth_in_lp * config.current_price + current_usdc_in_lp:,.2f}")
        print(f"  - Liquidity (L): {results.liquidity:,.2f}")
        print("-" * 50)
        
        print(f"Debt ({results.borrow_type}):")
        if results.borrow_type == 'ETH':
            print(f"  - Amount: {results.debt_eth:.6f} ETH")
            print(f"  - Value: ${results.debt_eth * config.current_price:,.2f}")
        else:
            print(f"  - Amount: ${results.debt_usdc:,.2f} USDC")
        
        print(f"LP Range: ${results.price_lower:,.0f} - ${results.price_upper:,.0f}")
        print("-" * 50)
        
        # Verify current position makes sense
        lp_val, final_eq, il = results.get_position_value(config.current_price)
        print(f"Position Verification at Current Price:")
        print(f"  - LP Value: ${lp_val:,.2f}")
        print(f"  - Final Equity: ${final_eq:,.2f}")
        print(f"  - Expected Equity: ${results.initial_equity_usd:,.2f}")
        if abs(final_eq - results.initial_equity_usd) > 10:
            print(f"  - ⚠️  WARNING: Large discrepancy detected!")
        else:
            print(f"  - ✅ Equity matches expectation")
        print("-" * 50)
        
        self._print_scenario_analysis(config, results)
        self._print_apr_earnings(config, results)
    
    def _print_scenario_analysis(self, config: PositionConfig, results: AnalysisResults) -> None:
        """Print PnL analysis for key price scenarios."""
        print("\n--- PnL at Key Price Points ---")
        
        key_prices = {
            "Lower Band": results.price_lower,
            "Current Price": results.current_price,
            "Upper Band": results.price_upper,
            "-5% Move": results.current_price * 0.95,
            "-10% Move": results.current_price * 0.90,
            "+5% Move": results.current_price * 1.05,
            "+10% Move": results.current_price * 1.10,
        }
        
        for name, price in key_prices.items():
            lp_val, final_eq, il = results.get_position_value(price)
            pnl = final_eq - results.initial_equity_usd
            pnl_percent = (pnl / results.initial_equity_usd) * 100
            
            print(f"\n{name} (ETH = ${price:,.0f}):")
            print(f"  LP Value: ${lp_val:,.2f}")
            print(f"  Final Equity: ${final_eq:,.2f}")
            print(f"  P&L: ${pnl:,.2f} ({pnl_percent:+.2f}%)")
            print(f"  Impermanent Loss: ${il:,.2f}")
    
    def _print_apr_earnings(self, config: PositionConfig, results: AnalysisResults) -> None:
        """Print estimated APR earnings."""
        daily_apr_earnings = results.total_position_usd * (config.apr_percent / 100) / 365
        print(f"\n" + "-" * 50)
        print(f"Additional Yield:")
        print(f"  - APR: {config.apr_percent}%")
        print(f"  - Daily Earnings: ${daily_apr_earnings:,.2f}")
        print(f"  - Monthly Earnings: ${daily_apr_earnings * 30:,.2f}")
        print("-" * 50)
    
    def plot_individual_position(self, config: PositionConfig, results: AnalysisResults) -> None:
        """Plot PnL profile for individual position."""
        price_range = np.linspace(config.current_price * 0.7, config.current_price * 1.3, 500)
        final_equities = [results.get_position_value(p)[1] for p in price_range]
        
        # Calculate reference lines
        lp_val_current, final_eq_current, _ = results.get_position_value(config.current_price)
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Main equity curve
        ax.plot(price_range, final_equities, label='Final Equity', 
                color='royalblue', linewidth=2.5)
        
        # Add daily fee breakeven line
        breakeven_line = self._calculate_daily_breakeven_line(config, results, price_range)
        daily_fee_income = results.total_position_usd * (config.apr_percent / 100.0) / 365.0
        ax.plot(price_range, breakeven_line, label=f'Daily Fee Breakeven (+${daily_fee_income:,.0f}/day)',
                color='purple', linestyle='-.', linewidth=2, alpha=0.8)
        
        # Reference lines
        ax.axhline(y=results.initial_equity_usd, color='red', linestyle='--', linewidth=2,
                   label=f'Initial Equity (${results.initial_equity_usd:,.0f})')
        ax.axhline(y=final_eq_current, color='orange', linestyle=':', linewidth=2,
                   label=f'Current Equity (${final_eq_current:,.0f})')
        ax.axvline(x=config.current_price, color='gray', linestyle=':', alpha=0.7,
                   label=f'Current Price (${config.current_price:,.0f})')
        
        # LP Range
        ax.axvspan(results.price_lower, results.price_upper, alpha=0.15, 
                   color='green', label=f'LP Range (${results.price_lower:,.0f}-${results.price_upper:,.0f})')
        
        # Profit/loss regions
        self._add_profit_loss_regions(ax, price_range, final_equities, results.initial_equity_usd)
        
        # Formatting
        ax.set_title(f"Leveraged LP Position P&L (Borrowing {results.borrow_type})", fontsize=16, fontweight='bold')
        ax.set_xlabel("ETH Price (USDC)", fontsize=12)
        ax.set_ylabel("Position Equity (USDC)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Format axes
        formatter = mticker.FormatStrFormatter('$%1.0f')
        ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_formatter(formatter)
        
        plt.tight_layout()
        plt.show()
    
    def plot_combined_positions(self, config1: PositionConfig, config2: PositionConfig) -> None:
        """Plot combined PnL profile of two positions."""
        results1 = self.calculate_position_analysis(config1)
        results2 = self.calculate_position_analysis(config2)
        
        if not results1 or not results2:
            print("Error: Could not calculate one or both positions")
            return
        
        price_range = np.linspace(config1.current_price * 0.7, config1.current_price * 1.3, 500)
        
        # Calculate individual and combined equities
        final_equities_1 = [results1.get_position_value(p)[1] for p in price_range]
        final_equities_2 = [results2.get_position_value(p)[1] for p in price_range]
        combined_equities = [eq1 + eq2 for eq1, eq2 in zip(final_equities_1, final_equities_2)]
        
        # Combined metrics
        combined_initial_equity = results1.initial_equity_usd + results2.initial_equity_usd
        _, final_eq_current_1, _ = results1.get_position_value(config1.current_price)
        _, final_eq_current_2, _ = results2.get_position_value(config2.current_price)
        combined_current_equity = final_eq_current_1 + final_eq_current_2
        
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Individual positions (background)
        ax.plot(price_range, final_equities_1, label=f'{config1.borrow_type} Position',
                color='lightblue', linewidth=1.5, alpha=0.7, linestyle='--')
        ax.plot(price_range, final_equities_2, label=f'{config2.borrow_type} Position',
                color='lightcoral', linewidth=1.5, alpha=0.7, linestyle='--')
        
        # Combined position (main focus)
        ax.plot(price_range, combined_equities, label='Combined Portfolio',
                color='darkblue', linewidth=3)
        
        # Add combined daily fee breakeven line
        combined_daily_fees = (results1.total_position_usd * (config1.apr_percent / 100.0) / 365.0 +
                              results2.total_position_usd * (config2.apr_percent / 100.0) / 365.0)
        combined_breakeven_equity = combined_initial_equity + combined_daily_fees
        breakeven_line = np.full_like(price_range, combined_breakeven_equity)
        ax.plot(price_range, breakeven_line, label=f'Combined Daily Fee Breakeven (+${combined_daily_fees:,.0f}/day)',
                color='purple', linestyle='-.', linewidth=2, alpha=0.8)
        
        # Reference lines
        ax.axhline(y=combined_initial_equity, color='red', linestyle='-', linewidth=2,
                   label=f'Combined Initial Equity (${combined_initial_equity:,.0f})')
        ax.axhline(y=combined_current_equity, color='orange', linestyle=':', linewidth=2,
                   label=f'Combined Current Equity (${combined_current_equity:,.0f})')
        ax.axvline(x=config1.current_price, color='gray', linestyle=':', alpha=0.7)
        
        # LP Range
        ax.axvspan(results1.price_lower, results1.price_upper, alpha=0.15,
                   color='green', label=f'LP Range')
        
        # Profit/loss regions for combined portfolio
        self._add_profit_loss_regions(ax, price_range, combined_equities, combined_initial_equity)
        
        # Formatting
        ax.set_title("Combined Leveraged LP Portfolio P&L", fontsize=16, fontweight='bold')
        ax.set_xlabel("ETH Price (USDC)", fontsize=12)
        ax.set_ylabel("Portfolio Equity (USDC)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        
        formatter = mticker.FormatStrFormatter('$%1.0f')
        ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_formatter(formatter)
        
        plt.tight_layout()
        plt.show()
        
        # Print combined analysis
        self._print_combined_analysis(config1, config2, results1, results2)
    
    def _calculate_daily_breakeven_line(self, config: PositionConfig, results: AnalysisResults, 
                                       price_range: np.ndarray) -> np.ndarray:
        """Calculate breakeven line where daily fee income compensates for IL."""
        daily_fee_income = results.total_position_usd * (config.apr_percent / 100.0) / 365.0
        
        # Calculate breakeven equity = initial equity + daily fees
        breakeven_equity = results.initial_equity_usd + daily_fee_income
        
        return np.full_like(price_range, breakeven_equity)
    
    def _add_profit_loss_regions(self, ax, price_range: np.ndarray, 
                                final_equities: List[float], initial_equity: float) -> None:
        """Add colored regions for profit and loss areas."""
        final_equities_array = np.array(final_equities)
        ax.fill_between(price_range, final_equities_array, initial_equity,
                       where=final_equities_array > initial_equity,
                       facecolor='green', alpha=0.2, interpolate=True, label='_nolegend_')
        ax.fill_between(price_range, final_equities_array, initial_equity,
                       where=final_equities_array < initial_equity,
                       facecolor='red', alpha=0.2, interpolate=True, label='_nolegend_')
    
    def _print_combined_analysis(self, config1: PositionConfig, config2: PositionConfig,
                               results1: AnalysisResults, results2: AnalysisResults) -> None:
        """Print combined analysis summary."""
        combined_initial = results1.initial_equity_usd + results2.initial_equity_usd
        combined_total_position = results1.total_position_usd + results2.total_position_usd
        
        print("=" * 60)
        print("COMBINED PORTFOLIO ANALYSIS")
        print("=" * 60)
        print(f"Total Initial Equity: ${combined_initial:,.2f}")
        print(f"Total Position Value: ${combined_total_position:,.2f}")
        print(f"Combined Leverage: {combined_total_position/combined_initial:.1f}x")
        print("-" * 60)
        print(f"Position 1: {config1.initial_eth} ETH → {config1.borrow_type} leverage")
        print(f"Position 2: ${config2.initial_usdc} USDC → {config2.borrow_type} leverage")
        print("-" * 60)
        
        # Key scenarios for combined position
        key_prices = [
            results1.price_lower, results1.current_price, results1.price_upper,
            results1.current_price * 0.9, results1.current_price * 1.1
        ]
        names = ["Lower Band", "Current", "Upper Band", "-10%", "+10%"]
        
        print("Combined Portfolio Scenarios:")
        for name, price in zip(names, key_prices):
            _, eq1, il1 = results1.get_position_value(price)
            _, eq2, il2 = results2.get_position_value(price)
            combined_eq = eq1 + eq2
            combined_pnl = combined_eq - combined_initial
            combined_pnl_pct = (combined_pnl / combined_initial) * 100
            
            print(f"  {name:12s} (${price:4.0f}): ${combined_eq:8,.0f} ({combined_pnl_pct:+6.2f}%)")
        
        print("=" * 60)


def create_position_config(current_price: float, range_lower: float, range_upper: float,
                          initial_eth: float, initial_usdc: float, leverage: float,
                          borrow_type: str, apr_percent: float) -> PositionConfig:
    """Factory function to create PositionConfig with validation."""
    if borrow_type not in ['ETH', 'USDC']:
        raise ValueError("borrow_type must be 'ETH' or 'USDC'")
    
    if leverage <= 1:
        raise ValueError("leverage must be greater than 1")
        
    if current_price <= 0 or range_lower <= 0 or range_upper <= 0:
        raise ValueError("All prices must be positive")
        
    if range_lower >= range_upper:
        raise ValueError("range_lower must be less than range_upper")
        
    if current_price <= range_lower or current_price >= range_upper:
        raise ValueError(f"current_price ({current_price}) must be within range [{range_lower}, {range_upper}]")
    
    return PositionConfig(
        current_price=current_price,
        range_lower=range_lower,
        range_upper=range_upper,
        initial_eth=initial_eth,
        initial_usdc=initial_usdc,
        leverage=leverage,
        borrow_type=borrow_type,
        apr_percent=apr_percent
    )