"""
Core data types for LP analysis.
All shared types, enums, and dataclasses.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class DebtAsset(Enum):
    """Which asset is borrowed in leveraged positions."""
    ASSET0 = "ASSET0"  # Base asset (e.g., ETH, WBTC)
    ASSET1 = "ASSET1"  # Quote asset (e.g., USDC, ETH)
    NONE = "NONE"      # No leverage


@dataclass
class AssetPair:
    """Asset pair configuration."""
    asset0_symbol: str  # Base (e.g., "ETH")
    asset1_symbol: str  # Quote (e.g., "USDC")
    
    def __str__(self):
        return f"{self.asset0_symbol}/{self.asset1_symbol}"


@dataclass
class PriceRange:
    """Price range for concentrated liquidity."""
    lower: float
    upper: float
    
    def __post_init__(self):
        assert 0 < self.lower < self.upper, "Invalid price range"
    
    def width_percent(self, current_price: float) -> float:
        """Range width as percentage of current price."""
        return ((self.upper - self.lower) / current_price) * 100
    
    def contains(self, price: float) -> bool:
        """Check if price is within range."""
        return self.lower <= price <= self.upper


@dataclass
class LPConfig:
    """
    Configuration for an LP position (non-leveraged).
    
    Convention: Price = asset1 per 1 asset0
    """
    capital_asset1: float       # Initial capital in quote asset
    price_initial: float        # Initial price
    price_range: PriceRange     # Concentrated liquidity range
    assets: AssetPair          # Asset pair info
    fee_tier: float = 0.3      # Pool fee tier (0.05, 0.3, 1.0%)
    
    def __post_init__(self):
        assert self.capital_asset1 > 0, "Capital must be positive"
        assert self.price_range.contains(self.price_initial), \
            "Initial price must be in range"


@dataclass
class LeveragedLPConfig(LPConfig):
    """
    Configuration for leveraged LP position with debt.
    """
    leverage: float = 1.0       # Leverage multiplier (1.0 = no leverage)
    debt_asset: DebtAsset = DebtAsset.NONE  # Which asset to borrow
    borrow_apr: float = 0.0    # Borrowing cost APR
    max_ltv: float = 0.8       # Max loan-to-value ratio (protocol specific)
    liquidation_threshold: float = 0.85  # Liquidation LTV
    
    def __post_init__(self):
        super().__post_init__()
        assert self.leverage >= 1.0, "Leverage must be >= 1.0"
        assert 0 <= self.borrow_apr <= 100, "APR must be 0-100%"
        assert 0 < self.max_ltv < 1, "LTV must be between 0 and 1"
        assert self.max_ltv < self.liquidation_threshold <= 1
        
        # Calculate actual leverage based on protocol LTV
        # self.effective_leverage = min(self.leverage, 1 + self.max_ltv)
        # if self.effective_leverage < self.leverage:
        #     import warnings
        #     warnings.warn(
        #         f"Requested leverage {self.leverage}x exceeds protocol max "
        #         f"{self.effective_leverage:.2f}x (LTV={self.max_ltv})"
        #     )


@dataclass
class LPPosition:
    """
    Actual LP position state at a given moment.
    """
    amount0: float             # Amount of asset0
    amount1: float             # Amount of asset1
    liquidity: float           # Liquidity constant L
    current_price: float       # Current market price
    config: LPConfig          # Original configuration


@dataclass
class LeveragedPosition(LPPosition):
    """
    Leveraged LP position with debt tracking.
    """
    debt_amount0: float        # Borrowed asset0
    debt_amount1: float        # Borrowed asset1
    config: LeveragedLPConfig  # Leveraged config
    
    @property
    def debt_value_asset1(self) -> float:
        """Total debt value in asset1 terms."""
        return self.debt_amount0 * self.current_price + self.debt_amount1
    
    @property
    def position_value_asset1(self) -> float:
        """Total position value in asset1 terms."""
        return self.amount0 * self.current_price + self.amount1
    
    @property
    def equity_asset1(self) -> float:
        """Net equity = position - debt."""
        return self.position_value_asset1 - self.debt_value_asset1
    
    @property
    def ltv(self) -> float:
        """Current loan-to-value ratio."""
        pv = self.position_value_asset1
        return self.debt_value_asset1 / pv if pv > 0 else float('inf')


@dataclass
class SimulationResult:
    """
    Results from price simulation.
    """
    prices: List[float]
    positions: List[LPPosition]  # Position at each price
    metrics: Dict               # Calculated metrics
    config: LPConfig           # Original config