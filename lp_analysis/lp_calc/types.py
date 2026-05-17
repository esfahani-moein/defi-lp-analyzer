"""
Core data types for LP analysis.
All shared types, enums, and dataclasses.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class DebtAsset(Enum):
    """Which asset is borrowed in leveraged positions."""
    ASSET0 = "ASSET0"  # Base asset (e.g., ETH, WBTC)
    ASSET1 = "ASSET1"  # Quote asset (e.g., USDC, ETH)
    EXTERNAL = "EXTERNAL"
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
    debt_asset_symbol: Optional[str] = None
    debt_price_initial_asset1: Optional[float] = None
    borrow_apr: float = 0.0    # Borrowing cost APR
    max_ltv: float = 0.8       # Max loan-to-value ratio (protocol specific)
    liquidation_threshold: float = 0.85  # Liquidation LTV
    collateral_factor_asset0: float = 0.80
    collateral_factor_asset1: float = 0.90
    liquidation_factor_asset0: float = 0.85
    liquidation_factor_asset1: float = 0.95
    lp_protocol_risk_factor: float = 1.0
    minimum_margin_asset1: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        assert self.leverage >= 1.0, "Leverage must be >= 1.0"
        assert 0 <= self.borrow_apr <= 1000, "APR must be 0-1000%"
        assert 0 < self.max_ltv < 1, "LTV must be between 0 and 1"
        assert self.max_ltv < self.liquidation_threshold <= 1
        assert 0 <= self.collateral_factor_asset0 <= self.liquidation_factor_asset0 <= 1
        assert 0 <= self.collateral_factor_asset1 <= self.liquidation_factor_asset1 <= 1
        assert 0 < self.lp_protocol_risk_factor <= 1
        assert self.minimum_margin_asset1 >= 0
        if self.leverage > 1.0 and self.debt_asset == DebtAsset.NONE:
            raise ValueError("Leveraged positions must specify a debt asset")
        if self.debt_asset == DebtAsset.EXTERNAL:
            if not self.debt_asset_symbol:
                raise ValueError("External debt requires debt_asset_symbol")
            if self.debt_price_initial_asset1 is None or self.debt_price_initial_asset1 <= 0:
                raise ValueError("External debt requires positive debt_price_initial_asset1")
        
        # Calculate actual leverage based on protocol LTV
        # self.effective_leverage = min(self.leverage, 1 + self.max_ltv)
        # if self.effective_leverage < self.leverage:
        #     import warnings
        #     warnings.warn(
        #         f"Requested leverage {self.leverage}x exceeds protocol max "
        #         f"{self.effective_leverage:.2f}x (LTV={self.max_ltv})"
        #     )

    def resolved_debt_symbol(self) -> Optional[str]:
        """Return the borrowed token symbol implied by debt_asset."""
        if self.debt_asset == DebtAsset.ASSET0:
            return self.assets.asset0_symbol
        if self.debt_asset == DebtAsset.ASSET1:
            return self.assets.asset1_symbol
        if self.debt_asset == DebtAsset.EXTERNAL:
            return self.debt_asset_symbol
        return None


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
    debt_amount_external: float = 0.0
    debt_price_asset1: float = 0.0
    
    @property
    def debt_value_asset1(self) -> float:
        """Total debt value in asset1 terms."""
        external_value = self.debt_amount_external * self.debt_price_asset1
        return self.debt_amount0 * self.current_price + self.debt_amount1 + external_value
    
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

    @property
    def collateral_value_asset1(self) -> float:
        """Arcadia-style collateral value after collateral-factor haircuts."""
        asset0_value = self.amount0 * self.current_price
        asset1_value = self.amount1
        collateral_value = (
            self.config.collateral_factor_asset0 * asset0_value
            + self.config.collateral_factor_asset1 * asset1_value
        )
        return collateral_value * self.config.lp_protocol_risk_factor

    @property
    def liquidation_value_asset1(self) -> float:
        """Arcadia-style liquidation value after liquidation-factor haircuts."""
        asset0_value = self.amount0 * self.current_price
        asset1_value = self.amount1
        liquidation_value = (
            self.config.liquidation_factor_asset0 * asset0_value
            + self.config.liquidation_factor_asset1 * asset1_value
        )
        return liquidation_value * self.config.lp_protocol_risk_factor

    @property
    def used_margin_asset1(self) -> float:
        """Open debt plus creditor-specific minimum margin."""
        return self.debt_value_asset1 + self.config.minimum_margin_asset1

    @property
    def free_margin_asset1(self) -> float:
        """Collateral value remaining after used margin."""
        return self.collateral_value_asset1 - self.used_margin_asset1

    @property
    def margin_health_ratio(self) -> float:
        """Liquidation value divided by used margin."""
        used_margin = self.used_margin_asset1
        return self.liquidation_value_asset1 / used_margin if used_margin > 0 else float('inf')

    @property
    def margin_state(self) -> str:
        """Arcadia account state based on collateral and liquidation values."""
        used_margin = self.used_margin_asset1
        if used_margin < self.collateral_value_asset1:
            return "healthy"
        if used_margin < self.liquidation_value_asset1:
            return "unhealthy"
        return "liquidatable"


@dataclass
class SimulationResult:
    """
    Results from price simulation.
    """
    prices: List[float]
    positions: List[LPPosition]  # Position at each price
    metrics: Dict               # Calculated metrics
    config: LPConfig           # Original config