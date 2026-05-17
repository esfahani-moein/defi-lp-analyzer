"""
LP Analysis Package - Comprehensive toolkit for analyzing liquidity positions.
"""
from .lp_calc.types import (
    DebtAsset, AssetPair, PriceRange, LPConfig, LeveragedLPConfig,
    LPPosition, LeveragedPosition, SimulationResult
)
from .lp_calc.lp_calc import LPCalculator
from .lp_calc.lev_lp_calc import LeveragedLPCalculator
from .analysis.lp_analysis import LPAnalyzer, ExposureAnalyzer
from .analysis.lp_risk import RiskAnalyzer
from .analysis.valuation import simulation_to_frame
from .protocols.lending import (
    aave_health_factor, arcadia_margin_metrics, moonwell_credit_metrics
)
from .visualization.plots import LPVisualizer, ExposureVisualizer

__version__ = "1.0.0"
__all__ = [
    # Types
    'DebtAsset', 'AssetPair', 'PriceRange', 'LPConfig', 'LeveragedLPConfig',
    'LPPosition', 'LeveragedPosition', 'SimulationResult',
    # Calculators
    'LPCalculator', 'LeveragedLPCalculator',
    # Analyzers
    'LPAnalyzer', 'ExposureAnalyzer', 'RiskAnalyzer', 'simulation_to_frame',
    'aave_health_factor', 'arcadia_margin_metrics', 'moonwell_credit_metrics',
    # Visualization
    'LPVisualizer', 'ExposureVisualizer'
]