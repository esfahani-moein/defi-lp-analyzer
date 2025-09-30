"""
LP Analysis Package - Comprehensive toolkit for analyzing liquidity positions.
"""
from .lp_calc.types import (
    DebtAsset, AssetPair, PriceRange, LPConfig, LeveragedLPConfig,
    LPPosition, LeveragedPosition, SimulationResult
)
from .lp_calc.lp_calc import LPCalculator
from .lp_calc.lev_lp_calc import LeveragedLPCalculator
from .analysis.lp_analysis import LPAnalyzer
from .analysis.lp_risk import RiskAnalyzer
from .visualization.plots import LPVisualizer

__version__ = "1.0.0"
__all__ = [
    # Types
    'DebtAsset', 'AssetPair', 'PriceRange', 'LPConfig', 'LeveragedLPConfig',
    'LPPosition', 'LeveragedPosition', 'SimulationResult',
    # Calculators
    'LPCalculator', 'LeveragedLPCalculator',
    # Analyzers
    'LPAnalyzer', 'RiskAnalyzer',
    # Visualization
    'LPVisualizer'
]