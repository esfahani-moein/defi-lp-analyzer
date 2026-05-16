# Start: Combined LP Position Analysis

## Features

### Two New Methods:

1. **`LPVisualizer.plot_combined_positions_dashboard()`**
   - Location: `/lp_analysis/visualization/plots.py`
   - Purpose: Visualize two LP positions combined
   - Shows: Delta hedging effectiveness, PnL, exposure metrics

2. **`ExposureAnalyzer.calculate_market_exposure()`**
   - Location: `/lp_analysis/analysis/lp_analysis.py`
   - Purpose: Calculate delta and gamma exposure
   - Returns: Delta/gamma arrays and statistics

## 🚀 How to Use (Already in Your Notebook!)

Your notebook already has the correct usage in the last cell:

```python
# Visualize combined positions
print("Combined Strategy 1: USDC Debt + ETH Debt (Same Range)")
LPVisualizer.plot_combined_positions_dashboard(
    result_usdc1, 
    result_eth1,
    position1_name="Long ETH (USDC Debt, 2x)",
    position2_name="Short ETH (ETH Debt, 3x)"
)
```

## 📋 Running Your Notebook

Simply run all cells in order:

1. **Cell 1**: Imports (with module reload)
2. **Cell 2**: Configure 4 LP positions
3. **Cell 3**: Simulate and analyze all positions
4. **Cell 4**: Individual dashboards (result_usdc1, result_usdc2)
5. **Cell 5**: Individual dashboards (result_eth1, result_eth2)
6. **Cell 6**: **NEW** Combined dashboards!

## 🎯 What You'll See

### Dashboard Panels (8 total):

1. **Combined Position Value** - Total portfolio value across prices
2. **Combined PnL** - Profit/loss with green/red zones
3. **Delta Exposure** - Shows hedging effectiveness (target: 0)
4. **Returns Distribution** - Risk profile comparison
5. **ETH Holdings** - Asset0 amounts by position
6. **USDC Holdings** - Asset1 amounts by position
7. **Gamma Exposure** - Convexity analysis
8. **Metrics Table** - All key statistics

### Terminal Output:

```
================================================================================
COMBINED POSITION SUMMARY
================================================================================

Positions:
  • Long ETH (USDC Debt, 2x): $10,000 @ 2.0x (ASSET1 debt)
  • Short ETH (ETH Debt, 3x): $10,000 @ 3.0x (ASSET0 debt)

Combined Portfolio:
  Total Capital: $20,000
  Current Value: $20,xxx
  Current PnL: $xxx (x.xx%)

Delta Neutrality:
  Average |Δ|: x.xxxx
  Maximum |Δ|: x.xxxx
  Assessment: ✓ Well Hedged / ⚠ Needs Optimization / ✗ Poorly Hedged
================================================================================
```

## 🔧 Customization Options

### Change Position Names:
```python
LPVisualizer.plot_combined_positions_dashboard(
    result1, 
    result2,
    position1_name="My Custom Long Position",
    position2_name="My Custom Short Position"
)
```

### Compare Different Combinations:
```python
# Same debt type comparison
LPVisualizer.plot_combined_positions_dashboard(
    result_usdc1, 
    result_usdc2,
    position1_name="USDC Debt (4800-5200)",
    position2_name="USDC Debt (4600-5200)"
)
```

## 📊 Interpreting Results

### Delta Neutrality:
- **< 0.05**: Excellent hedge ✓
- **0.05-0.20**: Needs tweaking ⚠
- **> 0.20**: Poor hedge ✗

### What to Adjust:
1. **Capital Split**: Change `capital_asset1` values
2. **Leverage Ratios**: Adjust `leverage` parameter
3. **Price Ranges**: Modify `PriceRange(lower, upper)`

## 🐛 Troubleshooting

### Error: "Both results must be analyzed first"
**Solution:** Make sure you run analysis before visualization:
```python
result.metrics['exposure'] = ExposureAnalyzer.calculate_market_exposure(result)
```

### Error: "Module not found"
**Solution:** Restart kernel and run the import cell with module reload

### No Delta/Gamma Shown
**Solution:** Ensure exposure analysis is run:
```python
result.metrics['exposure'] = ExposureAnalyzer.calculate_market_exposure(result)
```

## 💡 Next Steps

Once you see which combinations work best, you can:

1. Build optimization loops to find optimal parameters
2. Test different market scenarios (price ranges)
3. Analyze sensitivity to leverage changes
4. Compare strategies side-by-side

## 📝 Example Workflow

```python
# 1. Import
from lp_analysis import *

# 2. Configure
config1 = LeveragedLPConfig(capital_asset1=10000, leverage=2.0, debt_asset=DebtAsset.ASSET1, ...)
config2 = LeveragedLPConfig(capital_asset1=10000, leverage=3.0, debt_asset=DebtAsset.ASSET0, ...)

# 3. Simulate
calc = LeveragedLPCalculator()
result1 = calc.simulate_price_range(config1)
result2 = calc.simulate_price_range(config2)

# 4. Analyze
result1 = LPAnalyzer.analyze_simulation(result1)
result1.metrics['exposure'] = ExposureAnalyzer.calculate_market_exposure(result1)

result2 = LPAnalyzer.analyze_simulation(result2)
result2.metrics['exposure'] = ExposureAnalyzer.calculate_market_exposure(result2)

# 5. Visualize Combined
LPVisualizer.plot_combined_positions_dashboard(result1, result2)
```

## ✨ Features

- ✅ No code conflicts with existing repository
- ✅ Backward compatible (all existing code still works)
- ✅ Professional visualization
- ✅ Quantitative metrics
- ✅ Ready for optimization algorithms
- ✅ Handles any combination of positions

---

**Ready to run!** Just execute all cells in `project03_lp_strategy.ipynb`
