# 🎯 Exposure Analysis - Quick Reference

## New Classes Added

### `ExposureAnalyzer` (in `lp_analysis.analysis.lp_analysis`)
Analyzes net asset exposure accounting for debt.

### `ExposureVisualizer` (in `lp_analysis.visualization.plots`)
Specialized plots for exposure analysis.

---

## Quick Start

```python
from lp_analysis import (
    ExposureAnalyzer, ExposureVisualizer,
    LeveragedLPCalculator, LPAnalyzer
)

# 1. Simulate position
result = LeveragedLPCalculator().simulate_price_range(config)

# 2. Analyze (includes exposure)
result = LPAnalyzer.analyze_simulation(result)
result = ExposureAnalyzer.analyze_exposure(result)

# 3. Visualize
ExposureVisualizer.plot_exposure_dashboard(result)
```

---

## Key Methods

### Analysis
```python
# Single position exposure at one price
ExposureAnalyzer.calculate_net_exposure(position)

# Exposure across price range
ExposureAnalyzer.calculate_exposure_profile(result)

# Add all exposure metrics
ExposureAnalyzer.analyze_exposure(result)

# Compare multiple strategies
ExposureAnalyzer.compare_exposure_strategies([result1, result2])
```

### Visualization
```python
# Net token exposure plot
ExposureVisualizer.plot_net_exposure(result)

# Exposure percentage plot
ExposureVisualizer.plot_exposure_percentage(result)

# Value exposure plot
ExposureVisualizer.plot_exposure_value(result)

# Complete dashboard (4 plots)
ExposureVisualizer.plot_exposure_dashboard(result)

# Compare strategies side-by-side
ExposureVisualizer.compare_exposure_strategies(
    [result1, result2], 
    labels=['Strategy 1', 'Strategy 2']
)
```

---

## Access Exposure Data

After running `ExposureAnalyzer.analyze_exposure(result)`:

```python
exposure = result.metrics['exposure_profile']

# Available data
exposure['asset0_net_tokens']      # List of net token amounts
exposure['asset1_net_tokens']      # List of net quote amounts
exposure['asset0_net_value']       # List of net values (in asset1)
exposure['asset1_net_value']       # List of net values (in asset1)
exposure['asset0_exposure_pct']    # List of exposure percentages
exposure['asset1_exposure_pct']    # List of exposure percentages
exposure['total_net_value']        # List of total net values
exposure['is_leveraged']           # Boolean
exposure['prices']                 # Price points

# Exposure delta
delta = result.metrics['exposure_delta']
delta['token_delta']               # Sensitivity of tokens to price
delta['value_delta']               # Sensitivity of value to price
```

---

## Understanding Results

### USDC Debt (Long Volatile Asset)
```
✓ Borrowing USDC (stable)
✓ Holding ETH (volatile) in LP
✓ Net ETH exposure = POSITIVE
✓ Benefits when ETH rises
✓ Suffers when ETH falls
```

### ETH Debt (Short Volatile Asset)
```
✓ Borrowing ETH (volatile)
✓ Holding USDC (stable) in LP
✓ Net ETH exposure = NEGATIVE
✓ Benefits when ETH falls
✓ Suffers when ETH rises
```

### Delta-Neutral (Combined)
```
✓ 50% USDC debt + 50% ETH debt
✓ Net ETH exposure ≈ 0
✓ Minimal price sensitivity
✓ Profit from fees, not direction
```

---

## Exposure Metrics Explained

| Metric | Meaning | Interpretation |
|--------|---------|----------------|
| `net_asset0_tokens` | Holdings - Debt | Positive = Long, Negative = Short |
| `asset0_exposure_pct` | % of portfolio in asset0 | >100% = Leveraged long |
| `token_delta` | ∂tokens/∂price | How exposure changes |
| `exposure_stability` | σ of exposure % | Lower = more stable |

---

## Common Use Cases

### 1. Check Current Exposure
```python
exp = result.metrics['exposure_profile']
mid_idx = len(result.prices) // 2
print(f"ETH Exposure: {exp['asset0_exposure_pct'][mid_idx]:.1f}%")
```

### 2. Compare Debt Strategies
```python
ExposureVisualizer.compare_exposure_strategies(
    [result_usdc_debt, result_eth_debt],
    labels=['USDC Debt', 'ETH Debt']
)
```

### 3. Find Optimal Exposure
```python
for lev in [2, 3, 5, 10]:
    config = LeveragedLPConfig(..., leverage=lev)
    result = simulate_and_analyze(config)
    
    exposure_pct = get_initial_exposure(result)
    if abs(exposure_pct - 50) < 5:  # Target 50%
        print(f"Optimal: {lev}x")
```

### 4. Monitor Delta-Neutral
```python
combined_exposure = (
    exp_usdc['asset0_exposure_pct'] * 0.5 +
    exp_eth['asset0_exposure_pct'] * 0.5
)
is_neutral = abs(np.mean(combined_exposure)) < 10
```

---

## Notebook Examples

See `project01_lp_analysis.ipynb` cells:
- **Cell 4**: Single position exposure analysis
- **Cell 5**: Net exposure visualization
- **Cell 6**: Exposure percentage plots
- **Cell 7**: Complete exposure dashboard
- **Cell 8**: Strategy comparison

---

## Integration with Existing Code

**No modifications needed!** The new classes are **additions only**:

```python
# Old workflow still works
result = LPAnalyzer.analyze_simulation(result)
result.metrics['risk'] = RiskAnalyzer.analyze_risk(result)

# New exposure analysis is optional
result = ExposureAnalyzer.analyze_exposure(result)  # ← Add this
```

---

## Key Files Modified

✅ **Added** (no existing code changed):
- `lp_analysis/analysis/lp_analysis.py` - Added `ExposureAnalyzer` class
- `lp_analysis/visualization/plots.py` - Added `ExposureVisualizer` class
- `lp_analysis/__init__.py` - Added exports

🔧 **Fixed**:
- `lp_analysis/lp_calc/lev_lp_calc.py` - Changed `effective_leverage` → `leverage`

---

## Tips

1. **Always analyze exposure** for leveraged positions
2. **Compare strategies** before deploying capital
3. **Monitor exposure drift** as price moves
4. **Use dashboard** for quick overview
5. **Check stability** metric for rebalancing needs

---

**Quick Help**: See `EXPOSURE_ANALYSIS_GUIDE.md` for detailed documentation.
