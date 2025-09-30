# Exposure Analysis Guide

## 🎯 Overview

The **Exposure Analysis** module allows you to analyze and optimize asset exposure in leveraged liquidity positions. This is critical for understanding risk and building delta-neutral or targeted exposure strategies.

---

## 📊 Key Concepts

### **Net Exposure**
```
Net Exposure = LP Holdings - Debt
```

For a leveraged ETH/USDC position:
- **Long ETH Exposure**: Borrow USDC → You hold ETH, owe stable
- **Short ETH Exposure**: Borrow ETH → You owe ETH, hold stable

### **Exposure Calculation**

```python
# LP holds: 5 ETH + 10,000 USDC
# Debt: 0 ETH + 40,000 USDC (borrowed USDC)

Net ETH = 5 - 0 = 5 ETH          # Long exposure
Net USDC = 10,000 - 40,000 = -30,000 USDC  # Short exposure
```

At ETH = $5,000:
- ETH Value: 5 × $5,000 = $25,000
- USDC Value: -$30,000
- Net Portfolio Value: -$5,000

**Exposure %**:
- ETH: $25,000 / -$5,000 = 500% (!!)
- USDC: -$30,000 / -$5,000 = 600% (!!)

> ⚠️ When net value is negative or small, exposure % can be very large. This shows extreme leverage!

---

## 🔧 Usage

### **1. Basic Exposure Analysis**

```python
from lp_analysis import (
    ExposureAnalyzer, ExposureVisualizer,
    LeveragedLPConfig, DebtAsset, AssetPair, PriceRange
)

# Create leveraged position
config = LeveragedLPConfig(
    capital_asset1=10000,
    price_initial=5000,
    price_range=PriceRange(4800, 5200),
    assets=AssetPair("ETH", "USDC"),
    leverage=5.0,
    debt_asset=DebtAsset.ASSET1,  # Borrow USDC
    borrow_apr=10.0,
)

# Simulate and analyze
calc = LeveragedLPCalculator()
result = calc.simulate_price_range(config)
result = LPAnalyzer.analyze_simulation(result)
result = ExposureAnalyzer.analyze_exposure(result)

# Access exposure data
exposure = result.metrics['exposure_profile']
print(f"Net ETH tokens: {exposure['asset0_net_tokens']}")
print(f"ETH Exposure %: {exposure['asset0_exposure_pct']}")
```

### **2. Visualizations**

#### **Net Token Exposure**
Shows how many tokens (net of debt) you're exposed to:

```python
ExposureVisualizer.plot_net_exposure(result)
```

**Interpretation**:
- Positive values = Long exposure
- Negative values = Short exposure
- Flat line = Exposure doesn't change much with price (good for delta-neutral)

#### **Exposure Percentage**
Shows exposure as % of portfolio value:

```python
ExposureVisualizer.plot_exposure_percentage(result)
```

**Interpretation**:
- 50% / 50% = Balanced exposure
- >100% on one asset = Leveraged exposure to that asset
- Negative % = Short exposure

#### **Exposure Value**
Shows dollar value of exposures:

```python
ExposureVisualizer.plot_exposure_value(result)
```

#### **Complete Dashboard**
All metrics in one view:

```python
ExposureVisualizer.plot_exposure_dashboard(result)
```

### **3. Strategy Comparison**

Compare different debt strategies:

```python
# Create two strategies
config_usdc_debt = LeveragedLPConfig(
    capital_asset1=10000,
    price_initial=5000,
    price_range=PriceRange(4800, 5200),
    assets=AssetPair("ETH", "USDC"),
    leverage=5.0,
    debt_asset=DebtAsset.ASSET1,  # Borrow USDC
    borrow_apr=10.0,
)

config_eth_debt = LeveragedLPConfig(
    capital_asset1=10000,
    price_initial=5000,
    price_range=PriceRange(4800, 5200),
    assets=AssetPair("ETH", "USDC"),
    leverage=5.0,
    debt_asset=DebtAsset.ASSET0,  # Borrow ETH
    borrow_apr=15.0,
)

# Simulate both
result_usdc = calc.simulate_price_range(config_usdc_debt)
result_eth = calc.simulate_price_range(config_eth_debt)

# Analyze exposure
result_usdc = ExposureAnalyzer.analyze_exposure(result_usdc)
result_eth = ExposureAnalyzer.analyze_exposure(result_eth)

# Compare visually
ExposureVisualizer.compare_exposure_strategies(
    [result_usdc, result_eth],
    labels=['USDC Debt', 'ETH Debt']
)

# Compare metrics
comparison = ExposureAnalyzer.compare_exposure_strategies([result_usdc, result_eth])
```

---

## 💡 Use Cases

### **1. Delta-Neutral Strategies**

Combine USDC-debt and ETH-debt positions to minimize price risk:

```python
# Strategy 1: 50% in USDC debt (long ETH)
# Strategy 2: 50% in ETH debt (short ETH)
# → Net exposure ≈ 0

result_combined = combine_positions([result_usdc, result_eth], weights=[0.5, 0.5])
```

### **2. Directional Bets**

Maximize exposure to one asset:

```python
# Bullish on ETH: Use USDC debt only
# → High positive ETH exposure

# Bearish on ETH: Use ETH debt only
# → High negative ETH exposure
```

### **3. Exposure Optimization**

Find optimal leverage and range for target exposure:

```python
# Test different leverage levels
for leverage in [2, 3, 5, 10]:
    config = LeveragedLPConfig(..., leverage=leverage)
    result = analyze_exposure(config)
    
    # Find leverage that gives 50% ETH exposure
    if abs(result.avg_eth_exposure - 50) < 5:
        print(f"Optimal leverage: {leverage}x")
```

---

## 📈 Advanced Metrics

### **Exposure Delta**
How exposure changes with price:

```python
exposure_delta = result.metrics['exposure_delta']

# token_delta: Change in net tokens per $1 price move
# value_delta: Change in net value per $1 price move
```

**Low delta** = Stable exposure (good for hedging)  
**High delta** = Exposure changes rapidly with price (requires rebalancing)

### **Exposure Stability**
Standard deviation of exposure:

```python
comparison = ExposureAnalyzer.compare_exposure_strategies([result1, result2])
stability = comparison['strategies'][0]['exposure_stability']

# Lower = more stable exposure
```

---

## 🎨 Visualization Reference

### `plot_net_exposure()`
**What it shows**: Net token holdings across prices  
**Y-axis**: Number of tokens (asset0)  
**Use for**: Understanding if you're long or short

### `plot_exposure_percentage()`
**What it shows**: Exposure as % of portfolio value  
**Y-axis**: Percentage (can exceed 100%)  
**Use for**: Understanding leverage concentration

### `plot_exposure_value()`
**What it shows**: Dollar value of each exposure  
**Y-axis**: Value in asset1 (e.g., USDC)  
**Use for**: Seeing absolute risk amounts

### `plot_exposure_dashboard()`
**What it shows**: All metrics + PnL correlation  
**Use for**: Comprehensive analysis

### `compare_exposure_strategies()`
**What it shows**: Side-by-side comparison  
**Use for**: Choosing between strategies

---

## 🔍 Interpreting Results

### **Example 1: USDC Debt (Long ETH)**

```
Net ETH: +3.80 tokens
Net USDC: -$8,963
ETH Exposure: 188%
USDC Exposure: -88%
```

**Interpretation**:
- You're **long 3.80 ETH** (net of debt)
- You're **short $8,963 USDC** (borrowed)
- Your portfolio is **188% exposed to ETH price**
- If ETH rises 10%, your equity rises ~18.8%
- If ETH falls 10%, your equity falls ~18.8%
- **High risk, high reward**

### **Example 2: ETH Debt (Short ETH)**

```
Net ETH: -1.20 tokens
Net USDC: +$15,000
ETH Exposure: -40%
USDC Exposure: 140%
```

**Interpretation**:
- You're **short 1.20 ETH** (owe it)
- You're **long $15,000 USDC**
- Your portfolio benefits when ETH **falls**
- If ETH rises 10%, your equity falls ~4%
- If ETH falls 10%, your equity rises ~4%
- **Inverse exposure**

### **Example 3: Balanced**

```
Net ETH: +1.00 tokens
Net USDC: +$5,000
ETH Exposure: 50%
USDC Exposure: 50%
```

**Interpretation**:
- **Balanced** exposure
- Portfolio value stable across price moves
- Lower risk, but also lower potential gain
- **Good for delta-neutral strategies**

---

## ⚠️ Important Notes

1. **Negative Portfolio Value**: When debt > LP value, portfolio is underwater. Exposure % can be misleading. Focus on absolute values.

2. **Out of Range**: When price exits LP range, exposure changes dramatically as LP becomes single-sided.

3. **Rebalancing**: High leverage + concentrated range = frequent rebalancing needs.

4. **Liquidation Risk**: High exposure to volatile asset increases liquidation risk.

---

## 🚀 Next Steps

1. **Experiment** with different leverage levels
2. **Compare** USDC debt vs ETH debt strategies
3. **Optimize** for your target exposure (e.g., 50% for neutral)
4. **Monitor** exposure changes as price moves
5. **Build** multi-position portfolios for delta-neutral strategies

---

## 📚 API Reference

### `ExposureAnalyzer`

#### `calculate_net_exposure(position: LeveragedPosition) -> Dict`
Calculate net exposure for a single position.

#### `calculate_exposure_profile(result: SimulationResult) -> Dict`
Calculate exposure across entire price range.

#### `calculate_exposure_delta(result: SimulationResult) -> Dict`
Calculate how exposure changes with price.

#### `analyze_exposure(result: SimulationResult) -> SimulationResult`
Add all exposure metrics to result.

#### `compare_exposure_strategies(results_list: List[SimulationResult]) -> Dict`
Compare multiple strategies.

### `ExposureVisualizer`

All methods accept:
- `result: SimulationResult` - Simulation result with exposure analysis
- `ax: Optional[plt.Axes]` - Matplotlib axes (optional)
- `show: bool` - Whether to display plot (default: True)

---

**Created**: 2025-09-30  
**Version**: 1.0.0  
**Part of**: LP Analysis Project
