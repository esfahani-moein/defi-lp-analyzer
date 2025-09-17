# 🌊 DeFi LP Analyzer

> **Advanced Leveraged Liquidity Provider Analysis for DeFi Protocols**

A comprehensive Python toolkit for analyzing leveraged liquidity provider (LP) positions on decentralized exchanges (DEX). 
Calculate returns, impermanent loss, breakeven points, and risk metrics with interactive visualizations.
Explore and evaluate hedging ideas to manage risk and optimize LP performance.


## ✨ Features

- 📊 **Leveraged LP Position Analysis** - Calculate returns across different price scenarios
- 💰 **Impermanent Loss Calculations** - Understand IL impact with leverage
- 📈 **Interactive Visualizations** - Beautiful plots with breakeven lines and risk zones  
- ⚖️ **Multi-Asset Support** - Analyze all trading pairs
- 🎯 **Risk Metrics** - APR calculations, liquidation risks, and position optimization
- 🔄 **Scenario Modeling** - Compare different leverage ratios and price ranges
- 📈 **Hedge Strategy Modeling** - Assess and compare various hedging approaches for LP positions
- 📱 **Jupyter Integration** - Ready-to-use notebooks with examples

## 🎯 Who This Is For

- **DeFi Yield Farmers** seeking to optimize leveraged LP strategies
- **Quantitative Researchers** analyzing liquidity provision returns  
- **Portfolio Managers** evaluating DeFi investment opportunities
- **Developers** building DeFi analytics tools

### Uniswap v3 Liquidity Mathematics
- **Formula**: `L = V / [2*√P - P/√Pb - √Pa]` ✅ **ACCURATE**
- **Token Calculations**: Proper handling of concentrated liquidity ranges
- **Price Range Logic**: Correct handling of in-range, below-range, above-range scenarios

### Position Value Calculations
```python
# Within range (√Pa ≤ √P ≤ √Pb):
amount_eth = L * (1/√P - 1/√Pb)    
amount_usdc = L * (√P - √Pa)       

# Outside range scenarios  handled
```

#### 3. Leverage Mechanics
- **Debt Calculation**: Properly computes borrowed amounts
- **Equity Tracking**: Correctly tracks net position value
- **Verification**: Ensures equity at current price equals initial equity
## 🏗️ Built With

- **Python 3.8+** - Core calculations and analysis
- **NumPy & Pandas** - Mathematical computations  
- **Matplotlib** - Advanced visualizations
- **Jupyter** - Interactive analysis notebooks

## 🤝 Contributing

We welcome contributions!

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.