"""
Quick test script to verify all modules work correctly.
"""
import sys
sys.path.insert(0, '/Users/mesfahani1/myProjects/project_defi_analyzer')

print("="*60)
print("Testing LP Analysis Project")
print("="*60)

# Test 1: Imports
print("\n[1/4] Testing imports...")
try:
    from lp_analysis import (
        AssetPair, PriceRange, LPConfig, LeveragedLPConfig, DebtAsset,
        LPCalculator, LeveragedLPCalculator,
        LPAnalyzer, RiskAnalyzer,
        LPVisualizer
    )
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Basic LP
print("\n[2/4] Testing basic LP calculation...")
try:
    config = LPConfig(
        capital_asset1=10000,
        price_initial=5000,
        price_range=PriceRange(4800, 5200),
        assets=AssetPair("ETH", "USDC"),
        fee_tier=0.3
    )
    
    calc = LPCalculator()
    initial_pos = calc.calculate_initial_position(config)
    
    print(f"  ETH: {initial_pos.amount0:.6f}")
    print(f"  USDC: {initial_pos.amount1:.2f}")
    print(f"  Liquidity: {initial_pos.liquidity:.2f}")
    print("✓ Basic LP works")
except Exception as e:
    print(f"✗ Basic LP failed: {e}")
    sys.exit(1)

# Test 3: Simulation and Analysis
print("\n[3/4] Testing simulation and analysis...")
try:
    result = LeveragedLPCalculator().simulate_price_range(
        config, price_min=4500, price_max=5500
    )
    
    result = LPAnalyzer.analyze_simulation(result)
    
    print(f"  Max Gain: ${result.metrics['pnl']['max_gain']:.2f}")
    print(f"  Max Loss: ${result.metrics['pnl']['max_loss']:.2f}")
    print("✓ Simulation and analysis works")
except Exception as e:
    print(f"✗ Simulation failed: {e}")
    sys.exit(1)

# Test 4: Leveraged LP
print("\n[4/4] Testing leveraged LP...")
try:
    lev_config = LeveragedLPConfig(
        capital_asset1=10000,
        price_initial=5000,
        price_range=PriceRange(4800, 5200),
        assets=AssetPair("ETH", "USDC"),
        leverage=5.0,
        debt_asset=DebtAsset.ASSET1,
        borrow_apr=10.0,
        max_ltv=0.8,
        liquidation_threshold=0.85
    )
    
    result = LeveragedLPCalculator().simulate_price_range(lev_config)
    result = LPAnalyzer.analyze_simulation(result)
    result.metrics['risk'] = RiskAnalyzer.analyze_risk(result)
    
    liq = result.metrics['risk']['liquidation_prices']
    if liq['lower'] is not None:
        print(f"  Liquidation Range: [{liq['lower']:.2f}, {liq['upper']:.2f}]")
    else:
        print(f"  Liquidation Range: No liquidation in range")
    print(f"  Daily Cost: ${result.metrics['risk']['daily_cost']:.2f}")
    print("✓ Leveraged LP works")
except Exception as e:
    print(f"✗ Leveraged LP failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ All tests passed! Project is working correctly.")
print("="*60)
