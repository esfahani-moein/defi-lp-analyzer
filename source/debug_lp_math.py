"""
Debug script to verify Uniswap v3 concentrated liquidity mathematics
"""
import numpy as np

def debug_uniswap_v3_math():
    """Manual calculation to verify the math"""
    
    # Test case: ETH position with asymmetric range
    current_price = 4000.0
    lower_price = 3700.0  
    upper_price = 4500.0
    initial_eth = 1.0
    initial_usdc = 0.0
    leverage = 6.0
    
    # User's equity
    initial_equity_usd = initial_eth * current_price + initial_usdc
    print(f"Initial Equity: ${initial_equity_usd:,.2f}")
    
    # Total leveraged position value
    total_position_usd = initial_equity_usd * leverage
    print(f"Total Position Value: ${total_position_usd:,.2f}")
    
    # Borrowed amount
    borrowed_usd = total_position_usd - initial_equity_usd
    debt_eth = borrowed_usd / current_price
    print(f"Debt: {debt_eth:.6f} ETH (${borrowed_usd:,.2f})")
    
    # Calculate square roots
    sp_c = np.sqrt(current_price)  # sqrt(4000) ≈ 63.25
    sp_a = np.sqrt(lower_price)    # sqrt(3700) ≈ 60.83
    sp_b = np.sqrt(upper_price)    # sqrt(4500) ≈ 67.08
    
    print(f"\nSquare roots:")
    print(f"sqrt(P_c) = {sp_c:.4f}")
    print(f"sqrt(P_a) = {sp_a:.4f}")
    print(f"sqrt(P_b) = {sp_b:.4f}")
    
    # For Uniswap v3, given total USD value V, we need to solve:
    # V = x * P + y
    # where x and y are determined by the liquidity L and current price
    
    # The optimal allocation for price within range [Pa, Pb]:
    # x = L * (1/sqrt(P) - 1/sqrt(Pb))
    # y = L * (sqrt(P) - sqrt(Pa))
    
    # And L can be derived from the constraint that total value = V
    # V = L * (1/sqrt(P) - 1/sqrt(Pb)) * P + L * (sqrt(P) - sqrt(Pa))
    # V = L * [P * (1/sqrt(P) - 1/sqrt(Pb)) + (sqrt(P) - sqrt(Pa))]
    # V = L * [sqrt(P) - P/sqrt(Pb) + sqrt(P) - sqrt(Pa)]
    # V = L * [2*sqrt(P) - P/sqrt(Pb) - sqrt(Pa)]
    
    denominator = 2 * sp_c - current_price/sp_b - sp_a
    L = total_position_usd / denominator
    
    print(f"\nLiquidity calculation:")
    print(f"Denominator = 2*sqrt(P) - P/sqrt(Pb) - sqrt(Pa)")
    print(f"            = 2*{sp_c:.4f} - {current_price}/{sp_b:.4f} - {sp_a:.4f}")
    print(f"            = {2*sp_c:.4f} - {current_price/sp_b:.4f} - {sp_a:.4f}")
    print(f"            = {denominator:.4f}")
    print(f"L = V / denominator = {total_position_usd} / {denominator:.4f} = {L:.4f}")
    
    # Calculate token amounts
    eth_amount = L * (1/sp_c - 1/sp_b)
    usdc_amount = L * (sp_c - sp_a)
    
    print(f"\nToken amounts:")
    print(f"ETH = L * (1/sqrt(P) - 1/sqrt(Pb)) = {L:.4f} * ({1/sp_c:.6f} - {1/sp_b:.6f})")
    print(f"    = {L:.4f} * {1/sp_c - 1/sp_b:.6f} = {eth_amount:.6f}")
    print(f"USDC = L * (sqrt(P) - sqrt(Pa)) = {L:.4f} * ({sp_c:.4f} - {sp_a:.4f})")
    print(f"     = {L:.4f} * {sp_c - sp_a:.4f} = {usdc_amount:.2f}")
    
    # Verify total value
    total_value = eth_amount * current_price + usdc_amount
    print(f"\nVerification:")
    print(f"Total Value = {eth_amount:.6f} * {current_price} + {usdc_amount:.2f}")
    print(f"            = ${eth_amount * current_price:.2f} + ${usdc_amount:.2f}")
    print(f"            = ${total_value:.2f}")
    print(f"Expected: ${total_position_usd:.2f}")
    print(f"Difference: ${abs(total_value - total_position_usd):.2f}")
    
    # Now calculate position value at current price
    final_equity = total_value - debt_eth * current_price
    print(f"\nFinal equity at current price:")
    print(f"Final Equity = LP Value - Debt Value")
    print(f"             = ${total_value:.2f} - {debt_eth:.6f} * {current_price}")
    print(f"             = ${total_value:.2f} - ${debt_eth * current_price:.2f}")
    print(f"             = ${final_equity:.2f}")
    
    # This should equal the initial equity for no price change
    print(f"Initial Equity: ${initial_equity_usd:.2f}")
    print(f"Difference: ${final_equity - initial_equity_usd:.2f}")
    
    if abs(final_equity - initial_equity_usd) > 10:
        print("\n⚠️  WARNING: Large discrepancy detected!")
        print("The math appears to be incorrect.")
    else:
        print("\n✅ Math looks correct!")

if __name__ == "__main__":
    debug_uniswap_v3_math()