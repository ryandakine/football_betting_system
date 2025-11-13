#!/usr/bin/env python3
"""
Handle-Based Trap Detector
===========================

Detects public traps by comparing actual handle to expected handle.

Simple, focused, works.
"""

def calculate_trap_score(odds, handle_pct):
    """
    Calculate trap score: -100 (trap) to +100 (sharp consensus)
    
    Args:
        odds: American odds (e.g., -150)
        handle_pct: % of money on this side (0.85 = 85%)
    
    Returns:
        int: Trap score
    """
    # Expected handle by odds
    expected = {
        -300: 0.75, -250: 0.71, -200: 0.67, -175: 0.64,
        -150: 0.60, -130: 0.57, -110: 0.52, -100: 0.50,
        100: 0.50, 110: 0.48, 130: 0.43, 150: 0.40,
        200: 0.33, 250: 0.29, 300: 0.25
    }
    
    # Find closest odds
    closest = min(expected.keys(), key=lambda x: abs(x - odds))
    expected_handle = expected[closest]
    
    # Calculate divergence
    div = handle_pct - expected_handle
    
    # Score it
    if div > 0.15:      return -100  # STRONG TRAP
    elif div > 0.10:    return -60   # MODERATE TRAP
    elif div > 0.05:    return -30   # SLIGHT TRAP
    elif div < -0.10:   return +60   # SHARP CONSENSUS
    elif div < -0.05:   return +30   # SLIGHT SHARP
    else:               return 0     # NORMAL


def adjust_bet_size(original_amount, trap_score):
    """
    Adjust bet size based on trap score.
    
    Args:
        original_amount: Original bet size (e.g., 4.0)
        trap_score: Score from calculate_trap_score()
    
    Returns:
        float: Adjusted bet amount
    """
    if trap_score <= -60:
        return 0.0  # FADE - Don't bet
    elif trap_score <= -30:
        return original_amount * 0.5  # Cut 50%
    elif trap_score >= 60:
        return original_amount * 1.25  # Boost 25%
    else:
        return original_amount  # No change


if __name__ == "__main__":
    # Test cases
    print("\n🧪 HANDLE TRAP DETECTOR TESTS\n")
    
    # Test 1: Public trap (MNF scenario)
    print("Test 1: GB -1.5 with 85% handle")
    score = calculate_trap_score(-150, 0.85)
    adjusted = adjust_bet_size(4.0, score)
    print(f"  Trap Score: {score}")
    print(f"  Bet: $4.00 → ${adjusted:.2f}")
    print(f"  Action: {'FADE' if score <= -60 else 'REDUCE' if score <= -30 else 'NORMAL'}")
    print()
    
    # Test 2: Normal market
    print("Test 2: Normal -110 line with 54% handle")
    score = calculate_trap_score(-110, 0.54)
    adjusted = adjust_bet_size(4.0, score)
    print(f"  Trap Score: {score}")
    print(f"  Bet: $4.00 → ${adjusted:.2f}")
    print(f"  Action: NORMAL")
    print()
    
    # Test 3: Sharp consensus
    print("Test 3: Dog +130 with only 30% handle (sharps on it)")
    score = calculate_trap_score(130, 0.30)
    adjusted = adjust_bet_size(4.0, score)
    print(f"  Trap Score: {score}")
    print(f"  Bet: $4.00 → ${adjusted:.2f}")
    print(f"  Action: BOOST")
    print()
    
    print("✅ Tests complete!\n")
