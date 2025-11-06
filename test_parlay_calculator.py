#!/usr/bin/env python3
"""
Test script for the new ParlayCalculator
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from football_master_gui import ParlayCalculator

def test_parlay_calculator():
    """Test the parlay calculator functionality"""

    calculator = ParlayCalculator()

    # Test American to decimal conversion
    print("=== Testing Odds Conversion ===")
    assert calculator.american_to_decimal(-150) == 1.6666666666666667, "American -150 to decimal failed"
    assert calculator.american_to_decimal(+200) == 3.0, "American +200 to decimal failed"
    assert calculator.decimal_to_american(1.6666666666666667) == -150, "Decimal to American -150 failed"
    assert calculator.decimal_to_american(3.0) == 200, "Decimal to American +200 failed"
    print("✅ Odds conversion tests passed")

    # Test parlay calculation
    print("\n=== Testing Parlay Calculation ===")
    test_bets = [
        "Kansas City Chiefs -150",
        "Buffalo Bills +120",
        "San Francisco 49ers -180"
    ]

    result = calculator.calculate_parlay_odds(test_bets)
    print(f"Test parlay result: {result}")

    # Expected calculation:
    # Chiefs -150 = 1.667
    # Bills +120 = 2.2
    # 49ers -180 = 1.556
    # Total decimal: 1.667 * 2.2 * 1.556 ≈ 5.71
    # American: +471

    assert abs(result['decimal_odds'] - 5.71) < 0.1, f"Decimal odds calculation failed: {result['decimal_odds']}"
    assert result['legs'] == 3, f"Legs count failed: {result['legs']}"
    print("✅ Parlay calculation tests passed")

    # Test correlation detection
    print("\n=== Testing Correlation Detection ===")
    correlated_bets = [
        "Kansas City Chiefs -150",
        "Kansas City Chiefs -140",  # Same team twice
        "Buffalo Bills +120"
    ]

    correlated_result = calculator.calculate_parlay_odds(correlated_bets)
    warnings = correlated_result.get('correlation_warnings', [])
    assert len(warnings) > 0, "Correlation detection failed"
    print(f"✅ Correlation warnings: {warnings}")

    # Test payout calculation
    print("\n=== Testing Payout Calculation ===")
    payout_result = calculator.calculate_payout(result, stake=10.0)
    print(f"Payout result: {payout_result}")
    assert abs(payout_result['payout'] - 57.1) < 1.0, f"Payout calculation failed: {payout_result['payout']}"
    assert abs(payout_result['profit'] - 47.1) < 1.0, f"Profit calculation failed: {payout_result['profit']}"
    print("✅ Payout calculation tests passed")

    # Test risk assessment
    print("\n=== Testing Risk Assessment ===")
    risk = calculator.get_risk_assessment(result)
    print(f"Risk assessment: {risk}")
    assert "MODERATE" in risk or "HIGH" in risk, f"Risk assessment failed: {risk}"
    print("✅ Risk assessment tests passed")

    print("\n🎉 All ParlayCalculator tests passed!")

if __name__ == "__main__":
    test_parlay_calculator()
