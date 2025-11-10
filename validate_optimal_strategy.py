#!/usr/bin/env python3
"""
Validate Optimal LLM Strategy Configuration
============================================

Loads and validates optimal_llm_weights.json to ensure it's ready for production.
"""

import json
from pathlib import Path


def validate_optimal_strategy():
    """Validate the optimal strategy configuration."""
    print("="*80)
    print("VALIDATING OPTIMAL LLM STRATEGY CONFIGURATION")
    print("="*80)
    print()

    # Check if file exists
    weights_file = Path("optimal_llm_weights.json")
    if not weights_file.exists():
        print("❌ ERROR: optimal_llm_weights.json not found!")
        print("   Run: python backtest_llm_meta_models.py")
        return False

    # Load configuration
    try:
        with open(weights_file, 'r') as f:
            config = json.load(f)
        print("✅ Configuration file loaded successfully")
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON in configuration file: {e}")
        return False

    print()
    print("Configuration:")
    print("-" * 80)
    print(json.dumps(config, indent=2))
    print("-" * 80)
    print()

    # Validate required fields
    required_fields = [
        'strategy_name',
        'description',
        'weights',
        'min_confidence',
        'expected_roi',
        'expected_win_rate'
    ]

    print("Validating required fields...")
    all_valid = True
    for field in required_fields:
        if field in config:
            print(f"  ✅ {field}: {config[field]}")
        else:
            print(f"  ❌ Missing required field: {field}")
            all_valid = False

    if not all_valid:
        return False

    print()
    print("Validating weights...")
    weights = config.get('weights', {})

    # Check that weights sum to 1.0
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) < 0.01:
        print(f"  ✅ Weights sum to {total_weight:.2f}")
    else:
        print(f"  ⚠️  Warning: Weights sum to {total_weight:.2f} (expected 1.0)")

    # Check individual model weights
    for model, weight in weights.items():
        if 0 <= weight <= 1:
            print(f"  ✅ {model}: {weight:.2f}")
        else:
            print(f"  ❌ {model}: {weight:.2f} (should be between 0 and 1)")
            all_valid = False

    print()
    print("Validating confidence threshold...")
    min_conf = config.get('min_confidence', 0)
    if 50 <= min_conf <= 100:
        print(f"  ✅ Minimum confidence: {min_conf}%")
    else:
        print(f"  ❌ Invalid confidence threshold: {min_conf}% (should be 50-100)")
        all_valid = False

    print()
    print("Validating expected performance...")
    expected_roi = config.get('expected_roi', 0)
    expected_win_rate = config.get('expected_win_rate', 0)

    if expected_roi > 0:
        print(f"  ✅ Expected ROI: {expected_roi:.2f}%")
    else:
        print(f"  ⚠️  Expected ROI: {expected_roi:.2f}% (should be positive)")

    if 50 <= expected_win_rate <= 100:
        print(f"  ✅ Expected Win Rate: {expected_win_rate:.2f}%")
    else:
        print(f"  ⚠️  Expected Win Rate: {expected_win_rate:.2f}% (unexpected value)")

    print()
    print("="*80)
    if all_valid:
        print("✅ VALIDATION PASSED - Configuration is ready for production!")
        print()
        print("Recommended Strategy:")
        print(f"  Model: {config['description']}")
        print(f"  Expected ROI: {expected_roi:.2f}%")
        print(f"  Expected Win Rate: {expected_win_rate:.2f}%")
        print(f"  Min Confidence: {min_conf}%")
        print()
        print("To use in production:")
        print("  1. Load optimal_llm_weights.json")
        print("  2. Use the weights for model combination")
        print("  3. Apply min_confidence threshold before betting")
        print("  4. Size bets using bet_sizing configuration")
    else:
        print("❌ VALIDATION FAILED - Please fix errors above")

    print("="*80)

    return all_valid


def display_betting_guide():
    """Display quick betting guide."""
    print()
    print("="*80)
    print("QUICK BETTING GUIDE")
    print("="*80)
    print()
    print("For each game:")
    print("  1. Get prediction from DeepSeek-R1")
    print("  2. Check confidence level")
    print("  3. Size bet accordingly:")
    print()
    print("     IF confidence >= 80%  → Bet 6 units")
    print("     IF confidence >= 75%  → Bet 4 units")
    print("     IF confidence >= 70%  → Bet 2 units")
    print("     IF confidence <  70%  → PASS (no bet)")
    print()
    print("  4. Track results weekly")
    print()
    print("Expected Results (from backtest):")
    print("  - Win Rate: 74.57%")
    print("  - ROI: 37.03%")
    print("  - Bet Frequency: ~70% of games")
    print()
    print("="*80)


if __name__ == "__main__":
    valid = validate_optimal_strategy()
    if valid:
        display_betting_guide()
