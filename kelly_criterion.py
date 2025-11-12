#!/usr/bin/env python3
"""
Kelly Criterion Bet Sizing - Mathematically Optimal Position Sizing

THE PROBLEM:
- Betting too much = High risk of ruin
- Betting too little = Leaving money on the table
- Fixed bet sizes ignore your edge and bankroll

THE SOLUTION:
Kelly Criterion calculates optimal bet size based on:
1. Your win probability (edge)
2. The odds you're getting
3. Your current bankroll

FORMULA:
f* = (bp - q) / b

Where:
- f* = Fraction of bankroll to bet
- b = Decimal odds - 1 (e.g., +110 = 2.10 - 1 = 1.10)
- p = Win probability
- q = Loss probability (1 - p)

FRACTIONAL KELLY:
Most sharp bettors use 1/4 to 1/2 Kelly to reduce variance:
- Full Kelly: Maximum growth, high variance
- Half Kelly: 75% of growth, 50% of variance
- Quarter Kelly: 50% of growth, 25% of variance

USAGE:
    python kelly_criterion.py --odds -110 --win-prob 0.55 --bankroll 100
    python kelly_criterion.py --odds +150 --win-prob 0.45 --bankroll 100 --fraction 0.5
"""

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class KellyRecommendation:
    """Kelly criterion bet sizing recommendation"""
    full_kelly_pct: float
    full_kelly_amount: float
    fractional_kelly_pct: float
    fractional_kelly_amount: float
    fraction_used: float
    expected_value: float
    roi: float
    variance_reduction: float


class KellyCriterion:
    """Calculate optimal bet sizing using Kelly Criterion"""

    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """Convert American odds to decimal"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    @staticmethod
    def decimal_to_american(decimal_odds: float) -> int:
        """Convert decimal odds to American"""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))

    def calculate_kelly(
        self,
        win_probability: float,
        odds: int,
        bankroll: float,
        fraction: float = 0.25,
        min_edge: float = 0.02
    ) -> Optional[KellyRecommendation]:
        """
        Calculate Kelly Criterion bet size.

        Args:
            win_probability: Probability of winning (0-1)
            odds: American odds (e.g., -110, +150)
            bankroll: Current bankroll
            fraction: Fraction of Kelly to use (default 0.25 = quarter Kelly)
            min_edge: Minimum edge required to bet (default 2%)

        Returns:
            KellyRecommendation or None if no edge
        """
        # Convert to decimal odds
        decimal_odds = self.american_to_decimal(odds)

        # Calculate implied probability from odds
        if odds < 0:
            implied_prob = abs(odds) / (abs(odds) + 100)
        else:
            implied_prob = 100 / (odds + 100)

        # Calculate edge
        edge = win_probability - implied_prob

        # Don't bet if no edge or edge too small
        if edge < min_edge:
            return None

        # Kelly formula: f = (bp - q) / b
        # where b = decimal_odds - 1, p = win_prob, q = 1 - p
        b = decimal_odds - 1
        p = win_probability
        q = 1 - p

        # Full Kelly percentage
        kelly_fraction = (b * p - q) / b

        # Cap at 25% of bankroll (safety limit)
        kelly_fraction = min(kelly_fraction, 0.25)

        # Can't bet negative or more than 100%
        if kelly_fraction <= 0:
            return None

        # Calculate amounts
        full_kelly_pct = kelly_fraction * 100
        full_kelly_amount = bankroll * kelly_fraction

        fractional_pct = kelly_fraction * fraction * 100
        fractional_amount = bankroll * kelly_fraction * fraction

        # Calculate expected value
        ev = (win_probability * (decimal_odds - 1)) - (1 - win_probability)
        expected_value = fractional_amount * ev

        # ROI
        roi = ev * 100

        # Variance reduction from fractional Kelly
        # Variance reduction ≈ 1 - fraction²
        variance_reduction = (1 - fraction**2) * 100

        return KellyRecommendation(
            full_kelly_pct=full_kelly_pct,
            full_kelly_amount=full_kelly_amount,
            fractional_kelly_pct=fractional_pct,
            fractional_kelly_amount=fractional_amount,
            fraction_used=fraction,
            expected_value=expected_value,
            roi=roi,
            variance_reduction=variance_reduction
        )

    def recommend_bet_size(
        self,
        win_probability: float,
        odds: int,
        bankroll: float,
        confidence: str = "medium"
    ) -> Optional[KellyRecommendation]:
        """
        Recommend bet size based on confidence level.

        Args:
            win_probability: Estimated win probability
            odds: American odds
            bankroll: Current bankroll
            confidence: "low", "medium", or "high"

        Returns:
            Kelly recommendation with appropriate fraction
        """
        # Map confidence to Kelly fraction
        kelly_fractions = {
            "low": 0.125,      # 1/8 Kelly (very conservative)
            "medium": 0.25,    # 1/4 Kelly (recommended)
            "high": 0.5,       # 1/2 Kelly (aggressive)
        }

        fraction = kelly_fractions.get(confidence.lower(), 0.25)

        return self.calculate_kelly(
            win_probability=win_probability,
            odds=odds,
            bankroll=bankroll,
            fraction=fraction
        )


def format_currency(amount: float) -> str:
    """Format amount as currency"""
    return f"${amount:.2f}"


def format_percentage(pct: float) -> str:
    """Format percentage"""
    return f"{pct:.2f}%"


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Kelly Criterion bet sizing"
    )
    parser.add_argument(
        "--odds",
        type=int,
        required=True,
        help="American odds (e.g., -110, +150)"
    )
    parser.add_argument(
        "--win-prob",
        type=float,
        required=True,
        help="Win probability (0.0 to 1.0, e.g., 0.55 for 55%%)"
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        required=True,
        help="Current bankroll in dollars"
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.25,
        help="Fraction of Kelly to use (default 0.25 = quarter Kelly)"
    )
    parser.add_argument(
        "--confidence",
        choices=["low", "medium", "high"],
        help="Confidence level (overrides --fraction)"
    )

    args = parser.parse_args()

    kelly = KellyCriterion()

    # Use confidence if provided, otherwise use fraction
    if args.confidence:
        result = kelly.recommend_bet_size(
            win_probability=args.win_prob,
            odds=args.odds,
            bankroll=args.bankroll,
            confidence=args.confidence
        )
    else:
        result = kelly.calculate_kelly(
            win_probability=args.win_prob,
            odds=args.odds,
            bankroll=args.bankroll,
            fraction=args.fraction
        )

    if result is None:
        print("\n❌ NO BET - Insufficient edge or negative expectation")
        print("\nEither:")
        print("1. Your win probability is too low for these odds")
        print("2. The edge is less than minimum threshold (2%)")
        print("\nDon't bet! The math says this is -EV.\n")
        return

    print("\n" + "=" * 70)
    print("🎯 KELLY CRITERION BET SIZING")
    print("=" * 70)
    print(f"\nOdds:           {args.odds:+d}")
    print(f"Win Prob:       {format_percentage(args.win_prob * 100)}")
    print(f"Bankroll:       {format_currency(args.bankroll)}")
    print(f"Kelly Fraction: {result.fraction_used}× (")
    if result.fraction_used == 1.0:
        print("Full Kelly)")
    elif result.fraction_used == 0.5:
        print("Half Kelly)")
    elif result.fraction_used == 0.25:
        print("Quarter Kelly - RECOMMENDED)")
    else:
        print(f"{int(result.fraction_used * 100)}% Kelly)")

    print("\n" + "─" * 70)
    print("📊 RECOMMENDED BET SIZE")
    print("─" * 70)
    print(f"Full Kelly:     {format_currency(result.full_kelly_amount)} ({format_percentage(result.full_kelly_pct)} of bankroll)")
    print(f"Fractional:     {format_currency(result.fractional_kelly_amount)} ({format_percentage(result.fractional_kelly_pct)} of bankroll)")
    print(f"\n→ BET: {format_currency(result.fractional_kelly_amount)}")

    print("\n" + "─" * 70)
    print("💰 EXPECTED VALUE")
    print("─" * 70)
    print(f"EV per bet:     {format_currency(result.expected_value)}")
    print(f"ROI:            {format_percentage(result.roi)}")
    print(f"Variance Reduction: {format_percentage(result.variance_reduction)} vs Full Kelly")

    print("\n" + "─" * 70)
    print("💡 INTERPRETATION")
    print("─" * 70)

    if result.fractional_kelly_pct < 1:
        print("✅ Conservative bet size - Low risk")
    elif result.fractional_kelly_pct < 3:
        print("✅ Moderate bet size - Optimal risk/reward")
    elif result.fractional_kelly_pct < 5:
        print("⚠️  Aggressive bet size - Higher variance")
    else:
        print("🚨 Very aggressive bet - Consider reducing")

    print(f"\nExpected to win {format_currency(result.expected_value)} on this bet")
    print(f"Over 100 similar bets: ~{format_currency(result.expected_value * 100)} profit")

    print("\n" + "=" * 70)
    print()


if __name__ == "__main__":
    main()
