#!/usr/bin/env python3
"""
Kelly Criterion Bet Sizing Calculator for NFL
Calculates optimal bet sizes based on edge and bankroll
"""
import argparse
import json
from typing import Dict, List


class KellyCalculator:
    """
    Kelly Criterion calculator for sports betting

    Formula: Kelly % = (bp - q) / b
    Where:
        b = decimal odds - 1 (e.g., -110 = 1.909, so b = 0.909)
        p = probability of winning
        q = probability of losing (1 - p)

    Fractional Kelly: Use 0.25-0.5 of full Kelly for safety
    """

    def __init__(self, bankroll: float, fraction: float = 0.25):
        """
        Initialize calculator

        Args:
            bankroll: Total bankroll in dollars
            fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
        """
        self.bankroll = bankroll
        self.fraction = fraction

    def american_to_decimal(self, odds: int) -> float:
        """
        Convert American odds to decimal

        Args:
            odds: American odds (e.g., -110, +150)

        Returns:
            Decimal odds
        """
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1

    def calculate_kelly(self, win_prob: float, odds: int) -> Dict:
        """
        Calculate Kelly bet size

        Args:
            win_prob: Probability of winning (0.0 to 1.0)
            odds: American odds (e.g., -110)

        Returns:
            Dictionary with bet sizing info
        """
        # Convert odds to decimal
        decimal_odds = self.american_to_decimal(odds)
        b = decimal_odds - 1

        # Kelly calculation
        p = win_prob
        q = 1 - p

        # Full Kelly
        kelly_pct = (b * p - q) / b

        # Don't bet if no edge
        if kelly_pct <= 0:
            return {
                'bet_size': 0,
                'kelly_pct': 0,
                'recommendation': 'NO BET - No edge',
                'edge': ((win_prob * decimal_odds) - 1) * 100
            }

        # Apply fractional Kelly for safety
        fractional_kelly = kelly_pct * self.fraction

        # Calculate bet size
        bet_size = self.bankroll * fractional_kelly

        # Calculate edge
        edge = ((win_prob * decimal_odds) - 1) * 100

        return {
            'bet_size': round(bet_size, 2),
            'kelly_pct': round(kelly_pct * 100, 2),
            'fractional_kelly_pct': round(fractional_kelly * 100, 2),
            'edge': round(edge, 2),
            'recommendation': self._get_recommendation(fractional_kelly, edge),
            'max_loss': round(bet_size, 2),
            'potential_profit': round(bet_size * b, 2)
        }

    def _get_recommendation(self, kelly_pct: float, edge: float) -> str:
        """
        Get betting recommendation based on Kelly % and edge

        Args:
            kelly_pct: Fractional Kelly percentage
            edge: Edge percentage

        Returns:
            Recommendation string
        """
        if edge < 1.0:
            return "NO BET - Edge too small"
        elif edge < 2.5:
            return "MARGINAL - Consider passing"
        elif kelly_pct < 0.005:  # Less than 0.5% of bankroll
            return "SMALL BET - Minimal edge"
        elif kelly_pct < 0.015:  # 0.5% to 1.5% of bankroll
            return "MEDIUM BET - Decent edge"
        else:
            return "STRONG BET - Significant edge"

    def calculate_multiple_bets(self, bets: List[Dict]) -> List[Dict]:
        """
        Calculate bet sizes for multiple games

        Args:
            bets: List of dicts with 'game', 'win_prob', 'odds'

        Returns:
            List of bet calculations
        """
        results = []

        for bet in bets:
            calc = self.calculate_kelly(
                win_prob=bet['win_prob'],
                odds=bet['odds']
            )

            results.append({
                'game': bet.get('game', 'Unknown'),
                'pick': bet.get('pick', ''),
                'win_prob': bet['win_prob'],
                'odds': bet['odds'],
                **calc
            })

        # Sort by edge (highest first)
        results.sort(key=lambda x: x['edge'], reverse=True)

        return results

    def print_results(self, results: List[Dict]):
        """Print bet sizing results in readable format"""
        print("\n" + "="*80)
        print("🎯 KELLY CRITERION BET SIZING")
        print("="*80)
        print(f"\nBankroll: ${self.bankroll:.2f}")
        print(f"Kelly Fraction: {self.fraction} (fractional Kelly for safety)")
        print("\n" + "-"*80)

        total_risk = 0

        for i, bet in enumerate(results, 1):
            if bet['bet_size'] == 0:
                continue

            print(f"\n{i}. {bet['game']}")
            print(f"   Pick: {bet['pick']}")
            print(f"   Odds: {bet['odds']:+d}")
            print(f"   Win Probability: {bet['win_prob']*100:.1f}%")
            print(f"   Edge: {bet['edge']:.2f}%")
            print(f"   Full Kelly: {bet['kelly_pct']:.2f}%")
            print(f"   Fractional Kelly: {bet['fractional_kelly_pct']:.2f}%")
            print(f"   💰 BET SIZE: ${bet['bet_size']:.2f}")
            print(f"   Max Loss: ${bet['max_loss']:.2f}")
            print(f"   Potential Profit: ${bet['potential_profit']:.2f}")
            print(f"   ⚡ {bet['recommendation']}")

            total_risk += bet['bet_size']

        print("\n" + "-"*80)
        print(f"📊 TOTAL RISK: ${total_risk:.2f} ({total_risk/self.bankroll*100:.1f}% of bankroll)")
        print(f"💼 REMAINING: ${self.bankroll - total_risk:.2f}")
        print("="*80 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Kelly Criterion Bet Sizing Calculator'
    )
    parser.add_argument(
        '--bankroll',
        type=float,
        default=20.0,
        help='Total bankroll (default: $20 for NFL)'
    )
    parser.add_argument(
        '--fraction',
        type=float,
        default=0.25,
        help='Kelly fraction (0.25 = quarter Kelly, default)'
    )
    parser.add_argument(
        '--picks',
        type=str,
        help='JSON file with picks (optional)'
    )

    args = parser.parse_args()

    calculator = KellyCalculator(
        bankroll=args.bankroll,
        fraction=args.fraction
    )

    # Sample bets for demo
    sample_bets = [
        {
            'game': 'Chiefs vs Bills',
            'pick': 'Chiefs -3',
            'win_prob': 0.67,  # 67% confidence
            'odds': -110
        },
        {
            'game': 'Eagles vs Cowboys',
            'pick': 'Eagles -7',
            'win_prob': 0.65,  # 65% confidence
            'odds': -110
        },
        {
            'game': '49ers vs Seahawks',
            'pick': 'Over 47.5',
            'win_prob': 0.70,  # 70% confidence
            'odds': -105
        },
        {
            'game': 'Packers vs Bears',
            'pick': 'Packers -6',
            'win_prob': 0.60,  # 60% confidence - too low
            'odds': -110
        }
    ]

    # Load from file if provided
    if args.picks:
        try:
            with open(args.picks, 'r') as f:
                bets = json.load(f)
        except FileNotFoundError:
            print(f"❌ File not found: {args.picks}")
            print("📝 Using sample bets for demo\n")
            bets = sample_bets
    else:
        print("📝 No picks file provided. Using sample bets for demo\n")
        bets = sample_bets

    # Calculate bet sizes
    results = calculator.calculate_multiple_bets(bets)

    # Print results
    calculator.print_results(results)

    # Print usage instructions
    print("💡 USAGE:")
    print("   1. Create picks.json with your game selections")
    print("   2. Run: python3 kelly_calculator.py --bankroll 20 --picks picks.json")
    print("   3. Use recommended bet sizes")
    print("\n📋 picks.json format:")
    print("""   [
     {
       "game": "Team A vs Team B",
       "pick": "Team A -3",
       "win_prob": 0.67,
       "odds": -110
     }
   ]
""")


if __name__ == "__main__":
    main()
